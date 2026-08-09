"""The C unit against the Python one, on every corpus model that has both.

The differential style `tests/test_jit.py` established: two implementations of the same
`System`, the same inputs, and the answers compared. No reference numbers of the C
emitter's own -- a number this file invented would only ever prove that it still matches
itself.

`cc` is optional exactly as Numba is: no compiler, no test. What is *not* optional is the
list of models. Every corpus model either produces a unit and is checked, or is refused by
`cgen` with a reason, and `test_every_model_is_accounted_for` holds that there is no third
outcome -- a model that silently produced nothing would be a hole in this file rather than
a finding.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim.cgen import FUNCTIONS, unit
from modelica.sim.emit import EmitError

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MODELS = sorted((EXAMPLES / "models").glob("*.mo"))
COMPILER = shutil.which("cc") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(COMPILER is None, reason="no C compiler on this machine")

#: What `cgen` says it cannot write yet, and why. Listed here so that a model moving off it
#: -- because somebody wrote the missing C -- fails this file and gets its check turned on,
#: rather than staying quietly untested.
REFUSED: dict[str, str] = {}
"""Empty, and kept as the shape that would hold the next gap.

Three models passed through it and out. `AtwoodPendulum` left when the dense solve landed --
its 4x4 tears to a 2x2 that is *linear*, so it wanted `md_solve` and not Newton.
`TwoInertias` left with the mode dispatcher. `Pendulum` left with the iteration: it has a
second state set *and* a torn 1x1 that is nonlinear in the rod force, so it needed both.
"""


def _system(model: Path) -> System:
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    flatten = pytest.importorskip("modelica.sim").flatten
    return causalize(flatten(sources, model.stem))


def _library(system: System, tmp_path: Path) -> Any:
    """Compile the generated unit into a shared object and load it."""
    source = tmp_path / f"{system.name}.c"
    source.write_text(unit(system), encoding="utf-8")
    shared = tmp_path / f"{system.name}.so"
    assert COMPILER is not None
    subprocess.run(  # noqa: S603 - compiler from PATH, input generated here
        [
            COMPILER,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Werror",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(shared),
            "-lm",
        ],
        check=True,
        capture_output=True,
    )
    loaded = ctypes.CDLL(str(shared))
    for name in FUNCTIONS:
        getattr(loaded, name).restype = None
    return loaded


@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_every_model_is_accounted_for(model: Path) -> None:
    """A unit, or a refusal that names what is missing. There is no third answer."""
    if model.stem in REFUSED:
        with pytest.raises(EmitError, match=REFUSED[model.stem]):
            unit(_system(model))
        return
    assert unit(_system(model)).startswith("/* Generated from")


@needs_cc
@pytest.mark.parametrize(
    "model",
    [m for m in MODELS if m.stem not in REFUSED],
    ids=[m.stem for m in MODELS if m.stem not in REFUSED],
)
def test_the_c_right_hand_side_answers_what_the_python_one_answers(
    model: Path, tmp_path: Path
) -> None:
    """Same states, same parameters, same workspace: the derivative has to be the same.

    Taken at the model's own start point, which is the one place every model is guaranteed
    to be in a state it was designed to be in -- a random vector would compare two functions
    outside the domain either was written for.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)

    wanted = numpy.zeros(len(states))
    kernel.rhs(0.0, states, parameters, workspace, wanted)
    found = numpy.zeros_like(wanted)
    library = _library(system, tmp_path)
    library.rhs(
        ctypes.c_double(0.0),
        states.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        found.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert found == pytest.approx(wanted, rel=1e-12, abs=1e-12)


@needs_cc
@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_the_c_indicators_answer_what_the_python_ones_answer(model: Path, tmp_path: Path) -> None:
    """Every zero-crossing function, at the model's start point.

    A model with no events has an empty array and the check is that both agree it is empty,
    which is worth running: an emitter that wrote one indicator too many would pass every
    other test in this file.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)

    wanted = numpy.zeros(len(kernel.code.events))
    kernel.indicators(0.0, states, parameters, workspace, wanted)
    found = numpy.zeros_like(wanted)
    library = _library(system, tmp_path)
    library.indicators(
        ctypes.c_double(0.0),
        states.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        found.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert found == pytest.approx(wanted, rel=1e-12, abs=1e-12)


@needs_cc
def test_the_c_observation_answers_what_the_python_one_answers(tmp_path: Path) -> None:
    """One model is enough for `observe`: it is `rhs` plus the aliases, in the same order.

    `RoastingMachine` is the one chosen, because it has the most observed variables in the
    corpus and its schedule puts a latched relation in the middle of them.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(EXAMPLES / "models" / "RoastingMachine.mo")
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)

    wanted = numpy.zeros(len(kernel.code.observed))
    kernel.observe(0.0, states, parameters, workspace, wanted)
    found = numpy.zeros_like(wanted)
    library = _library(system, tmp_path)
    library.observe(
        ctypes.c_double(0.0),
        states.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        found.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert found == pytest.approx(wanted, rel=1e-12, abs=1e-12)
