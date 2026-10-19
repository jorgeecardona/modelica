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
import dataclasses
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

# Whole file is slow: it emits and compiles C for every corpus model that has a twin. `make test`
# skips it; CI and `make test-all` run it.
pytestmark = pytest.mark.slow

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
@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_the_c_transition_answers_what_the_python_one_answers(model: Path, tmp_path: Path) -> None:
    """Every event of every model, taken from the start point, compared state by state.

    Taking an event the model is not actually at is the point rather than a flaw: the
    function is a pure map from (event, state) to a new state, and the two implementations
    have to agree on it everywhere, not only where the solver would call it.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    if not kernel.code.events:
        pytest.skip(f"{model.stem} has no events")
    library = _library(system, tmp_path)
    for event in range(len(kernel.code.events)):
        parameters, states, workspace = kernel.start(0.0)
        wanted = numpy.zeros(len(states))
        kernel.transition(event, 0.0, states, parameters, workspace, wanted)
        parameters, states, workspace = kernel.start(0.0)
        found = numpy.zeros(len(states))
        library.transition(
            ctypes.c_int(event),
            ctypes.c_double(0.0),
            states.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            found.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        assert found == pytest.approx(wanted, rel=1e-12, abs=1e-12), f"event {event}"


@needs_cc
@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_the_c_jacobian_answers_what_the_python_one_answers(model: Path, tmp_path: Path) -> None:
    """The whole ``n``-by-``n`` matrix, entry by entry, in every state set the model has.

    Every state set and not only the first, which the other four functions here do not do.
    The reason is `Emitter.rebase`: a model with a second set has a *different* tangent in
    it -- different blocks, different rows, different sparsity -- so checking one set would
    leave the other entirely unexecuted.

    Reached the way a solver reaches it: by taking the event that switches, which rebases
    the states into the new set's coordinates. Setting the slot by hand and leaving the
    states alone was the first attempt and it is not a state the model is ever in --
    `Pendulum`'s second set reads ``y`` as an angle, so Cartesian values in those slots put
    a zero in the pivot and both implementations answered NaN.

    Exact rather than close, to 1e-12: this is symbolic differentiation on both sides of the
    comparison, not a difference quotient, so a disagreement here is a wrong derivative and
    never a rounding one.

    ``equal_nan`` is on, and it is not a way of passing. `Pendulum` starts horizontal, which
    is exactly where its ``(x, vx)`` set is singular -- taking the switch there divides by a
    zero and half the matrix is genuinely undefined. Both implementations have to be
    undefined in the *same entries* and equal in all the others, which is what this checks;
    a C emitter that lost a term and answered ``0.0`` where the interpreter answered NaN
    would fail here.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    library = _library(system, tmp_path)
    switches = [k for k, event in enumerate(kernel.code.events) if event.switch is not None]
    for chosen in [None, *switches]:
        parameters, states, workspace = kernel.start(0.0)
        if chosen is not None:
            moved = numpy.zeros_like(states)
            kernel.transition(chosen, 0.0, states, parameters, workspace, moved)
            states = moved
        width = len(states)
        wanted = numpy.zeros((width, width))
        kernel.jacobian(0.0, states, parameters, workspace, wanted)
        found = numpy.zeros_like(wanted)
        library.jacobian(
            ctypes.c_double(0.0),
            states.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            found.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        numpy.testing.assert_allclose(
            found, wanted, rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=f"after event {chosen}"
        )


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


def _bound(library: Any, kernel: Any) -> Any:
    """The five generated functions, in C, behind the interface the solver already calls.

    `modelica.sim.kernel.Kernel` is a record of callables, so swapping five of them for
    ctypes wrappers gives `simulate` a model whose arithmetic runs in the shared object with
    no change to the solver at all. That is the point of the workspace layout being shared
    rather than reinvented.

    The other four stay Python and are named so rather than glossed: `parameters`,
    `initial`, `latch` and `observe_jacobian` are not in the unit. The first three are
    start-up and the fourth is the output Jacobian, which no solver in this repo calls.

    `numpy.ascontiguousarray` on every input, because `_Run.observed` hands `observe` a
    *column* of the trajectory. A column of a C-order matrix is strided, and passing its
    data pointer to C would read the wrong doubles quietly.
    """
    numpy = pytest.importorskip("numpy")
    pointer = ctypes.POINTER(ctypes.c_double)

    def sent(*arrays: Any) -> list[Any]:
        return [numpy.ascontiguousarray(a).ctypes.data_as(pointer) for a in arrays]

    def call(name: str) -> Any:
        entry = getattr(library, name)

        def run(t: float, y: Any, p: Any, w: Any, out: Any) -> Any:
            entry(ctypes.c_double(t), *sent(y, p, w), out.ctypes.data_as(pointer))
            return out

        return run

    def transition(  # noqa: PLR0913, PLR0917 - the signature is the solver's, not this file's
        event: int, t: float, y: Any, p: Any, w: Any, out: Any
    ) -> Any:
        library.transition(
            ctypes.c_int(event), ctypes.c_double(t), *sent(y, p, w), out.ctypes.data_as(pointer)
        )
        return out

    return dataclasses.replace(
        kernel,
        rhs=call("rhs"),
        jacobian=call("jacobian"),
        observe=call("observe"),
        indicators=call("indicators"),
        transition=transition,
    )


#: One model per feature the C unit had to grow, with how long to run it for. Four rather
#: than the whole corpus because this test integrates: it is the slowest thing in the file,
#: and the entry-point tests above already cover every model at a point.
END_TO_END: list[tuple[str, float]] = [
    ("RLCCircuit", 0.05),  # a plain linear ODE through the whole solver
    ("BouncingBall", 3.0),  # events, and a state reinitialised at each one
    ("RoastingMachine", 600.0),  # latched relations and a schedule of them
    ("Pendulum", 5.0),  # two state sets, a switch between them, and a Newton block
]


@needs_cc
@pytest.mark.parametrize(("name", "stop"), END_TO_END, ids=[n for n, _ in END_TO_END])
def test_the_corpus_simulates_through_the_compiled_unit(
    name: str, stop: float, tmp_path: Path
) -> None:
    """The same simulation, once through the interpreter and once through the shared object.

    Every test above this one compares the two implementations *at a point*. This one lets
    the solver drive: thousands of calls, adaptive steps, event location, and -- for
    `Pendulum` -- a switch of state set in the middle. A disagreement small enough to hide
    in a single evaluation has a whole trajectory to grow in.

    Compared on a fixed grid rather than on the solver's own steps, because the two runs do
    not take the same steps: the arithmetic differs in the last bit, the error estimate
    differs with it, and the step sizes diverge. The interpolated values at requested times
    are what a user reads, and they are what has to agree.

    1e-6 relative, which is the solver's own tolerance and not the emitter's. Holding two
    adaptive runs to 1e-12 would be a test of the integrator's determinism.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    system = _system(EXAMPLES / "models" / f"{name}.mo")
    kernel = build(system)
    grid = numpy.linspace(0.0, stop, 50)

    wanted = simulate(kernel, stop=stop, times=grid)
    found = simulate(_bound(_library(system, tmp_path), kernel), stop=stop, times=grid)

    assert found.names == wanted.names
    # The same events, in the same order, and each located to the solver's own tolerance.
    # A C unit whose indicator was off by a sign would still integrate; it would bounce at a
    # different instant, or not at all, and only this line would notice.
    assert [e.index for e in found.events] == [e.index for e in wanted.events]
    for mine, theirs in zip(found.events, wanted.events, strict=True):
        assert mine.time == pytest.approx(theirs.time, rel=1e-6, abs=1e-6), mine.cause
    for variable in wanted.names:
        numpy.testing.assert_allclose(
            found[variable][: len(grid)],
            wanted[variable][: len(grid)],
            rtol=1e-6,
            atol=1e-6,
            err_msg=variable,
        )
