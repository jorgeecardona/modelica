"""The Rust unit against the Python one, on every corpus model rustgen can write.

The same differential style as `tests/test_cgen.py`: two implementations of one `System`, the
same inputs, the answers compared. No reference numbers of the Rust emitter's own. Where the
C test compiles a shared object and calls it through ctypes, this one compiles a `main` that
computes the right-hand side at the model's start point and prints it -- the pattern
`tests/test_rust_dialect.py` already uses, which keeps the first increment off the C ABI.

Every corpus model either produces a unit and is checked, or is refused by `rustgen` with a
reason. `test_every_model_is_accounted_for` holds that there is no third outcome.

`rustc` is optional exactly as `cc` is in `test_cgen`: no compiler, no test.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim.emit import EmitError
from modelica.sim.rustgen import unit

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MODELS = sorted((EXAMPLES / "models").glob("*.mo"))
COMPILER = shutil.which("rustc")
needs_rustc = pytest.mark.skipif(COMPILER is None, reason="no rustc on this machine")


def _system(model: Path) -> System:
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    flatten = pytest.importorskip("modelica.sim").flatten
    return causalize(flatten(sources, model.stem))


def _supported() -> list[Path]:
    """The models rustgen writes today, decided by trying rather than by a hand list."""
    out: list[Path] = []
    for model in MODELS:
        try:
            unit(_system(model))
        except EmitError:
            continue
        out.append(model)
    return out


def _floats(values: object) -> str:
    return ", ".join(f"{float(v)!r}f64" for v in values)  # type: ignore[union-attr]


def _call_in_rust(
    system: System, name: str, out_len: int, start: tuple[object, object, object]
) -> list[float]:
    """Emit the unit, wrap it in a main that runs one function at the start point, compile, run.

    ``name`` is the generated function to call (``rhs`` or ``observe``), ``out_len`` is the
    length of the output vector it fills -- the states for ``rhs``, the observed variables for
    ``observe`` -- and ``start`` is ``(states, parameters, workspace)``, the shared workspace the
    two functions read and differ only in that last argument.
    """
    numpy = pytest.importorskip("numpy")
    states, parameters, workspace = start
    n = len(numpy.asarray(states))
    program = "\n".join(
        [
            unit(system),
            "fn main() {",
            f"    let y: [f64; {n}] = [{_floats(states)}];",
            f"    let p: [f64; {len(numpy.asarray(parameters))}] = [{_floats(parameters)}];",
            f"    let mut w: [f64; {len(numpy.asarray(workspace))}] = [{_floats(workspace)}];",
            f"    let mut out = [0.0f64; {out_len}];",
            f"    {name}(0.0, &y, &p, &mut w, &mut out);",
            "    let printed: Vec<String> = out.iter().map(|v| v.to_string()).collect();",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    return _compile_and_run(program, system.name)


def _compile_and_run(program: str, name: str) -> list[float]:
    assert COMPILER is not None
    with tempfile.TemporaryDirectory() as tmp:
        unit_path = Path(tmp) / f"{name}.rs"
        unit_path.write_text(program, encoding="utf-8")
        binary = Path(tmp) / name
        subprocess.run(  # noqa: S603 - compiler from PATH, input generated here
            [
                COMPILER,
                "--edition",
                "2021",
                "-O",
                "-D",
                "warnings",
                str(unit_path),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
        )
        done = subprocess.run([str(binary)], check=True, capture_output=True, text=True)  # noqa: S603
    return [float(token) for token in done.stdout.split()]


@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_every_model_is_accounted_for(model: Path) -> None:
    """A unit, or an EmitError that names what is missing. There is no third answer."""
    try:
        source = unit(_system(model))
    except EmitError as error:
        assert str(error).startswith("rust:")  # noqa: PT017 - asserting the message shape is the point
        return
    assert source.startswith("// Generated from")


@needs_rustc
@pytest.mark.parametrize("model", _supported(), ids=[m.stem for m in _supported()])
def test_the_rust_right_hand_side_answers_what_the_python_one_answers(model: Path) -> None:
    """Same states, same parameters, same workspace: the derivative has to be the same.

    Taken at the model's own start point, the one place every model is in a state it was
    designed to be in.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)

    wanted = numpy.zeros(len(states))
    kernel.rhs(0.0, states, parameters, workspace, wanted)
    found = _call_in_rust(system, "rhs", len(states), (states, parameters, workspace))
    assert found == pytest.approx(list(wanted), rel=1e-12, abs=1e-12)


@needs_rustc
@pytest.mark.parametrize("model", _supported(), ids=[m.stem for m in _supported()])
def test_the_rust_observation_answers_what_the_python_one_answers(model: Path) -> None:
    """Every observed variable, at the start point: `rhs`'s blocks plus the aliases, in order."""
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)

    wanted = numpy.zeros(len(kernel.code.observed))
    kernel.observe(0.0, states, parameters, workspace, wanted)
    found = _call_in_rust(
        system, "observe", len(kernel.code.observed), (states, parameters, workspace)
    )
    assert found == pytest.approx(list(wanted), rel=1e-12, abs=1e-12)


@needs_rustc
@pytest.mark.parametrize("model", _supported(), ids=[m.stem for m in _supported()])
def test_the_rust_indicators_answer_what_the_python_ones_answer(model: Path) -> None:
    """Every zero-crossing function at the start point.

    A supported model has no events, so both tiers produce an empty array -- worth running,
    because an emitter that wrote one indicator too many would pass every other test here.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)

    wanted = numpy.zeros(len(kernel.code.events))
    kernel.indicators(0.0, states, parameters, workspace, wanted)
    found = _call_in_rust(
        system, "indicators", len(kernel.code.events), (states, parameters, workspace)
    )
    assert found == pytest.approx(list(wanted), rel=1e-12, abs=1e-12)


@needs_rustc
@pytest.mark.parametrize("model", _supported(), ids=[m.stem for m in _supported()])
def test_the_rust_jacobian_answers_what_the_python_one_answers(model: Path) -> None:
    """The whole n-by-n matrix, entry by entry, at the start point.

    Exact to 1e-12: this is symbolic differentiation on both sides, not a difference quotient,
    so a disagreement is a wrong derivative and never a rounding one. ``equal_nan`` because a
    model can start where its Jacobian is genuinely singular; both tiers must be undefined in
    the same entries and equal in all the others. The supported models have one state set and
    no switch, so the start point is the one place to compare.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)
    width = len(states)

    wanted = numpy.zeros((width, width))
    kernel.jacobian(0.0, states, parameters, workspace, wanted)
    found = _call_in_rust(system, "jacobian", width * width, (states, parameters, workspace))
    numpy.testing.assert_allclose(
        numpy.array(found).reshape(width, width), wanted, rtol=1e-12, atol=1e-12, equal_nan=True
    )
