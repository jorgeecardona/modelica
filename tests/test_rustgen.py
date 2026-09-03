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
from modelica.sim.dialect import RUST
from modelica.sim.emit import EmitError, Emitter
from modelica.sim.rust_radau import CONSTANTS, FIRST_STEP, INTEGRATE, LINALG, NEWTON, STEP
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


def _reference_models() -> list[Path]:
    """Supported models whose .toml records reference points for at least one state.

    The integrator is checked against these: the model's own recorded trajectory, not a number
    the Rust or the Python radau invented. A model whose references name only observed variables
    (not states) is left out, since the integrator returns states.
    """
    out: list[Path] = []
    for model in _supported():
        system = _system(model)
        if Emitter(system, dialect=RUST).events():
            continue  # md_integrate has no event handling yet, so an event model cannot integrate
        spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
        points = [
            point
            for point in spec.get("simulate", {}).get("point", [])
            if point.get("source") == "reference" and "values" in point
        ]
        names = set(system.modes[0].states)
        if any(any(name in names for name in point["values"]) for point in points):
            out.append(model)
    return out


_REFERENCE_MODELS = _reference_models()


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


def _transition_in_rust(
    system: System, event: int, start: tuple[object, object, object]
) -> tuple[list[float], list[float]]:
    """Run ``transition`` for one event at the start point, returning the post-event ``(out, w)``.

    Both are needed, and this is why it is not `_call_in_rust`: a reinit lands in ``out`` (the new
    state) but a latch flip or discrete assignment lands in ``w``, so a test that read only ``out``
    would let every Thermostat-style crossing pass without checking anything it does.
    """
    numpy = pytest.importorskip("numpy")
    states, parameters, workspace = start
    n = len(numpy.asarray(states))
    m = len(numpy.asarray(workspace))
    program = "\n".join(
        [
            unit(system),
            "fn main() {",
            f"    let y: [f64; {n}] = [{_floats(states)}];",
            f"    let p: [f64; {len(numpy.asarray(parameters))}] = [{_floats(parameters)}];",
            f"    let mut w: [f64; {m}] = [{_floats(workspace)}];",
            f"    let mut out = [0.0f64; {n}];",
            f"    transition({event}, 0.0, &y, &p, &mut w, &mut out);",
            "    let printed: Vec<String> = out.iter().chain(w.iter()).map(|v| v.to_string()).collect();",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    both = _compile_and_run(program, system.name)
    return both[:n], both[n:]


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
def test_the_rust_transition_answers_what_the_python_one_answers(model: Path) -> None:
    """The post-event state for every crossing, at the start point.

    A no-event model has an empty event list and both tiers only copy the state through -- still
    run, because a transition that reinitialised a slot it should not have would pass every other
    test here. An event model fires each crossing in turn: BouncingBall's ``reinit(v, -e*pre(v))``
    and Thermostat's latch flip both have to land on the same numbers the interpreter tier does.
    The workspace matters here, not just the last argument: a crossing writes latches into ``w``.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)
    n = len(states)

    for event in range(max(kernel.events, 1)):
        wanted = numpy.zeros(n)
        after = workspace.copy()  # transition writes latches into the workspace in place
        kernel.transition(event, 0.0, states, parameters, after, wanted)
        found_out, found_w = _transition_in_rust(system, event, (states, parameters, workspace))
        assert found_out == pytest.approx(list(wanted), rel=1e-12, abs=1e-12), (
            f"{system.name} event {event} state"
        )
        assert found_w == pytest.approx(list(after), rel=1e-12, abs=1e-12), (
            f"{system.name} event {event} workspace"
        )


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


@needs_rustc
@pytest.mark.parametrize("model", _REFERENCE_MODELS, ids=[m.stem for m in _REFERENCE_MODELS])
def test_the_rust_integrator_reproduces_the_toml_reference(model: Path) -> None:
    """The milestone: emit the model, bundle the radau runtime, integrate, match the .toml.

    The AOT "integrator and model compile together" unit. rustgen emits rhs and jacobian, the
    radau runtime calls them by name, and rustc compiles the whole into one binary that
    integrates from the model's start to its stop. The reference is the model's own recorded
    trajectory in its .toml, at the tolerance (1e-8) that reproduces it to well within the file's
    1e-6 comparison bar -- so this checks the Rust integrator against a number neither it nor the
    Python radau invented. Run over every supported model with state references, so the stiff
    ones (Robertson, VanDerPol) and the implicit tangent (AtwoodPendulum) are exercised too, not
    only the linear RLCCircuit.
    """
    pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    refs = sorted(
        (p for p in spec["simulate"]["point"] if p.get("source") == "reference" and "values" in p),
        key=lambda p: p["t"],
    )
    times = [float(pt["t"]) for pt in refs]

    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)
    names = system.modes[0].states
    n = len(states)
    stop = float(spec["simulate"]["stop"])

    program = "\n".join(
        [
            "#![allow(dead_code)]",
            unit(system),
            *CONSTANTS,
            *LINALG,
            *STEP,
            *NEWTON,
            *FIRST_STEP,
            *INTEGRATE,
            "fn main() {",
            f"    let y0: [f64; {n}] = [{_floats(states)}];",
            f"    let p: [f64; {len(parameters)}] = [{_floats(parameters)}];",
            f"    let mut w: [f64; {len(workspace)}] = [{_floats(workspace)}];",
            f"    let grid: [f64; {len(times)}] = [{_floats(times)}];",
            f"    let mut out = [0.0f64; {len(times) * n}];",
            f"    md_integrate(0.0, {stop!r}f64, &y0, &p, &mut w, 1e-8, 1e-8, &grid, &mut out);",
            "    let printed: Vec<String> = out.iter().map(|v| v.to_string()).collect();",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    out = _compile_and_run(program, system.name)
    for i, point in enumerate(refs):
        for q, name in enumerate(names):
            if name not in point["values"]:
                continue
            assert out[i * n + q] == pytest.approx(point["values"][name], rel=1e-6, abs=1e-9), (
                f"{system.name} t={point['t']} {name}"
            )
