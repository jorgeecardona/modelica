"""The tape against the emitted Python, on every corpus model, in Python and in the Rust engine.

The differential style of `tests/test_rustgen.py`: two implementations of one `System`, the same
inputs at the model's start point, the answers compared. `vm.evaluate` is checked on every
model; the Rust engine (``engine/src/lib.rs``) is compiled once with ``rustc`` and checked on the
same calls, and skipped when there is no ``rustc``.
"""

from __future__ import annotations

import functools
import shutil
import tomllib
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import vm
from modelica.sim.emit import Code, EmitError, emit

if TYPE_CHECKING:
    from modelica.ir.system import System

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
MODELS = sorted((EXAMPLES / "models").glob("*.mo"))
COMPILER = shutil.which("rustc")


@functools.cache
def _system(model: Path) -> System:
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    flatten = pytest.importorskip("modelica.sim").flatten
    return causalize(flatten(sources, model.stem))


@functools.cache
def _models() -> tuple[Path, ...]:
    """The corpus models the interpreter emits; the others are not the tape's to answer."""
    out: list[Path] = []
    for model in MODELS:
        try:
            emit(_system(model))
        except Exception:  # noqa: BLE001, S112 - any model the emitter cannot write is out of scope
            continue
        out.append(model)
    return tuple(out)


Kernel = Any  # a function of the emitted module


class _Point:
    """A model's start point, read from the emitted module, and the reference answers."""

    def __init__(self, code: Code) -> None:
        self.module = types.ModuleType(code.name)
        exec(code.source, self.module.__dict__)  # noqa: S102 - the emitter's own output
        self.p = self.module.parameters(np.zeros(len(code.parameters)))
        self.y = np.zeros(len(code.states))
        self.w = np.zeros(code.workspace)
        self.module.initial(self.p, self.y, self.w)
        self.module.latch(0.0, self.y, self.p, self.w)
        self.code = code

    def _jacobian(self, t: float, y: Any, p: Any, w: Any, out: Any) -> None:
        """The emitted ``jacobian`` writes rows; the tape writes the same numbers row-major."""
        n = len(y)
        rows = np.zeros((n, n))
        self.module.jacobian(t, y, p, w, rows)
        out[:] = rows.reshape(-1)

    def calls(self) -> list[tuple[str, int, int | None, int, Kernel, tuple[Any, ...]]]:
        """(label, which, then, length, reference, extra args) for every kernel call checked."""
        code = self.code
        n, events = len(code.states), len(code.events)
        found: list[tuple[str, int, int | None, int, Kernel, tuple[Any, ...]]] = [
            ("rhs", 0, None, n, self.module.rhs, ()),
            ("observe", 1, None, len(code.observed), self.module.observe, ()),
            ("jacobian", 4, None, n * n, self._jacobian, ()),
        ]
        if events:
            found.append(("indicators", 2, None, events, self.module.indicators, ()))
            found.extend(
                (f"transition {e}", 3, e, n, self.module.transition, (e,)) for e in range(events)
            )
        return found


def _reference(
    point: _Point, kernel: Kernel, extra: tuple[Any, ...], length: int
) -> tuple[Any, Any]:
    w = point.w.copy()
    out = np.zeros(length)
    kernel(*extra, 0.0, point.y, point.p, w, out)
    return out, w


def _tol(label: str) -> float:
    """Exact to rounding, except the Jacobian: forward mode orders the products differently."""
    return 1e-9 if label == "jacobian" else 1e-12


def _index(encoded: vm.Encoded, which: int, mode: int) -> int:
    pick = (
        encoded.rhs,
        encoded.observe,
        encoded.indicators,
        encoded.transition_prefix,
        encoded.jacobian,
    )
    return pick[which](mode)


@pytest.mark.parametrize("model", _models(), ids=[m.stem for m in _models()])
def test_the_tape_equals_the_interpreter(model: Path) -> None:
    """``rhs``, ``observe``, ``indicators`` and every ``transition`` agree at the start point."""
    system = _system(model)
    point = _Point(emit(system))
    program = vm.lower(system)
    mode = vm.mode(program, point.w)
    groups = (
        program.rhs,
        program.observe,
        program.indicators,
        program.transition_prefix,
        program.jacobian,
    )
    for label, which, event, length, kernel, extra in point.calls():
        expected, expected_w = _reference(point, kernel, extra, length)
        out, w = [0.0] * length, list(point.w)
        then = None if event is None else program.transition[mode][event]
        vm.evaluate(
            program, groups[which][mode], 0.0, list(point.y), list(point.p), w, out, then=then
        )
        np.testing.assert_allclose(out, expected, rtol=_tol(label), atol=_tol(label), err_msg=label)
        np.testing.assert_allclose(w, expected_w, rtol=1e-12, atol=1e-12, err_msg=label)


def test_an_unknown_function_is_refused() -> None:
    """A call the tape has no builtin for raises `EmitError`, not a wrong number."""
    from modelica import ir  # noqa: PLC0415 - only this test builds nodes by hand

    lowering = vm._Lowering(vm.Emitter(_system(_models()[0])))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    scope = vm._Scope([], {}, {}, [])  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(EmitError, match="frobnicate"):
        lowering.value(scope, ir.Call("frobnicate", (ir.RealLiteral(1.0),)))


# -- the Rust engine ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The wheel's engine library, else one built once from ``engine/src`` into a temporary cache."""
    from modelica.sim.engine import installed, library  # noqa: PLC0415 - only the engine tests

    if (bundled := installed()) is not None:
        return bundled
    if COMPILER is None:
        pytest.skip("no engine in this install and no rustc to build one")
    return library(tmp_path_factory.mktemp("engine"))


@pytest.mark.parametrize("model", _models(), ids=[m.stem for m in _models()])
def test_the_engine_equals_the_interpreter(engine: Path, model: Path) -> None:
    """The same calls as `test_the_tape_equals_the_interpreter`, run by the Rust engine."""
    from modelica.sim.engine import load  # noqa: PLC0415 - only the engine tests need it

    system = _system(model)
    point = _Point(emit(system))
    program = vm.lower(system)
    loaded = load(program, engine)
    try:
        encoded = loaded.encoded
        mode = vm.mode(program, point.w)
        for label, which, event, length, kernel, extra in point.calls():
            expected, expected_w = _reference(point, kernel, extra, length)
            w, out = point.w.copy(), np.zeros(length)
            then = -1 if event is None else encoded.transition(mode, event)
            p = np.asarray(point.p, dtype=np.float64)
            loaded.call(_index(encoded, which, mode), 0.0, point.y.copy(), p, w, out, then)
            tolerance = _tol(label)
            np.testing.assert_allclose(out, expected, rtol=tolerance, atol=tolerance, err_msg=label)
            np.testing.assert_allclose(w, expected_w, rtol=1e-12, atol=1e-12, err_msg=label)
    finally:
        loaded.close()


@pytest.mark.parametrize("model", _models(), ids=[m.stem for m in _models()])
def test_the_engine_equals_the_tape_at_random_points(engine: Path, model: Path) -> None:
    """Random states, times, latch bits and every mode: Rust and the Python reference agree.

    The start point never leaves mode 0, never flips a latch, and always converges its Newton
    blocks; these points do all three, and reach branches a SELECT discards (NaN, inf).
    """
    from modelica.sim.engine import load  # noqa: PLC0415 - only the engine tests need it

    system = _system(model)
    code = emit(system)
    program = vm.lower(system)
    loaded = load(program, engine)
    encoded = loaded.encoded
    rng = np.random.default_rng(len(model.stem))
    n, events = len(code.states), len(code.events)
    point = _Point(code)
    for _ in range(8):
        y = point.y + rng.normal(0.0, 1.0, n)
        t = float(rng.uniform(0.0, 10.0))
        w = point.w.copy()
        for trigger in code.events:
            if trigger.latch is not None:
                w[trigger.latch] = float(rng.integers(0, 2))
        for mode in range(encoded.modes):
            if program.switching is not None:
                w[program.switching] = float(mode)
            calls: list[tuple[int, vm.Function, vm.Function | None, int, int]] = [
                (encoded.rhs(mode), program.rhs[mode], None, -1, n),
                (encoded.observe(mode), program.observe[mode], None, -1, len(code.observed)),
                (encoded.jacobian(mode), program.jacobian[mode], None, -1, n * n),
            ]
            if events:
                calls.append((encoded.indicators(mode), program.indicators[mode], None, -1, events))
                calls.extend(
                    (
                        encoded.transition_prefix(mode),
                        program.transition_prefix[mode],
                        program.transition[mode][e],
                        encoded.transition(mode, e),
                        n,
                    )
                    for e in range(events)
                )
            for index, function, then, then_index, length in calls:
                expected, expected_w = [0.0] * length, list(w)
                vm.evaluate(
                    program, function, t, list(y), list(point.p), expected_w, expected, then=then
                )
                out, got_w = np.zeros(length), w.copy()
                loaded.call(
                    index, t, y, np.asarray(point.p, dtype=np.float64), got_w, out, then_index
                )
                np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12, equal_nan=True)
                np.testing.assert_allclose(got_w, expected_w, rtol=1e-9, atol=1e-12, equal_nan=True)
    loaded.close()


_TRAJECTORY_MODELS = ("BouncingBall", "Lorenz", "Pendulum", "PhaseChange", "BumpStop", "TwoTank")


@pytest.mark.slow  # compiles each model with rustsim for the comparison
@pytest.mark.parametrize("name", _TRAJECTORY_MODELS)
def test_the_engine_integrates_like_rustsim(engine: Path, name: str) -> None:
    """A whole run: the same Radau runtime on the tape and on compiled code, step for step."""
    from modelica.sim import engine as native  # noqa: PLC0415 - only the engine tests need it
    from modelica.sim import rustsim  # noqa: PLC0415

    model = EXAMPLES / "models" / f"{name}.mo"
    system = _system(model)
    compiled = rustsim.simulate(system, stop=2.0, points=40)
    tape = native.simulate(system, stop=2.0, points=40, path=engine)
    assert tape.event_list == compiled.event_list
    assert tape.steps == compiled.steps
    np.testing.assert_allclose(tape.result.values, compiled.result.values, rtol=1e-9, atol=1e-12)


def test_the_engine_runtime_is_the_rustsim_runtime() -> None:
    """``engine/src/radau.rs`` is `rust_radau`'s text; rewrite it with ``vm.write_runtime()``."""
    written = (ROOT / "engine" / "src" / "radau.rs").read_text(encoding="utf-8")
    assert written == vm.runtime()


@pytest.mark.parametrize("model", _models(), ids=[m.stem for m in _models()])
def test_the_engine_starts_where_the_interpreter_does(engine: Path, model: Path) -> None:
    """``parameters``, ``initial`` and ``latch`` on the tape give `Kernel.start`'s three vectors."""
    from modelica.sim.engine import load  # noqa: PLC0415 - only the engine tests need it
    from modelica.sim.kernel import build  # noqa: PLC0415

    system = _system(model)
    expected = build(system).start(0.0)
    loaded = load(vm.lower(system), engine)
    try:
        for got, want in zip(loaded.start(), expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
        p = expected[0] * 1.5
        overridden = build(system).start(0.0, p)
        for got, want in zip(loaded.start(p), overridden, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    finally:
        loaded.close()


_GRIDS = {
    "points": {"points": 41},
    "times": {"times": np.linspace(0.05, 1.95, 20)},
    "steps": {},
}


@pytest.mark.parametrize("grid", list(_GRIDS))
@pytest.mark.parametrize("name", ["BouncingBall", "RCCircuit", "Pendulum"])
def test_simulate_on_the_engine_agrees_with_scipy(engine: Path, name: str, grid: str) -> None:
    """The engine returns the rows SciPy returns: grid, start, stop and both sides of events."""
    from modelica.sim import engine as native  # noqa: PLC0415 - only the engine tests need it
    from modelica.sim.solve import simulate  # noqa: PLC0415

    system = _system(EXAMPLES / "models" / f"{name}.mo")
    knobs: dict[str, Any] = {"stop": 2.0, "tolerance": 1e-8, **_GRIDS[grid]}
    scipy = simulate(system, backend="python", **knobs)
    ran = native.result(system, path=engine, **knobs)
    assert (scipy.backend, ran.backend) == ("python", "engine")
    assert ran.names == scipy.names
    assert [e.index for e in ran.events] == [e.index for e in scipy.events]
    assert (ran.time[0], ran.time[-1]) == (scipy.time[0], scipy.time[-1])
    for event in ran.events:
        assert np.count_nonzero(np.isclose(ran.time, event.time, rtol=0, atol=1e-12)) >= 2
    if grid != "steps":
        assert len(ran.time) == len(scipy.time)
        np.testing.assert_allclose(ran.time, scipy.time, rtol=0, atol=1e-7)
    # Compare away from the events, which the two locate a few 1e-9 apart. Without a grid,
    # SciPy is asked for the engine's own step times: interpolating between steps is not exact.
    sample = np.linspace(0.0, 2.0, 41) if grid != "steps" else np.unique(ran.time)
    for event in ran.events:
        sample = sample[np.abs(sample - event.time) > 1e-3]
    if grid == "steps":
        scipy = simulate(system, backend="python", stop=2.0, tolerance=1e-8, times=sample)
    for variable in ran.names:
        np.testing.assert_allclose(
            _between(ran, variable, sample),
            _between(scipy, variable, sample),
            rtol=1e-4,
            atol=1e-6,
            err_msg=variable,
        )


def _between(result: Any, variable: str, sample: Any) -> Any:
    """``variable`` at ``sample``, read on the side of each duplicated instant before it."""
    time = result.time
    keep = np.concatenate([np.diff(time) > 0, [True]])
    return np.interp(sample, time[keep], np.asarray(result[variable])[keep])


def test_auto_falls_back_to_scipy_for_another_method() -> None:
    """``backend="auto"`` with a method the engine does not have runs on SciPy, and says so."""
    from modelica.sim.solve import simulate  # noqa: PLC0415

    system = _system(EXAMPLES / "models" / "RCCircuit.mo")
    assert simulate(system, stop=0.1, method="LSODA").backend == "python"


def test_a_shared_name_used_after_a_loop_is_defined_again_after_it() -> None:
    """The value from inside a Newton loop is from before its last step, so it is recomputed."""
    from modelica.sim.emit import _define  # noqa: PLC0415 # pyright: ignore[reportPrivateUsage]

    lines = [
        "for _ in range(50):",
        "    step = f(x) / __s0",
        "    x = x + step",
        "w[3] = x",
        "vx = 1.0 / __s0",
    ]
    assert _define(lines, "__s0", "__s0 = 2.0 * x") == [
        "for _ in range(50):",
        "    __s0 = 2.0 * x",
        "    step = f(x) / __s0",
        "    x = x + step",
        "w[3] = x",
        "__s0 = 2.0 * x",
        "vx = 1.0 / __s0",
    ]


@pytest.mark.parametrize("name", ["Pendulum", "TwoTank"])
def test_the_emitted_rhs_equals_the_tape_away_from_the_start(name: str) -> None:
    """At random points near the start, Python and the tape agree after the Newton blocks."""
    system = _system(EXAMPLES / "models" / f"{name}.mo")
    point = _Point(emit(system))
    program = vm.lower(system)
    mode = vm.mode(program, point.w)
    rng = np.random.default_rng(7)
    n = len(point.y)
    for _ in range(30):
        y = point.y * (1.0 + rng.uniform(-0.2, 0.2, size=n)) + rng.normal(scale=0.01, size=n)
        expected = np.zeros(n)
        point.module.rhs(0.0, y, point.p, point.w.copy(), expected)
        out = [0.0] * n
        vm.evaluate(program, program.rhs[mode], 0.0, list(y), list(point.p), list(point.w), out)
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)
