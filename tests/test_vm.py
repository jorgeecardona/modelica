"""The tape against the emitted Python, on every corpus model, in Python and in the Rust engine.

The differential style of `tests/test_rustgen.py`: two implementations of one `System`, the same
inputs at the model's start point, the answers compared. `vm.evaluate` is checked on every
model; the Rust engine (``engine/src/lib.rs``) is compiled once with ``rustc`` and checked on the
same calls, and skipped when there is no ``rustc``.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
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


#: The position of the jacobian in `_index` and the call groups: it writes the pattern's values.
_JACOBIAN = 4


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
        if which == _JACOBIAN:
            out = vm.dense(program, out, len(point.y))
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
            if which == _JACOBIAN:
                out = np.array(vm.dense(program, list(out), len(point.y)))
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


@pytest.mark.parametrize("name", ["Lorenz", "BouncingBall"])
def test_the_engine_starts_at_start(engine: Path, name: str) -> None:
    """``start = 1``: the same rows and values as SciPy from that instant."""
    from modelica.sim import engine as native  # noqa: PLC0415 - only the engine tests need it
    from modelica.sim.solve import simulate  # noqa: PLC0415

    system = _system(EXAMPLES / "models" / f"{name}.mo")
    knobs: dict[str, Any] = {"start": 1.0, "stop": 1.5, "tolerance": 1e-9, "points": 21}
    scipy = simulate(system, backend="python", **knobs)
    ran = native.result(system, path=engine, **knobs)
    assert [e.index for e in ran.events] == [e.index for e in scipy.events]
    np.testing.assert_allclose(ran.time, scipy.time, rtol=0, atol=1e-7)
    keep = np.ones(len(ran.time), dtype=bool)
    for event in ran.events:
        keep &= np.abs(ran.time - event.time) > 1e-3
    for variable in ran.names:
        np.testing.assert_allclose(
            np.asarray(ran[variable])[keep],
            np.asarray(scipy[variable])[keep],
            rtol=1e-5,
            atol=1e-6,
            err_msg=variable,
        )


def test_a_sweep_runs_on_the_engine_like_scipy(engine: Path) -> None:
    """``sweep(..., backend="engine")`` gives each point the values SciPy gives it."""
    from modelica.sim.sweep import grid, sweep  # noqa: PLC0415

    del engine  # the fixture skips without a Rust compiler; the sweep finds its own library
    system = _system(EXAMPLES / "models" / "RCCircuit.mo")
    points = grid(**{"r.R": [0.5, 1.0, 2.0]})
    knobs: dict[str, Any] = {"stop": 1.0, "points": 11, "tolerance": 1e-9}
    on_engine = sweep(system, points, backend="engine", **knobs)
    on_scipy = sweep(system, points, backend="python", **knobs)
    assert {r.backend for r in on_engine} == {"engine"}
    assert {r.backend for r in on_scipy} == {"python"}
    for mine, theirs in zip(on_engine, on_scipy, strict=True):
        for name in mine.names:
            np.testing.assert_allclose(mine[name], theirs[name], rtol=1e-6, atol=1e-8)
    assert not np.allclose(on_engine[0]["c.v"], on_engine[2]["c.v"])


def test_a_constant_subexpression_leaves_no_instruction() -> None:
    """``sin(2.0) * 3.0`` is one preloaded constant: no call, no constant load, one multiply."""
    from modelica.sim import flatten  # noqa: PLC0415

    source = "model Fold Real x(start = 1); equation der(x) = sin(2.0) * 3.0 * x; end Fold;"
    system = causalize(flatten([parse(source)], "Fold"))
    program = vm.lower(system)
    ops = [i.op for i in program.rhs[0].code]
    assert vm.Op.CALL1 not in ops
    assert vm.Op.CONST not in ops
    assert ops.count(vm.Op.MUL) == 1
    out = [0.0]
    vm.evaluate(program, program.rhs[0], 0.0, [1.0], [], [0.0] * 8, out)
    assert out[0] == pytest.approx(np.sin(2.0) * 3.0, rel=1e-15)


def test_the_engine_refuses_a_register_out_of_range(engine: Path) -> None:
    """The engine reads registers without bounds checks, so it checks them when it loads."""
    from dataclasses import replace  # noqa: PLC0415

    from modelica.sim import engine as native  # noqa: PLC0415
    from modelica.sim.solve import SimulationError  # noqa: PLC0415

    program = vm.lower(_system(EXAMPLES / "models" / "RCCircuit.mo"))
    clipped = replace(program, registers=len(program.constants) + 1)
    with pytest.raises(SimulationError, match="refused"):
        native.load(clipped, engine)


@pytest.mark.parametrize("name", ["RCCircuit", "BouncingBall"])
def test_parallel_runs_equal_single_runs(engine: Path, name: str) -> None:
    """16 parameter vectors on all cores give exactly what 16 runs one at a time give."""
    from modelica.sim import engine as native  # noqa: PLC0415
    from modelica.sim.sweep import vector  # noqa: PLC0415

    system = _system(EXAMPLES / "models" / f"{name}.mo")
    base = vector(system)
    vectors = [base * (1.0 + 0.05 * k) for k in range(16)]
    knobs: dict[str, Any] = {"stop": 2.0, "tolerance": 1e-8, "points": 21, "path": engine}
    together = native.results(system, vectors, **knobs)
    for p, mine in zip(vectors, together, strict=True):
        (alone,) = native.results(system, [p], **knobs)
        np.testing.assert_array_equal(mine.time, alone.time)
        np.testing.assert_array_equal(mine.values, alone.values)
        assert mine.events == alone.events


@pytest.mark.parametrize("name", ["Lorenz", "TwoTank", "Pendulum"])
def test_any_state_order_integrates_the_same(
    engine: Path, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Radau factored in the reversed state order gives the natural order's trajectory."""
    from modelica.sim import engine as native  # noqa: PLC0415

    system = _system(EXAMPLES / "models" / f"{name}.mo")
    knobs: dict[str, Any] = {"stop": 2.0, "tolerance": 1e-8, "points": 21, "path": engine}
    n = len(system.states)

    def natural_order(_: object) -> tuple[int, ...]:
        return ()

    def reversed_order(_: object) -> tuple[int, ...]:
        return tuple(reversed(range(n)))

    monkeypatch.setattr(vm, "bandwidth_order", natural_order)
    natural = native.simulate(system, **knobs).result
    monkeypatch.setattr(vm, "bandwidth_order", reversed_order)
    assert vm.lower(system).order == tuple(reversed(range(n)))
    reordered = native.simulate(system, **knobs).result
    np.testing.assert_array_equal(natural.time, reordered.time)
    np.testing.assert_allclose(reordered.values, natural.values, rtol=1e-9, atol=1e-12)


def test_a_chain_gets_a_banded_order() -> None:
    """A chain numbered 0, 2, 4, ..., 1, 3, 5 has bandwidth n/2; the order brings it to 1."""
    n = 20
    chain = [*range(0, n, 2), *range(n - 1, 0, -2)]
    sparsity: list[list[int]] = [[] for _ in range(n)]
    for a, b in itertools.pairwise(chain):
        sparsity[a].append(b)
        sparsity[b].append(a)
    order = vm.bandwidth_order(sparsity)
    position = {k: row for row, k in enumerate(order)}
    assert sorted(order) == list(range(n))
    assert max(abs(position[i] - position[j]) for i in range(n) for j in sparsity[i]) == 1
    assert vm.bandwidth_order([[0, 1], [0, 1, 2], [1, 2]]) == ()


def test_the_engine_refuses_an_order_that_is_not_a_permutation(engine: Path) -> None:
    """A repeated state in the order is refused when the program loads."""
    from modelica.sim import engine as native  # noqa: PLC0415
    from modelica.sim.solve import SimulationError  # noqa: PLC0415

    program = vm.lower(_system(EXAMPLES / "models" / "Lorenz.mo"))
    with pytest.raises(SimulationError, match="state order"):
        native.load(dataclasses.replace(program, order=(0, 0, 1)), engine)


_DRIVEN = """
model Driven
  Real x(start = 1, fixed = true);
  Real v(start = 0, fixed = true);
  Real noise = sin(3 * time) + 0.5 * sin(7 * time + 0.3 * sin(2 * time)) + cos(5 * time) ^ 2;
equation
  der(x) = v;
  der(v) = -4 * x - 0.1 * v * abs(v) + noise;
  when x < -0.5 then
    reinit(v, -0.8 * pre(v));
  end when;
end Driven;
"""


def _driven() -> System:
    flatten = pytest.importorskip("modelica.sim").flatten
    return causalize(flatten([parse(_DRIVEN)], "Driven"))


def test_time_only_work_moves_before_hoist() -> None:
    """The noise reads only ``time``: it moves ahead of `Op.HOIST`, the states stay after it."""
    function = vm.lower(_driven()).rhs[0]
    ops = [i.op for i in function.code]
    split = ops.index(vm.Op.HOIST)
    assert vm.Op.CALL1 in ops[:split]
    assert vm.Op.LOAD_Y not in ops[:split]
    assert {vm.Op.LOAD_Y, vm.Op.VLOAD_Y} & set(ops[split:])
    assert vm.Op.HOIST not in [i.op for i in vm.lower(_driven(), hoist=False).rhs[0].code]


def test_hoisting_changes_no_result(engine: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine's cache of time-only values gives the run without it, bit for bit."""
    from modelica.sim import engine as native  # noqa: PLC0415

    system = _driven()
    knobs: dict[str, Any] = {"stop": 6.0, "tolerance": 1e-8, "points": 61, "path": engine}
    first = native.simulate(system, **knobs)
    real = vm.lower

    def plain(system: System, *, hoist: bool = True) -> vm.Program:
        del hoist
        return real(system, hoist=False)

    monkeypatch.setattr(vm, "lower", plain)
    second = native.simulate(system, **knobs)
    assert first.events == second.events == 2
    assert first.event_times == second.event_times
    np.testing.assert_array_equal(first.result.time, second.result.time)
    np.testing.assert_array_equal(first.result.values, second.result.values)


def test_compaction_puts_rhs_registers_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """After `_compact` the rhs reads and writes the lowest registers, and none is lost."""
    system = _system(EXAMPLES / "models" / "TorusHeat.mo")
    compact = vm.lower(system, arrays=False)

    def spread_out(program: vm.Program, base: int) -> tuple[vm.Program, int]:
        return program, base

    monkeypatch.setattr(vm, "_compact", spread_out)
    spread = vm.lower(system, arrays=False)
    assert compact.registers < spread.registers
    base = compact.registers - len(compact.constants)
    used = {
        getattr(i, name)
        for i in compact.rhs[0].code
        for name in vm.register_fields(i)
        if getattr(i, name) < base
    }
    assert used == set(range(len(used)))


def test_the_peephole_reads_copies_through_and_folds_negations() -> None:
    """``MOVE`` readers read the source; ``x + (-y)`` is ``x - y``; ``x - (-y)`` is ``x + y``."""
    op, instr = vm.Op, vm.Instr
    code = (
        instr(op.LOAD_Y, 0, 0),
        instr(op.LOAD_Y, 1, 1),
        instr(op.MOVE, 2, 0),
        instr(op.NEG, 3, 1),
        instr(op.ADD, 4, 2, 3),
        instr(op.SUB, 5, 4, 3),
        instr(op.STORE_OUT, 0, 0, 5),
    )
    folded = vm._peephole(vm.Function(code))  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    assert folded.code[4] == instr(op.SUB, 4, 0, 1)
    assert folded.code[5] == instr(op.ADD, 5, 4, 1)
    (kept,) = vm._prune([folded])  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    assert op.MOVE not in [i.op for i in kept.code]
    assert op.NEG not in [i.op for i in kept.code]


def _array_program() -> tuple[vm.Program, vm.Function]:
    """Lorenz's program with a hand-written ``rhs`` that uses every array opcode once."""
    op, instr = vm.Op, vm.Instr
    program = vm.lower(_system(EXAMPLES / "models" / "Lorenz.mo"))
    # 64 fresh registers past the old file; the constants move up past them.
    s = program.registers
    program = dataclasses.replace(
        program, registers=program.registers + 64 + len(program.constants)
    )
    code = (
        instr(op.LOAD_Y, s + 0, 0),
        instr(op.LOAD_Y, s + 1, 1),
        instr(op.LOAD_Y, s + 2, 2),
        instr(op.VGATHER, s + 10, 0, 0, 3),  # reversed y
        instr(op.VBROADCAST, s + 20, s + 0, 0, 3),  # y0, three times
        instr(op.VMUL, s + 30, s + 10, s + 20, 3),
        instr(op.VSUB, s + 33, s + 30, s + 10, 3),
        instr(op.VADD, s + 36, s + 33, s + 10, 3),
        instr(op.VDIV, s + 40, s + 36, s + 20, 3),
        instr(op.VNEG, s + 43, s + 40, 0, 3),
        instr(op.VCALL1, s + 46, s + 43, 3, int(vm.Builtin.SIN)),
        instr(op.VSCATTER, 0, 3, s + 46, 3),  # into s + 50, 51, 52, through table 3
        *(instr(op.STORE_OUT, 0, k, s + 50 + k) for k in range(3)),
    )
    tables = (s + 2, s + 1, s + 0, s + 50, s + 51, s + 52)
    return program, vm.Function(code, (), tables)


def test_array_opcodes_run_the_same_in_the_engine_and_in_python(engine: Path) -> None:
    """Each array opcode, element by element, as `vm.evaluate` and the engine run it."""
    from modelica.sim.engine import load  # noqa: PLC0415

    program, function = _array_program()
    program = dataclasses.replace(program, rhs=(function,))
    y = [0.7, -1.3, 2.1]
    expected = [0.0] * 3
    vm.evaluate(program, function, 0.0, y, [10.0, 28.0, 8.0 / 3.0], [], expected)
    x = np.array(y)[::-1]
    np.testing.assert_allclose(expected, np.sin(-((x * y[0] - x + x) / y[0])), rtol=1e-15)
    loaded = load(program, engine)
    out = np.zeros(3)
    loaded.call(
        loaded.encoded.rhs(0), 0.0, np.array(y), np.array([10.0, 28.0, 8.0 / 3.0]), np.zeros(0), out
    )
    loaded.close()
    np.testing.assert_array_equal(out, expected)


def test_the_engine_refuses_a_table_that_names_no_register(engine: Path) -> None:
    """A gather table entry past the register file is refused when the program loads."""
    from modelica.sim.engine import load  # noqa: PLC0415
    from modelica.sim.solve import SimulationError  # noqa: PLC0415

    program, function = _array_program()
    bad = dataclasses.replace(function, tables=(program.registers + 5, *function.tables[1:]))
    with pytest.raises(SimulationError, match="refused"):
        load(dataclasses.replace(program, rhs=(bad,)), engine)


def test_an_array_block_runs_as_array_opcodes_with_the_same_bits(
    engine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TorusHeat's grid loop is a few array opcodes, and the run is bit for bit the scalar one."""
    from modelica.sim import engine as native  # noqa: PLC0415

    system = _system(EXAMPLES / "models" / "TorusHeat.mo")
    arrays, scalar = vm.lower(system), vm.lower(system, arrays=False)
    ops = [i.op for i in arrays.rhs[0].code]
    assert vm.Op.VGATHER in ops
    assert vm.Op.VSCATTER in ops
    assert ops.count(vm.Op.VLOAD_Y) == 1  # every state in one load (`vm._fuse`)
    assert vm.Op.LOAD_Y not in ops
    assert vm.Op.STORE_OUT not in ops
    assert len(arrays.rhs[0].code) < len(scalar.rhs[0].code) // 10
    observe = [i.op for i in arrays.observe[0].code]
    assert vm.Op.VGATHER in observe
    assert len(observe) < len(scalar.observe[0].code) // 10
    jacobian = [i.op for i in arrays.jacobian[0].code]
    assert vm.Op.VGATHER in jacobian  # built from the runs, not a scalar lowering
    assert len(jacobian) < len(scalar.jacobian[0].code) // 2
    knobs: dict[str, Any] = {"stop": 0.5, "tolerance": 1e-8, "points": 11, "path": engine}
    first = native.simulate(system, **knobs).result
    real = vm.lower

    def plain(system: System, **_: object) -> vm.Program:
        return real(system, arrays=False)

    monkeypatch.setattr(vm, "lower", plain)
    second = native.simulate(system, **knobs).result
    np.testing.assert_array_equal(first.values, second.values)


def test_a_listing_prints_one_numbered_line_per_instruction() -> None:
    """`vm.listing` numbers each instruction and shows constants by value and runs by length."""
    program = vm.lower(_system(EXAMPLES / "models" / "TorusHeat.mo"))
    text = vm.listing(program, program.rhs[0])
    lines = text.splitlines()
    assert len(lines) == len(program.rhs[0].code)
    assert lines[0].split()[0] == "0"
    assert "= gather r[" in text
    assert "scatter r" in text
    assert "4.0" in text


def test_a_jacobian_built_from_array_opcodes_has_the_scalar_bits() -> None:
    """TorusHeat's jacobian from runs is bit for bit the one from the scalar tape."""
    from modelica.sim.kernel import build  # noqa: PLC0415

    system = _system(EXAMPLES / "models" / "TorusHeat.mo")
    p = list(build(system).start(0.0)[0])
    rng = np.random.default_rng(7)
    y = list(rng.random(len(system.states)))
    results: list[list[float]] = []
    for arrays in (True, False):
        program = vm.lower(system, arrays=arrays)
        out = [0.0] * len(y) ** 2
        w = [0.0] * program.layout.workspace
        vm.evaluate(program, program.jacobian[0], 0.1, y, p, w, out)
        results.append(out)
    assert any(results[0])
    assert results[0] == results[1]


def _mesh(cells: int, count: int, seed: int) -> vm.Faces:
    """A random mesh: ``count`` faces, each joining two different cells, owner below neighbour."""
    rng = np.random.default_rng(seed)
    pairs = [sorted(rng.choice(cells, size=2, replace=False).tolist()) for _ in range(count)]
    return vm.Faces(cells, tuple(a for a, _ in pairs), tuple(b for _, b in pairs))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ldu_amul_equals_the_dense_product(engine: Path, seed: int) -> None:
    """`vm.ldu_amul` on a random mesh gives ``A x`` within 1e-14, in Python and in the engine."""
    from modelica.sim.engine import load  # noqa: PLC0415

    faces = _mesh(20, 45, seed)
    n, count = faces.cells, len(faces.owner)
    rng = np.random.default_rng(seed + 10)
    x, diag = rng.standard_normal(n), rng.standard_normal(n)
    upper, lower = rng.standard_normal(count), rng.standard_normal(count)
    dense = np.diag(diag)
    for f, (o, c) in enumerate(zip(faces.owner, faces.neighbour, strict=True)):
        dense[o, c] += upper[f]
        dense[c, o] += lower[f]
    p = np.concatenate([x, diag, upper, lower])

    base = vm.lower(_system(EXAMPLES / "models" / "Lorenz.mo"))
    s = base.registers
    fresh = [s]

    def register(size: int) -> int:
        first = fresh[0]
        fresh[0] += size
        return first

    xs, ds, us, ls, ys = (register(k) for k in (n, n, count, count, n))
    tables: list[int] = []
    code = [vm.Instr(vm.Op.LOAD_P, xs + k, k) for k in range(len(p))]
    code += vm.ldu_amul(faces, ys, xs, ds, us, ls, tables, register)
    code += [vm.Instr(vm.Op.STORE_OUT, 0, k, ys + k) for k in range(n)]
    function = vm.Function(tuple(code), (), tuple(tables))
    program = dataclasses.replace(base, registers=fresh[0] + len(base.constants), rhs=(function,))
    expected = [0.0] * n
    vm.evaluate(program, function, 0.0, [0.0] * 3, list(p), [], expected)
    np.testing.assert_allclose(expected, dense @ x, rtol=0, atol=1e-14)
    loaded = load(program, engine)
    out = np.zeros(n)
    loaded.call(loaded.encoded.rhs(0), 0.0, np.zeros(3), p, np.zeros(0), out)
    loaded.close()
    np.testing.assert_array_equal(out, expected)


def _face_rhs(program: vm.Program, n: int) -> tuple[vm.Function, int]:
    """HeatBar's ``rhs`` as a finite-volume tape, and the register file size it needs.

    ``der(T) = A T + b``, with ``A T`` from `vm.ldu_amul`. Faces join cell ``i`` to ``i + 1``. Each face conducts with coefficient ``a``, so
    ``upper = lower = a`` and ``diag = -a`` per face of the cell. The held left end adds one
    more ``-a`` to cell 1 and the source ``a * T_left``; the right end is insulated.
    """
    op, instr = vm.Op, vm.Instr
    fresh = [program.registers]

    def register(size: int = 1) -> int:
        first = fresh[0]
        fresh[0] += size
        return first

    slots = {name: k for k, name in enumerate(program.layout.parameters)}
    faces = vm.Faces(n, tuple(range(n - 1)), tuple(range(1, n)))
    a, hot = register(), register()
    code = [instr(op.LOAD_P, a, slots["a"]), instr(op.LOAD_P, hot, slots["T_left"])]
    minus = register()
    twice = register()
    code += [instr(op.NEG, minus, a), instr(op.ADD, twice, minus, minus)]
    xs, ds, us, ls, ys, out = (register(k) for k in (n, n, n - 1, n - 1, n, n))
    code += [instr(op.LOAD_Y, xs + k, k) for k in range(n)]
    code += [instr(op.MOVE, ds + k, twice) for k in range(n - 1)]
    code += [instr(op.MOVE, ds + n - 1, minus)]
    code += [instr(op.MOVE, us + f, a) for f in range(n - 1)]
    code += [instr(op.MOVE, ls + f, a) for f in range(n - 1)]
    tables: list[int] = []
    code += vm.ldu_amul(faces, ys, xs, ds, us, ls, tables, register)
    source = register()
    code += [instr(op.MUL, source, a, hot), instr(op.ADD, out, ys, source)]
    code += [instr(op.STORE_OUT, 0, 0, out)]
    code += [instr(op.STORE_OUT, 0, k, ys + k) for k in range(1, n)]
    return vm.Function(tuple(code), (), tuple(tables)), fresh[0]


def test_a_heat_bar_written_with_face_lists_equals_heat_bar(
    engine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HeatBar with its ``rhs`` as an LDU product over faces runs as the ``for`` loop does."""
    from modelica.sim import engine as native  # noqa: PLC0415

    system = _system(EXAMPLES / "models" / "HeatBar.mo")
    knobs: dict[str, Any] = {"stop": 40.0, "tolerance": 1e-10, "points": 21, "path": engine}
    loops = native.simulate(system, **knobs).result
    real = vm.lower

    def faces(system: System, **options: Any) -> vm.Program:
        program = real(system, **options)
        function, used = _face_rhs(program, len(system.states))
        # The constants sit at the top of the register file. The new registers go past the
        # old file, so the pool is padded down over them to keep each constant where the
        # other tapes read it, and repeated at the new top.
        pool = program.constants
        padding = (0.0,) * (used - program.registers)
        return dataclasses.replace(
            program,
            rhs=(function,),
            registers=used + len(pool),
            constants=(*pool, *padding, *pool),
        )

    monkeypatch.setattr(vm, "lower", faces)
    lists = native.simulate(system, **knobs).result
    np.testing.assert_allclose(lists.values, loops.values, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("model", _models(), ids=[m.stem for m in _models()])
def test_the_jacobian_pattern_is_inside_the_sparsity(model: Path) -> None:
    """Every entry a jacobian tape stores is one `Mode.sparsity` allows, and none twice."""
    system = _system(model)
    program = vm.lower(system)
    allowed = {
        (row, column)
        for mode in system.modes
        for row, columns in enumerate(mode.sparsity)
        for column in columns
    }
    assert set(program.pattern) <= allowed
    assert len(set(program.pattern)) == len(program.pattern)


def test_torus_heat_stores_five_entries_per_row() -> None:
    """TorusHeat's jacobian tape writes its 5 * 64 nonzeros, not the 64 * 64 matrix."""
    program = vm.lower(_system(EXAMPLES / "models" / "TorusHeat.mo"))
    assert len(program.pattern) == 5 * 64


def test_a_2500_state_torus_runs_in_band_storage_and_decays_exactly(engine: Path) -> None:
    """TorusHeat at n = 50 on the engine: each cell decays as ``exp(-8 a sin^2(pi / n) t)``.

    The start is the lowest Fourier mode, so the method-of-lines solution is that one
    exponential. Radau factors in band storage from the jacobian pattern, never ``n * n``.
    """
    from modelica.sim import engine as native  # noqa: PLC0415

    n = 50
    source = (EXAMPLES / "models" / "TorusHeat.mo").read_text(encoding="utf-8")
    model = parse(source.replace("Integer n = 8", f"Integer n = {n}"))
    system = causalize(pytest.importorskip("modelica.sim").flatten([model], "TorusHeat"))
    knobs: dict[str, Any] = {"stop": 5.0, "tolerance": 1e-9, "points": 3, "path": engine}
    result = native.simulate(system, **knobs).result
    cells = [k for k, name in enumerate(result.names) if name.startswith("T[")]
    decay = np.exp(-8.0 * np.sin(np.pi / n) ** 2 * 5.0)
    values = result.values[cells]
    np.testing.assert_allclose(values[:, -1], values[:, 0] * decay, rtol=0, atol=1e-12)


def _mass_program(
    base: vm.Program, rhs: list[vm.Instr], jacobian: list[tuple[int, int]], mass: tuple[float, ...]
) -> vm.Program:
    """Lorenz's program with a hand-written ``rhs``, a dense jacobian of constants, and ``mass``.

    The pool is preloaded at the top of the register file. The constants -1, 0 and 1 and 60
    zeros go after the old ones and the file grows by 63, so the old constants keep their
    registers. ``rhs`` and the jacobian use the zeroed registers from ``base.registers + 3``;
    ``jacobian`` lists ``(slot, constant)`` with constant 0, 1, 2 for -1, 0, 1.
    """
    op, instr = vm.Op, vm.Instr
    c, top = len(base.constants), base.registers + 3
    stores: list[vm.Instr] = [instr(op.CLEAR_OUT, 0, 9)]
    for slot, k in jacobian:
        stores += [
            instr(op.CONST, top + 20 + slot, c + k),
            instr(op.STORE_OUT, 0, slot, top + 20 + slot),
        ]
    return dataclasses.replace(
        base,
        registers=base.registers + 63,
        constants=(*base.constants, -1.0, 0.0, 1.0, *(0.0,) * 60),
        rhs=(vm.Function(tuple(rhs)),),
        jacobian=(vm.Function(tuple(stores)),),
        pattern=(),
        mass=mass,
    )


@pytest.mark.parametrize("case", ["lumped", "algebraic"])
def test_radau_with_a_mass_matrix(engine: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """``M y' = f`` on the engine: a lumped mass, and a singular one (an index-1 DAE).

    Lumped: ``f = -y``, ``M = diag(2, 4, 0.5)``, so ``y_i = exp(-t / m_i)``. Algebraic:
    ``x' = -y``, ``0 = y - x``, ``z' = 0`` with ``M = diag(1, 0, 1)``, so ``x = y = exp(-t)``.
    """
    from modelica.sim import engine as native  # noqa: PLC0415

    op, instr = vm.Op, vm.Instr
    system = _system(EXAMPLES / "models" / "Lorenz.mo")
    base = vm.lower(system)
    s = base.registers + 3  # the zeroed registers `_mass_program` adds
    loads = [instr(op.LOAD_Y, s + k, k) for k in range(3)]
    if case == "lumped":
        mass = (2.0, 4.0, 0.5)
        rhs = [*loads, *(instr(op.NEG, s + 3 + k, s + k) for k in range(3))]
        rhs += [instr(op.STORE_OUT, 0, k, s + 3 + k) for k in range(3)]
        jacobian = [(0, 0), (4, 0), (8, 0)]
        expected = np.exp(-5.0 / np.array(mass))
    else:
        mass = (1.0, 0.0, 1.0)
        rhs = [*loads, instr(op.NEG, s + 3, s + 1), instr(op.SUB, s + 4, s + 1, s + 0)]
        rhs += [instr(op.CONST, s + 5, len(base.constants) + 1)]
        rhs += [instr(op.STORE_OUT, 0, k, s + 3 + k) for k in range(3)]
        jacobian = [(1, 0), (3, 0), (4, 2)]
        expected = np.array([np.exp(-5.0), np.exp(-5.0), 1.0])
    program = _mass_program(base, rhs, jacobian, mass)

    def lowered(*_: object, **__: object) -> vm.Program:
        return program

    monkeypatch.setattr(vm, "lower", lowered)
    knobs: dict[str, Any] = {"stop": 5.0, "tolerance": 1e-10, "points": 2, "path": engine}
    result = native.simulate(system, **knobs).result
    rows = [result.names.index(name) for name in ("x", "y", "z")]
    np.testing.assert_allclose(result.values[rows, -1], expected, rtol=1e-7)
