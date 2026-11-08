"""The tape against the emitted Python, on every corpus model, in Python and in the Rust engine.

The differential style of `tests/test_rustgen.py`: two implementations of one `System`, the same
inputs at the model's start point, the answers compared. `vm.evaluate` is checked on every
model; the Rust engine (``engine/src/lib.rs``) is compiled once with ``rustc`` and checked on the
same calls, and skipped when there is no ``rustc``.
"""

from __future__ import annotations

import ctypes
import functools
import shutil
import subprocess
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


@functools.cache
def _engine(directory: Path) -> ctypes.CDLL:
    assert COMPILER is not None
    library = directory / "libmd_engine.so"
    subprocess.run(  # noqa: S603 - our own source, our own compiler path
        [
            COMPILER,
            "--edition",
            "2021",
            "-O",
            "--crate-type",
            "cdylib",
            "-D",
            "warnings",
            str(ROOT / "engine" / "src" / "lib.rs"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    engine = ctypes.CDLL(str(library))
    engine.md_vm_new.restype = ctypes.c_void_p
    engine.md_vm_free.argtypes = [ctypes.c_void_p]
    size, i32p, f64p = (
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
    )
    engine.md_vm_new.argtypes = [size, f64p, size, i32p, size, i32p, size, i32p, size, i32p, size]
    engine.md_vm_call.argtypes = [
        ctypes.c_void_p,
        size,
        ctypes.c_int64,
        ctypes.c_double,
        f64p,
        size,
        f64p,
        size,
        f64p,
        size,
        f64p,
        size,
    ]
    return engine


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    if COMPILER is None:
        pytest.skip("no rustc on this machine")
    return _engine(tmp_path_factory.mktemp("engine"))


def _pointer(array: Any, kind: Any) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(kind))


@pytest.mark.parametrize("model", _models(), ids=[m.stem for m in _models()])
def test_the_engine_equals_the_interpreter(engine: ctypes.CDLL, model: Path) -> None:
    """The same calls as `test_the_tape_equals_the_interpreter`, run by the Rust engine."""
    system = _system(model)
    point = _Point(emit(system))
    program = vm.lower(system)
    encoded = vm.encode(program)
    arrays = [
        np.asarray(encoded.constants, dtype=np.float64),
        np.asarray(encoded.code, dtype=np.int32),
        np.asarray(encoded.functions, dtype=np.int32),
        np.asarray(encoded.newton, dtype=np.int32),
        np.asarray(encoded.unknowns, dtype=np.int32),
    ]
    constants, code, table, newton, unknowns = arrays
    handle = engine.md_vm_new(
        encoded.registers,
        _pointer(constants, ctypes.c_double),
        len(constants),
        _pointer(code, ctypes.c_int32),
        len(code),
        _pointer(table, ctypes.c_int32),
        len(table) // 4,
        _pointer(newton, ctypes.c_int32),
        len(newton) // 7,
        _pointer(unknowns, ctypes.c_int32),
        len(unknowns),
    )
    try:
        mode = vm.mode(program, point.w)
        for label, which, event, length, kernel, extra in point.calls():
            expected, expected_w = _reference(point, kernel, extra, length)
            y, p = point.y.copy(), np.asarray(point.p, dtype=np.float64)
            w, out = point.w.copy(), np.zeros(length)
            then = -1 if event is None else encoded.transition(mode, event)
            engine.md_vm_call(
                handle,
                _index(encoded, which, mode),
                then,
                0.0,
                _pointer(y, ctypes.c_double),
                len(y),
                _pointer(p, ctypes.c_double),
                len(p),
                _pointer(w, ctypes.c_double),
                len(w),
                _pointer(out, ctypes.c_double),
                length,
            )
            np.testing.assert_allclose(
                out, expected, rtol=_tol(label), atol=_tol(label), err_msg=label
            )
            np.testing.assert_allclose(w, expected_w, rtol=1e-12, atol=1e-12, err_msg=label)
    finally:
        engine.md_vm_free(handle)
