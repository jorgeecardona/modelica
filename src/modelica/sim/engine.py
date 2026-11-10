"""Run a model in the precompiled engine (``engine/src/lib.rs``): the tape, no compiler per model.

`modelica.sim.vm` lowers the model to a tape; this module hands the tape to the engine library
and integrates with the same Radau runtime `rustsim` compiles. The result is `rustsim.Run`.

The engine is one shared library for every model. A platform wheel of ``modelica`` carries it as
``modelica/_engine.<suffix>`` (``hatch_build.py``): Linux x86_64 and aarch64, Windows x86_64 and
macOS arm64. With a pure wheel or a source checkout,
`library` builds it once with ``rustc`` from ``engine/src`` and keeps it in the cache directory
`rustsim` uses.
"""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from modelica.sim import vm
from modelica.sim.result import Result
from modelica.sim.rustsim import (
    _RUN_ARGUMENTS,  # pyright: ignore[reportPrivateUsage] - the same entry point arguments
    _default_cache,  # pyright: ignore[reportPrivateUsage] - the same cache directory
    _suffix,  # pyright: ignore[reportPrivateUsage]
    assemble,
    drive,
    rows_for,
    triggers,
)
from modelica.sim.solve import SimulationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.rustsim import Run

__all__ = ["Loaded", "installed", "library", "load", "result", "results", "simulate"]

SOURCE = Path(__file__).resolve().parents[3] / "engine" / "src"


def installed() -> Path | None:
    """The engine library inside the installed wheel, or ``None`` for a pure wheel or a checkout.

    A platform wheel carries it as ``modelica/_engine.<suffix>`` (``hatch_build.py``).
    """
    bundled = Path(__file__).resolve().parents[1] / f"_engine{_suffix()}"
    return bundled if bundled.exists() else None


def library(cache: Path | None = None) -> Path:
    """The engine's shared library: the installed wheel's, else built once from ``engine/src``.

    The build is for a source checkout without the wheel. It is kept by the hash of the
    sources, so it happens once per change to the engine, never per model.
    """
    if cache is None and (wheel := installed()) is not None:
        return wheel
    sources = sorted(SOURCE.glob("*.rs"))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    folder = cache if cache is not None else _default_cache()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"engine-{digest.hexdigest()[:20]}{_suffix()}"
    if target.exists():
        return target
    rustc = shutil.which("rustc")
    if rustc is None:
        message = f"no rustc on PATH, and no engine library in {folder}"
        raise SimulationError(message)
    with tempfile.NamedTemporaryFile(dir=folder, suffix=_suffix(), delete=False) as handle:
        partial = Path(handle.name)
    done = subprocess.run(  # noqa: S603 - the compiler from PATH, on the engine's own source
        [
            rustc,
            "--edition",
            "2021",
            "-O",
            "--crate-type",
            "cdylib",
            str(SOURCE / "lib.rs"),
            "-o",
            str(partial),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        partial.unlink(missing_ok=True)
        message = f"rustc failed on the engine\n{done.stderr}"
        raise SimulationError(message)
    partial.replace(target)
    return target


def _open(path: Path) -> ctypes.CDLL:
    engine = ctypes.CDLL(str(path))
    size, pointer = ctypes.c_size_t, ctypes.c_void_p
    engine.md_vm_new.restype = pointer
    engine.md_vm_new.argtypes = [
        size, size, size, ctypes.c_int64,
        pointer, size, pointer, size, pointer, size, pointer, size, pointer, size,
    ]  # fmt: skip
    engine.md_vm_free.argtypes = [pointer]
    engine.md_vm_order.restype = ctypes.c_int32
    engine.md_vm_order.argtypes = [pointer, pointer, size]
    engine.md_vm_pattern.restype = ctypes.c_int32
    engine.md_vm_pattern.argtypes = [pointer, pointer, pointer, size, size]
    engine.md_vm_mass.restype = ctypes.c_int32
    engine.md_vm_mass.argtypes = [pointer, pointer, size]
    engine.md_vm_times.restype = ctypes.c_int32
    engine.md_vm_times.argtypes = [pointer, pointer, size]
    engine.md_vm_run_many.restype = None
    engine.md_vm_run_many.argtypes = [
        pointer, size, size, *_RUN_ARGUMENTS[:-1], pointer, pointer,
    ]  # fmt: skip
    engine.md_vm_call.restype = ctypes.c_int32
    engine.md_vm_call.argtypes = [
        pointer, size, ctypes.c_int64, ctypes.c_double,
        pointer, size, pointer, size, pointer, size, pointer, size,
    ]  # fmt: skip
    return engine


@dataclass
class Loaded:
    """A program inside the engine: the handle, and the arrays it was decoded from."""

    engine: ctypes.CDLL
    handle: int
    encoded: vm.Encoded
    layout: vm.Layout

    def close(self) -> None:
        """Free the program. The handle is not valid afterwards."""
        if self.handle:
            self.engine.md_vm_free(self.handle)
            self.handle = 0

    def __del__(self) -> None:
        self.close()

    def start(
        self, parameters: NDArray[np.float64] | None = None, t: float = 0.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The parameter vector, the initial state and a latched workspace, as `Kernel.start`.

        ``parameters`` replaces the vector the model computes; the start values follow it.
        ``t`` is the start time: a start value or a latch can read ``time``.
        """
        layout = self.layout
        empty = np.zeros(0)
        p = np.zeros(len(layout.parameters))
        if parameters is None:
            self.call(self.encoded.parameters, t, empty, empty, empty, p)
        else:
            p[:] = parameters
        y = np.zeros(len(layout.states))
        w = np.zeros(layout.workspace)
        self.call(self.encoded.initial, t, empty, p, w, y)
        self.call(self.encoded.latch, t, y, p, w, empty)
        return p, y, w

    def call(  # noqa: PLR0913, PLR0917 - the kernel signature
        self,
        index: int,
        t: float,
        y: NDArray[np.float64],
        p: NDArray[np.float64],
        w: NDArray[np.float64],
        out: NDArray[np.float64],
        then: int = -1,
    ) -> None:
        """Run function ``index`` (an `Encoded` index), then ``then`` if not negative."""
        self.engine.md_vm_call(
            self.handle, index, then, t,
            y.ctypes.data, len(y), p.ctypes.data, len(p),
            w.ctypes.data, len(w), out.ctypes.data, len(out),
        )  # fmt: skip


def load(program: vm.Program, path: Path | None = None) -> Loaded:
    """Decode ``program`` in the engine at ``path`` (default: `library`)."""
    engine = _open(path if path is not None else library())
    encoded = vm.encode(program)
    arrays: list[Any] = [
        np.ascontiguousarray(encoded.constants, dtype=np.float64),
        np.ascontiguousarray(encoded.code, dtype=np.int32),
        np.ascontiguousarray(encoded.functions, dtype=np.int32),
        np.ascontiguousarray(encoded.newton, dtype=np.int32),
        np.ascontiguousarray(encoded.unknowns, dtype=np.int32),
    ]
    constants, code, functions, newton, unknowns = arrays
    handle = engine.md_vm_new(
        encoded.registers,
        encoded.modes,
        encoded.events,
        -1 if program.switching is None else program.switching,
        constants.ctypes.data, len(constants),
        code.ctypes.data, len(code),
        functions.ctypes.data, len(functions) // 4,
        newton.ctypes.data, len(newton) // 7,
        unknowns.ctypes.data, len(unknowns),
    )  # fmt: skip
    if not handle:
        message = "the engine refused the program"
        raise SimulationError(message)
    if program.order:
        order = np.ascontiguousarray(program.order, dtype=np.uint64)
        if engine.md_vm_order(handle, order.ctypes.data, len(order)) != 0:
            engine.md_vm_free(handle)
            message = "the engine refused the state order"
            raise SimulationError(message)
    if program.pattern:
        rows = np.ascontiguousarray([i for i, _ in program.pattern], dtype=np.uint64)
        cols = np.ascontiguousarray([j for _, j in program.pattern], dtype=np.uint64)
        n = len(program.layout.states)
        if engine.md_vm_pattern(handle, rows.ctypes.data, cols.ctypes.data, len(rows), n) != 0:
            engine.md_vm_free(handle)
            message = "the engine refused the jacobian pattern"
            raise SimulationError(message)
    if program.mass:
        mass = np.ascontiguousarray(program.mass, dtype=np.float64)
        if engine.md_vm_mass(handle, mass.ctypes.data, len(mass)) != 0:
            engine.md_vm_free(handle)
            message = "the engine refused the mass matrix"
            raise SimulationError(message)
    if program.times:
        times = np.ascontiguousarray(program.times, dtype=np.int64)
        if engine.md_vm_times(handle, times.ctypes.data, len(times)) != 0:
            engine.md_vm_free(handle)
            message = "the engine refused the time-event slots"
            raise SimulationError(message)
    return Loaded(engine, handle, encoded, program.layout)


def simulate(  # noqa: PLR0913, PLR0917 - the same knobs as `rustsim.simulate`
    system: System,
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None = None,
    points: int | None = None,
    tolerance: float = 1e-6,
    absolute: float | None = None,
    parameters: NDArray[np.float64] | None = None,
    path: Path | None = None,
    *,
    every: bool = False,
    start: float = 0.0,
) -> Run:
    """Integrate ``system`` in the engine: lower it to a tape, then run the Radau runtime on it.

    ``every`` returns the rows `modelica.sim.simulate` returns (`rustsim.drive`).
    """
    return _runs(
        system,
        [parameters],
        stop=stop,
        times=times,
        points=points,
        tolerance=tolerance,
        absolute=absolute,
        path=path,
        every=every,
        start=start,
    )[0]


def _runs(  # noqa: PLR0913 - the knobs of `simulate`
    system: System,
    vectors: Sequence[NDArray[np.float64] | None],
    *,
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None,
    points: int | None,
    tolerance: float,
    absolute: float | None,
    path: Path | None,
    every: bool,
    start: float,
) -> list[Run]:
    """One run per parameter vector, from one lowered and loaded program.

    Several vectors run in parallel, one thread per core (`_parallel`).
    """
    program = vm.lower(system)
    loaded = load(program, path)
    try:
        if len(vectors) > 1:
            return _parallel(
                system,
                loaded,
                program,
                [loaded.start(parameters, start) for parameters in vectors],
                stop=stop,
                times=times,
                points=points,
                tolerance=tolerance,
                absolute=absolute,
                every=every,
                t0=start,
            )
        return [
            drive(
                system,
                loaded.engine.md_vm_run,
                (loaded.handle,),
                start=loaded.start(parameters, start),
                events=program.layout.events,
                observed=program.layout.observed,
                stop=stop,
                times=times,
                points=points,
                tolerance=tolerance,
                absolute=absolute,
                every=every,
                t0=start,
            )
            for parameters in vectors
        ]
    finally:
        loaded.close()


def _parallel(  # noqa: PLR0913 - the knobs of `simulate`
    system: System,
    loaded: Loaded,
    program: vm.Program,
    starts: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    *,
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None,
    points: int | None,
    tolerance: float,
    absolute: float | None,
    every: bool,
    t0: float,
    threads: int = 0,
) -> list[Run]:
    """Every start integrated at once by ``md_vm_run_many``, one thread per core by default.

    A run that needs more rows than the first guess runs again on its own, through `drive`.
    """
    events, observed = program.layout.events, program.layout.observed
    dirs, lat, term = triggers(events)
    grid, cap, ev_cap = rows_for(stop, times, points, t0, every=every)
    count = len(starts)
    p = np.ascontiguousarray(np.stack([s[0] for s in starts]), dtype=np.float64)
    y0 = np.ascontiguousarray(np.stack([s[1] for s in starts]), dtype=np.float64)
    w = np.ascontiguousarray(np.stack([s[2] for s in starts]), dtype=np.float64)
    n, nobs = y0.shape[1], len(observed)
    t_out = np.zeros((count, cap), dtype=np.float64)
    out = np.zeros((count, cap * n), dtype=np.float64)
    obs = np.zeros((count, cap * nobs), dtype=np.float64)
    ev_out = np.zeros((count, ev_cap), dtype=np.float64)
    ev_idx = np.zeros((count, ev_cap), dtype=np.float64)
    counters = np.zeros((count, 9), dtype=np.float64)
    status = np.zeros(count, dtype=np.int32)
    loaded.engine.md_vm_run_many(
        loaded.handle, threads, count,
        float(t0), float(stop),
        y0.ctypes.data, n,
        p.ctypes.data, p.shape[1],
        w.ctypes.data, w.shape[1],
        float(tolerance), float(tolerance if absolute is None else absolute),
        dirs.ctypes.data, lat.ctypes.data, term.ctypes.data, len(dirs),
        grid.ctypes.data, len(grid),
        1 if every else 0, t_out.ctypes.data, cap,
        out.ctypes.data, obs.ctypes.data, nobs,
        ev_out.ctypes.data, ev_idx.ctypes.data, ev_cap,
        counters.ctypes.data, status.ctypes.data,
    )  # fmt: skip
    runs: list[Run] = []
    for k, begin in enumerate(starts):
        if counters[k, 8] > cap:
            runs.append(
                drive(
                    system,
                    loaded.engine.md_vm_run,
                    (loaded.handle,),
                    start=begin,
                    events=events,
                    observed=observed,
                    stop=stop,
                    times=times,
                    points=points,
                    tolerance=tolerance,
                    absolute=absolute,
                    every=every,
                    t0=t0,
                )
            )
            continue
        runs.append(
            assemble(
                system,
                status=int(status[k]),
                counters=counters[k],
                rows=(t_out[k], out[k], obs[k]),
                found=(ev_out[k], ev_idx[k]),
                events=events,
                observed=observed,
                grid=grid,
                every=every,
            )
        )
    return runs


def results(  # noqa: PLR0913 - the knobs of `simulate`
    system: System,
    vectors: Sequence[NDArray[np.float64] | None],
    *,
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None = None,
    points: int | None = None,
    tolerance: float = 1e-6,
    absolute: float | None = None,
    path: Path | None = None,
    start: float = 0.0,
) -> tuple[Result, ...]:
    """One `Result` per parameter vector, as `modelica.sim.simulate` returns it.

    The model is lowered and loaded once; only the parameter vector changes between runs.
    ``None`` is the vector the model computes.
    """
    runs = _runs(
        system,
        vectors,
        stop=stop,
        times=times,
        points=points,
        tolerance=tolerance,
        absolute=absolute,
        path=path,
        every=True,
        start=start,
    )
    states = len(system.modes[0].states)
    return tuple(
        Result(
            name=run.result.name,
            time=run.result.time,
            names=run.result.names[states:],
            values=run.result.values[states:],
            events=run.event_list,
            backend="engine",
        )
        for run in runs
    )


def result(  # noqa: PLR0913 - the knobs of `simulate`
    system: System,
    *,
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None = None,
    points: int | None = None,
    tolerance: float = 1e-6,
    absolute: float | None = None,
    parameters: NDArray[np.float64] | None = None,
    path: Path | None = None,
    start: float = 0.0,
) -> Result:
    """The engine's run as `modelica.sim.simulate` returns one: observed rows and the events."""
    return results(
        system,
        [parameters],
        stop=stop,
        times=times,
        points=points,
        tolerance=tolerance,
        absolute=absolute,
        path=path,
        start=start,
    )[0]
