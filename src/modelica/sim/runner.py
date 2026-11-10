"""Call the engine's ``md_vm_run`` and read its buffers back into a `Run`.

`drive` passes NumPy arrays to the entry point by address, with their lengths, and `assemble`
reads the rows, events and counters back. `engine` uses both.
"""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from modelica.sim.result import Event, Result
from modelica.sim.solve import SimulationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.emit import Trigger

__all__ = ["Run", "assemble", "default_cache", "drive", "rows_for", "suffix", "triggers"]

#: What the driver's status means, by the constants `rust_radau.STEP` declares.
STATUS = {
    1: "the required step size fell below the spacing between numbers",
    2: "the Newton iteration did not converge",
    3: "the model is chattering: events stopped carrying the run forward",
    4: "the model reached a terminate() and the run stopped there",
}


def default_cache() -> Path:
    """``$PYMODELICA_CACHE/rust``, else the user's cache directory."""
    root = os.environ.get("PYMODELICA_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "pymodelica"
    return base / "rust"


def suffix() -> str:
    """What the platform calls a shared library."""
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    if system == "Windows":
        return ".dll"
    return ".so"


@dataclass(frozen=True)
class Run:
    """A Rust run: the trajectory, and the counters the driver kept while making it."""

    result: Result
    """States and observed variables on the requested grid, by name, as the interpreter's."""

    events: int
    """How many crossings fired, checked exactly against the interpreter's since a wrong tally is
    a structural bug."""

    event_times: tuple[float, ...]
    """When each crossing fired, in order -- the instants the driver located, carried back beside
    the count. Several crossings that settle at one instant each carry that same time."""

    event_list: tuple[Event, ...]
    """The crossings as the interpreter reports them: the instant, the index of the indicator that
    fired, and its cause in Modelica -- the same `Event` the interpreter's `Result.events` holds.
    The index comes back beside the instant over the FFI; the cause is read from the model."""

    steps: int
    rejected: int
    evaluations: int
    jacobians: int
    factorisations: int
    end: float
    """Where the run stopped: the horizon, or earlier when the status said so."""

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        """One variable's trajectory, as ``Result`` gives it, so ``run["x"]`` works like ``result["x"]``."""
        return self.result[name]

    @property
    def time(self) -> NDArray[np.float64]:
        """The output instants, as ``Result.time``."""
        return self.result.time


def _grid(
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None,
    points: int | None,
    t0: float = 0.0,
) -> NDArray[np.float64]:
    if times is not None:
        return np.ascontiguousarray(np.asarray(times, dtype=np.float64))
    return np.linspace(t0, stop, points or 300)


#: The ctypes argument types of ``md_run``, after any leading arguments a caller adds.
_RUN_ARGUMENTS: Final = [
    ctypes.c_double, ctypes.c_double,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p,
]  # fmt: skip


def triggers(
    events: Sequence[Trigger],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """The direction, latch slot and terminate flag of each crossing, as the runtime takes them."""
    dirs = np.ascontiguousarray([float(e.direction) for e in events], dtype=np.float64)
    lat = np.ascontiguousarray(
        [e.latch if e.latch is not None else -1 for e in events], dtype=np.int64
    )
    # 1 for a crossing whose `when` holds `terminate(...)`, 0 otherwise: the driver stops the
    # run at the first such crossing, the same as `solve.py` breaking its loop there.
    term = np.ascontiguousarray([1 if e.terminates else 0 for e in events], dtype=np.int64)
    return dirs, lat, term


def rows_for(
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None,
    points: int | None,
    t0: float,
    *,
    every: bool,
) -> tuple[NDArray[np.float64], int, int]:
    """The output grid, a first guess at the row capacity, and the event capacity (`drive`)."""
    if every and times is None and points is None:
        grid = np.zeros(0, dtype=np.float64)
    else:
        grid = _grid(stop, times, points, t0)
    cap = max(1, len(grid)) if not every else max(4096, 4 * len(grid) + 64)
    return grid, cap, max(4096, 8 * len(grid))


def assemble(  # noqa: PLR0913 - what one run wrote
    system: System,
    *,
    status: int,
    counters: NDArray[np.float64],
    rows: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    found: tuple[NDArray[np.float64], NDArray[np.float64]],
    events: Sequence[Trigger],
    observed: Sequence[str],
    grid: NDArray[np.float64],
    every: bool,
) -> Run:
    """Read one run's buffers into a `Run`; raise `SimulationError` on a failed status."""
    t_out, out, obs = rows
    ev_out, ev_idx = found
    ev_cap = len(ev_out)
    n = len(out) // max(1, len(t_out))
    nobs = len(observed)
    rows_written = int(counters[8])
    _TERMINATED = 4  # noqa: N806 - a status code, matching rust_radau's constant
    if status not in (0, _TERMINATED):
        reason = STATUS.get(status, f"status {status}")
        message = f"{system.name}: the Rust run stopped at t = {counters[7]:.12g}: {reason}"
        raise SimulationError(message)
    # A terminate() is a clean early stop, not a failure: the run is defined only up to the
    # instant it fired, and the fixed sample grid was filled only that far. Keep the grid
    # points at or before that instant and drop the unwritten tail, so the result ends where
    # the model does, the same truncation `solve.py` gets for free by sampling per segment.
    rows_written = min(rows_written, len(t_out))
    if status == _TERMINATED and not every:
        rows_written = int(np.count_nonzero(grid <= float(counters[7])))
    time = t_out[:rows_written].copy()
    names = (*system.modes[0].states, *observed)
    values = np.vstack(
        [
            out.reshape(-1, n)[:rows_written].T if n else np.empty((0, rows_written)),
            obs.reshape(-1, nobs)[:rows_written].T if nobs else np.empty((0, rows_written)),
        ]
    )
    count = int(counters[6])
    kept = min(count, ev_cap)
    # The full events, as the interpreter reports them: the instant and index come back over the
    # FFI, the cause `str(...)` is read from the model's own indicator list by that index.
    event_list = tuple(
        Event(float(ev_out[k]), int(ev_idx[k]), str(events[int(ev_idx[k])])) for k in range(kept)
    )
    return Run(
        result=Result(system.name, time, names, values),
        events=count,
        event_times=tuple(event.time for event in event_list),
        event_list=event_list,
        steps=int(counters[1]),
        rejected=int(counters[2]),
        evaluations=int(counters[3]),
        jacobians=int(counters[4]),
        factorisations=int(counters[5]),
        end=float(counters[7]),
    )


def drive(  # noqa: PLR0913 - the knobs of `simulate`, plus the entry point
    system: System,
    entry: Any,  # noqa: ANN401 - a ctypes function pointer
    leading: tuple[Any, ...],
    *,
    start: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    events: Sequence[Trigger],
    observed: Sequence[str],
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None = None,
    points: int | None = None,
    tolerance: float = 1e-6,
    absolute: float | None = None,
    every: bool = False,
    t0: float = 0.0,
) -> Run:
    """Call an ``md_run``-shaped entry point and read its buffers back into a `Run`.

    ``entry`` is ``md_run`` from a compiled model, or ``md_vm_run`` from the engine with the
    program handle as ``leading``. Both take the same arrays and return the same status.

    With ``every`` the rows are what `modelica.sim.simulate` returns: the grid points, the
    start and stop instants, and both sides of each event; with no ``times`` or ``points``,
    every accepted step instead of a grid. Without it the rows are the grid points only.
    """
    p, y0, w0 = start
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    y0 = np.ascontiguousarray(np.asarray(y0, dtype=np.float64))
    w0 = np.asarray(w0, dtype=np.float64)
    dirs, lat, term = triggers(events)
    # The rows: exactly the grid without `every`. With it, a guess; the runtime counts the rows
    # it could not write, and one more run with that many fills them all. The two event buffers
    # are far more than a well-behaved run needs; the chattering guard stops a runaway first.
    grid, cap, ev_cap = rows_for(stop, times, points, t0, every=every)
    n, nobs = len(y0), len(observed)
    ev_out = np.zeros(ev_cap, dtype=np.float64)
    ev_idx = np.zeros(ev_cap, dtype=np.float64)
    counters = np.zeros(9, dtype=np.float64)
    entry.restype = ctypes.c_int32
    entry.argtypes = [*(ctypes.c_void_p for _ in leading), *_RUN_ARGUMENTS]
    while True:
        w = np.ascontiguousarray(w0.copy())
        t_out = np.zeros(cap, dtype=np.float64)
        out = np.zeros(cap * n, dtype=np.float64)
        obs = np.zeros(cap * nobs, dtype=np.float64)
        # Every array goes across as its address: the entry point casts each to a slice of the
        # length passed beside it, and the arrays stay alive in this frame for the whole call.
        status = entry(
            *leading,
            float(t0), float(stop),
            y0.ctypes.data, n,
            p.ctypes.data, len(p),
            w.ctypes.data, len(w),
            float(tolerance), float(tolerance if absolute is None else absolute),
            dirs.ctypes.data, lat.ctypes.data, term.ctypes.data, len(dirs),
            grid.ctypes.data, len(grid),
            1 if every else 0, t_out.ctypes.data, cap,
            out.ctypes.data, obs.ctypes.data, nobs,
            ev_out.ctypes.data, ev_idx.ctypes.data, ev_cap,
            counters.ctypes.data,
        )  # fmt: skip
        rows = int(counters[8])
        if rows <= cap:
            break
        cap = rows
    return assemble(
        system,
        status=status,
        counters=counters,
        rows=(t_out, out, obs),
        found=(ev_out, ev_idx),
        events=events,
        observed=observed,
        grid=grid,
        every=every,
    )
