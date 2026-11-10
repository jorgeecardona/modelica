"""Simulate a model through the Rust backend: emit it, compile it once, call it.

The execution model `docs/issues/03-09-2026-a-rust-backend-is-a-third-dialect-plus-a-port-of-radau.md`
decided: ahead of time and native, no compiler on the run path. `program` writes the model's
unit (`rustgen`) and the radau runtime (`rust_radau`) into one Rust source behind a C entry
point; `build` compiles that source into a shared library **once**, keyed by the source's
content hash, so a model that has not changed is never compiled again; `simulate` loads the
library through `ctypes` and runs it, handing back the same `Result` the interpreter does, so
the two tiers can be compared by the same tests and drawn by the same panel.

`rustc` is the whole toolchain -- no cargo, no crate, no runtime library -- and it is needed
only when the cache misses. The browser form (WebAssembly) is the same source under a different
target and is not built here yet.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from modelica.sim.kernel import build as build_kernel
from modelica.sim.result import Event, Result
from modelica.sim.rust_radau import CONSTANTS, FIRST_STEP, INTEGRATE, LINALG, NEWTON, STEP
from modelica.sim.rustgen import unit
from modelica.sim.solve import SimulationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.emit import Trigger

__all__ = [
    "Run",
    "assemble",
    "build",
    "compiler",
    "drive",
    "program",
    "rows_for",
    "simulate",
    "triggers",
]

#: The C entry point over `md_integrate`. Every pointer comes from a NumPy array the caller
#: keeps alive for the duration of the call, with its length beside it; a zero length means
#: the array may be empty, and an empty slice is made without reading the pointer. The eight
#: counters the driver returns come back through ``counters`` so the status can be the return
#: value, which is what a C caller checks first.
ENTRY = """\
#[no_mangle]
pub extern "C" fn md_run(
    t0: f64,
    t1: f64,
    y0: *const f64,
    n: usize,
    p: *const f64,
    np: usize,
    w: *mut f64,
    nw: usize,
    rtol: f64,
    atol: f64,
    dirs: *const f64,
    lat: *const i64,
    term: *const i64,
    ne: usize,
    grid: *const f64,
    npts: usize,
    every: i32,
    t_out: *mut f64,
    cap: usize,
    out: *mut f64,
    obs: *mut f64,
    nobs: usize,
    events_out: *mut f64,
    ev_index_out: *mut f64,
    nev_out: usize,
    counters: *mut f64,
) -> i32 {
    unsafe fn view<'a>(ptr: *const f64, len: usize) -> &'a [f64] {
        if len == 0 { &[] } else { std::slice::from_raw_parts(ptr, len) }
    }
    unsafe fn view_mut<'a>(ptr: *mut f64, len: usize) -> &'a mut [f64] {
        if len == 0 { &mut [] } else { std::slice::from_raw_parts_mut(ptr, len) }
    }
    let (y0, p, w) = unsafe { (view(y0, n), view(p, np), view_mut(w, nw)) };
    let dirs = unsafe { view(dirs, ne) };
    let lat: &[i64] = if ne == 0 { &[] } else { unsafe { std::slice::from_raw_parts(lat, ne) } };
    let term: &[i64] = if ne == 0 { &[] } else { unsafe { std::slice::from_raw_parts(term, ne) } };
    let grid = unsafe { view(grid, npts) };
    let t_out = unsafe { view_mut(t_out, cap) };
    let out = unsafe { view_mut(out, cap * n) };
    let obs = unsafe { view_mut(obs, cap * nobs) };
    let events_out = unsafe { view_mut(events_out, nev_out) };
    let ev_index_out = unsafe { view_mut(ev_index_out, nev_out) };
    let r = md_integrate(t0, t1, y0, p, w, rtol, atol, dirs, lat, term, &[], grid, &[], &[], every != 0, t_out, out, obs, events_out, ev_index_out);
    let counters = unsafe { view_mut(counters, 9) };
    counters[0] = r.0 as f64;
    counters[1] = r.1 as f64;
    counters[2] = r.2 as f64;
    counters[3] = r.3 as f64;
    counters[4] = r.4 as f64;
    counters[5] = r.5 as f64;
    counters[6] = r.6 as f64;
    counters[7] = r.7;
    counters[8] = r.8 as f64;
    r.0
}
"""

#: What the driver's status means, by the constants `rust_radau.STEP` declares.
STATUS = {
    1: "the required step size fell below the spacing between numbers",
    2: "the Newton iteration did not converge",
    3: "the model is chattering: events stopped carrying the run forward",
    4: "the model reached a terminate() and the run stopped there",
}


def compiler() -> str | None:
    """The ``rustc`` on PATH, or ``None``: the backend needs it only when the cache misses."""
    return shutil.which("rustc")


def program(system: System) -> str:
    """The model and the integrator as one Rust source, behind the C entry point."""
    return "\n".join(
        [
            "#![allow(dead_code)]",
            unit(system),
            *CONSTANTS,
            *LINALG,
            *STEP,
            *NEWTON,
            *FIRST_STEP,
            *INTEGRATE,
            ENTRY,
        ]
    )


def _default_cache() -> Path:
    """``$PYMODELICA_CACHE/rust``, else the user's cache directory."""
    root = os.environ.get("PYMODELICA_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "pymodelica"
    return base / "rust"


def _suffix() -> str:
    """What the platform calls a shared library."""
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    if system == "Windows":
        return ".dll"
    return ".so"


def build(system: System, cache: Path | None = None) -> Path:
    """The shared library for ``system``, compiled once and kept under its content hash.

    The key is the hash of the complete source, so any change to the model, to the emitter or
    to the runtime is a different artifact, and an unchanged model costs a hash and a stat. The
    compile goes to a temporary name in the same directory and is renamed into place, so two
    processes building the same model cannot leave each other a half-written library.
    """
    source = program(system)
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    folder = cache if cache is not None else _default_cache()
    folder.mkdir(parents=True, exist_ok=True)
    library = folder / f"{key}{_suffix()}"
    if library.exists():
        return library
    rustc = compiler()
    if rustc is None:
        message = f"{system.name}: no rustc on PATH, and no compiled library for it in {folder}"
        raise SimulationError(message)
    source_path = folder / f"{key}.rs"
    source_path.write_text(source, encoding="utf-8")
    with tempfile.NamedTemporaryFile(dir=folder, suffix=_suffix(), delete=False) as handle:
        partial = Path(handle.name)
    done = subprocess.run(  # noqa: S603 - the compiler from PATH, on source generated here
        [
            rustc,
            "--edition",
            "2021",
            "-O",
            "--crate-type",
            "cdylib",
            str(source_path),
            "-o",
            str(partial),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        partial.unlink(missing_ok=True)
        message = f"{system.name}: rustc failed on {source_path}\n{done.stderr}"
        raise SimulationError(message)
    partial.replace(library)
    return library


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


def simulate(  # noqa: PLR0913, PLR0917 - a solver has knobs, the same ones `solve.simulate` has
    system: System,
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None = None,
    points: int | None = None,
    tolerance: float = 1e-6,
    absolute: float | None = None,
    parameters: NDArray[np.float64] | None = None,
    cache: Path | None = None,
) -> Run:
    """Integrate ``system`` from its start to ``stop`` in the compiled Rust runtime.

    The same knobs as `modelica.sim.simulate`, the same `Result` back: states and observed
    variables at ``times`` (or ``points`` evenly spaced), by name. The library is built on the
    first call for a model and reused after; a status other than success raises
    `SimulationError` with what the driver reported.
    """
    library = build(system, cache)
    kernel = build_kernel(system)
    return drive(
        system,
        ctypes.CDLL(str(library)).md_run,
        (),
        start=kernel.start(0.0, parameters),
        events=kernel.code.events,
        observed=kernel.code.observed,
        stop=stop,
        times=times,
        points=points,
        tolerance=tolerance,
        absolute=absolute,
    )


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
