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
from typing import TYPE_CHECKING

import numpy as np

from modelica.sim.kernel import build as build_kernel
from modelica.sim.result import Result
from modelica.sim.rust_radau import CONSTANTS, FIRST_STEP, INTEGRATE, LINALG, NEWTON, STEP
from modelica.sim.rustgen import unit
from modelica.sim.solve import SimulationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System

__all__ = ["Run", "build", "compiler", "program", "simulate"]

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
    ne: usize,
    grid: *const f64,
    npts: usize,
    out: *mut f64,
    obs: *mut f64,
    nobs: usize,
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
    let grid = unsafe { view(grid, npts) };
    let out = unsafe { view_mut(out, npts * n) };
    let obs = unsafe { view_mut(obs, npts * nobs) };
    let r = md_integrate(t0, t1, y0, p, w, rtol, atol, dirs, lat, grid, out, obs);
    let counters = unsafe { view_mut(counters, 8) };
    counters[0] = r.0 as f64;
    counters[1] = r.1 as f64;
    counters[2] = r.2 as f64;
    counters[3] = r.3 as f64;
    counters[4] = r.4 as f64;
    counters[5] = r.5 as f64;
    counters[6] = r.6 as f64;
    counters[7] = r.7;
    r.0
}
"""

#: What the driver's status means, by the constants `rust_radau.STEP` declares.
STATUS = {
    1: "the required step size fell below the spacing between numbers",
    2: "the Newton iteration did not converge",
    3: "the model is chattering: events stopped carrying the run forward",
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
    """How many crossings fired. The instants are not carried back yet; the count is checked
    exactly against the interpreter's, since a wrong tally is a structural bug."""

    steps: int
    rejected: int
    evaluations: int
    jacobians: int
    factorisations: int
    end: float
    """Where the run stopped: the horizon, or earlier when the status said so."""


def _grid(
    stop: float, times: Sequence[float] | NDArray[np.float64] | None, points: int | None
) -> NDArray[np.float64]:
    if times is not None:
        return np.ascontiguousarray(np.asarray(times, dtype=np.float64))
    return np.linspace(0.0, stop, points or 300)


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
    kernel = build_kernel(system)
    library = build(system, cache)
    p, y0, w = kernel.start(0.0, parameters)
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    y0 = np.ascontiguousarray(np.asarray(y0, dtype=np.float64))
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
    events = kernel.code.events
    dirs = np.ascontiguousarray([float(e.direction) for e in events], dtype=np.float64)
    lat = np.ascontiguousarray(
        [e.latch if e.latch is not None else -1 for e in events], dtype=np.int64
    )
    grid = _grid(stop, times, points)
    n, nobs = len(y0), len(kernel.code.observed)
    out = np.zeros(len(grid) * n, dtype=np.float64)
    obs = np.zeros(len(grid) * nobs, dtype=np.float64)
    counters = np.zeros(8, dtype=np.float64)

    # Every array goes across as its address: the entry point casts each to a slice of the
    # length passed beside it, and the arrays stay alive in this frame for the whole call.
    lib = ctypes.CDLL(str(library))
    run = lib.md_run
    run.restype = ctypes.c_int32
    run.argtypes = [
        ctypes.c_double, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p,
    ]  # fmt: skip
    status = run(
        0.0, float(stop),
        y0.ctypes.data, n,
        p.ctypes.data, len(p),
        w.ctypes.data, len(w),
        float(tolerance), float(tolerance if absolute is None else absolute),
        dirs.ctypes.data, lat.ctypes.data, len(dirs),
        grid.ctypes.data, len(grid),
        out.ctypes.data, obs.ctypes.data, nobs,
        counters.ctypes.data,
    )  # fmt: skip
    if status != 0:
        reason = STATUS.get(status, f"status {status}")
        message = f"{system.name}: the Rust run stopped at t = {counters[7]:.12g}: {reason}"
        raise SimulationError(message)
    names = (*system.modes[0].states, *kernel.code.observed)  # type: ignore[attr-defined]
    values = np.vstack([out.reshape(len(grid), n).T, obs.reshape(len(grid), nobs).T])
    return Run(
        result=Result(system.name, grid, names, values),
        events=int(counters[6]),
        steps=int(counters[1]),
        rejected=int(counters[2]),
        evaluations=int(counters[3]),
        jacobians=int(counters[4]),
        factorisations=int(counters[5]),
        end=float(counters[7]),
    )
