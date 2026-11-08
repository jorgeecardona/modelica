"""Run a model in the precompiled engine (``engine/src/lib.rs``): the tape, no compiler per model.

`modelica.sim.vm` lowers the model to a tape; this module hands the tape to the engine library
and integrates with the same Radau runtime `rustsim` compiles. The result is `rustsim.Run`.

The engine is one shared library for every model. A platform wheel of ``modelica`` carries it as
``modelica/_engine.<suffix>`` (``hatch_build.py``). With a pure wheel or a source checkout,
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
    _default_cache,  # pyright: ignore[reportPrivateUsage] - the same cache directory
    _suffix,  # pyright: ignore[reportPrivateUsage]
    drive,
)
from modelica.sim.solve import SimulationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.rustsim import Run

__all__ = ["Loaded", "installed", "library", "load", "result", "simulate"]

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
        self, parameters: NDArray[np.float64] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The parameter vector, the initial state and a latched workspace, as `Kernel.start`.

        ``parameters`` replaces the vector the model computes; the start values follow it.
        """
        layout = self.layout
        empty = np.zeros(0)
        p = np.zeros(len(layout.parameters))
        if parameters is None:
            self.call(self.encoded.parameters, 0.0, empty, empty, empty, p)
        else:
            p[:] = parameters
        y = np.zeros(len(layout.states))
        w = np.zeros(layout.workspace)
        self.call(self.encoded.initial, 0.0, empty, p, w, y)
        self.call(self.encoded.latch, 0.0, y, p, w, empty)
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
) -> Run:
    """Integrate ``system`` in the engine: lower it to a tape, then run the Radau runtime on it.

    ``every`` returns the rows `modelica.sim.simulate` returns (`rustsim.drive`).
    """
    program = vm.lower(system)
    loaded = load(program, path)
    try:
        return drive(
            system,
            loaded.engine.md_vm_run,
            (loaded.handle,),
            start=loaded.start(parameters),
            events=program.layout.events,
            observed=program.layout.observed,
            stop=stop,
            times=times,
            points=points,
            tolerance=tolerance,
            absolute=absolute,
            every=every,
        )
    finally:
        loaded.close()


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
) -> Result:
    """The engine's run as `modelica.sim.simulate` returns one: observed rows and the events."""
    run = simulate(
        system,
        stop,
        times=times,
        points=points,
        tolerance=tolerance,
        absolute=absolute,
        parameters=parameters,
        path=path,
        every=True,
    )
    states = len(system.modes[0].states)
    return Result(
        name=run.result.name,
        time=run.result.time,
        names=run.result.names[states:],
        values=run.result.values[states:],
        events=run.event_list,
        backend="engine",
    )
