"""One compiled model, many parameter values.

`docs/ideas/vulkan.md:111` puts this first and says why: "The first GPU-worthy target is
**not** the solver. It is the ensemble driver: run one compiled model over a matrix of
parameters. Design it single-threaded first. The shape is already the shape a GPU wants."

Single-threaded is what this is, and deliberately. It is a loop with no opinion about where
the loop runs, which is what makes it the interface a process pool or a device queue can be
put behind without changing a caller. There is no threading here, no multiprocessing, and no
attempt to be clever about ordering.

What it *does* own is the thing a Python `for` loop over `simulate` gets wrong twice.

**The compile happens once.** `modelica.sim.kernel.build` is cached on the content of the
model, so the second point of a sweep would hit that cache anyway -- but a caller who edits a
parameter *in the model* to sweep it changes the content and misses. Sweeping parameter
*values* against one kernel is what keeps the compile out of the loop by construction rather
than by luck.

**A parameter is not an array slot.** Parameters are computed in dependency order --
`RoastingMachine` writes ``m_dry = m_green * (1 - moisture)`` -- so setting ``m_green`` by
writing into the parameter vector leaves ``m_dry`` holding a value derived from the old one.
`modelica.sim.emit.settings` recomputes everything downstream of a change, and this module
goes through it. A sweep that got that wrong would produce a smooth, plausible, wrong curve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modelica.sim.emit import EmitError, settings
from modelica.sim.kernel import build
from modelica.sim.solve import SimulationError, simulate

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.result import Result

__all__ = ["grid", "sweep", "vector"]


def vector(system: System, values: Mapping[str, float] | None = None) -> NDArray[np.float64]:
    """The parameter vector a kernel expects, with ``values`` overriding the model's own.

    In `modelica.sim.emit.Code.parameters` order, which is the order the emitted functions
    read ``p`` in. Everything downstream of an overridden parameter is recomputed from it.
    """
    computed = settings(system, values)
    return np.array([float(computed[variable.name]) for variable in system.parameters], dtype=float)


def grid(**axes: Sequence[float]) -> tuple[dict[str, float], ...]:
    """The full cross product of the axes given, as one mapping per point.

    ``grid(R=[1, 2], C=[0.5, 1.0])`` is four points. Written out rather than left to the
    caller because a sweep over two axes is the common case and `itertools.product` with the
    names zipped back on is the part people get subtly wrong.

    The order is the one `itertools.product` gives: the last axis varies fastest, so a
    reshape of the results to the shape of the axes lands the way a reader expects.
    """
    from itertools import product  # noqa: PLC0415 - one function needs it

    names = list(axes)
    return tuple(
        dict(zip(names, choice, strict=True)) for choice in product(*(axes[n] for n in names))
    )


def sweep(
    system: System,
    over: Iterable[Mapping[str, float]],
    **options: Any,  # noqa: ANN401 - these are `simulate`'s knobs, and it owns their types
) -> tuple[Result, ...]:
    """Integrate ``system`` once per point of ``over``, in the order given.

    ``over`` and not ``points``, because `modelica.sim.solve.simulate` already has a ``points``
    and it means the number of output samples of *one* run. Two meanings for one word in one
    call is how a caller writes ``sweep(system, points=..., points=...)`` and finds out at
    runtime.

    ``options`` go straight to `modelica.sim.solve.simulate` -- ``stop`` is required there and
    so it is required here, and every other knob a run has is that function's.

    One build for the whole sweep. Each point differs only in the parameter vector, which is
    not part of the compiled code, so there is nothing to recompile between them -- that is
    the compile mode `docs/ideas/backends.md:66` calls "leave them as an argument so that one
    compiled artefact serves a whole sweep", and it is the default one.

    ``backend`` picks who integrates, as in `modelica.sim.simulate`. With the engine, the model
    is lowered and loaded once and every point runs on that program.

    Single-threaded, in order, no shared state between points: each gets its own parameter,
    state and workspace arrays. That is what a caller who wants to put a pool
    or a device behind this needs to be able to assume.
    """
    jit = bool(options.pop("jit", False))
    backend = options.pop("backend", "auto")
    vectors = [vector(system, point) for point in over]
    ran = _on_engine(system, vectors, backend=backend, jit=jit, options=options)
    if ran is not None:
        return ran
    kernel = build(system, jit=jit)
    return tuple(simulate(kernel, parameters=p, backend="python", **options) for p in vectors)


#: The `simulate` knobs the engine takes. A sweep with any other runs on SciPy.
_ENGINE_KNOBS = frozenset({"stop", "times", "points", "tolerance", "absolute", "start", "method"})


def _on_engine(
    system: System,
    vectors: list[NDArray[np.float64]],
    *,
    backend: str,
    jit: bool,
    options: dict[str, Any],
) -> tuple[Result, ...] | None:
    """Every point on the engine, loaded once; ``None`` when ``"auto"`` falls back to SciPy.

    The same rule as `modelica.sim.simulate`: ``"engine"`` raises where ``"auto"`` falls back.
    """
    from modelica.sim import engine  # noqa: PLC0415 - the engine is optional

    if backend == "python":
        return None
    supported = (
        not jit and options.get("method", "Radau") == "Radau" and set(options) <= _ENGINE_KNOBS
    )
    if backend == "auto" and (not supported or engine.installed() is None):
        return None
    if not supported:
        message = "backend='engine' sweeps with method='Radau', no jit and the engine's knobs"
        raise SimulationError(message)
    knobs = {key: value for key, value in options.items() if key != "method"}
    try:
        return engine.results(system, vectors, **knobs)
    except EmitError:
        if backend == "engine":
            raise
        return None
