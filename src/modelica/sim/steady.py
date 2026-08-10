"""Where a model settles: ``rhs(t, y, p, w) = 0``, solved by Newton.

`simulate` answers what a model *does*. This answers where it *ends up*, and they are the two
things an engineer does with one thermal model -- sizing and transient -- from one set of
equations. Nothing new is emitted for it: `rhs` already is the residual, and `jacobian` is
d(rhs)/dy exactly, by source transformation rather than by difference quotient, so a Newton
iteration calls two functions that every other part of this package already calls.

**An equilibrium belongs to a mode.** A latched relation is part of the state of the discrete
half, and a model with more than one admissible state set has one equilibrium per set. The
latches are set the way `simulate` sets them at the initial point, and `Steady.mode` says
which set the answer is an equilibrium *of* rather than leaving a caller to assume there is
only one.

**It reports what it found and claims nothing about stability.** `VanDerPol` has a limit cycle
around an unstable fixed point; Newton lands on the fixed point and this says so without
suggesting anything settles there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from modelica.sim.kernel import Kernel, build
from modelica.sim.sweep import vector

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from modelica.ir.system import System

__all__ = ["Steady", "SteadyError", "steady"]

ITERATIONS = 60
"""Newton steps before giving up. Generous: a step is one `rhs` and one `jacobian`."""

TOLERANCE = 1e-10
"""How small the residual has to be, in the infinity norm, to call it an equilibrium."""


class SteadyError(Exception):
    """No equilibrium was found, and the message says which way it failed."""


@dataclass(frozen=True)
class Steady:
    """An equilibrium, and enough context to know what it is an equilibrium of."""

    name: str
    states: tuple[str, ...]
    values: NDArray[np.float64]
    """The state vector where the residual vanished, in `states` order."""

    observed: Mapping[str, float]
    """Every variable of the model at that point, the way `simulate` reports a sample."""

    residual: float
    """The infinity norm of ``rhs`` there. Reported rather than asserted."""

    steps: int
    mode: tuple[str, ...]
    """The state set this is an equilibrium of. A model with one has one."""

    def __getitem__(self, name: str) -> float:
        return self.observed[name]

    def __str__(self) -> str:
        where = ", ".join(f"{n} = {v:.6g}" for n, v in zip(self.states, self.values, strict=True))
        return f"{self.name} settles at {where} (residual {self.residual:.2e}, {self.steps} steps)"


def steady(  # noqa: PLR0913 - a model, an operating point, a guess, and two Newton knobs
    system: System,
    *,
    t: float = 0.0,
    guess: NDArray[np.float64] | None = None,
    values: Mapping[str, float] | None = None,
    tolerance: float = TOLERANCE,
    iterations: int = ITERATIONS,
) -> Steady:
    """Solve ``rhs = 0`` from the model's own start point, or from ``guess``.

    ``values`` overrides parameters the way `modelica.sim.sweep` does, so an equilibrium can
    be asked for at a different operating point without editing the model.

    Newton with no line search and no continuation. That is the first pass the entry behind
    this asked for: a model that needs a path to find its equilibrium is a different problem
    and gets a different entry point.
    """
    kernel = build(system)
    parameters, start, workspace = kernel.start(
        t, None if values is None else vector(system, values)
    )
    width = len(start)
    y = np.array(start if guess is None else guess, dtype=float)
    if y.shape != (width,):
        message = f"{system.name}: a guess must have {width} entries, not {y.shape}"
        raise SteadyError(message)

    residual = np.empty(width)
    matrix = np.zeros((width, width))
    for step in range(iterations):
        kernel.rhs(t, y, parameters, workspace, residual)
        worst = float(np.abs(residual).max()) if width else 0.0
        if worst <= tolerance:
            return _found(system, kernel, t, y, parameters, workspace, worst, step)
        kernel.jacobian(t, y, parameters, workspace, matrix)
        try:
            y = y - np.linalg.solve(matrix, residual)
        except np.linalg.LinAlgError as error:
            message = (
                f"{system.name}: the Jacobian is singular at step {step}, so the residual does"
                f" not determine an equilibrium on its own. A model whose states are tied by a"
                f" conservation law has a family of them, and picking one needs the conserved"
                f" quantity as an extra equation -- which this does not have. ({error})"
            )
            raise SteadyError(message) from error

    kernel.rhs(t, y, parameters, workspace, residual)
    worst = float(np.abs(residual).max()) if width else 0.0
    message = (
        f"{system.name}: {iterations} Newton steps and the residual is still {worst:.3e},"
        f" against a tolerance of {tolerance:.0e}"
    )
    raise SteadyError(message)


def _found(  # noqa: PLR0913, PLR0917 - the state, the arrays it was found with, and the count
    system: System,
    kernel: Kernel,
    t: float,
    y: NDArray[np.float64],
    parameters: NDArray[np.float64],
    workspace: NDArray[np.float64],
    residual: float,
    steps: int,
) -> Steady:
    """Read every variable at the point Newton stopped, and record which mode it was."""
    code = kernel.code
    seen = np.empty(len(code.observed))
    kernel.observe(t, y, parameters, workspace, seen)
    switching = code.switching
    chosen = system.modes[0] if switching is None else system.modes[int(workspace[switching])]
    return Steady(
        name=system.name,
        states=chosen.states,
        values=y.copy(),
        observed=dict(zip(code.observed, (float(v) for v in seen), strict=True)),
        residual=residual,
        steps=steps,
        mode=chosen.states,
    )
