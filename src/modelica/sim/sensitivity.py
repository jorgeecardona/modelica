"""How a trajectory moves when a parameter does: ``dy/dp`` integrated alongside ``y``.

`modelica.sim.sweep` answers the same question by running the model twice and subtracting.
That is honest and it is what a difference quotient always is: the answer is limited by the
step you chose, it costs one full run per parameter, and near a stiff transient it is limited
by the *solver's* tolerance rather than by the step. This integrates the sensitivity instead.

The construction is the forward-sensitivity ODE. With ``y' = f(t, y, p)`` and
``S = dy/dp``, differentiating under the integral gives

    S' = (df/dy) S + (df/dp),        S(0) = dy0/dp

and both partials are already emitted, exactly, by source transformation:
`Kernel.jacobian` is ``df/dy`` and `Kernel.jacobian_parameters` is ``df/dp`` over the
parameters `Code.sensitivity` names. So the augmented system is the model's own right-hand
side stacked with a matrix product, and nothing here differences anything.

**``S(0)`` is the caller's, and defaults to zero.** A start value written in terms of a
parameter -- ``Real T(start = T_amb)`` -- has ``dy0/dp`` non-zero, and no function in the
emitted module computes it: `initial` maps ``p`` to ``y0`` and its derivative is not written
anywhere. Passing ``initial=`` is how a caller says what it is; leaving it out asserts that
the fitted parameters do not reach the initial state, which is true of most models and false
of some. `sensitivity` does not guess, because a wrong ``S(0)`` is wrong for the whole run and
looks like a modelling error rather than an unset argument.

**A model with events is refused.** Not approximated: the latches would never be refreshed
and ``S`` has no jump condition at a crossing, so the answer would be wrong in a way that
looks like an answer. `RoastingMachine` is the measurement -- integrated this way its own
trajectory comes out 8.8e-2 from `simulate`'s, before any sensitivity is asked for.

This is the first of the three pieces
`docs/issues/09-08-2026-nothing-fits-a-parameter-to-data-and-the-reason-that-is-hard.md`
asks for, and deliberately only the first: it says how the trajectory moves, not which
parameters can be told apart, and not what values fit a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.integrate import (  # pyright: ignore[reportMissingTypeStubs]
    solve_ivp,  # pyright: ignore[reportUnknownVariableType]
)

from modelica.sim.kernel import Kernel, build
from modelica.sim.solve import (
    SimulationError,
    _Solution,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.emit import Code

__all__ = ["Sensitivity", "sensitivity"]


@dataclass(frozen=True)
class Sensitivity:
    """``dy/dp`` and ``d(observed)/dp`` along one trajectory."""

    name: str
    time: NDArray[np.float64]
    parameters: tuple[str, ...]
    """The parameters differentiated against, in the column order of every array here."""

    states: tuple[str, ...]
    values: NDArray[np.float64]
    """The trajectory itself, shape ``(len(states), len(time))``."""

    state: NDArray[np.float64]
    """``dy/dp``, shape ``(len(states), len(parameters), len(time))``."""

    observed: tuple[str, ...]
    output: NDArray[np.float64]
    """``d(observed)/dp``, shape ``(len(observed), len(parameters), len(time))``.

    Not the same question as :attr:`state` and usually the one a caller wants. A measurement
    is almost never a state: it is a probe temperature, a branch current, a lag -- an
    *observed* variable, often an alias causalization eliminated. The chain rule closes over
    the two partials the kernel already has,
    ``d(obs)/dp = (d(obs)/dy) S + (d(obs)/dp)``, so this costs one matrix product per sample
    and no extra integration.
    """

    def __getitem__(self, key: tuple[str, str]) -> NDArray[np.float64]:
        """``s["T_bean", "m_green"]`` -- one variable's sensitivity to one parameter."""
        name, parameter = key
        try:
            column = self.parameters.index(parameter)
        except ValueError:
            raise KeyError(f"{self.name} was not differentiated against {parameter!r}") from None
        if name in self.observed:
            return self.output[self.observed.index(name), column]
        try:
            return self.state[self.states.index(name), column]
        except ValueError:
            raise KeyError(f"{self.name} has no variable {name!r}") from None


def sensitivity(  # noqa: PLR0913 - a solver has knobs, and this one has the same ones
    model: System | Code | Kernel,
    parameters: Sequence[str],
    *,
    stop: float,
    start: float = 0.0,
    tolerance: float = 1e-6,
    absolute: float | None = None,
    times: Sequence[float] | NDArray[np.float64] | None = None,
    method: str = "Radau",
    initial: NDArray[np.float64] | None = None,
) -> Sensitivity:
    """Integrate ``dy/dp`` alongside ``y`` and return both.

    ``parameters`` names what to differentiate against. It is passed to `build` as its
    ``sensitivity`` argument, so a kernel handed in directly must already have been built with
    the same list -- otherwise ``jacobian_parameters`` has no columns to fill and this raises
    rather than returning zeros.

    ``initial`` is ``S(0)``, shape ``(states, parameters)``, and defaults to zero. The module
    docstring says why it is not guessed.

    The augmented system is integrated as one vector, ``[y; S.ravel()]``, by the same SciPy
    method `simulate` uses. Its Jacobian is *not* supplied: the augmented Jacobian is block
    triangular with ``df/dy`` on the diagonal repeated ``m + 1`` times, and building that
    ``n(m+1)`` square array per step costs more than the extra Newton iterations it saves at
    the sizes this is for. That is a judgement, not a measurement, and the honest place to
    revisit it is a model with many parameters.
    """
    kernel = model if isinstance(model, Kernel) else build(model, sensitivity=tuple(parameters))
    named = tuple(parameters)
    if kernel.code.sensitivity != named:
        message = (
            f"{kernel.name}: the kernel differentiates against {kernel.code.sensitivity} and"
            f" {named} was asked for -- build it with `sensitivity=` to match"
        )
        raise SimulationError(message)
    if not named:
        message = f"{kernel.name}: no parameters to differentiate against"
        raise SimulationError(message)
    if kernel.events:
        # Refused rather than approximated, and for two separate reasons. The latches are set
        # once here and never refreshed, so a model with a watched relation is integrated on
        # whichever side of it the run started -- measured on `RoastingMachine`, whose own
        # trajectory comes out 8.8e-2 from `simulate`'s, before any sensitivity is asked for.
        # And `S` needs a jump condition at an event: a `when` that reinitialises a state
        # moves `dy/dp` too, by an amount that depends on the crossing time, which itself
        # depends on `p`. Neither is hard and both are their own piece of work; returning a
        # number that is quietly wrong for eight of the corpus models is not an option.
        message = (
            f"{kernel.name}: {kernel.events} watched relations, and forward sensitivity here"
            " does not handle events -- the latches are never refreshed and `S` has no jump"
            " condition at a crossing"
        )
        raise SimulationError(message)

    n, m = kernel.states, len(named)
    p, y0, w = kernel.start(start)
    s0 = np.zeros((n, m)) if initial is None else np.asarray(initial, dtype=float).reshape(n, m)

    slope = np.zeros((n, n))
    partial = np.zeros((n, m))

    def augmented(t: float, u: NDArray[np.float64]) -> NDArray[np.float64]:
        y, s = u[:n], u[n:].reshape(n, m)
        kernel.jacobian(t, y, p, w, slope)
        kernel.jacobian_parameters(t, y, p, w, partial)
        return np.concatenate([kernel.rhs(t, y, p, w, np.empty(n)), (slope @ s + partial).ravel()])

    solution = cast(
        "_Solution",
        solve_ivp(  # pyright: ignore[reportUnknownArgumentType]
            augmented,
            (start, stop),
            np.concatenate([y0, s0.ravel()]),
            method=method,
            rtol=tolerance,
            atol=tolerance if absolute is None else absolute,
            dense_output=True,
        ),
    )
    if not solution.success:
        message = f"{kernel.name}: {solution.message}"
        raise SimulationError(message)

    # Requested times are *added* to the solver's own steps and filled in from the
    # interpolant, which is what `simulate` does. Replacing the steps would be the other
    # reasonable choice and the two functions disagreeing about it would be a trap.
    grid = np.asarray(solution.t, dtype=float)
    if times is not None:
        grid = np.unique(np.concatenate([grid, np.asarray(times, dtype=float)]))
    trajectory = solution.sol(grid)
    states, rows = trajectory[:n], trajectory[n:].reshape(n, m, len(grid))

    # `d(obs)/dp` closes over the two partials the kernel already carries. `observe_jacobian`
    # is `None` when the kernel was built without it, and then the output half is empty rather
    # than wrong -- a caller who wants measurements differentiated has to ask for H.
    observed = kernel.code.observed
    output = np.zeros((len(observed), m, len(grid)))
    if kernel.observe_jacobian is not None:
        d_obs_dy = np.zeros((len(observed), n))
        d_obs_dp = np.zeros((len(observed), m))
        for column, t in enumerate(grid):
            y = np.ascontiguousarray(states[:, column])
            scratch = w.copy()
            kernel.latch(float(t), y, p, scratch)
            kernel.observe_jacobian(float(t), y, p, scratch, d_obs_dy)
            kernel.observe_jacobian_parameters(float(t), y, p, scratch, d_obs_dp)
            output[:, :, column] = d_obs_dy @ rows[:, :, column] + d_obs_dp

    return Sensitivity(
        name=kernel.name,
        time=grid,
        parameters=named,
        states=kernel.code.states,
        values=states,
        state=rows,
        observed=observed,
        output=output,
    )
