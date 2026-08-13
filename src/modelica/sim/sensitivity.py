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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.integrate import (  # pyright: ignore[reportMissingTypeStubs]
    solve_ivp,  # pyright: ignore[reportUnknownVariableType]
)

from modelica.sim.emit import HELD, Trigger
from modelica.sim.kernel import Kernel, build
from modelica.sim.solve import (
    SimulationError,
    _Solution,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.emit import Code

__all__ = [
    "Fit",
    "FitError",
    "Identifiability",
    "Sensitivity",
    "fit",
    "identifiability",
    "sensitivity",
]


@dataclass(frozen=True)
class Sensitivity:
    """``dy/dp`` and ``d(observed)/dp`` along one trajectory."""

    name: str
    time: NDArray[np.float64]
    parameters: tuple[str, ...]
    """The parameters differentiated against, in the column order of every array here."""

    states: tuple[str, ...]
    values: NDArray[np.float64]
    """The state trajectory, shape ``(len(states), len(time))``."""

    trajectory: NDArray[np.float64]
    """Every observed variable, shape ``(len(observed), len(time))``.

    Carried because the caller who wants ``d(obs)/dp`` almost always wants ``obs`` beside it
    -- a fit compares one against a measurement and multiplies by the other -- and running the
    model a second time to get it would be a second answer to the same question.
    """

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
    values: NDArray[np.float64] | None = None,
) -> Sensitivity:
    """Integrate ``dy/dp`` alongside ``y`` and return both.

    ``parameters`` names what to differentiate against. It is passed to `build` as its
    ``sensitivity`` argument, so a kernel handed in directly must already have been built with
    the same list -- otherwise ``jacobian_parameters`` has no columns to fill and this raises
    rather than returning zeros.

    ``initial`` is ``S(0)``, shape ``(states, parameters)``, and defaults to zero. The module
    docstring says why it is not guessed.

    ``values`` runs the model at a parameter vector other than the one it declares, and is
    `modelica.sim.sweep.vector`'s output exactly as `simulate` takes it. A fit moves the
    parameters every iteration and this is how.

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
    unhandled = [trigger.condition for trigger in kernel.code.events if trigger.switch is not None]
    if unhandled:
        # What is left refused is a mode switch, where the crossing does not merely move the
        # state but changes *which variables the state is*. `Mode.derivative` differs either
        # side of it, so `S` is not even the same shape, and that is a different piece of work
        # from a jump.
        message = (
            f"{kernel.name}: {unhandled} switches state sets, and a sensitivity across one is"
            " not a jump -- the state vector itself changes meaning"
        )
        raise SimulationError(message)

    n, m = kernel.states, len(named)
    p, y0, w = kernel.start(start, values)
    s0 = np.zeros((n, m)) if initial is None else np.asarray(initial, dtype=float).reshape(n, m)

    slope = np.zeros((n, n))
    partial = np.zeros((n, m))

    def augmented(t: float, u: NDArray[np.float64]) -> NDArray[np.float64]:
        y, s = u[:n], u[n:].reshape(n, m)
        kernel.jacobian(t, y, p, w, slope)
        kernel.jacobian_parameters(t, y, p, w, partial)
        return np.concatenate([kernel.rhs(t, y, p, w, np.empty(n)), (slope @ s + partial).ravel()])

    pieces = _segments(
        kernel,
        augmented,
        np.concatenate([y0, s0.ravel()]),
        (start, stop),
        p,
        w,
        (n, m),
        method,
        tolerance,
        tolerance if absolute is None else absolute,
    )

    grid = np.unique(np.concatenate([piece for piece, _ in pieces]))
    if times is not None:
        grid = np.unique(np.concatenate([grid, np.asarray(times, dtype=float)]))
    trajectory = np.column_stack([_at(pieces, float(t)) for t in grid])
    states, rows = trajectory[:n], trajectory[n:].reshape(n, m, len(grid))

    # `d(obs)/dp` closes over the two partials the kernel already carries. `observe_jacobian`
    # is `None` when the kernel was built without it, and then the output half is empty rather
    # than wrong -- a caller who wants measurements differentiated has to ask for H.
    observed = kernel.code.observed
    output = np.zeros((len(observed), m, len(grid)))
    seen = np.zeros((len(observed), len(grid)))
    d_obs_dy = np.zeros((len(observed), n))
    d_obs_dp = np.zeros((len(observed), m))
    for column, t in enumerate(grid):
        y = np.ascontiguousarray(states[:, column])
        scratch = w.copy()
        kernel.latch(float(t), y, p, scratch)
        seen[:, column] = kernel.observe(float(t), y, p, scratch, np.empty(len(observed)))
        if kernel.observe_jacobian is not None:
            kernel.observe_jacobian(float(t), y, p, scratch, d_obs_dy)
            kernel.observe_jacobian_parameters(float(t), y, p, scratch, d_obs_dp)
            output[:, :, column] = d_obs_dy @ rows[:, :, column] + d_obs_dp

    return Sensitivity(
        name=kernel.name,
        time=grid,
        parameters=named,
        states=kernel.code.states,
        values=states,
        trajectory=seen,
        state=rows,
        observed=observed,
        output=output,
    )


@dataclass(frozen=True)
class Identifiability:
    """Which parameters a set of measurements can tell apart, from the model alone."""

    name: str
    parameters: tuple[str, ...]
    measurements: tuple[str, ...]
    singular: NDArray[np.float64]
    """Singular values of the scaled sensitivity matrix, descending."""

    directions: NDArray[np.float64]
    """Right singular vectors, one per row, in the same order as :attr:`singular`.

    The last row is the combination of parameters the measurements see *least*. When
    :attr:`condition` is large that row is the answer to "which two cannot be separated": it
    says how to move the parameters so that nothing measured changes.
    """

    @property
    def condition(self) -> float:
        """Largest singular value over smallest. Infinite when a direction is invisible."""
        smallest = float(self.singular[-1])
        return float("inf") if smallest == 0.0 else float(self.singular[0]) / smallest

    def inseparable(self, threshold: float = 1e-6) -> tuple[tuple[str, float], ...]:
        """The parameters in the weakest direction, largest weight first, or nothing.

        Empty when every direction is seen -- when the smallest singular value is more than
        ``threshold`` of the largest. The two corpus readings that bracket the default are
        `RCCircuit`'s ``R`` and ``C`` from the capacitor voltage at **5.1e-15** and
        `MassSpringDamper`'s stiffness and damping from the position at **2.0e-1**. Fourteen
        orders of daylight, so the default sits in the middle of it on a log scale rather than
        at either edge.
        """
        if float(self.singular[-1]) > threshold * float(self.singular[0]):
            return ()
        weights = self.directions[-1]
        order: list[int] = [int(index) for index in np.argsort(-np.abs(weights))]
        return tuple((self.parameters[index], float(weights[index])) for index in order)

    def __str__(self) -> str:
        weak = self.inseparable()
        head = (
            f"{self.name}: {len(self.parameters)} parameters from"
            f" {len(self.measurements)} measurements, condition {self.condition:.3g}"
        )
        if not weak:
            return head + " -- separable"
        combination = " ".join(f"{value:+.3f}*{name}" for name, value in weak)
        return head + f"\n  invisible combination: {combination}"


def identifiability(  # noqa: PLR0913 - the integration knobs, spelled out rather than **kwargs
    model: System | Code | Kernel,
    parameters: Sequence[str],
    measurements: Sequence[str],
    *,
    stop: float,
    points: int = 40,
    tolerance: float = 1e-6,
    absolute: float | None = None,
    method: str = "Radau",
    initial: NDArray[np.float64] | None = None,
    values: NDArray[np.float64] | None = None,
) -> Identifiability:
    """Which of ``parameters`` the ``measurements`` can separate -- **no data required**.

    This is the diagnostic the fit is dangerous without. A least-squares fit over parameters
    that move every measurement together converges to *a* pair, matches the data it was shown,
    is wrong everywhere else, and reports a small residual while being wrong. Whether that can
    happen is a property of the model and the sensor set, not of the data, so it is answerable
    before any data exists.

    The matrix is ``d(measurement)/d(log p)`` stacked over parameters and sample times, and
    the two scalings are the whole of what makes the condition number mean anything.

    **Columns are scaled by the parameter's own value**, which makes them
    ``d m / d log p`` -- a resistance in ohms and a capacitance in farads are otherwise being
    compared by their units, and the condition number would say more about SI than about the
    model.

    **Each measurement's rows are scaled by that measurement's own largest sensitivity.** A
    bath temperature in kelvin and a current in milliamps would otherwise be weighted by how
    big their numbers happen to be, and a well-placed small sensor would count for nothing.

    ``points`` samples the trajectory uniformly. More rows cannot make a visible direction
    invisible; they can only fill in a direction that a coarse grid missed.

    **A large condition number is bounded by ``tolerance``, and a caller reading one should
    know it.** When a direction is genuinely invisible its singular value is round-off, so
    what comes out is how accurately the sensitivity was integrated rather than a property of
    the model: `RCCircuit`'s ``R`` and ``C`` from the voltage read 1.7e10 at the default 1e-6
    and 1.9e14 at 1e-9. The *conclusion* is the same at both -- fourteen orders from anything
    separable -- and the number is not a measurement of the model. A condition number in the
    single digits, which is the useful other answer, does not move with the tolerance.
    """
    named, watched = tuple(parameters), tuple(measurements)
    grid = np.linspace(0.0, stop, points + 1)[1:]
    exact = sensitivity(
        model,
        named,
        stop=stop,
        times=grid,
        tolerance=tolerance,
        absolute=absolute,
        method=method,
        initial=initial,
        values=values,
    )
    taken = np.searchsorted(exact.time, grid)

    kernel = model if isinstance(model, Kernel) else build(model, sensitivity=named)
    declared = list(kernel.code.parameters)
    scale = kernel.parameters(np.zeros(len(declared))) if values is None else values

    blocks: list[NDArray[np.float64]] = []
    for measurement in watched:
        if measurement not in exact.observed:
            message = f"{kernel.name} has no variable {measurement!r} to measure"
            raise SimulationError(message)
        block = exact.output[exact.observed.index(measurement)][:, taken]
        scaled = block * np.array([scale[declared.index(name)] for name in named])[:, None]
        blocks.append(scaled.T / max(float(np.abs(scaled).max()), np.finfo(float).tiny))

    matrix = np.vstack(blocks)
    singular = np.linalg.svd(matrix, compute_uv=False)
    _, _, directions = np.linalg.svd(matrix, full_matrices=False)
    return Identifiability(
        name=kernel.name,
        parameters=named,
        measurements=watched,
        singular=singular,
        directions=directions,
    )


@dataclass(frozen=True)
class Fit:
    """Fitted parameters, the residual they leave, and the conditioning they were fit at."""

    name: str
    parameters: tuple[str, ...]
    values: NDArray[np.float64]
    """The fitted values, in :attr:`parameters` order."""

    residual: float
    """Root-mean-square of the *scaled* residual over every measurement and every sample.

    Each measurement is scaled by its own spread, the way `identifiability` scales its rows,
    so a kelvin and a milliamp weigh the same. An unscaled residual is a number about units.
    """

    iterations: int
    conditioning: Identifiability
    """What `identifiability` said **at the fitted values**, carried rather than summarised.

    This is why `Fit` is a record and not a vector. A residual on its own is exactly the thing
    that makes a bad fit look like a good one: parameters that move every measurement together
    drive the residual to nothing and land anywhere along the invisible direction. The two
    numbers only mean something read together, so they are returned together.
    """

    def __str__(self) -> str:
        fitted = ", ".join(
            f"{name} = {value:.6g}"
            for name, value in zip(self.parameters, self.values, strict=True)
        )
        return (
            f"{self.name}: {fitted}\n  residual {self.residual:.3e} after {self.iterations}"
            f" iterations, at condition {self.conditioning.condition:.3g}"
        )


class FitError(Exception):
    """A fit that was refused, or that did not settle, and why."""


ITERATIONS = 40
"""Gauss-Newton steps before giving up. One step is one sensitivity integration."""

SETTLED = 1e-10
"""Relative parameter movement below which the iteration has stopped moving."""


def fit(  # noqa: PLR0913 - what is being fitted, plus the integration knobs
    model: System | Code | Kernel,
    parameters: Sequence[str],
    measurements: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    stop: float,
    tolerance: float = 1e-8,
    method: str = "Radau",
    separable: float = 1e-6,
) -> Fit:
    """Fit ``parameters`` to ``measurements`` by Gauss-Newton on exact sensitivities.

    ``measurements`` maps an observed variable to ``(times, values)``. The times must lie in
    ``[0, stop]``; they need not be the same for two measurements, and are not resampled.

    **A parameter set the measurements cannot separate is refused**, which is the reason this
    lives beside `identifiability` rather than in a caller's script. The report is taken at the
    starting parameters, and a subset whose weakest direction falls below ``separable`` gets a
    `FitError` naming the invisible combination instead of a number that looks like an answer.
    ``separable=0`` fits anyway: the refusal is a default and not a prohibition, because a
    caller who knows the product is what they want is not making a mistake.

    Gauss-Newton, on an exact Jacobian, which is the case it is for. Not a global optimiser and
    not a trust region: the entry that asked for this rules the first out because a global
    search on an unidentifiable problem only finds one of the many answers faster, and the
    second is machinery this has no evidence of needing. When Gauss-Newton fails it fails by
    not settling, and that is reported rather than returned.
    """
    named, watched = tuple(parameters), tuple(measurements)
    kernel = model if isinstance(model, Kernel) else build(model, sensitivity=named)
    for measurement in watched:
        if measurement not in kernel.code.observed:
            message = f"{kernel.name} has no variable {measurement!r} to measure"
            raise FitError(message)

    grid = np.unique(
        np.concatenate([np.asarray(times, dtype=float) for times, _ in measurements.values()])
    )
    if grid.min() < 0.0 or grid.max() > stop:
        message = (
            f"{kernel.name}: measurement times {grid.min():g}..{grid.max():g} leave [0, {stop:g}]"
        )
        raise FitError(message)

    declared = list(kernel.code.parameters)
    slots = [declared.index(name) for name in named]
    point = kernel.parameters(np.zeros(len(declared)))

    def moved(current: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = point.copy()
        vector[slots] = current
        return vector

    def conditioning(current: NDArray[np.float64]) -> Identifiability:
        return identifiability(
            kernel,
            named,
            watched,
            stop=stop,
            points=max(len(grid), 8),
            tolerance=tolerance,
            method=method,
            values=moved(current),
        )

    values = point[slots].astype(float)
    start = conditioning(values)
    if separable and start.inseparable(separable):
        combination = " ".join(f"{w:+.3f}*{n}" for n, w in start.inseparable(separable))
        message = (
            f"{kernel.name}: {list(watched)} cannot separate {list(named)} -- condition"
            f" {start.condition:.3g}, invisible combination {combination}. Add a measurement,"
            " fit fewer parameters, or pass `separable=0` and read the conditioning yourself"
        )
        raise FitError(message)

    residual = np.zeros(0)
    taken_steps = 0
    for taken_steps in range(1, ITERATIONS + 1):  # noqa: B007 - the count is reported
        exact = sensitivity(
            kernel,
            named,
            stop=stop,
            times=grid,
            tolerance=tolerance,
            method=method,
            values=moved(values),
        )
        taken = np.searchsorted(exact.time, grid)
        errors: list[NDArray[np.float64]] = []
        rows: list[NDArray[np.float64]] = []
        for measurement, (times, wanted) in measurements.items():
            row = exact.observed.index(measurement)
            where = np.searchsorted(grid, np.asarray(times, dtype=float))
            target = np.asarray(wanted, dtype=float)
            spread = max(float(np.ptp(target)), 1.0)
            errors.append((exact.trajectory[row][taken][where] - target) / spread)
            rows.append(exact.output[row][:, taken][:, where].T / spread)
        residual = np.concatenate(errors)
        step, *_ = np.linalg.lstsq(np.vstack(rows), -residual, rcond=None)
        values = values + step
        if float(np.max(np.abs(step) / np.maximum(np.abs(values), 1e-30))) < SETTLED:
            break
    else:
        message = f"{kernel.name}: {ITERATIONS} Gauss-Newton steps without settling"
        raise FitError(message)

    return Fit(
        name=kernel.name,
        parameters=named,
        values=values,
        residual=float(np.sqrt(np.mean(residual**2))),
        iterations=taken_steps,
        conditioning=conditioning(values),
    )


def _segments(  # noqa: PLR0913, PLR0917 - the run, and everything one segment is read against
    kernel: Kernel,
    augmented: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    u: NDArray[np.float64],
    span: tuple[float, float],
    p: NDArray[np.float64],
    w: NDArray[np.float64],
    shape: tuple[int, int],
    method: str,
    relative: float,
    absolute: float,
) -> list[tuple[NDArray[np.float64], _Solution]]:
    """Integrate to the horizon, stopping at each crossing to move ``S`` across it.

    One segment per crossing, which is what `modelica.sim.solve` does for the state alone. The
    latches live in ``w`` and are flipped by `Kernel.transition`, so the segment after a
    crossing integrates the model on the other side of it -- which is the half the refusal
    this replaced was really about.
    """
    n, m = shape
    start, stop = span
    pieces: list[tuple[NDArray[np.float64], _Solution]] = []
    at = start
    while at < stop:
        try:
            solution = cast(
                "_Solution",
                solve_ivp(  # pyright: ignore[reportUnknownArgumentType]
                    augmented,
                    (at, stop),
                    u,
                    method=method,
                    rtol=relative,
                    atol=absolute,
                    dense_output=True,
                    events=_crossings(kernel, p, w, n) or None,
                ),
            )
        except ValueError as error:
            # The same guard `modelica.sim.solve._segment` carries, for the same reason: a
            # `ValueError` out of SciPy's internals tells a modeller nothing. `DrainingTank`
            # is the case here -- past the moment it runs dry its `sqrt` is of a negative
            # level, and the augmented system reaches further into that region than the state
            # alone does because the Jacobian is evaluated there too.
            message = (
                f"{kernel.name}: the augmented system left the model's domain (scipy: {error})"
            )
            raise SimulationError(message) from error
        if not solution.success:
            message = f"{kernel.name}: {solution.message}"
            raise SimulationError(message)
        pieces.append((np.asarray(solution.t, dtype=float), solution))
        fired = _fired(solution)
        if fired is None:
            break
        index, at = fired
        u = _jump(kernel, index, at, solution.sol(np.array([at]))[:, 0], p, w, n, m)
    return pieces


def _crossings(kernel: Kernel, p: NDArray[np.float64], w: NDArray[np.float64], n: int) -> list[Any]:
    """One SciPy event function per watched relation, read off the augmented state.

    The indicator is nudged to the side its latch is holding, the same `_REARM` trick
    `modelica.sim.solve` uses and for the same reason: a segment restarts exactly on the root
    the previous one located, so the fired indicator reads as zero give or take noise and
    SciPy is entitled to report that noise as a fresh crossing forever.
    """
    from modelica.sim.solve import (  # noqa: PLC0415 - one constant, and it is theirs
        _REARM,  # pyright: ignore[reportPrivateUsage]
    )

    scratch = np.empty(kernel.events)

    def watcher(index: int, trigger: Trigger) -> Callable[[float, NDArray[np.float64]], float]:
        def crossing(t: float, u: NDArray[np.float64]) -> float:
            # The side a just-fired crossing restarts on, copied from `solve._offset` rather
            # than guessed: a latched relation sits on the side its latch is holding, and a
            # `when` branch fired as its condition became true, so it restarts on the true
            # side. Writing `trigger.latch or 0` here instead would silently read slot zero,
            # which is a *real* latch, and the wrongness would look like a working run.
            side = -_REARM if trigger.latch is not None and not w[trigger.latch] > HELD else _REARM
            return float(kernel.indicators(t, u[:n], p, w, scratch)[index] + side)

        # A `when` fires only as its condition becomes true; a relation is an event whichever
        # way it turns. `Trigger.direction` already carries that and SciPy spells it the same.
        crossing.terminal = True  # pyright: ignore[reportFunctionMemberAccess]
        crossing.direction = trigger.direction  # pyright: ignore[reportFunctionMemberAccess]
        return crossing

    return [watcher(index, trigger) for index, trigger in enumerate(kernel.code.events)]


def _fired(solution: _Solution) -> tuple[int, float] | None:
    """Which crossing ended this segment, and when. ``None`` if it reached the horizon.

    ``t_events`` is `None` and not empty when the model has no crossings to watch, because
    that is what SciPy returns when ``events=`` was not passed. Nine of the corpus's
    twenty-five models have events and the other sixteen came through here.
    """
    if solution.t_events is None:
        return None
    for index, times in enumerate(solution.t_events):
        if len(times):
            return index, float(times[0])
    return None


def _jump(  # noqa: PLR0913, PLR0917 - the crossing, and everything it is read against
    kernel: Kernel,
    index: int,
    at: float,
    u: NDArray[np.float64],
    p: NDArray[np.float64],
    w: NDArray[np.float64],
    n: int,
    m: int,
) -> NDArray[np.float64]:
    """Apply the transition and move ``S`` with it.

    ``S+ = H S- + dh/dp + (H f- - f+) dt*/dp`` with ``dt*/dp = -(g_y S-) / (dg/dt)``, where
    ``H`` is `Kernel.transition_jacobian` -- the identity for a crossing that only flips a
    latch or sets a discrete variable, and something else where a ``reinit`` moves the state.

    ``dh/dp`` is not optional and `BouncingBall` is why. Its ``reinit(v, -e * pre(v))`` puts
    ``e`` *only* in the reset, so with that term missing the sensitivity to ``g`` came out at
    5.5e-10 and the sensitivity to ``e`` at 1.0 -- wrong by the whole of itself.

    **The order inside the bracket is not a detail.** Writing ``f+ - H f-`` is the obvious
    mistake and it is not obviously wrong: it leaves the sensitivity continuous where it
    should be, and it moves in the right direction. The derivation is that just after the
    crossing ``y(t) = h(y(t*)) + f+ (t - t*)``, so differentiating at fixed ``t`` gives
    ``S+ = H [S- + f- dt*/dp] - f+ dt*/dp``. On the model in
    `tests/test_sensitivity_forward.py` the two differ by a factor of three -- -0.75 against
    the closed form's -0.25 -- which is how it was caught while ``H`` was still the identity.
    """
    y, s = u[:n], u[n:].reshape(n, m)
    gradient = kernel.indicator_jacobian(at, y, p, w, np.zeros((kernel.events, n)))[index].copy()
    slope = float(kernel.indicator_slope(at, y, p, w, np.zeros(kernel.events))[index])
    before = kernel.rhs(at, y, p, w, np.empty(n)).copy()
    h = kernel.transition_jacobian(index, at, y, p, w, np.zeros((n, n))).copy()
    direct = kernel.transition_jacobian_parameters(index, at, y, p, w, np.zeros((n, m))).copy()
    # `transition` returns the state *after* the crossing, which for a `reinit` is not the one
    # it was handed. Everything above is read at the pre-event point on purpose: `H`, `dh/dp`,
    # `g_y`, the slope and `f-` are all properties of where the run arrived, not of where it
    # leaves. Reading `dh/dp` one line too late was worth a factor of exactly `-e` on
    # `BouncingBall` -- `-pre(v)` evaluated at the reflected velocity instead of the incoming
    # one -- and a constant factor across every sample is what made it findable.
    y = np.ascontiguousarray(kernel.transition(index, at, y, p, w, np.empty(n)))
    after = kernel.rhs(at, y, p, w, np.empty(n)).copy()
    if slope == 0.0:  # pragma: no cover - a crossing reached with no speed cannot be located
        return u
    moved = -(gradient @ s) / slope
    jumped = h @ s + direct + np.outer(h @ before - after, moved)
    return np.concatenate([y, np.asarray(jumped, dtype=float).ravel()])


def _at(pieces: list[tuple[NDArray[np.float64], _Solution]], t: float) -> NDArray[np.float64]:
    """The augmented state at ``t``, from whichever segment covers it.

    The *last* segment whose span reaches ``t``, so a time that is exactly a crossing reads
    the value after the jump. That is the same convention `modelica.sim.solve` uses for its
    samples and the reason `tests/test_solution_set.py` skips an event instant: one timestamp
    names two states and something has to choose.
    """
    for grid, solution in reversed(pieces):
        if grid[0] <= t <= grid[-1]:
            return np.asarray(solution.sol(np.array([t]))[:, 0], dtype=float)
    return np.asarray(pieces[-1][1].sol(np.array([t]))[:, 0], dtype=float)
