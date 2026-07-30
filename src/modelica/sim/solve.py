"""Integrating a causalized model, events and all.

The loop here is short because everything hard happened upstream: causalization already
decided what the states are and in what order to compute the rest, and the emitter already
turned that into straight-line arithmetic. What is left is to hand the derivative function
to SciPy, stop when a ``when``-condition crosses zero, apply the discontinuity, and start
again from the other side.

Restarting is not an optimisation, it is the only correct thing to do. An integrator
carries an implicit assumption that the function it is integrating is smooth over the step
it is taking; a bouncing ball violates that assumption exactly once per bounce, and a
solver that is not told about it will either smear the discontinuity over a step or cut the
step size to nothing trying to resolve it.

SciPy is behind the ``sim`` extra, so this module is not imported by :mod:`modelica.sim`
until someone actually calls :func:`simulate`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from scipy.integrate import (  # pyright: ignore[reportMissingTypeStubs]
    solve_ivp,  # pyright: ignore[reportUnknownVariableType]
)

from modelica.sim.emit import HELD
from modelica.sim.kernel import Kernel, build
from modelica.sim.result import Event, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.emit import Code, Trigger

__all__ = ["SimulationError", "simulate"]

_CHATTER = 100
"""Consecutive zero-length segments before we call it a Zeno point and give up."""

_STANDSTILL = 1e-12
"""Relative segment length below which a segment counts as not having advanced at all."""

_ITERATIONS = 100
"""Events allowed to act at one instant before the iteration is called non-convergent."""

_REARM = 1e-9
"""How far a crossing that just fired is held off zero while the run leaves it behind.

A segment restarts exactly on the root the previous one located, so the fired indicator
reads as zero give or take noise -- and SciPy, asked for crossings in either direction,
is entitled to report that noise as a fresh event at the same instant, forever. A ball's
``when`` moves the state off the root by reinitialising it and a mode switch carries its
own hysteresis margin, but a relation crossed in passing -- beans rising through boiling,
a gas schedule reaching its cut -- leaves nothing behind to step over. So the crossing
that fired is nudged to the side its latch says it is on, for the segment that follows,
and the nudge is money: it moves the *next* root of that relation by ``_REARM`` over the
indicator's slope, which is orders below the tolerance any spec asserts an event time to.
"""


class SimulationError(Exception):
    """The integration did not reach the end, and why."""


class _Solution(Protocol):
    """What :func:`scipy.integrate.solve_ivp` hands back, spelled out.

    SciPy returns an untyped bunch, so without this every array that comes out of it is
    unknown to the type checker and every NumPy call downstream of it goes unchecked.
    Naming the attributes we actually read is cheaper than an ignore on each of them, and
    it states what this module depends on: if a SciPy release changes any of it, the
    failure lands here rather than somewhere in the middle of the loop.
    """

    success: bool
    message: str
    status: int
    t: NDArray[np.float64]
    """The steps the integrator took, ends included."""

    y: NDArray[np.float64]
    """The state at each of them, one row per state."""

    t_events: list[NDArray[np.float64]]
    y_events: list[NDArray[np.float64]]

    def sol(self, t: NDArray[np.float64], /) -> NDArray[np.float64]:
        """The dense interpolant, evaluated wherever we ask."""
        ...


def simulate(  # noqa: PLR0913 - a solver has knobs
    model: System | Code | Kernel,
    *,
    stop: float,
    start: float = 0.0,
    tolerance: float = 1e-6,
    absolute: float | None = None,
    times: Sequence[float] | NDArray[np.float64] | None = None,
    points: int | None = None,
    method: str = "Radau",
    jit: bool = False,
) -> Result:
    """Integrate ``model`` from ``start`` to ``stop`` and return every variable it has.

    ``tolerance`` is the relative tolerance; ``absolute`` defaults to it. ``times`` asks
    for output at exactly those instants, ``points`` for a uniform grid of that many; with
    neither, the solver's own steps are what comes back, which is the honest picture of
    where it had to work hard.

    Requested times are added to the solver's steps rather than replacing them, and are
    filled in from the integrator's own interpolant, so asking for a point costs nothing
    in accuracy and asking for a coarse grid still shows every bounce.

    ``method`` goes straight to SciPy. The default is ``Radau`` because a model that came
    out of a Modelica flattener is stiff more often than not -- Robertson and van der Pol
    are both in the example set -- and an implicit method that is merely slower on an easy
    problem beats an explicit one that fails on a hard one.
    """
    kernel = model if isinstance(model, Kernel) else build(model, jit=jit)
    grid = _grid(start, stop, times, points)
    p, y, w = kernel.start(start)
    plan = _Run(kernel, p, w, tolerance, absolute if absolute is not None else tolerance, method)
    return plan.run(start, stop, y, grid)


def _grid(
    start: float,
    stop: float,
    times: Sequence[float] | NDArray[np.float64] | None,
    points: int | None,
) -> NDArray[np.float64] | None:
    if times is not None and points is not None:
        message = "give `times` or `points`, not both"
        raise ValueError(message)
    if points is not None:
        return np.linspace(start, stop, points)
    if times is None:
        return None
    grid = np.sort(np.asarray(times, dtype=float))
    return grid[(grid >= start) & (grid <= stop)]


class _Run:
    """One integration, segment by segment, with the bookkeeping the loop needs."""

    def __init__(  # noqa: PLR0913, PLR0917 - state carried between segments
        self,
        kernel: Kernel,
        p: NDArray[np.float64],
        w: NDArray[np.float64],
        relative: float,
        absolute: float,
        method: str,
    ) -> None:
        self.kernel = kernel
        self.p = p
        self.w = w
        self.relative = relative
        self.absolute = absolute
        self.method = method
        self.rearm: dict[int, float] = {}
        self.triggers = [self._trigger(k) for k in range(kernel.events)]

    # -- the functions SciPy calls ------------------------------------------------------

    def derivative(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.kernel.rhs(t, y, self.p, self.w, np.empty(self.kernel.states))

    def _trigger(self, index: int) -> Callable[[float, NDArray[np.float64]], float]:
        """A terminal zero crossing for one event.

        The direction comes from the model: a ``when`` branch fires only as its condition
        becomes true, while a relation inside an equation is an event whichever way it
        turns, since either way the equation it sits in changes shape.
        """

        def crossing(t: float, y: NDArray[np.float64]) -> float:
            z = self.kernel.indicators(t, y, self.p, self.w, np.empty(self.kernel.events))
            return float(z[index]) + self.rearm.get(index, 0.0)

        crossing.terminal = True  # pyright: ignore[reportFunctionMemberAccess]
        crossing.direction = self.kernel.code.events[index].direction  # pyright: ignore[reportFunctionMemberAccess]
        return crossing

    # -- the loop ----------------------------------------------------------------------

    def run(
        self,
        start: float,
        stop: float,
        y: NDArray[np.float64],
        grid: NDArray[np.float64] | None,
    ) -> Result:
        times: list[NDArray[np.float64]] = []
        values: list[NDArray[np.float64]] = []
        events: list[Event] = []
        t, chatter = start, 0
        while t < stop:
            segment = self._segment(t, stop, y)
            sampled, trajectory = self._sample(segment, grid)
            times.append(sampled)
            values.append(self._observe(sampled, trajectory))
            if segment.status != 1:
                break
            when, at, before = self._fired(segment)
            y = self._settle(at, before, when, events)
            chatter = chatter + 1 if at - t < _STANDSTILL * max(1.0, abs(at)) else 0
            if chatter > _CHATTER:
                message = (
                    f"{self.kernel.name}: {_CHATTER} events at t = {at:.12g} without advancing"
                    " -- the model is chattering"
                )
                raise SimulationError(message)
            t = at
        return self._result(times, values, events)

    def _segment(self, t: float, stop: float, y: NDArray[np.float64]) -> _Solution:
        """Integrate until the end or until something fires."""
        solution = cast(
            "_Solution",
            solve_ivp(  # pyright: ignore[reportUnknownArgumentType]
                self.derivative,
                (t, stop),
                y,
                method=self.method,
                rtol=self.relative,
                atol=self.absolute,
                dense_output=True,
                events=self.triggers or None,
            ),
        )
        if not solution.success:
            message = f"{self.kernel.name}: {solution.message}"
            raise SimulationError(message)
        return solution

    def _sample(
        self, segment: _Solution, grid: NDArray[np.float64] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The segment's own steps, plus whichever requested times fall inside it."""
        steps: NDArray[np.float64] = np.asarray(segment.t, dtype=float)
        if grid is None:
            return steps, np.asarray(segment.y, dtype=float)
        inside = grid[(grid >= steps[0]) & (grid <= steps[-1])]
        times = np.union1d(steps, inside)
        return times, np.asarray(segment.sol(times), dtype=float)

    # -- one instant --------------------------------------------------------------------

    def _settle(
        self, at: float, y: NDArray[np.float64], first: int, events: list[Event]
    ) -> NDArray[np.float64]:
        """Apply the event that fired, and then everything it made stale.

        This is Modelica's *event iteration*, and it is not an optimisation. A latched
        relation holds its value between events, so the blocks that read it change shape
        the moment its latch flips -- and that can leave a *second* relation on the wrong
        side of its own latch. The integrator will never catch that second one: on the next
        segment its indicator does not *cross* zero, it starts out already past it, so no
        event is reported and the stale value is carried for the rest of the run.

        So the instant is not over when one event has been applied. It is over when nothing
        disagrees any more, which is what this loop establishes.

        An event acts at most once per instant. That is what makes the loop terminate today
        -- there is no ``pre`` on a discrete variable yet, so no relation can be driven back
        and forth within a single instant. When discrete state does land, that guarantee
        goes away and :data:`_ITERATIONS` becomes load-bearing; that is the point at which
        :mod:`modelica.verify` earns its keep, because "the iteration converges" stops being
        obvious and becomes something to prove.
        """
        fired: set[int] = set()
        pending: int | None = first
        while pending is not None:
            events.append(Event(at, pending, str(self.kernel.code.events[pending])))
            y = self.kernel.transition(pending, at, y, self.p, self.w, np.empty(len(y)))
            fired.add(pending)
            if len(fired) > _ITERATIONS:
                message = (
                    f"{self.kernel.name}: event iteration did not converge at t = {at:.12g}"
                    f" after {len(fired)} events"
                )
                raise SimulationError(message)
            pending = self._stale(at, y, fired)
        self.rearm = {
            index: offset
            for index in fired
            if (offset := self._offset(self.kernel.code.events[index])) is not None
        }
        return y

    def _offset(self, trigger: Trigger) -> float | None:
        """Which side of its own zero a crossing that just fired restarts on.

        A latched relation sits on the side its latch was left holding. A ``when`` branch
        fired as its condition became true, so it restarts on the true side. A mode switch
        gets nothing: its indicator carries a hysteresis margin of its own, and is only
        watched from the state set it would leave.
        """
        if trigger.switch is not None:
            return None
        if trigger.latch is not None and not self.w[trigger.latch] > HELD:
            return -_REARM
        return _REARM

    def _stale(self, t: float, y: NDArray[np.float64], fired: set[int]) -> int | None:
        """The first latch that no longer agrees with the relation it is holding.

        Events that already acted at this instant are skipped, and not only to bound the
        loop: the one that just fired sits *on* its own zero crossing, where the sign of
        its indicator is numerical noise, so asking whether it agrees with itself is not a
        question with an answer.
        """
        z = self.kernel.indicators(t, y, self.p, self.w, np.empty(self.kernel.events))
        for index, trigger in enumerate(self.kernel.code.events):
            if trigger.latch is None or index in fired:
                continue
            if bool(z[index] > 0.0) != bool(self.w[trigger.latch] > HELD):
                return index
        return None

    def _fired(self, segment: _Solution) -> tuple[int, float, NDArray[np.float64]]:
        """Which branch stopped the segment, when, and the state just before it did."""
        candidates = [
            (float(found[-1]), index) for index, found in enumerate(segment.t_events) if len(found)
        ]
        at, when = min(candidates)
        return when, at, np.asarray(segment.y_events[when][-1], dtype=float)

    # -- output ------------------------------------------------------------------------

    def _observe(
        self, times: NDArray[np.float64], trajectory: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Every variable at every sample of one segment.

        Here, and not once over the whole trajectory at the end, because the workspace is
        part of the answer: a latched relation holds what it decided at the *last* event,
        and reading the tank's flow at t = 10 with the latch that fired at t = 22 would
        report it empty while it was still draining.
        """
        values = np.empty((self.kernel.observed, len(times)))
        buffer = np.empty(self.kernel.observed)
        for column, moment in enumerate(times):
            self.kernel.observe(float(moment), trajectory[:, column], self.p, self.w, buffer)
            values[:, column] = buffer
        return values

    def _result(
        self,
        times: list[NDArray[np.float64]],
        values: list[NDArray[np.float64]],
        events: list[Event],
    ) -> Result:
        time = np.concatenate(times) if times else np.empty(0)
        observed = np.concatenate(values, axis=1) if values else np.empty((self.kernel.observed, 0))
        return Result(
            name=self.kernel.name,
            time=time,
            names=self.kernel.code.observed,
            values=observed,
            events=tuple(events),
        )
