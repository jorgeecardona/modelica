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

import warnings
from collections import Counter
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

import numpy as np
from scipy.integrate import (  # pyright: ignore[reportMissingTypeStubs]
    solve_ivp,  # pyright: ignore[reportUnknownVariableType]
)

from modelica import ir
from modelica.ir.evaluate import EvaluationError, evaluate
from modelica.sim.emit import HELD, EmitError
from modelica.sim.kernel import Kernel, build
from modelica.sim.result import Event, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.emit import Code, Trigger

__all__ = ["SimulationError", "simulate"]

_IMPLICIT = frozenset({"Radau", "BDF", "LSODA"})
"""The methods that solve with a Jacobian, and so are the ones worth handing an exact one."""

_CHATTER = 100
"""Consecutive zero-length segments before we call it a Zeno point and give up."""

_STANDSTILL = 1e-4
"""How far `_CHATTER` consecutive events must carry the run, relative to where it is.

A *span*, not a segment length, and that is the whole of the fix in `13-08-2026`. The test
used to ask whether each segment on its own was shorter than ``1e-12 * max(1, |t|)``, and no
segment of a chattering run ever is: a relay with no hysteresis advances by `_REARM` per
event, because the nudge that keeps a fired crossing off zero is what decides where the next
root lands. `_REARM` is ``1e-9``, a thousand times the old threshold, so every segment read as
progress and the counter reset on every one of them. Measured on the relay in
`tests/test_chattering.py`: 24,273 events, 45 seconds, ``t`` reached 1.00004854 of a
5-second horizon, and the counter never got past 0 of 100.

Asking about the span instead needs no knowledge of the indicator's slope, which is what an
absolute floor tied to `_REARM` would have needed -- the nudge is a distance in *indicator*
units and becomes a distance in time only after dividing by ``d indicator / dt``, which is
the model's business and not this loop's.

``1e-4`` is chosen with both sides measured, over the corpus and over the relay, in exactly
the units the test computes:

    tightest 100-event span, corpus     7.9e-1   (Pendulum; Thermostat 9.9e-1)
    the relay                           2.0e-7

Six and a half orders of magnitude of daylight, and ``1e-4`` sits near the middle of it on a
log scale: 500x clear of the relay, 7,900x clear of the tightest real model. No corpus model
has more than 33 events in its entire run, so none of them comes close to needing 100 inside
any window at all.
"""

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

    t_events: list[NDArray[np.float64]] | None
    """The crossings, or `None` when ``events=`` was not passed at all.

    `None` rather than an empty list, which is a SciPy detail worth writing down here rather
    than rediscovering: a model with nothing to watch comes back with no list to read. This
    module always passes events, so it never sees the `None`; `modelica.sim.sensitivity`
    integrates event-free models through the same protocol and does.
    """

    y_events: list[NDArray[np.float64]] | None

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
    parameters: NDArray[np.float64] | None = None,
    backend: Literal["auto", "python", "engine"] = "auto",
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

    ``parameters`` runs the model at values other than the ones it declares. Build the vector
    with `modelica.sim.sweep.vector`, which recomputes everything downstream of a change --
    writing into the array by hand leaves a derived parameter holding a value from the old
    one. The initial conditions are computed *from* it, so a start value written in terms of
    a parameter follows the override, which is what makes this usable for a sweep.

    ``backend`` picks who integrates. ``"auto"`` (the default) uses the precompiled engine
    (`modelica.sim.engine`, inside a platform wheel) when it is installed and can run the call,
    else SciPy. ``"python"`` is always SciPy on the emitted Python. ``"engine"`` raises when the
    engine cannot run the call. The engine runs a `System` with ``method="Radau"``, no ``jit``,
    ``start = 0`` and an output grid. With a grid, SciPy returns the grid instants plus the two
    sides of each event, never its own steps. The engine returns the grid instants only.
    """
    if backend != "python":
        ran = _on_engine(
            model,
            backend=backend,
            stop=stop,
            start=start,
            tolerance=tolerance,
            absolute=absolute,
            times=times,
            points=points,
            method=method,
            jit=jit,
            parameters=parameters,
        )
        if ran is not None:
            _check_asserts(ran, getattr(model, "asserts", ()))
            return ran
    # `observed_jacobian=False`: this kernel is internal to the run and nothing here calls
    # `d(observed)/dy`. It is the largest function the emitter writes -- 17 % of the module on
    # a 1000-cell bar -- so a caller who only wants a trajectory should not pay to emit it, to
    # compile it, or to hold it. A caller who *does* want it builds the kernel itself.
    kernel = model if isinstance(model, Kernel) else build(model, jit=jit, observed_jacobian=False)
    grid = _grid(start, stop, times, points)
    p, y, w = kernel.start(start, parameters)
    plan = _Run(kernel, p, w, tolerance, absolute if absolute is not None else tolerance, method)
    result = plan.run(start, stop, y, grid)
    # A `System` carries the model's asserts; a pre-built `Code`/`Kernel` does not expose them,
    # so `getattr` leaves those callers unchecked rather than reaching for a field they lack.
    _check_asserts(result, getattr(model, "asserts", ()))
    return result


def _on_engine(  # noqa: PLR0913 - the knobs of `simulate`
    model: System | Code | Kernel,
    *,
    backend: str,
    stop: float,
    start: float,
    tolerance: float,
    absolute: float | None,
    times: Sequence[float] | NDArray[np.float64] | None,
    points: int | None,
    method: str,
    jit: bool,
    parameters: NDArray[np.float64] | None,
) -> Result | None:
    """The run on the engine, or ``None`` when ``"auto"`` should fall back to SciPy."""
    from modelica.ir.system import System as SystemType  # noqa: PLC0415 - runtime check
    from modelica.sim import engine  # noqa: PLC0415 - the engine is optional

    supported = (
        isinstance(model, SystemType)
        and method == "Radau"
        and not jit
        and start == 0.0
        and (times is not None or points is not None)
    )
    if backend == "auto" and (not supported or engine.installed() is None):
        return None
    if not supported:
        message = (
            "backend='engine' runs a System with method='Radau', no jit, start=0 and an output"
            " grid (times or points)"
        )
        raise SimulationError(message)
    assert isinstance(model, SystemType)  # noqa: S101 - narrowed by `supported`
    try:
        return engine.result(
            model,
            stop=stop,
            times=times,
            points=points,
            tolerance=tolerance,
            absolute=absolute,
            parameters=parameters,
        )
    except EmitError:
        if backend == "engine":
            raise
        return None


def _check_asserts(result: Result, asserts: tuple[ir.Assert, ...]) -> None:
    """Evaluate each carried assert along the reported trajectory.

    An error-level assert (the default) that goes false stops the run at the first instant it
    does, with its message; a warning-level one is reported once and the run goes on. The
    condition is read from what the run reports at each output point -- states, parameters and
    derivatives are all there -- so an assert that names only those is checked, and one that
    names something the run does not expose is skipped rather than guessed at. The check sees
    the output grid, not every internal step, so a violation that opens and closes between two
    reported points can slip through; that is the honest limit of a post-hoc check.
    """
    if not asserts or result.values.size == 0:
        return
    names = result.names
    warned: set[int] = set()
    skip: set[int] = set()
    for column in range(len(result.time)):
        env: dict[str, float] | None = None
        for index, check in enumerate(asserts):
            if index in skip or index in warned:
                continue
            if env is None:
                env = {name: float(result.values[row, column]) for row, name in enumerate(names)}
                env["time"] = float(result.time[column])
            try:
                holds = evaluate(check.condition, env)
            except EvaluationError:
                skip.add(index)  # names a variable the run does not report; cannot be checked
                continue
            if holds:
                continue
            text = (
                f"{result.name}: assertion failed at t = {env['time']:.6g}: "
                f"{_assert_message(check.message, env)}"
            )
            if _is_warning(check.level):
                warnings.warn(text, stacklevel=2)
                warned.add(index)
            else:
                raise SimulationError(text)


def _assert_message(message: ir.Expr, env: dict[str, float]) -> str:
    """The assert's message. A string literal is the common case; anything else is evaluated."""
    if isinstance(message, ir.StringLiteral):
        return message.value
    try:
        return str(evaluate(message, env))
    except EvaluationError:
        return "assertion failed"


def _is_warning(level: ir.Expr | None) -> bool:
    """``AssertionLevel.warning`` downgrades the assert to a warning; the default is an error."""
    return isinstance(level, ir.Ref) and bool(level.parts) and level.parts[-1] == "warning"


_SPARSE_STATES: Final = 16
"""States below which the Jacobian is handed over dense, whatever its pattern says.

Measured on a heat bar swept in ``n``, and on the corpus. Sparse first wins at 25 states
(2.16x) and reaches 88x at 400; at every corpus size -- 1 to 10 states -- the two are within
noise of each other, 0.98x to 1.04x. So 16 is above every model here and below where the win
starts, and nothing in the corpus changes behaviour because of it.
"""

_SPARSE_FILL: Final = 0.25
"""Fill fraction above which a dense factorisation is the better one anyway.

A CSR of a nearly full matrix is index arrays and SuperLU wrapped around the same
factorisation. Not measured at large ``n`` -- no dense large model exists here -- so this is
a guard against a shape nobody has, chosen rather than found.
"""


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
        pattern = kernel.code.sparsity
        self.indices = np.array([j for row in pattern for j in row], dtype=np.int32)
        self.indptr = np.cumsum([0, *(len(row) for row in pattern)], dtype=np.int32)
        # Sparse only where sparse wins. A dense factorisation beats a sparse one on a small
        # matrix by a wide margin -- the sparse path allocates index arrays, builds a CSR and
        # calls SuperLU, and for a 2x2 all of that is overhead around four multiplies. Both
        # numbers were measured on the corpus and on a heat bar swept in `n`.
        self.sparse = (
            kernel.states >= _SPARSE_STATES
            and len(self.indices) <= _SPARSE_FILL * kernel.states * kernel.states
        )
        self.relative = relative
        self.absolute = absolute
        self.method = method
        self.rearm: dict[int, float] = {}
        self.triggers = [self._trigger(k) for k in range(kernel.events)]

    # -- the functions SciPy calls ------------------------------------------------------

    def derivative(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.kernel.rhs(t, y, self.p, self.w, np.empty(self.kernel.states))

    def sparse_slope(
        self,
        t: float,
        y: NDArray[np.float64],
    ) -> Any:  # noqa: ANN401 - a `csr_array`; scipy.sparse ships no stub worth naming
        """The same Jacobian, as a CSR matrix, for a model whose pattern is worth it.

        SciPy switches on what the callable *returns*: `scipy.integrate._ivp.radau` checks
        `issparse(J)` and factors with `splu` instead of `lu_factor`. The `jac_sparsity`
        argument is not the way in -- its own documentation says it "is ignored if `jac` is
        not `None`", and this project always has an exact Jacobian, so that argument could
        never have applied here.

        The indices and the row pointers are built once and reused; only the data changes.
        """
        from scipy.sparse import (  # noqa: PLC0415 - only the sparse path pays  # pyright: ignore[reportMissingTypeStubs]
            csr_array,
        )

        data = np.empty(len(self.indices))
        self.kernel.jacobian_sparse(t, y, self.p, self.w, data)
        return csr_array((data, self.indices, self.indptr), shape=(n := self.kernel.states, n))

    def slope(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """The exact Jacobian, into an array of its own every time.

        Every other call here fills a scratch buffer, and this one deliberately does not.
        An implicit solver keeps the last Jacobian it was given across steps -- Radau
        reuses ``self.J`` until it decides the iteration has gone stale -- so handing back
        one buffer would let the next evaluation rewrite a matrix the solver still believes
        it is holding. One allocation per Jacobian is nothing: they are asked for orders of
        magnitude less often than the right-hand side.
        """
        n = self.kernel.states
        return self.kernel.jacobian(t, y, self.p, self.w, np.zeros((n, n)))

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
        t, chatter, stalled_from, stalled_at = start, 0, 0, start
        while t < stop:
            segment = self._segment(t, stop, y)
            sampled, trajectory = self._sample(segment, grid)
            times.append(sampled)
            values.append(self._observe(sampled, trajectory))
            if segment.status != 1:
                break
            when, at, before = self._fired(segment)
            if self.kernel.code.events[when].terminates:
                # `terminate(...)` fired: the run is defined only up to here. Record the
                # instant and its reason, keep the samples of this segment (which ended at the
                # crossing), and stop -- no state to settle, nothing meaningful past it.
                events.append(Event(at, when, str(self.kernel.code.events[when])))
                break
            # Measured from where the standstill began, not from the previous event. A
            # chattering run advances every time -- by `_REARM` over the indicator's slope,
            # which is the engine's own nudge and not the model making progress -- so a
            # per-segment test resets on every event and never counts to one. The span is
            # what does not grow.
            if at - stalled_at >= _STANDSTILL * max(1.0, abs(at)):
                chatter, stalled_from, stalled_at = 0, len(events), at
            else:
                chatter += 1
            y = self._settle(at, before, when, events)
            if chatter > _CHATTER:
                raise SimulationError(self._chattering(at, events[stalled_from:]))
            t = at
        return self._result(times, values, events)

    def _unlocatable(self, t: float, y: NDArray[np.float64], error: ValueError) -> str:
        """Name the relations sitting on zero, rather than repeat SciPy's arithmetic.

        A root finder that cannot bracket a crossing has almost always been handed one that
        is already *at* zero, so the useful thing to report is which. Every indicator is
        evaluated at the start of the failing segment and the ones the solver cannot tell from
        zero are listed by the Modelica they came from -- `Trigger.condition` is on the public
        record for the same reason `_chattering` uses `Event.cause`.

        "Cannot tell from zero" is the absolute tolerance and not ``== 0.0``. Measured on the
        two-relation model in `tests/test_chattering.py`, the failing segment starts with both
        indicators at 1e-9 against an ``atol`` of 1e-6: they were exactly zero at ``t = 0``, an
        event fired, and the segment after it begins a nudge away. An exact test names nothing
        in the case this method exists for.

        SciPy's own message is kept. If the near-zero reading is the wrong explanation for some
        future failure, the reader still has what the root finder said.
        """
        z = self.kernel.indicators(t, y, self.p, self.w, np.empty(self.kernel.events))
        flat = [
            trigger.condition
            for trigger, value in zip(self.kernel.code.events, z, strict=True)
            if abs(value) <= self.absolute
        ]
        sitting = (
            " -- "
            + ", ".join(repr(condition) for condition in flat)
            + (" sits" if len(flat) == 1 else " sit")
            + f" at zero to within the solver's tolerance of {self.absolute:g}, so there is no"
            " sign change to bracket"
            if flat
            else ""
        )
        return (
            f"{self.kernel.name}: no crossing could be located at t = {t:.12g}{sitting}"
            f" (scipy: {error})"
        )

    def _chattering(self, at: float, stalled: Sequence[Event]) -> str:
        """Name what kept firing, rather than only that something did.

        The counter knows which crossings it stopped at -- ``Event.cause`` is the relation in
        Modelica, kept for exactly this -- so the message can say *which* condition will not
        settle instead of leaving the modeller to find it. A model chatters because two
        conditions disagree, or because one sits on its own threshold, and the tally
        separates those two cases at a glance.
        """
        tally = Counter(event.cause for event in stalled)
        worst = ", ".join(f"{cause!r} x{count}" for cause, count in tally.most_common(4))
        more = "" if len(tally) <= 4 else f", and {len(tally) - 4} more"  # noqa: PLR2004
        return (
            f"{self.kernel.name}: {len(stalled)} events at t = {at:.12g} without advancing"
            f" -- the model is chattering. What kept firing: {worst}{more}"
        )

    def _segment(self, t: float, stop: float, y: NDArray[np.float64]) -> _Solution:
        """Integrate until the end or until something fires.

        The Jacobian goes to the implicit methods and to those only: an explicit one has no
        use for it and SciPy warns about arguments a solver ignores, which is a warning
        worth keeping meaningful.
        """
        chosen = self.sparse_slope if self.sparse else self.slope
        implicit: dict[str, Any] = {"jac": chosen} if self.method in _IMPLICIT else {}
        try:
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
                    **implicit,
                ),
            )
        except ValueError as error:
            # SciPy's root finder, not this package's code. `brentq` needs a sign change
            # across the bracket it is handed and refuses one where the indicator is zero at
            # both ends -- and a `ValueError` reading "f(a) and f(b) must have different
            # signs", with a traceback ending in `scipy/optimize/_zeros_py.py`, tells a
            # modeller nothing about their model. `_unlocatable` says which relation, when.
            #
            # `ValueError` and not `Exception`: turning every SciPy failure into a
            # `SimulationError` would hide a genuine bug in the source this package generates,
            # and those tracebacks are how such a bug gets found.
            raise SimulationError(self._unlocatable(t, y, error)) from error
        if not solution.success:
            message = f"{self.kernel.name}: {solution.message}"
            raise SimulationError(message)
        return solution

    def _sample(
        self, segment: _Solution, grid: NDArray[np.float64] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The requested times inside the segment, plus its two ends; all steps with no grid.

        The ends stay so an event shows both its left and right value.
        """
        steps: NDArray[np.float64] = np.asarray(segment.t, dtype=float)
        if grid is None:
            return steps, np.asarray(segment.y, dtype=float)
        inside = grid[(grid >= steps[0]) & (grid <= steps[-1])]
        times = np.union1d(steps[[0, -1]], inside)
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
        """Which branch stopped the segment, when, and the state just before it did.

        Only reached when a segment stopped on a crossing, so the two event lists are there;
        the `None` the protocol allows is what SciPy returns when nothing was watched at all,
        and this loop is what runs when something was.
        """
        assert segment.t_events is not None  # noqa: S101 - see the docstring
        assert segment.y_events is not None  # noqa: S101 - the same
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
