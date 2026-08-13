"""Model-checking the discrete half of a model.

Simulating is sampling: one initial condition, one trajectory, one order of events. The
discrete half of a model deserves better than that, because it is **finite**. At a single
instant of simulated time the continuous state does not move, so the only things that can
change are which watched relations are currently latched true -- and there are two to the
power of not-very-many of those -- and what the ``when`` bodies have written into the
discrete-time variables. Finite means it can be enumerated rather than sampled.

Those two halves are not the same kind of thing, and :class:`Instant` keeps both because the
difference is the finding. A latch is a cached answer to a question the model can re-ask, so
the iteration repairs whatever you set it to: all 512 latch configurations of
``RoastingMachine`` settle in one place. A discrete-time variable is *memory* -- what the
last ``when`` left there -- and nothing outside a ``when`` can reconstruct it, so two orders
of applying the same pending events can leave it holding different values and stay there.

What gets enumerated here is Modelica's *event iteration*: flipping one latch changes the
blocks that read it, which can leave a second relation disagreeing with its own latch,
which flips, and so on. :mod:`modelica.sim.solve` runs that loop forwards from wherever the
simulation happens to be. This module runs it from *every* configuration the model could be
in, in *every* order the pending events could be applied, and asks two questions:

* **does it stop?** -- a cycle of events that never settles is a hung simulation, and no
  test finds it unless the test happens to start in the configuration that triggers it;
* **does the order matter?** -- if two orders of applying the same pending events reach
  different fixed points, then "apply the first stale one" (which is what the engine does)
  is not a choice, it is a bug waiting for a model that notices.

There is no abstraction in the first of those and none in the second: the emitted
``indicators`` and ``transition`` are the functions being called, on real ``float`` arrays,
so a counterexample is a genuine execution rather than something an approximation admitted.
The one thing that *is* approximate is a ``when`` branch, which fires on a rising edge in
the engine and is allowed to fire here whenever its condition merely holds -- conservative,
so it can report an event sequence the integrator would not produce, never miss one it
would.

The state space is bounded by the instant: each event acts at most once, so this says
nothing about chattering *across* instants. That is what the ``_STANDSTILL`` counter in
:mod:`modelica.sim.solve` watches at runtime.

Behind the ``verify`` extra (``pip install modelica[verify,sim]``), so nothing here is
imported by :mod:`modelica` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from musil import (
    Action,
    Assumption,
    EnvironmentSpec,
    Model,
    check,
    check_liveness,
    check_open,
    explore,
    fixed_points,
    protocol_actions,
    reaches,
)

from modelica.fmi.modes import Interface, Mode, table
from modelica.sim.emit import HELD
from modelica.sim.kernel import Kernel, build
from modelica.ssp.master import Component, Phase, Progress, moves, start

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from musil import LivenessResult, OpenResult
    from musil import Result as CheckResult
    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.emit import Code
    from modelica.sim.result import Result

__all__ = [
    "Instant",
    "MasterReport",
    "ModeReport",
    "Report",
    "Selection",
    "SelectionReport",
    "check_iteration",
    "check_master",
    "check_modes",
    "check_run",
    "check_selection",
    "iteration",
    "master",
    "modes",
    "numerics",
    "selection",
    "settled",
]

_CONFIGURATIONS = 12
"""Watched relations beyond which starting from every latch configuration is refused.

Two to the twelfth is four thousand starting points, which is still seconds; past that the
caller should say which configurations they actually care about rather than have the module
quietly check fewer than it claims.
"""


@dataclass(frozen=True, slots=True)
class Instant:
    """Where a model's discrete half stands at one instant of simulated time."""

    latches: tuple[bool, ...]
    """One per watched relation, in the order :attr:`~modelica.sim.emit.Code.events` gives
    them: what the model currently *believes* that relation says."""

    state: tuple[float, ...]
    """The continuous state. Constant through the instant unless a ``when`` branch
    reinitialises it."""

    fired: frozenset[int]
    """The events that have already acted here. An event acts at most once per instant."""

    memory: tuple[float, ...] = ()
    """The discrete-time variables, in :attr:`~modelica.sim.emit.Code.remembered` order.

    The other half of the discrete state, and it behaves nothing like a latch. A latch is a
    cached answer to a question the model can re-ask, so the iteration repairs whatever you
    set it to -- which is why starting from all 512 latch configurations of
    ``RoastingMachine`` settles in one place. A remembered variable is what a ``when`` body
    left there and nothing else can reconstruct it, so a move that writes one is a move to a
    genuinely different state.

    Defaulted, so that an `Instant` written by hand for a model with no memory reads the way
    it always did.
    """

    def __str__(self) -> str:
        latches = "".join("1" if held else "0" for held in self.latches) or "-"
        memory = " memory=" + ",".join(f"{v:g}" for v in self.memory) if self.memory else ""
        return f"latches={latches}{memory} fired={sorted(self.fired)}"


@dataclass(frozen=True, slots=True)
class Report:
    """What the checker found: whether the iteration settles, and whether it settles once."""

    name: str
    explored: int
    """Reachable discrete states, summed over every starting configuration."""

    outcomes: Mapping[Instant, tuple[Instant, ...]]
    """Each starting configuration, mapped to the settled states reachable from it.

    Different configurations settling in different places is not a problem -- a latch that
    remembers is the whole point of one. *One* configuration settling in two places is,
    because the engine applies the first stale event it finds and nothing chooses that
    order on purpose.
    """

    liveness: LivenessResult[Instant]

    @property
    def divergent(self) -> tuple[tuple[Instant, tuple[Instant, ...]], ...]:
        """The configurations whose fixed point depends on the order events are applied."""
        return tuple((start, ends) for start, ends in self.outcomes.items() if len(ends) > 1)

    @property
    def ok(self) -> bool:
        return self.liveness.ok and not self.divergent

    def __str__(self) -> str:
        if self.ok:
            return (
                f"{self.name}: event iteration settles, and settles in one place, from all"
                f" {len(self.outcomes)} configurations ({self.explored} discrete states)"
            )
        if not self.liveness.ok:
            return f"{self.name}: {self.liveness}"
        lines = [
            f"  from {start} -> " + " | ".join(str(end) for end in ends)
            for start, ends in self.divergent
        ]
        return (
            f"{self.name}: the event iteration settles somewhere different depending on the"
            " order events are applied in:\n" + "\n".join(lines)
        )


class _Discrete:
    """The model's discrete transition relation, evaluated by the real compiled functions.

    Held as an object rather than a pile of closures because every predicate needs the same
    four things (the kernel, the instant, the parameter vector, the workspace template) and
    because the indicator values are worth caching: the checker asks "is this event enabled"
    once per event per state, and each question is otherwise a full pass over the model's
    blocks.
    """

    def __init__(
        self,
        kernel: Kernel,
        t: float,
        y: NDArray[np.float64] | None,
        memory: NDArray[np.float64] | None = None,
    ) -> None:
        self.kernel = kernel
        self.t = t
        parameters, start, workspace = kernel.start(t)
        self.p = parameters
        self.w = workspace
        self.y = start if y is None else np.asarray(y, dtype=float)
        self.slots = tuple(
            trigger.latch for trigger in kernel.code.events if trigger.latch is not None
        )
        # Which workspace slots hold discrete-time variables. `w` begins with one slot per
        # observed variable, and `Code.remembered` names the ones a `when` writes.
        self.memory = tuple(kernel.code.observed.index(name) for name in kernel.code.remembered)
        # What they hold at the start: `initial` put it there, unless the caller is checking
        # an instant part-way through a run and read the values out of the trajectory.
        self.start = (
            tuple(float(self.w[slot]) for slot in self.memory)
            if memory is None
            else tuple(float(value) for value in memory)
        )
        self.cache: dict[
            tuple[tuple[bool, ...], tuple[float, ...], tuple[float, ...]], NDArray[np.float64]
        ] = {}

    # -- reading a state ----------------------------------------------------------------

    def workspace(self, instant: Instant) -> NDArray[np.float64]:
        """``w`` as it would be with these latches held and this much remembered."""
        w = self.w.copy()
        for position, slot in enumerate(self.slots):
            w[slot] = 1.0 if instant.latches[position] else 0.0
        for position, slot in enumerate(self.memory):
            w[slot] = instant.memory[position]
        return w

    def indicators(self, instant: Instant) -> NDArray[np.float64]:
        # The memory is in the key because it can be *read* by a condition -- a `when` that
        # fires on `cycles > 3` watches a variable another `when` writes. Leaving it out
        # would serve one instant's answer to a state that had moved.
        key = (instant.latches, instant.state, instant.memory)
        cached = self.cache.get(key)
        if cached is None:
            cached = self.kernel.indicators(
                self.t,
                np.asarray(instant.state, dtype=float),
                self.p,
                self.workspace(instant),
                np.empty(self.kernel.events),
            )
            self.cache[key] = cached
        return cached

    def holds(self, instant: Instant, event: int) -> bool:
        """Whether the condition watched by ``event`` is true right now."""
        return bool(self.indicators(instant)[event] > 0.0)

    def stale(self, instant: Instant, event: int, position: int) -> bool:
        """Whether a latched relation disagrees with what its latch is holding."""
        return (
            event not in instant.fired and self.holds(instant, event) != instant.latches[position]
        )

    def rising(self, instant: Instant, event: int) -> bool:
        """Whether a ``when`` branch could act here.

        This is the one conservative step, and the module docstring says why: the engine
        also demands that the condition just *became* true, which is not a question a
        single instant can answer.
        """
        return event not in instant.fired and self.holds(instant, event)

    # -- taking a step ------------------------------------------------------------------

    def act(self, instant: Instant, event: int) -> Instant:
        """Apply one event, exactly as the engine would."""
        w = self.workspace(instant)
        after = self.kernel.transition(
            event,
            self.t,
            np.asarray(instant.state, dtype=float),
            self.p,
            w,
            np.empty(len(instant.state)),
        )
        return Instant(
            latches=tuple(bool(w[slot] > HELD) for slot in self.slots),
            state=tuple(float(value) for value in after),
            fired=instant.fired | {event},
            # Read back off `w`, the same array `transition` just wrote. A `when` body
            # assigning a discrete variable is the only way one ever changes, so this is
            # where the move becomes visible -- and before this it was not visible at all:
            # firing `Thermostat`'s heat-on branch produced an `Instant` equal to the one it
            # started from, and the checker reported "settles in one place" about a model
            # that had incremented a counter.
            memory=tuple(float(w[slot]) for slot in self.memory),
        )

    # -- the pieces musil wants ---------------------------------------------------------

    def actions(self) -> list[Action[Instant]]:
        actions: list[Action[Instant]] = []
        position = 0
        for event, trigger in enumerate(self.kernel.code.events):
            if trigger.latch is None:
                actions.append(
                    Action(
                        name=f"when {trigger.condition}",
                        enabled=lambda s, _e=event: self.rising(s, _e),
                        apply=lambda s, _e=event: self.act(s, _e),
                    )
                )
                continue
            actions.append(
                Action(
                    name=f"settle {trigger.condition}",
                    enabled=lambda s, _e=event, _p=position: self.stale(s, _e, _p),
                    apply=lambda s, _e=event: self.act(s, _e),
                )
            )
            position += 1
        return actions

    def settled(self, instant: Instant) -> bool:
        """Nothing left to do: every latch agrees, and no ``when`` branch is waiting."""
        position = 0
        for event, trigger in enumerate(self.kernel.code.events):
            if trigger.latch is None:
                if self.rising(instant, event):
                    return False
                continue
            if self.stale(instant, event, position):
                return False
            position += 1
        return True

    def configurations(self) -> list[Instant]:
        """Every latch configuration, because the simulation only ever visits some of them.

        A model that hangs from a configuration no trajectory reaches is not a bug today;
        it is a bug the first time somebody changes a start value.
        """
        if len(self.slots) > _CONFIGURATIONS:
            message = (
                f"{self.kernel.name}: {len(self.slots)} watched relations is"
                f" {2 ** len(self.slots)} starting configurations -- pass `start=` to say"
                " which ones matter"
            )
            raise ValueError(message)
        state = tuple(float(value) for value in self.y)
        # The memory starts at what `initial` put there and is *not* enumerated. A latch is a
        # bit and there are `2**k` of them; a discrete `Real` is a counter with no finite
        # range, so the tractable question is what is reachable from a start rather than what
        # is possible. `check` already answers that one.
        memory = self.start
        return [
            Instant(latches=latches, state=state, fired=frozenset(), memory=memory)
            for latches in product((False, True), repeat=len(self.slots))
        ]


def iteration(
    model: System | Code | Kernel,
    *,
    t: float = 0.0,
    y: NDArray[np.float64] | None = None,
    memory: NDArray[np.float64] | None = None,
    start: list[Instant] | None = None,
) -> Model[Instant]:
    """The event iteration at ``t``, as a model checker's model.

    ``start`` defaults to every latch configuration; the settled states are declared
    terminal, so anything else with no way out is reported as a deadlock rather than
    mistaken for an answer.

    ``memory`` is what the discrete-time variables hold, in
    :attr:`~modelica.sim.emit.Code.remembered` order, and defaults to what ``initial`` put
    there. It is a starting *point* rather than a set to enumerate, and
    :meth:`_Discrete.configurations` says why: a latch is a bit, a counter is not.
    """
    kernel = model if isinstance(model, Kernel) else build(model)
    discrete = _Discrete(kernel, t, y, memory)
    return Model(
        init=discrete.configurations() if start is None else start,
        actions=discrete.actions(),
        terminal=discrete.settled,
    )


def settled(model: System | Code | Kernel, instant: Instant, *, t: float = 0.0) -> bool:
    """Whether ``instant`` is a fixed point of ``model``'s event iteration at ``t``."""
    kernel = model if isinstance(model, Kernel) else build(model)
    return _Discrete(kernel, t, None).settled(instant)


def check_iteration(
    model: System | Code | Kernel,
    *,
    t: float = 0.0,
    y: NDArray[np.float64] | None = None,
    memory: NDArray[np.float64] | None = None,
    start: list[Instant] | None = None,
) -> Report:
    """Prove that the event iteration settles at ``t``, and settles in one place.

    The two questions are asked differently on purpose. *Settling* is a property of the
    whole model, so it is one liveness check over every configuration at once. *Settling in
    one place* is a property of each configuration separately -- two configurations are
    allowed to disagree, that is what a latch is for -- which is exactly what
    :func:`musil.fixed_points` reports, per initial state.
    """
    kernel = model if isinstance(model, Kernel) else build(model)
    discrete = _Discrete(kernel, t, y, memory)
    checkable = iteration(kernel, t=t, y=y, memory=memory, start=start)
    settling = fixed_points(checkable)
    return Report(
        name=kernel.name,
        explored=len(explore(checkable).states),
        # `fired` is bookkeeping for the iteration, not part of where it ended up: the same
        # latches reached by one event or by three are the same answer, so the outcomes are
        # deduplicated with it dropped. The memory is *not* dropped -- reaching the same
        # latches with a counter at 1 and at 2 is two answers, and it is the case this
        # deduplication would hide.
        outcomes={
            configuration: tuple(
                dict.fromkeys(
                    Instant(end.latches, end.state, frozenset(), end.memory)
                    for end in sorted(ends, key=str)
                )
            )
            for configuration, ends in settling.by_init.items()
        },
        liveness=check_liveness(checkable, goal=discrete.settled, goal_name="settled"),
    )


def check_run(model: System | Code | Kernel, result: Result) -> tuple[Report, ...]:
    """Check the iteration at the start of a run, and at every instant it had an event.

    Checking at ``t = 0`` alone is thin for a model whose only event is a ``when``: nothing
    is pending at the start, so there is nothing to settle and the check passes without
    having asked anything. The instants worth asking about are the ones the run actually
    reached -- the bounce, the switch, the moment the tank runs dry -- and at each of those
    this still enumerates *every* configuration, not just the one the trajectory was in.

    Simulate first, then verify what the simulation walked past.
    """
    kernel = model if isinstance(model, Kernel) else build(model)
    rows = [result.names.index(name) for name in kernel.code.states]
    # The discrete-time variables are read out of the trajectory too, and for a stronger
    # reason than the continuous state. Starting them from `initial` at an instant halfway
    # through a run would check a model that had never run: `Thermostat` at t = 40 has come
    # on twice, and a check that starts its counter at zero is asking about a different
    # state than the one the run reached.
    held = [result.names.index(name) for name in kernel.code.remembered]
    reports = [check_iteration(kernel)]
    for event in result.events:
        column = int(np.searchsorted(result.time, event.time))
        if column >= len(result.time):  # pragma: no cover - an event past the last sample
            continue
        y = np.array([result.values[row, column] for row in rows], dtype=float)
        memory = np.array([result.values[row, column] for row in held], dtype=float)
        reports.append(check_iteration(kernel, t=event.time, y=y, memory=memory))
    return tuple(reports)


# --------------------------------------------------------------------------------------
# Dynamic state selection
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Selection:
    """Which state set is in use, which are singular, and which the engine has ruled out."""

    chosen: tuple[str, ...]
    singular: frozenset[tuple[str, ...]]
    """The sets whose constraint Jacobian is currently rank-deficient. The *truth*, which
    the engine does not get to read -- it finds out by trying."""

    rejected: frozenset[tuple[str, ...]]
    """What the engine has learned at this point of the trajectory, and forgets when it
    advances: a factorisation that failed here says nothing about anywhere else."""

    def __str__(self) -> str:
        def show(sets: frozenset[tuple[str, ...]]) -> str:
            return "{" + ", ".join(sorted("(" + ",".join(s) + ")" for s in sets)) + "}"

        return (
            f"chosen=({','.join(self.chosen)}) singular={show(self.singular)}"
            f" rejected={show(self.rejected)}"
        )


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """Whether a dynamic state selection can get stuck, and what it is trusting."""

    name: str
    candidates: tuple[tuple[str, ...], ...]
    safety: OpenResult[Selection]
    liveness: LivenessResult[Selection]

    @property
    def ok(self) -> bool:
        return self.safety.ok and self.liveness.ok

    def __str__(self) -> str:
        if len(self.candidates) == 1:
            return (
                f"{self.name}: one admissible state set, so there is nothing to select"
                " between -- it has to stay regular for the whole run"
            )
        if not self.safety.ok:
            return f"{self.name}: {self.safety}"
        if not self.liveness.ok:
            return f"{self.name}: {self.liveness}"
        assumptions = ", ".join(a.name for a in self.safety.unverified_assumptions)
        return (
            f"{self.name}: switching between {len(self.candidates)} state sets always"
            f" recovers a regular one, assuming {assumptions}"
        )


def _switching(system: System) -> list[Action[Selection]]:
    """The engine's side: find out the hard way, switch, and move on.

    Modelled the way it would be implemented rather than the way it would be wished for.
    Nothing tells a solver in advance that a state set has gone singular -- it finds out
    when the factorisation fails -- so "detect" is a step of its own, and what it learns is
    only valid where it learned it.
    """
    actions = [
        Action(
            name="detect",
            enabled=lambda s: s.chosen in s.singular and s.chosen not in s.rejected,
            apply=lambda s: Selection(s.chosen, s.singular, s.rejected | {s.chosen}),
        ),
        Action(
            name="advance",
            enabled=lambda s: s.chosen not in s.singular and s.chosen not in s.rejected,
            apply=lambda s: Selection(s.chosen, s.singular, frozenset()),
        ),
    ]
    actions.extend(
        Action(
            name=f"switch to ({','.join(candidate)})",
            enabled=lambda s, _c=candidate: s.chosen in s.rejected and _c not in s.rejected,
            apply=lambda s, _c=candidate: Selection(_c, s.singular, s.rejected),
        )
        for candidate in system.candidates
    )
    return actions


def numerics(system: System, *, quiescent: bool = True) -> EnvironmentSpec[Selection]:
    """The linear algebra, as an adversary that decides what goes singular and when.

    Which state set is usable is not the engine's to choose: it is a property of the
    constraint Jacobian at the point the trajectory has reached, and the engine only ever
    learns it by failing. So the numerics is the environment, free to degenerate and recover
    any set in any order -- under two restrictions, both of which are real assumptions and
    both of which are recorded as such rather than baked in silently.

    ``quiescent=False`` drops the second of them, which is what makes it an assumption you
    can *look at* rather than one you have to believe: it is the difference between a
    passing check and a counterexample, and a test asserts exactly that.
    """
    everything = frozenset(system.candidates)

    def searching(state: Selection) -> bool:
        """Whether the environment is allowed to move right now."""
        return not quiescent or not state.rejected

    return EnvironmentSpec[Selection](
        name="numerics",
        behaviors=[
            *(
                Action(
                    name=f"({','.join(candidate)}) goes singular",
                    enabled=lambda s, _c=candidate: (
                        searching(s)
                        and _c not in s.singular
                        and len(s.singular) + 1 < len(everything)
                    ),
                    apply=lambda s, _c=candidate: Selection(
                        s.chosen, s.singular | {_c}, s.rejected
                    ),
                )
                for candidate in system.candidates
            ),
            *(
                Action(
                    name=f"({','.join(candidate)}) recovers",
                    enabled=lambda s, _c=candidate: searching(s) and _c in s.singular,
                    apply=lambda s, _c=candidate: Selection(
                        s.chosen, s.singular - {_c}, s.rejected
                    ),
                )
                for candidate in system.candidates
            ),
        ],
        guarantees={
            "one-set-is-regular": lambda s: len(s.singular) < len(everything),
        },
        assumptions={
            "index-reduction-leaves-a-usable-set": Assumption(
                name="index-reduction-leaves-a-usable-set",
                description=(
                    "at every point of the trajectory at least one admissible state set has"
                    " a non-singular constraint Jacobian. Structural analysis cannot"
                    " establish this -- it is a statement about numbers, not about incidence"
                ),
                status="unverified",
                source="Mattsson & Soderlind (1993), dummy derivatives",
            ),
            "jacobians-hold-still-during-a-search": Assumption(
                name="jacobians-hold-still-during-a-search",
                description=(
                    "the Jacobians change only as the step advances, not while the engine is"
                    " part-way through trying candidates. Drop it (`quiescent=False`) and"
                    " liveness fails, not safety: a set can recover *after* the engine ruled"
                    " it out, so the engine exhausts every candidate at a point where one of"
                    " them was usable all along, and no step is ever accepted again"
                ),
                status="unverified",
                source="modelica.verify.numerics",
            ),
        },
    )


def selection(system: System) -> Model[Selection]:
    """A dynamic state selection, as a model checker's model.

    The engine's actions only; pair it with :func:`numerics` through
    :func:`~musil.check_open` to check it against a hostile linear algebra.
    """
    return Model(
        init=Selection(system.states, frozenset(), frozenset()),
        actions=_switching(system),
    )


def check_selection(system: System, *, quiescent: bool = True) -> SelectionReport:
    """Prove that switching state sets always recovers a usable one, and never chatters.

    Two properties, and the second is the one that matters. Safety says the engine never
    runs out of things to try. Liveness with ``everywhere=True`` says something stronger and
    harder: from *any* state it can reach, a usable selection is still reachable -- so there
    is no loop it can fall into where it switches back and forth forever without ever
    getting a step accepted.
    """
    model = selection(system)
    environment = numerics(system, quiescent=quiescent)
    combined = Model(
        init=model.init,
        actions=[*model.actions, *environment.behaviors],
        invariants=dict(environment.guarantees),
    )
    return SelectionReport(
        name=system.name,
        candidates=system.candidates,
        safety=check_open(model, environment),
        liveness=check_liveness(
            combined,
            goal=lambda s: s.chosen not in s.singular and not s.rejected,
            goal_name="a usable state set",
            everywhere=True,
            fair=[action.name for action in model.actions],
        ),
    )


# --------------------------------------------------------------------------------------
# The FMI 3.0 calling sequence
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModeReport:
    """What every legal call sequence of an FMI interface does and does not reach."""

    interface: Interface
    reached: frozenset[Mode]
    """The modes some legal sequence actually gets to. A mode in the table but not in here
    is a mode no importer can enter -- either the table is wrong or the mode is dead."""

    unreached: frozenset[Mode]
    stranded: frozenset[Mode]
    """Modes from which the instance can never be freed. Empty is the only acceptable
    answer: an FMU an importer cannot dispose of is a leak the standard does not allow for."""

    safety: CheckResult[Mode]

    @property
    def ok(self) -> bool:
        return self.safety.ok and not self.unreached and not self.stranded

    def __str__(self) -> str:
        if self.unreached:
            unreachable = ", ".join(sorted(self.unreached))
            return f"{self.interface}: unreachable modes -- {unreachable}"
        if self.stranded:
            stuck = ", ".join(sorted(self.stranded))
            return f"{self.interface}: cannot free an instance from -- {stuck}"
        if not self.safety.ok:
            return f"{self.interface}: {self.safety}"
        return (
            f"{self.interface}: all {len(self.reached)} modes reachable, no dead ends, and"
            " an instance can always still be freed"
        )


def modes(interface: Interface) -> Model[Mode]:
    """The FMI calling sequence, as a model checker's model.

    The state is the mode and nothing else, because that is exactly what the specification's
    machine says: what may be called next is a function of where you are, and of nothing you
    did to get there. If that ever stops being true the table is the wrong shape, and this
    is where it shows.

    The actions come straight out of the table through :func:`musil.protocol_actions`, so
    there is no second description of the sequence for the table to drift from -- calls that
    are legal without moving contribute no action, which is the same reading a guard gives
    them.
    """
    return Model(
        init=Mode.INSTANTIATED,
        actions=protocol_actions(
            table(interface), get=lambda s: s, with_state=lambda _s, target: target
        ),
        terminal=lambda s: s is Mode.FREED,
    )


def check_modes(interface: Interface) -> ModeReport:
    """Prove the calling sequence has no traps, no dead modes, and always an exit.

    Three questions. *No dead ends* is ``check``'s deadlock report. *Every mode reachable*
    is the reachable set compared against the table, which catches a mode written down that
    nothing can enter -- an easy mistake when transcribing a specification and an impossible
    one to spot by reading; it found Event Mode sitting in the Scheduled Execution table,
    where the standard has Clock Activation Mode instead.

    *Always an exit* is deliberately **reachability**, not liveness. "Every run eventually
    frees the instance" is false and should be: an importer that sits in Event Mode forever
    is doing nothing wrong, and asking a checker to rule that out only produces a
    counterexample that describes a legal caller. What the standard owes is that
    ``fmi3FreeInstance`` is *available* from wherever you are, which is a question about the
    graph and is answered on the graph.
    """
    model = modes(interface)
    graph = explore(model)
    declared = frozenset(table(interface))
    reached = frozenset(graph.states)
    return ModeReport(
        interface=interface,
        reached=reached,
        unreached=declared - reached,
        stranded=reached - reaches(graph, lambda s: s is Mode.FREED),
        safety=check(model),
    )


@dataclass(frozen=True, slots=True)
class MasterReport:
    """Whether a co-simulation master can get stuck, and whether it always finishes."""

    components: tuple[Component, ...]
    safety: CheckResult[Progress]
    liveness: LivenessResult[Progress]
    declined: bool
    """Whether the master refused the system at load time rather than stepping it.

    An outcome, not a failure: the components did not offer what the algorithm needs, and
    saying so before anything moves is the correct behaviour. ``ok`` stays true.
    """

    @property
    def ok(self) -> bool:
        return self.safety.ok and self.liveness.ok

    def __str__(self) -> str:
        who = ", ".join(
            f"{c.name}{'' if c.rollback else ' (no rollback)'}" for c in self.components
        )
        if not self.safety.ok:
            return f"master over {who}: {self.safety}"
        if not self.liveness.ok:
            return f"master over {who}: {self.liveness}"
        if self.declined:
            return (
                f"master over {who}: declined before stepping anything -- a rollback-based"
                " algorithm needs `canGetAndSetFMUState` from every component"
            )
        return f"master over {who}: every component reaches the horizon together"


def master(
    components: tuple[Component, ...],
    *,
    horizon: int = 2,
    retries: int = 1,
    require_rollback: bool = True,
) -> Model[Progress]:
    """A co-simulation master, as a model checker's model.

    ``retries`` bounds how often a component may refuse a step before the last accepted
    point. Unbounded refusal is legal FMI and makes progress impossible, which is a true
    statement about the standard and an uninteresting counterexample about the master: real
    masters give up after a fixed number, so the bound is the honest thing to model.

    The invariants are the two things a master is *for*. Components must never be spread
    across more than one communication point once the step is over, and a component must
    never be asked to restore a snapshot it does not hold -- the second reads as a tautology
    only because :mod:`modelica.ssp.master` refuses to offer the move; the deadlock check is
    what turns the refusal into a finding.
    """

    def out(state: Progress) -> list[tuple[str, Progress]]:
        return list(
            moves(
                state,
                components,
                horizon=horizon,
                retries=retries,
                require_rollback=require_rollback,
            )
        )

    catalogue = {name for state in _walk(start(components), out) for name, _ in out(state)}
    return Model(
        init=[start(components)],
        actions=[
            Action(
                name=name,
                enabled=lambda s, _n=name: any(move == _n for move, _ in out(s)),
                apply=lambda s, _n=name: next(nxt for move, nxt in out(s) if move == _n),
            )
            for name in sorted(catalogue)
        ],
        invariants={
            "agreed-point-is-never-passed-twice": lambda s: all(
                s.accepted <= time <= s.accepted + 1 for time in s.times
            ),
            # Not "every snapshot is current": a snapshot left over from the previous
            # communication point is harmless, because `restore` will not touch it. What has
            # to hold is that when a rollback is actually needed, everyone who moved can be
            # put back -- and that is the one an FMU declining `canGetAndSetFMUState`
            # breaks, in a run where somebody else stepped before it refused.
            "a-rollback-can-put-everyone-back": lambda s: (
                s.phase is not Phase.ROLLBACK
                or all(
                    saved == s.accepted
                    for time, saved in zip(s.times, s.saved, strict=True)
                    if time != s.accepted
                )
            ),
        },
        terminal=lambda s: s.phase in (Phase.DONE, Phase.DECLINED),
    )


def _walk(origin: Progress, out: Callable[[Progress], list[tuple[str, Progress]]]) -> set[Progress]:
    """Every state the protocol can reach, so the move names can be collected once.

    musil wants a fixed list of actions and asks each whether it is enabled; the protocol
    offers moves per state. Walking it here bridges the two without letting the action list
    depend on where the search happens to be.
    """
    seen = {origin}
    frontier = [origin]
    while frontier:
        for _, target in out(frontier.pop()):
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    return seen


def check_master(
    components: tuple[Component, ...],
    *,
    horizon: int = 2,
    retries: int = 1,
    require_rollback: bool = True,
) -> MasterReport:
    """Prove the master finishes, or declines before it has moved anything.

    Both endings count as settling, and that is the point of the load-time precondition: a
    system whose FMUs cannot all roll back is not one this algorithm can step, and the honest
    place to say so is before the first `fmi3DoStep`.

    ``require_rollback=False`` drops the precondition and the run that motivated it comes
    back: a component that cannot roll back has already stepped when another refuses, there
    is no move that puts it back, and the master has nothing left to offer.
    """
    model = master(components, horizon=horizon, retries=retries, require_rollback=require_rollback)
    return MasterReport(
        components=components,
        safety=check(model),
        liveness=check_liveness(
            model,
            goal=lambda s: s.phase in (Phase.DONE, Phase.DECLINED),
            goal_name="the horizon, or a refusal before moving",
            fair=[action.name for action in model.actions],
        ),
        declined=require_rollback and not all(c.rollback for c in components),
    )
