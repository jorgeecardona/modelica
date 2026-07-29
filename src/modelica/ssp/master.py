"""The co-simulation master's protocol, before the master exists.

An SSP system is several FMUs stepped together, and that makes a co-simulation master a
distributed algorithm wearing a numerical hat. The numerics is the easy half. The hard half
is the bookkeeping: everybody has to end each communication step at the *same* time, any
component may refuse the step it was given, and undoing a step someone else already took
requires that they kept a copy of themselves before taking it.

FMI is explicit that the copy is optional -- ``canGetAndSetFMUState`` is a capability flag,
and an FMU is allowed to say no. So a master that assumes it can always roll back is
correct right up until it meets a conforming FMU that cannot, at which point it has no move
left. That is the failure this module exists to make visible, and it is visible because the
protocol is written as a value with pure transitions rather than as a loop with an
early ``return`` in the middle.

Nothing here integrates anything. This is the *protocol*: what state the master is in, what
moves are legal, and what each one does. :func:`modelica.verify.check_master` enumerates
every interleaving of it; the eventual real master is expected to make these same moves,
which is what keeps the two from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Component", "Phase", "Progress", "moves", "start"]

NOTHING = -1
"""The stored-state marker for a component holding no snapshot."""


class Phase(StrEnum):
    """Where the master is within one communication step."""

    SAVE = "save"
    """Snapshotting whoever can be snapshotted, before anybody moves."""

    STEP = "step"
    """Handing out ``fmi3DoStep``. Components answer in any order, or refuse."""

    ROLLBACK = "rollback"
    """Somebody refused, so everybody who moved has to be put back."""

    DONE = "done"


@dataclass(frozen=True, slots=True)
class Component:
    """One FMU in the system, described only by what the master needs to know."""

    name: str
    rollback: bool
    """``canGetAndSetFMUState``. False is legal, and is the interesting case."""


@dataclass(frozen=True, slots=True)
class Progress:
    """Where every component is, what each has stored, and what the master is doing.

    Time is counted in communication points rather than seconds: the protocol questions are
    about who is ahead of whom, and integers answer those exactly while keeping the state
    space finite.
    """

    phase: Phase
    times: tuple[int, ...]
    saved: tuple[int, ...]
    """Per component, the point its stored state was taken at, or :data:`NOTHING`."""

    accepted: int
    """The last communication point every component agreed on."""

    retries: int
    """Refusals since the last accepted point."""

    def __str__(self) -> str:
        times = ",".join(str(t) for t in self.times)
        saved = ",".join("-" if s == NOTHING else str(s) for s in self.saved)
        return f"{self.phase} at {self.accepted} times=({times}) saved=({saved})"


def start(components: tuple[Component, ...]) -> Progress:
    """Every component at the origin, nothing stored, about to snapshot."""
    return Progress(
        phase=Phase.SAVE,
        times=(0,) * len(components),
        saved=(NOTHING,) * len(components),
        accepted=0,
        retries=0,
    )


def _replace(values: tuple[int, ...], index: int, value: int) -> tuple[int, ...]:
    return (*values[:index], value, *values[index + 1 :])


def moves(  # noqa: C901, PLR0912 - one branch per protocol move; splitting hides the machine
    state: Progress,
    components: tuple[Component, ...],
    *,
    horizon: int,
    retries: int,
) -> Iterator[tuple[str, Progress]]:
    """Every move legal in ``state``, as ``(name, next)``.

    Written as one generator rather than a class per phase because the whole point is to be
    able to read the protocol in one sitting: four phases, and what may happen in each.

    The refusals are in here too, not in a separate adversary, because a refusal is a legal
    answer to ``fmi3DoStep`` rather than a fault -- an FMU that discards a step is behaving,
    and a master that only works when nobody discards does not.
    """
    if state.phase is Phase.DONE:
        return

    if state.phase is Phase.SAVE:
        for index, component in enumerate(components):
            if component.rollback and state.saved[index] != state.accepted:
                yield (
                    f"save {component.name}",
                    Progress(
                        state.phase,
                        state.times,
                        _replace(state.saved, index, state.accepted),
                        state.accepted,
                        state.retries,
                    ),
                )
        if all(
            not component.rollback or state.saved[index] == state.accepted
            for index, component in enumerate(components)
        ):
            yield (
                "begin",
                Progress(Phase.STEP, state.times, state.saved, state.accepted, state.retries),
            )
        return

    if state.phase is Phase.STEP:
        for index, component in enumerate(components):
            if state.times[index] != state.accepted:
                continue
            yield (
                f"step {component.name}",
                Progress(
                    state.phase,
                    _replace(state.times, index, state.accepted + 1),
                    state.saved,
                    state.accepted,
                    state.retries,
                ),
            )
            if state.retries < retries:
                yield (
                    f"{component.name} discards",
                    Progress(
                        Phase.ROLLBACK,
                        state.times,
                        state.saved,
                        state.accepted,
                        state.retries + 1,
                    ),
                )
        if all(time == state.accepted + 1 for time in state.times):
            done = state.accepted + 1 >= horizon
            yield (
                "commit",
                Progress(
                    Phase.DONE if done else Phase.SAVE,
                    state.times,
                    state.saved,
                    state.accepted + 1,
                    0,
                ),
            )
        return

    for index, component in enumerate(components):
        if state.times[index] == state.accepted:
            continue
        # The move that is not always available: putting a component back requires that it
        # kept a copy, and `canGetAndSetFMUState` is a capability an FMU may decline. When it
        # did, there is no restore to offer here -- which is exactly the state the master
        # gets stuck in, and exactly what the deadlock check reports.
        if state.saved[index] == state.accepted:
            yield (
                f"restore {component.name}",
                Progress(
                    state.phase,
                    _replace(state.times, index, state.accepted),
                    state.saved,
                    state.accepted,
                    state.retries,
                ),
            )
    if all(time == state.accepted for time in state.times):
        yield (
            "retry",
            Progress(Phase.STEP, state.times, state.saved, state.accepted, state.retries),
        )
