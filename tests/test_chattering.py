"""A model that chatters, and what the engine says about it.

`docs/ideas/verification.md:245` names this as the first of the model checker's three hard
limits: freezing ``y`` is what buys exactness within an instant, and it is also the boundary,
so "this says nothing about chattering *across* instants. That stays a runtime question,
watched by the standstill counter in `modelica.sim.solve`."

No corpus model chatters, so until this file nothing exercised either side of that sentence.
`SLIDING` is the standard example -- a relay with no hysteresis, which cannot settle on its
threshold because crossing it reverses the flow that crossed it.

Measured on 2026-08-13, and both halves of the sentence came out worse than it says.

**The checker is right and useless here, exactly as predicted.** `check_iteration` reports
``ok`` at ``x = +1``, ``x = 0`` and ``x = -1``: every instant settles, and settles in one
place. It is the *sequence* that does not progress, which is a liveness property of the run
and not of the instant.

**The runtime watch does not fire.** 24,273 events in 45 seconds, and the run reached
``t = 1.00004854`` of a 5-second horizon -- it stalled the moment ``x`` reached zero, and
crawled from there at a thousandth of a nanosecond per event. Every segment advanced by
1e-9 or 2e-9, which is `_REARM`, the distance the engine itself holds a just-fired crossing
off zero. `_STANDSTILL` is 1e-12. **Zero of 24,272 segments were short enough to count**, so
the counter reset on every one and never reached 1 of the 100 it needs.

Both bugs are filed. The tests below are the regression hooks and are expected failures until
they close; neither simulates anything, because a test that proves a run does not terminate by
waiting for it is a test that does not terminate either.
"""

from __future__ import annotations

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.sim.solve import (
    _CHATTER,  # pyright: ignore[reportPrivateUsage]
    _REARM,  # pyright: ignore[reportPrivateUsage]
    _STANDSTILL,  # pyright: ignore[reportPrivateUsage]
    SimulationError,
    simulate,
)

SLIDING = """model Chattering "A relay with no hysteresis, which cannot settle on its threshold"
  Real x(start = 1, fixed = true);
equation
  der(x) = if x > 0 then -1 else 1;
end Chattering;
"""

FIGHTING = """model Fight "Two relations, each reversing the flow that decides the other"
  Real x(start = 0, fixed = true);
  Real y(start = 0, fixed = true);
equation
  der(x) = if y > 0 then -1 else 1;
  der(y) = if x > 0 then 1 else -1;
end Fight;
"""


def _system(source: str, name: str):  # noqa: ANN202 - the return type is System, imported lazily
    return causalize(flatten([parse(source)], name))


def test_the_relay_is_a_model_with_one_watched_relation() -> None:
    """The shape the two expected failures below are about, asserted so they cannot drift.

    If a change to the front end ever stops reading `SLIDING` as one watched relation over
    one state, the two tests below would start failing for a reason that has nothing to do
    with what they are named for.
    """
    from modelica.sim.kernel import build  # noqa: PLC0415 - keeps this file importable without sim

    kernel = build(_system(SLIDING, "Chattering"))
    assert kernel.code.states == ("x",)
    assert [trigger.condition for trigger in kernel.code.events] == ["x > 0"]


@pytest.mark.xfail(
    reason="the-standstill-counter-cannot-fire-because-the-rearm-distance-is-larger",
    strict=True,
)
def test_a_segment_no_longer_than_the_rearm_distance_counts_as_standstill() -> None:
    """The two constants are three orders of magnitude apart, and the wrong way round.

    `_REARM` is how far the engine holds a crossing that just fired away from zero, so a
    segment that advanced by no more than that did not advance on the model's account -- the
    engine put it there. `_STANDSTILL` is the length below which a segment is called a
    standstill. With ``1e-12`` against ``1e-9`` no re-arm-paced segment is ever short enough,
    and a relay with no hysteresis runs forever without the counter reaching 1 of `_CHATTER`.

    Written as a statement about the constants rather than as a simulation on purpose: the
    failing run takes 45 seconds to produce 24,273 events and still does not raise, so a test
    that waited for it would be the same bug in the test suite.
    """
    assert _CHATTER > 0
    assert _STANDSTILL * 1.0 >= _REARM, (
        f"a segment of {_REARM:g} is the re-arm distance and counts as progress, because the"
        f" standstill threshold is {_STANDSTILL:g}"
    )


@pytest.mark.xfail(
    reason="a-scipy-valueerror-escapes-simulate-when-two-crossings-fire-at-one-instant",
    raises=ValueError,
    strict=True,
)
def test_two_relations_fighting_raise_a_simulation_error() -> None:
    """`FIGHTING` fails in 0.01 s, and not with an error this package defines.

    ``ValueError: f(a) and f(b) must have different signs`` -- SciPy's `brentq`, reached
    through `solve_ivp`'s root finder, surfacing out of `simulate` with no model name, no
    time, and nothing a caller can act on. A model whose two crossings are both at zero at
    ``t = 0`` is a real thing to write by accident, and this is what writing it says.

    `pytest.raises` is inside the expected failure rather than around it because the claim is
    *which* exception: today it is `ValueError` and the test asserts `SimulationError`, so the
    marker's ``raises`` records what actually comes out.
    """
    with pytest.raises(SimulationError):
        simulate(_system(FIGHTING, "Fight"), stop=5.0)
