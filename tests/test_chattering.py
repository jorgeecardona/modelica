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

**The runtime watch did not fire.** 24,273 events in 45 seconds, and the run reached
``t = 1.00004854`` of a 5-second horizon -- it stalled the moment ``x`` reached zero, and
crawled from there at a nanosecond per event. Every segment advanced by 1e-9 or 2e-9, which
is `_REARM`, the distance the engine itself holds a just-fired crossing off zero.
`_STANDSTILL` was 1e-12, so **zero of 24,272 segments were short enough to count** and the
counter reset on every one, never reaching 1 of the 100 it needs.

Fixed by asking about the *span* of `_CHATTER` consecutive events rather than the length of
each one: `SLIDING` now raises in 0.19 s, naming ``x > 0``. `FIGHTING` found a second thing on
the way -- a SciPy `ValueError` out of `simulate` with no model name and no time -- which is
now a `SimulationError` naming the relations that sit flat at zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from modelica.ir.causalize import causalize
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.sim.solve import (
    _CHATTER,  # pyright: ignore[reportPrivateUsage]
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


def test_a_relay_with_no_hysteresis_is_reported_as_chattering() -> None:
    """The run stops, in a fraction of a second, and names the relation that kept firing.

    It used to run forever. Every segment of a chattering run advances by `_REARM` over the
    indicator's slope -- the nudge that keeps a just-fired crossing off zero -- and the
    standstill test asked whether each segment alone was shorter than ``1e-12 * max(1, |t|)``.
    A thousandth of the nudge, so nothing ever qualified: 24,273 events in 45 seconds with the
    counter still on 0 of `_CHATTER`.

    The test is a span now, so what is asserted here is the whole chain: the counter reaches
    `_CHATTER`, `SimulationError` is raised, and `Plan._chattering`'s tally reaches the
    message with `Event.cause` in it. Naming the relation is the point -- "the model is
    chattering" alone leaves the modeller to find which one.
    """
    with pytest.raises(SimulationError) as caught:
        simulate(_system(SLIDING, "Chattering"), stop=5.0)
    message = str(caught.value)
    assert "chattering" in message
    assert "x > 0" in message, f"the relation is not named: {message}"
    assert f"x{_CHATTER}" in message or f"x{_CHATTER + 2}" in message, message


def test_the_standstill_span_clears_the_corpus_by_orders_of_magnitude() -> None:
    """`_STANDSTILL` is a chosen number, and these are the two measurements that chose it.

    Both sides in the units the loop computes -- the span of `_CHATTER` consecutive events
    over ``max(1, |t|)``:

    ======================================  ========
    tightest 100-event span across the corpus  7.9e-1
    the relay above                            2.0e-7
    ======================================  ========

    A test rather than a comment because the margin is the justification. If a future model or
    a future solver setting narrows it, this is where that shows up, rather than in a run that
    stops for a reason nobody expected.
    """
    assert 2e-7 < _STANDSTILL < 7.9e-1
    assert _STANDSTILL / 2e-7 > 100, "too close to the model this is meant to catch"
    assert 7.9e-1 / _STANDSTILL > 100, "too close to a model that is not chattering"


ON_THE_THRESHOLD = {
    "rises through it": """model Rising
  Real x(start = 0, fixed = true);
equation
  der(x) = 1 + (if x > 0 then 1 else 0);
end Rising;
""",
    "falls away from it": """model Falling
  Real x(start = 0, fixed = true);
equation
  der(x) = -1 - (if x > 0 then 1 else 0);
end Falling;
""",
    "another state watches it": """model Watch
  Real x(start = 0, fixed = true);
  Real q(start = 0, fixed = true);
equation
  der(x) = 1;
  der(q) = if x > 0 then 2 else 0;
end Watch;
""",
}
"""Models whose watched relation reads exactly zero at ``t = 0`` and are perfectly ordinary.

``x(start = 0)`` with a relation ``x > 0`` is not a strange thing to write, and none of these
is ill-posed: the state leaves the threshold, or stays on the side its latch says it is on.
They are here as the guard on the two tests above -- whatever is done about `FIGHTING`, these
have to keep simulating, and a nudge or a bracket guard that broke them would be a cure worse
than the disease.
"""


@pytest.mark.parametrize("source", ON_THE_THRESHOLD.values(), ids=ON_THE_THRESHOLD.keys())
def test_a_relation_at_zero_at_the_start_is_not_by_itself_a_problem(source: str) -> None:
    """An indicator of exactly 0.0 at ``t = 0`` simulates, and gets the right answer.

    Measured: `Rising` reaches ``x = 4`` at ``t = 2`` (one unit of flow for the instant it is
    at the threshold, two after), `Falling` reaches ``-2``, and `Watch` reaches ``x = 2``,
    ``q = 4``. What breaks `FIGHTING` is not the zero -- it is that two relations are at zero
    and each decides the flow the other crosses on.
    """
    from modelica.sim.kernel import build  # noqa: PLC0415 - keeps this file importable without sim

    system = _system(source, source.split()[1])
    kernel = build(system)
    p, y, w = kernel.start(0.0)
    indicators = kernel.indicators(0.0, y, p, w, np.empty(kernel.events))
    assert list(indicators) == [0.0], "the point of these models is the exact zero"

    result = simulate(system, stop=2.0)
    assert abs(result.values[result.names.index("x"), -1]) > 1.0, "the state left the threshold"


def test_two_relations_fighting_raise_a_simulation_error() -> None:
    """`FIGHTING` fails in 0.01 s, and now with an error this package defines.

    It used to be ``ValueError: f(a) and f(b) must have different signs`` -- SciPy's `brentq`,
    reached through `solve_ivp`'s root finder, out of `simulate` with no model name, no time
    and nothing a caller can act on. A model whose two crossings are both at zero at ``t = 0``
    is a real thing to write by accident, and that is what writing it said.

    The message now names the relations that are flat. SciPy's own text is kept on the end: if
    the near-zero reading is the wrong explanation for some future failure, the reader still
    has what the root finder said.
    """
    with pytest.raises(SimulationError) as caught:
        simulate(_system(FIGHTING, "Fight"), stop=5.0)
    message = str(caught.value)
    assert "no crossing could be located" in message
    assert "'y > 0'" in message, f"the relations are not named: {message}"
    assert "'x > 0'" in message, f"the relations are not named: {message}"
    assert "f(a) and f(b) must have different signs" in message, "scipy's own text is dropped"
