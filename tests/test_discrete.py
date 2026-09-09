"""Discrete variables: what a ``when`` clause can remember.

Modelica's discrete-time variability (specification 3.x, section 8.3.5) is a variable that
holds its value between events. Nothing in the continuous part determines it, so it is
neither a state nor a block unknown, and causalization leaves it alone entirely -- which is
why it reached the emitter untouched and failed there for years.

The distinction these tests are about is between a **latch** and a **memory**, and the two
look alike until you ask the question this file asks. A latch is the cached value of a
relation the model can evaluate at any time: set it wrong and the event iteration repairs it.
A memory is a fact about the past, and nothing can repair it, because nothing else in the
model knows what it was.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelica.lang import parse
from modelica.sim import build, causalize, emit, flatten, simulate
from modelica.sim.emit import EmitError

if TYPE_CHECKING:
    from modelica.ir.system import System


def system(source: str, name: str = "M") -> System:
    return causalize(flatten([parse(source)], name))


COUNTER = """model M
  Real x(start = 1, fixed = true);
  discrete Real n(start = 0, fixed = true);
equation
  der(x) = -1;
  when x <= 0 then
    reinit(x, 1);
    n = pre(n) + 1;
  end when;
end M;
"""


def test_a_discrete_variable_is_neither_a_state_nor_a_block() -> None:
    """It is memory, and the workspace is where this engine keeps memory."""
    built = system(COUNTER)
    assert built.states == ("x",)
    assert all("n" not in block.unknowns for block in built.blocks)


def test_the_when_body_writes_the_workspace_and_the_reads_come_back_from_it() -> None:
    source = emit(system(COUNTER)).source
    assert "n = w[" in source, "a discrete variable is read from the workspace"
    assert any(
        line.strip().startswith("w[") and "= n + 1" in line for line in source.splitlines()
    ), "the when body assigns into the workspace slot"


def test_a_counter_counts() -> None:
    """``n = pre(n) + 1`` over five events. The value has to survive the event to be right.

    ``pre(n)`` needs no special handling in the emitter and this is the test that says so:
    ``transition`` runs against the *pre-event* workspace, so the name is already bound to
    the old value when the body reads it.
    """
    result = simulate(system(COUNTER), stop=5.5, tolerance=1e-9)
    assert len(result.events) == 5
    for index, moment in enumerate((0.5, 1.5, 2.5, 3.5, 4.5)):
        assert result.at(moment)["n"] == pytest.approx(float(index)), f"at t = {moment}"


def test_a_discrete_variable_starts_where_it_says() -> None:
    source = """model M
      Real x(start = 1, fixed = true);
      discrete Real held(start = 7, fixed = true);
    equation
      der(x) = -1;
      when x <= -1 then
        held = 9;
      end when;
    end M;
    """
    result = simulate(system(source), stop=0.5, tolerance=1e-9)
    assert not result.events
    assert result.at(0.25)["held"] == pytest.approx(7.0), "never assigned, so it is its start"


def test_a_discrete_variable_drives_the_continuous_part() -> None:
    """The point of the feature: an equation reads it, and it changes what is integrated."""
    source = """model M
      Real x(start = 0, fixed = true);
      discrete Real gain(start = 1, fixed = true);
    equation
      der(x) = gain;
      when x >= 1 then
        gain = -1;
      end when;
    end M;
    """
    result = simulate(system(source), stop=1.6, tolerance=1e-10)
    assert result.at(0.5)["x"] == pytest.approx(0.5, abs=1e-7), "rising at gain = 1"
    assert result.at(1.5)["x"] == pytest.approx(0.5, abs=1e-6), "falling after the flip"


def test_a_when_on_a_boolean_relation_variable_fires_at_the_crossing() -> None:
    """``when b`` where ``b = x > c`` fires the same as ``when x > c``.

    A crossing written once and named is what the event machinery could not read: it needs the
    relation, not a variable standing for it. Flattening now inlines the defining relation into
    the condition, so the ``when`` fires at the crossing and the body runs, and ``b`` is still a
    variable of the model, readable at every point.
    """
    source = """model M
      Real x(start = 0, fixed = true);
      Boolean above(start = false);
      discrete Real count(start = 0, fixed = true);
    equation
      der(x) = 1;
      above = x > 2.5;
      when above then
        count = pre(count) + 1;
      end when;
    end M;
    """
    result = simulate(system(source), stop=5.0, tolerance=1e-9)
    assert any(event.time == pytest.approx(2.5, abs=1e-6) for event in result.events), "at x = 2.5"
    assert result.at(2.4)["count"] == pytest.approx(0.0), "before the crossing"
    assert result.at(2.6)["count"] == pytest.approx(1.0), "fired once, after it"
    assert result.at(2.6)["above"] == pytest.approx(1.0), "the named relation is still a variable"


def test_a_when_on_a_compound_boolean_is_still_refused() -> None:
    """Only a single crossing is inlined; ``when a and b`` is not one relation, so it is refused.

    The variables it names are still substituted, so the message shows the relations rather than
    the opaque names -- but a conjunction is not a threshold the crossing detector can watch, and
    it stays refused, the catch-all in front of the inlining.
    """
    source = """model M
      Real x(start = 0, fixed = true);
      Boolean a(start = false);
      Boolean b(start = false);
      discrete Real count(start = 0, fixed = true);
    equation
      der(x) = 1;
      a = x > 2.5;
      b = x < 4.0;
      when a and b then
        count = pre(count) + 1;
      end when;
    end M;
    """
    with pytest.raises(EmitError, match="has to be a relation"):
        build(system(source))


def test_two_when_clauses_may_disagree_which_is_what_hysteresis_is() -> None:
    """One `when` turns it on and another turns it off, and neither can see the other.

    This is the case a latch cannot express. A relation is a question the model asks now;
    two thresholds with a memory between them is a question about which one was crossed
    last, and only a value that persists can answer it.
    """
    source = """model M
      Real x(start = 0, fixed = true);
      discrete Real on(start = 0, fixed = true);
    equation
      der(x) = 1 - 2 * on;
      when x >= 1 then
        on = 1;
      end when;
      when x <= 0 then
        on = 0;
      end when;
    end M;
    """
    result = simulate(system(source), stop=4.0, tolerance=1e-10)
    assert len(result.events) >= 3, "it should cycle rather than settle"
    values = {round(float(v), 9) for v in result["on"]}
    assert values <= {0.0, 1.0}, f"an interpolated regime would show something else: {values}"


# --------------------------------------------------------------------------------------
# What stays refused, and why each has to be
# --------------------------------------------------------------------------------------


def test_assigning_a_continuous_state_in_a_when_is_refused_with_the_remedy() -> None:
    source = """model M
      Real x(start = 1, fixed = true);
    equation
      der(x) = -1;
      when x <= 0 then
        x = 1;
      end when;
    end M;
    """
    with pytest.raises(EmitError, match=r"reinit\(x, \.\.\.\)"):
        build(system(source))


def test_an_unknown_statement_in_a_when_is_still_refused() -> None:
    """The catch-all did not go away when two cases were added in front of it."""
    source = """model M
      Real x(start = 1, fixed = true);
      Real y;
    equation
      der(x) = -1;
      y = x;
      when x <= 0 then
        y = 2;
      end when;
    end M;
    """
    with pytest.raises(EmitError, match="declare it `discrete`"):
        build(system(source))
