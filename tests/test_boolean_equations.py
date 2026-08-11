"""A Boolean set by a relation, which is latched rather than differentiated.

``hot = x > 0.5`` is an ordinary Modelica equation and it used to fail with
``DerivativeError: a '>' expression has no derivative``. Two passes asked for a derivative of
it: causalization, to find the coefficient of the unknown it was solving for, and the
emitter's forward-mode walk, to build a tangent row. Neither question is worth asking of a
relation, and each had a correct answer available without one.

* An unknown standing alone on one side of an equation has slope exactly +/-1. No
  differentiation finds that out; the shape does.
* A boolean-valued expression is piecewise constant between events, so its tangent is zero
  everywhere the tangent exists. The one place it does not exist is a crossing, and this
  engine handles a crossing as an event.

What falls out is that the value is **latched**. The emitted line is ``hot = w[k] > 0.5``,
reading the workspace slot that ``indicators`` watches and ``transition`` flips -- not
``x > 0.5`` recomputed at every evaluation. That is the behaviour the model asks for: a
relation inside an equation is frozen between events, and a named one is no different.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelica.lang import parse
from modelica.sim import causalize, emit, flatten, simulate

if TYPE_CHECKING:
    from modelica.ir.system import System

FALLING = """model M
  Real x(start = 1, fixed = true);
  Boolean hot;
equation
  der(x) = -1;
  hot = x > 0.5;
end M;
"""


def system(source: str, name: str = "M") -> System:
    return causalize(flatten([parse(source)], name))


def test_a_boolean_set_by_a_relation_builds() -> None:
    """The whole bug, in one line of Modelica."""
    assert emit(system(FALLING)).source


def test_it_reads_the_latch_rather_than_recomputing_the_relation() -> None:
    """``hot = w[k] > 0.5``, not ``hot = x > 0.5``.

    This is the difference between a value that is frozen between events and one that the
    integrator can watch change under its own feet mid-step. `DrainingTank` is in the corpus
    because the second kind drains a tank to minus one machine epsilon and then oscillates.
    """
    where: dict[str, list[str]] = {}
    current = ""
    for line in emit(system(FALLING)).source.splitlines():
        if line.startswith("def "):
            current = line[4:].split("(")[0]
        if line.strip().startswith("hot = "):
            where.setdefault(current, []).append(line.strip())

    assert where, "nothing assigns hot"
    # `latch` is the one function that reads the relation live, and it has to: it is what
    # establishes the latched value at the initial point, where there is no previous event
    # to inherit one from.
    assert where.pop("latch") == ["hot = x > 0.5"]
    assert where, "every other function should read it too"
    for function, assigned in where.items():
        assert all("w[" in line for line in assigned), f"{function}: {assigned}"


def test_the_value_is_what_the_relation_says_and_it_changes_at_the_crossing() -> None:
    result = simulate(system(FALLING), stop=1.0, tolerance=1e-10)
    assert [round(float(event.time), 9) for event in result.events] == [0.5]
    assert result.at(0.2)["hot"] == pytest.approx(1.0)
    assert result.at(0.8)["hot"] == pytest.approx(0.0)


def test_a_logical_combination_of_relations_works_and_latches_each_one() -> None:
    """``a and b`` over two relations is two crossings, not one.

    The entry that filed this asked for the combination case to work *or* be refused by
    name, because silence there would be the same defect one level up.
    """
    source = """model M
      Real x(start = 0, fixed = true);
      Boolean band;
    equation
      der(x) = 1;
      band = x > 0.5 and x < 1.5;
    end M;
    """
    result = simulate(system(source), stop=2.0, tolerance=1e-10)
    assert [round(float(event.time), 9) for event in result.events] == [0.5, 1.5]
    assert result.at(0.2)["band"] == pytest.approx(0.0)
    assert result.at(1.0)["band"] == pytest.approx(1.0)
    assert result.at(1.8)["band"] == pytest.approx(0.0)


def test_a_boolean_does_not_become_a_state_or_a_jacobian_row() -> None:
    """It is not continuous, so nothing integrates it and nothing differentiates it."""
    built = system(FALLING)
    assert built.states == ("x",)
    source = emit(built).source
    tangent = [line for line in source.splitlines() if "hot" in line and "_d" in line]
    assert not tangent, tangent


def test_the_isolated_shape_is_used_for_ordinary_equations_too() -> None:
    """The fast path is not a special case for Booleans; it is the commonest shape there is.

    ``y = 2 * x`` has its unknown alone on the left, so the answer is the right-hand side
    and no derivative is taken to discover that. This asserts the ordinary case still comes
    out right, since it now travels a different road than it did.
    """
    source = """model M
      Real x(start = 3, fixed = true);
      Real y;
    equation
      der(x) = -x;
      y = 2 * x;
    end M;
    """
    result = simulate(system(source), stop=1.0, tolerance=1e-10)
    assert result.at(0.5)["y"] == pytest.approx(2 * result.at(0.5)["x"], rel=1e-12)
