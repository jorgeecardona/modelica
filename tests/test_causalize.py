"""Causalization on models small enough to reason about by hand.

`tests/test_examples.py` checks causalization the way it is really used -- against the
corpus, end to end, with numbers at the far side. That is the test that matters, and it is
also the test that cannot say *why* something broke: a corpus model exercises index
reduction, alias elimination, matching and blocking at once, so a failure names the model
rather than the stage.

This file is the other half. Every model here is four lines long and exists to put exactly
one pass under load.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from modelica.ir.causalize import CausalizationError, causalize
from modelica.ir.source import flatten
from modelica.lang import parse

if TYPE_CHECKING:
    from modelica.ir.system import System


def _system(source: str, name: str = "M") -> System:
    return causalize(flatten([parse(source)], name))


def _decay(fixed: str) -> str:
    """``der(x) = -x`` alongside an integrator of it, with x's start attribute as given."""
    attribute = "" if not fixed else f", fixed = {fixed}"
    return f"""
model M
  Real x(start = 1.0{attribute});
  Real v(start = 0.0, fixed = true);
equation
  der(x) = -x;
  der(v) = x;
end M;
"""


@pytest.mark.parametrize("fixed", ["true", "false", ""], ids=["fixed", "unfixed", "absent"])
def test_a_state_is_never_eliminated_in_favour_of_its_own_derivative(fixed: str) -> None:
    """``der(x) = -x`` is an affine relation between two unknowns, and must not be read as one.

    Alias elimination looks for ``a = +/-b + k`` and substitutes one away. ``der(x) = -x`` has
    exactly that shape, so nothing structural stops it from choosing ``x := -der_x`` -- which
    does not eliminate a variable. It deletes the equation of motion and leaves ``x`` to be
    found from a derivative that is now determined by nothing at all. Index reduction then
    differentiates looking for the equation that went missing, invents ``der_der_x``, and
    reports the model as structurally singular: a true statement about the wreckage and a
    false one about the model, which is an ordinary exponential decay.

    Parametrised over the ``fixed`` attribute because that is what hid it. Every model in the
    corpus writes ``fixed = true`` on its states, and `_permits` refuses to substitute a fixed
    variable for an unrelated reason -- so the guard that mattered was never the one being
    tested, and the bug was invisible until a model declined to pin its start value.
    """
    system = _system(_decay(fixed))
    assert sorted(system.states) == ["v", "x"]


@pytest.mark.parametrize(
    ("equation", "closed_form"),
    [
        ("der(x) = -x", math.exp(-1.0)),
        ("der(x) = x", math.exp(1.0)),
        ("der(x) = 2*x", math.exp(2.0)),
        ("der(x) = -2.5*x", math.exp(-2.5)),
    ],
)
def test_the_simplest_ode_there_is_integrates_to_the_answer(
    equation: str, closed_form: float
) -> None:
    """And the trajectory, not just the state selection, because wrong states integrate too.

    ``der(x) = a*x`` from ``x(0) = 1`` is ``e^a`` at ``t = 1``. If alias elimination has eaten
    the equation of motion the run does not necessarily fail -- it can quite happily integrate
    something else -- so the closed form is what pins it.
    """
    simulate = pytest.importorskip("modelica.sim").simulate
    source = f"model M Real x(start = 1.0, fixed = true); equation {equation}; end M;"
    result = simulate(_system(source), stop=1.0, tolerance=1e-11)
    assert float(result["x"][-1]) == pytest.approx(closed_form, rel=1e-7)


def test_pinning_a_start_value_does_not_change_what_gets_integrated() -> None:
    """``fixed`` is an initialization attribute and must not reach the structural decisions.

    Whether a start value is pinned says what the *initial* values are. It says nothing about
    which variables are states or in what order the blocks run, and a stage that lets it reach
    those decisions is a stage where two models that differ only in their initial conditions
    can causalize into different systems. That is the general statement of the specific bug
    above, and it is the one worth holding on to.
    """
    shapes = [_system(_decay(fixed)) for fixed in ("true", "false", "")]
    first = shapes[0]
    for other in shapes[1:]:
        assert other.states == first.states
        assert other.index == first.index
        assert len(other.blocks) == len(first.blocks)


def test_a_genuinely_singular_model_is_still_reported_as_singular() -> None:
    """The other side of the fix: refusing that substitution must not refuse the diagnosis.

    Two equations saying the same thing about one unknown, and nothing determining the other.
    No amount of care in alias elimination makes this solvable, and a stage that answered
    anyway would be worse than one that raises.
    """
    source = """
model M
  Real x;
  Real y;
equation
  x = 1.0;
  x = 1.0;
end M;
"""
    with pytest.raises(CausalizationError):
        _system(source)
