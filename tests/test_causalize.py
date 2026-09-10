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

from modelica.ir.causalize import (
    CausalizationError,
    _Work,  # pyright: ignore[reportPrivateUsage]
    causalize,
)
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


def test_a_second_derivative_can_be_written_directly() -> None:
    """``der(der(x)) = -x`` is a harmonic oscillator, and the model may say so directly.

    Index reduction already names ``der_x`` and ``der_der_x`` when it differentiates an equation
    twice, so a modeller may write the second derivative out instead of introducing ``v = der(x)``
    by hand. The naming recurses: the inner ``der(x)`` becomes ``der_x`` and this der is then
    ``der(der_x)`` -> ``der_der_x``. Two states, and the trajectory is the cosine that pins it.
    """
    simulate = pytest.importorskip("modelica.sim").simulate
    source = (
        "model M\n  Real x(start = 1.0, fixed = true);\n  Real v(start = 0.0, fixed = true);\n"
        "equation\n  der(der(x)) = -x;\n  v = der(x);\nend M;\n"
    )
    system = _system(source)
    assert len(system.states) == 2
    result = simulate(system, stop=2.0, tolerance=1e-11)
    assert float(result["x"][-1]) == pytest.approx(math.cos(2.0), abs=1e-7)
    assert float(result["v"][-1]) == pytest.approx(-math.sin(2.0), abs=1e-7)


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


def test_a_runaway_differentiation_chain_is_reported_before_it_explodes() -> None:
    """An over-determined chain of derivatives is caught by the depth cap, not left to run away.

    Sixteen integrators with two constraints on the first state is over-determined, and index
    reduction differentiates the first equation without ever resolving it. Each further derivative
    doubles an already-large expression, so a real machine model reaching this took minutes; the
    depth cap stops it at once, with the same over-determined diagnosis the budget guard gives.
    """
    states = "".join(f"  Real x{i};\n" for i in range(1, 17))
    chain = "".join(f"  der(x{i}) = x{i + 1};\n" for i in range(1, 16))
    source = (
        "model M\n" + states + "equation\n" + chain + "  x1 = time;\n  x1 = 2 * time;\nend M;\n"
    )
    with pytest.raises(CausalizationError, match="never left it a variable"):
        _system(source)


def test_a_singular_algebraic_loop_is_named_not_a_division_by_zero() -> None:
    """An inconsistent algebraic loop is caught here, where its equation can still be named.

    ``a = b + 1`` and ``b = a + 1`` say ``a - b = 1`` and ``a - b = -1``: no solution. Alias
    elimination reduces it to a single equation whose unknown has a zero coefficient, and without
    the check that used to reach the emitted code as ``a = .../0.0`` -- a bare ``ZeroDivisionError``
    from the generated function with no clue which equations were at fault. Now it is refused at
    causalization, naming the singular loop, and a *consistent* loop still solves.
    """
    inconsistent = """
    model M
      Real x(start = 1, fixed = true);
      Real a;
      Real b;
    equation
      der(x) = -x;
      a = b + 1;
      b = a + 1;
    end M;
    """
    with pytest.raises(CausalizationError, match="singular algebraic loop"):
        _system(inconsistent)

    # A genuinely solvable loop must still build: this one is a - 2*b = x, b = a - 1.
    solvable = """
    model M
      Real x(start = 1, fixed = true);
      Real a;
      Real b;
    equation
      der(x) = -x;
      a = 2 * b + x;
      b = a - 1;
    end M;
    """
    assert _system(solvable) is not None


def test_walking_the_differentiation_chain_stops_even_if_it_loops() -> None:
    """`_Work.chain` is bounded, so a cycle in ``integral`` is a wrong answer and not a hang.

    `_permits` is what keeps ``integral`` a forest, and it did not always: two eliminations
    could make ``x2`` the derivative of ``x0`` and then ``x0`` the derivative of ``x2``, and
    `_Work.level` walked that with an unbounded ``while`` -- a three-equation linear ODE that
    never came back. The guard is the fix and this is the second half of it.

    The cycle is installed by hand rather than reached through a model, because the point is
    that the walk survives one however it got there.
    """
    work = _Work(name="M", equations=[])
    work.integral = {"x2": "x0", "x0": "x2"}
    assert list(work.chain("x2")) == ["x2", "x0"]
    assert work.level("x2") == 1
