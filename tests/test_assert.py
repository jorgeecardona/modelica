"""``assert(condition, message);`` in an equation section: carried, not counted.

The MSL measurement (docs/ideas/coverage.md) found 48 example models whose first refusal
was an assert in a dependency's equation section -- `Modelica.Electrical.Analog.Basic.
Resistor` opens its equations with one. An assert binds no unknown, so it must never be
counted by causalization; flattening routes it to ``FlatModel.asserts``, prefixed like
everything else, where it stays part of the model's value. `causalize` carries it onto the
`System`, and `simulate` evaluates it along the trajectory: a false error-level assert stops
the run with its message, a warning-level one reports once and lets it finish.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

from modelica import ir
from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import ModelicaSyntaxError, parse, unparse

if TYPE_CHECKING:
    from modelica.sim.result import Result

SOURCE = """\
model M
  Real x(start = 0, fixed = true);
equation
  assert(x < 10, "x ran away");
  der(x) = 1;
end M;
"""


def test_an_assert_is_read_and_round_trips() -> None:
    program = parse(SOURCE)
    m = program.find("M")
    assert m is not None
    check = m.equations[0]
    assert isinstance(check, ir.Assert)
    assert check.message == ir.StringLiteral("x ran away")
    assert parse(unparse(program)) == program


def test_the_level_may_be_positional_or_named() -> None:
    positional = 'model M\nequation\n  assert(true, "m", AssertionLevel.warning);\nend M;\n'
    named = 'model M\nequation\n  assert(true, "m", level = AssertionLevel.warning);\nend M;\n'
    assert parse(positional) == parse(named)


def test_anything_else_in_the_call_is_refused() -> None:
    with pytest.raises(ModelicaSyntaxError, match="assert takes a condition"):
        parse("model M\nequation\n  assert(true);\nend M;\n")


def test_flattening_carries_the_assert_and_never_counts_it() -> None:
    flat = flatten(parse(SOURCE), "M")
    assert len(flat.asserts) == 1
    assert len(flat.equations) == 1
    causalize(flat)  # one unknown, one equation: the assert did not unbalance the model


def test_an_assert_inside_a_component_is_prefixed() -> None:
    source = """\
model Sub
  Real y;
equation
  assert(y > 0, "gone");
  y = 1;
end Sub;
model M
  Sub s;
end M;
"""
    flat = flatten(parse(source), "M")
    check = flat.asserts[0]
    assert check.message == ir.StringLiteral("gone")
    assert check.condition == ir.Binary(op=">", lhs=ir.Ref(("s", "y")), rhs=ir.IntLiteral(0))


# A cooling body that decays from 300 K toward 250 K; T crosses 270 K at t = 5 ln(2.5) ~= 4.58.
_COOLING = """\
model Cooling
  Real T(start = 300, fixed = true);
  parameter Real tau = 5;
equation
  der(T) = (250 - T) / tau;
  {check}
end Cooling;
"""


def _simulate_cooling(check: str) -> Result:
    causalize = pytest.importorskip("modelica.sim").causalize
    simulate = pytest.importorskip("modelica.sim").simulate
    system = causalize(flatten(parse(_COOLING.format(check=check)), "Cooling"))
    return simulate(system, stop=20.0, tolerance=1e-8)


def test_a_violated_error_assert_stops_the_run_with_its_message() -> None:
    """The default level is an error: a false condition ends the run, naming the time and text."""
    solve = pytest.importorskip("modelica.sim.solve")
    with pytest.raises(solve.SimulationError, match="assertion failed") as caught:
        _simulate_cooling('assert(T > 270, "temperature below the floor");')
    message = str(caught.value)
    assert "temperature below the floor" in message
    assert "t = 4.6" in message, message  # the crossing, read off the output grid


def test_a_warning_assert_reports_but_lets_the_run_finish() -> None:
    """``AssertionLevel.warning`` downgrades it: the run completes, with one warning."""
    pytest.importorskip("modelica.sim")
    with pytest.warns(UserWarning, match="temperature dipped"):
        result = _simulate_cooling('assert(T > 270, "temperature dipped", AssertionLevel.warning);')
    assert result.at(20.0)["T"] < 270  # it really did dip, and the run went on anyway


def test_a_satisfied_assert_is_silent() -> None:
    """A condition that holds the whole way raises nothing and warns about nothing."""
    pytest.importorskip("modelica.sim")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        result = _simulate_cooling('assert(T > 0, "temperature went non-physical");')
    assert result.at(20.0)["T"] > 0
