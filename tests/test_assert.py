"""``assert(condition, message);`` in an equation section: carried, not counted.

The MSL measurement (docs/ideas/coverage.md) found 48 example models whose first refusal
was an assert in a dependency's equation section -- `Modelica.Electrical.Analog.Basic.
Resistor` opens its equations with one. An assert binds no unknown, so it must never be
counted by causalization; flattening routes it to ``FlatModel.asserts``, prefixed like
everything else, where it stays part of the model's value. The simulator does not
evaluate asserts yet, and that is recorded on the node rather than discovered.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import ModelicaSyntaxError, parse, unparse

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
