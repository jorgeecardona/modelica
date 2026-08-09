"""``enumeration``: a type whose values are names, and a number underneath.

The specification is blunt about the encoding -- a literal *is* its position in the
declaration, counted from one -- and that is what this toolchain emits. So an enumeration
needs no fifth base type: it flattens to an `Integer` that has been told its range, and
`modelica.sim.emit`, which puts every variable in a float workspace, never learns that the
type existed.

Two consequences are what the tests below are about. The ordinal has to be *right*, because
it is the whole meaning of the type and nothing downstream can check it. And `Schedule.day`
has to become a number before prefixing, because it is not a component reference and
prefixing it would produce a variable that does not exist.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse, unparse_equation

MODEL = """\
model M
  type Mode = enumeration(red, green, blue);
  parameter Mode chosen = Mode.green;
  Real x;
equation
  x = if chosen == Mode.blue then 1.0 else 2.0;
end M;
"""


def test_the_literals_are_kept_in_order() -> None:
    """Order is the content: reordering them changes what every literal means."""
    mode = parse(MODEL).classes[0].classes[0]
    assert mode.literals == ("red", "green", "blue")
    assert mode.restriction == "type"
    # An enumeration defines a type; it does not rename one.
    assert mode.extends == ()


def test_it_round_trips() -> None:
    program = parse(MODEL)
    assert parse(unparse(program)) == program


def test_it_prints_as_it_was_written() -> None:
    source = 'model M\n  type E = enumeration(a, b) "Two of them";\nend M;\n'
    assert '  type E = enumeration(a, b) "Two of them";' in unparse(parse(source)).splitlines()


def test_a_literal_becomes_its_position_counted_from_one() -> None:
    flat = flatten(parse(MODEL), "M")
    assert unparse_equation(flat.equations[0])[0] == "x = if chosen == 3 then 1.0 else 2.0;"


def test_the_variable_is_an_integer_that_knows_its_range() -> None:
    chosen = next(v for v in flatten(parse(MODEL), "M").variables if v.name == "chosen")
    assert chosen.base_type == "Integer"
    assert (chosen.minimum, chosen.maximum) == (ir.IntLiteral(1), ir.IntLiteral(3))
    assert chosen.binding == ir.IntLiteral(2)


def test_a_literal_in_a_binding_is_resolved_too() -> None:
    """A binding is not an equation, so it takes the same rewrite in a second place."""
    source = "model M\n  type E = enumeration(a, b);\n  parameter E e = E.b;\nend M;\n"
    binding = next(v for v in flatten(parse(source), "M").variables if v.name == "e")
    assert binding.binding == ir.IntLiteral(2)


def test_a_component_of_that_name_wins_over_the_type() -> None:
    """A local name is a local name, whatever else it might resolve to."""
    source = (
        "model M\n"
        "  type Mode = enumeration(day);\n"
        "  Real Mode;\n"
        "  Real y;\n"
        "equation\n"
        "  Mode = 1.0;\n"
        "  y = Mode;\n"
        "end M;\n"
    )
    assert unparse_equation(flatten(parse(source), "M").equations[1])[0] == "y = Mode;"


def test_an_enumeration_with_no_literals_is_refused() -> None:
    with pytest.raises(ModelicaSyntaxError, match="has no literals"):
        parse("model M\n  type E = enumeration();\nend M;\n")


def test_a_literal_that_was_never_declared_stays_a_reference_and_fails_later() -> None:
    """Not silently zero: an unknown member of a known type has to be an error somewhere."""
    source = "model M\n  type E = enumeration(a);\n  Real x;\nequation\n  x = E.z;\nend M;\n"
    with pytest.raises(FlattenError):
        flatten(parse(source), "M")
