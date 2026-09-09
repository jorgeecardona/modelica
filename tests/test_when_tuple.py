"""A multi-output call inside a `when` equation, written with `=`.

A `when` in an equation section takes equations, and an equation that reads two results from
one call is written `(a, b) = f(x)`. The parser reads a `when` body as statements (the only
form the specification allows there), and its tuple case required `:=`, the algorithm operator,
so the equation spelling was refused. `Modelica.Blocks.Interfaces.PartialNoise` draws its random
pair that way (`(r_raw, state) = generator.random(pre(state))`), and that one file is imported
across the standard library, so refusing it blocked well over a hundred example models at the
point they looked it up. Either operator is accepted now, as the scalar assignment beside it
already was.
"""

from __future__ import annotations

from modelica import ir
from modelica.lang import parse
from modelica.lang.unparse import unparse

EQUAL = """\
model M
  Real a;
  Real b;
equation
  when time > 1 then
    (a, b) = f(a);
  end when;
end M;
"""

ASSIGN = EQUAL.replace("(a, b) = f(a)", "(a, b) := f(a)")


def _when_body(source: str) -> ir.Statement:
    when = parse(source, "<test>").find("M").equations[0]  # type: ignore[union-attr]
    assert isinstance(when, ir.When)
    return when.branches[0].body[0]


def test_a_when_tuple_equation_reads_both_operators_the_same() -> None:
    """`(a, b) = f(a)` and `(a, b) := f(a)` inside a `when` parse to the same call statement."""
    equal = _when_body(EQUAL)
    assign = _when_body(ASSIGN)
    assert isinstance(equal, ir.CallStatement)
    assert equal.targets == (ir.Ref(("a",)), ir.Ref(("b",)))
    assert equal.call.name == "f"
    assert equal == assign


def test_the_when_tuple_equation_round_trips() -> None:
    program = parse(EQUAL, "<test>")
    assert parse(unparse(program), "<roundtrip>") == program
