"""``connect(p[2], q)``: one element of an array of connectors, named and wired.

The MSL measurement (docs/ideas/coverage.md) counted 83 example models whose first
refusal was the parser meeting the ``[`` in a connect endpoint. The elements themselves
already existed -- array expansion declares ``p[1]``, ``p[2]`` like any component -- so
the missing piece was naming one from a ``connect``. The subscript is structural the way
a dimension is: it decides which element carries the wire, so it is evaluated during
flattening against the parameters declared so far. Connecting a whole array at once is
still refused, by name, with the elements suggested.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import parse, unparse

LIB = """\
connector Pin
  Real v;
  flow Real i;
end Pin;
model Lamp
  Pin p;
equation
  p.v = p.i;
end Lamp;
"""


def test_a_subscripted_endpoint_is_read_and_round_trips() -> None:
    source = (
        LIB + "model M\n  Pin p[2];\n  Lamp lamp;\nequation\n  connect(p[2], lamp.p);\nend M;\n"
    )
    program = parse(source)
    m = program.find("M")
    assert m is not None
    wire = m.equations[0]
    assert isinstance(wire, ir.Connect)
    assert wire.a == ir.Subscript(ir.Ref(("p",)), (ir.IntLiteral(2),))
    assert parse(unparse(program)) == program


def test_the_element_carries_the_wire_and_its_sibling_does_not() -> None:
    source = (
        LIB + "model M\n  Pin p[2];\n  Lamp lamp;\nequation\n  connect(p[2], lamp.p);\nend M;\n"
    )
    flat = flatten(parse(source), "M")
    assert ir.Equal(ir.Ref(("p[2]", "v")), ir.Ref(("lamp", "p", "v"))) in flat.equations
    assert "p[1]" not in repr(flat.equations)


def test_the_subscript_may_be_a_parameter() -> None:
    source = (
        LIB
        + "model M\n  parameter Integer k = 1;\n  Pin p[2];\n  Lamp lamp;\n"
        + "equation\n  connect(p[k], lamp.p);\nend M;\n"
    )
    flat = flatten(parse(source), "M")
    assert ir.Equal(ir.Ref(("p[1]", "v")), ir.Ref(("lamp", "p", "v"))) in flat.equations


def test_a_whole_array_connect_is_refused_by_name() -> None:
    source = LIB + "model M\n  Pin p[2];\n  Lamp lamp;\nequation\n  connect(p, lamp.p);\nend M;\n"
    with pytest.raises(FlattenError, match="whole array of connectors is not supported yet"):
        flatten(parse(source), "M")


def test_a_mid_path_subscript_is_refused_by_name() -> None:
    with pytest.raises(Exception, match="must be on the last name"):
        parse(LIB + "model M\nequation\n  connect(a[1].p, b);\nend M;\n")
