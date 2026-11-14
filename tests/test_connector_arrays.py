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


def test_a_for_loop_connects_each_element_of_an_array_of_connectors() -> None:
    """MSL's polyphase `Star` wires a plug's pins to a common node in a `for` loop.

    The loop index rides on the endpoints, which are indexed, so it is substituted into each and the
    loop comes out one connect per element -- ``connect(p[1], common)``, ``connect(p[2], common)`` --
    which the endpoint resolver reads like any other.
    """
    source = (
        LIB + "model M\n  Pin p[2];\n  Pin common;\n"
        "equation\n  for j in 1:2 loop\n    connect(p[j], common);\n  end for;\nend M;\n"
    )
    flat = flatten(parse(source), "M")
    equations = repr(flat.equations)
    # Every element's potential is tied to the common node; the flows sum to zero at it.
    assert "p[1]" in equations
    assert "p[2]" in equations
    assert ir.Equal(ir.Ref(("p[1]", "v")), ir.Ref(("common", "v"))) in flat.equations


def test_a_whole_array_connect_joins_the_elements() -> None:
    """`connect(p, q)` between two arrays of connectors is the element-wise connect it stands for.

    The power-converter models write `connect(internalEnable, booleanReplicator.y)` between two arrays
    rather than looping; each side is one array of the same length, so it joins p[i] to q[i].
    """
    source = LIB + "model M\n  Pin p[2];\n  Pin q[2];\nequation\n  connect(p, q);\nend M;\n"
    flat = flatten(parse(source), "M")
    # Each element pair is joined: the potentials equal, the flows sum to zero, for element 1 and 2.
    assert ir.Equal(ir.Ref(("p[1]", "v")), ir.Ref(("q[1]", "v"))) in flat.equations
    assert ir.Equal(ir.Ref(("p[2]", "v")), ir.Ref(("q[2]", "v"))) in flat.equations


def test_an_array_connected_to_a_scalar_is_refused_by_name() -> None:
    """An array of connectors and a single one have unequal length and cannot be joined."""
    source = LIB + "model M\n  Pin p[2];\n  Lamp lamp;\nequation\n  connect(p, lamp.p);\nend M;\n"
    with pytest.raises(FlattenError, match="unequal array length"):
        flatten(parse(source), "M")


def test_a_connect_reaches_a_connector_on_each_element_of_an_array_of_components() -> None:
    """`connect(r.p, plug)` where `r` is an array of components joins `r[i].p` to `plug[i]`.

    The array need not be the last name: the polyphase machines write `connect(resistor.p,
    plug_p.pin)`, one resistor per phase, so `resistor.p` is a whole array of connectors reached
    through the array of components it belongs to.
    """
    source = (
        LIB
        + "model M\n  Lamp lamp[2];\n  Pin plug[2];\nequation\n  connect(lamp.p, plug);\nend M;\n"
    )
    flat = flatten(parse(source), "M")
    assert ir.Equal(ir.Ref(("lamp[1]", "p", "v")), ir.Ref(("plug[1]", "v"))) in flat.equations
    assert ir.Equal(ir.Ref(("lamp[2]", "p", "v")), ir.Ref(("plug[2]", "v"))) in flat.equations


CHAIN = """\
model Chain
  Modelica.Electrical.Analog.Basic.Resistor r[2](each R = 1);
  Modelica.Electrical.Analog.Sources.ConstantVoltage s(V = 2);
  Modelica.Electrical.Analog.Basic.Ground g;
equation
  connect(s.p, r[1].p); connect(r[1].n, r[2].p);
  connect(r[2].n, s.n); connect(s.n, g.p);
end Chain;
"""

FLAT_CHAIN = """\
model Chain
  Modelica.Electrical.Analog.Basic.Resistor r1(R = 1);
  Modelica.Electrical.Analog.Basic.Resistor r2(R = 1);
  Modelica.Electrical.Analog.Sources.ConstantVoltage s(V = 2);
  Modelica.Electrical.Analog.Basic.Ground g;
equation
  connect(s.p, r1.p); connect(r1.n, r2.p);
  connect(r2.n, s.n); connect(s.n, g.p);
end Chain;
"""


def test_a_subscript_on_an_inner_component_names_the_element() -> None:
    """``connect(r[1].n, r[2].p)`` joins the pins of two elements of an array of components."""
    flat = flatten(parse(CHAIN), "Chain")
    renamed = flatten(parse(FLAT_CHAIN), "Chain")
    assert repr(flat.equations).replace("r[1]", "r1").replace("r[2]", "r2") == repr(
        renamed.equations
    )


def test_the_chain_divides_the_voltage() -> None:
    from modelica.ir.causalize import causalize  # noqa: PLC0415
    from modelica.sim import simulate  # noqa: PLC0415

    result = simulate(causalize(flatten(parse(CHAIN), "Chain")), stop=1.0)
    assert abs(float(result["r[2].v"][-1]) - 1.0) < 1e-9


def test_an_inner_subscript_in_an_equation_reads_the_element() -> None:
    source = LIB + "model M\n  Lamp lamp[2];\n  Real x;\nequation\n  x = lamp[2].p.v;\nend M;\n"
    flat = flatten(parse(source), "M")
    assert ir.Equal(ir.Ref(("x",)), ir.Ref(("lamp[2]", "p", "v"))) in flat.equations


def test_an_inner_subscript_on_a_statement_target_parses_and_flatten_refuses_it() -> None:
    """``lamp[2].p.v := 1`` parses; an algorithm cannot assign through it yet, and says so."""
    source = LIB + "model M\n  Lamp lamp[2];\nalgorithm\n  lamp[2].p.v := 1;\nend M;\n"
    program = parse(source)
    assert parse(unparse(program)) == program
    with pytest.raises(FlattenError, match="through a subscripted component"):
        flatten(program, "M")
