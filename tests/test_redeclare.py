"""``replaceable`` and ``redeclare``: the language half.

A redeclaration says *which class* a component is, which is a question settled before any
equation exists. So it is not a value and not an entry -- it rides on the modification,
where Modelica writes it, and it carries a whole declaration rather than a type name,
because ``redeclare Capacitor element(C = 0.01)`` is one redeclaration and not a
redeclaration plus a binding.

What this file pins is that both halves survive the text: the declaration that offers the
slot (``replaceable``, with the ``constrainedby`` that says what fits) and the modification
that fills it. Flattening does not apply them yet, and the round trip is what has to hold
first -- a redeclaration read as something else would be applied to the wrong component.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse

LIBRARY = """\
package Parts "Two things that fit the same slot, and the slot"

  partial model Element "What a replacement has to provide"
    Real v;
    Real i;
  end Element;

  model Resistor
    extends Element;
    parameter Real R = 1;
  equation
    v = R * i;
  end Resistor;

  model Capacitor
    extends Element;
    parameter Real C = 1;
  equation
    i = C * der(v);
  end Capacitor;

  model Rig "One slot, with a resistor in it by default"
    replaceable Resistor element(R = 100) constrainedby Element "The swappable slot";
  end Rig;

  model WithCapacitor
    Rig rig(redeclare Capacitor element(C = 0.01));
  end WithCapacitor;

end Parts;
"""


def _rig() -> ir.Class:
    found = parse(LIBRARY).find("Parts.Rig")
    assert found is not None
    return found


def test_a_replaceable_component_carries_its_constraining_clause() -> None:
    element = _rig().components[0]
    assert element.replaceable
    assert element.constraining == "Element"
    assert element.type_name == "Resistor"
    assert element.comment == "The swappable slot"


def test_the_default_modification_survives_beside_the_constraint() -> None:
    """``replaceable Resistor element(R = 100)`` offers a slot *and* fills it."""
    element = _rig().components[0]
    assert element.modification.entries["R"].value == ir.IntLiteral(100)


def test_a_redeclaration_reaches_the_ir_as_a_declaration() -> None:
    found = parse(LIBRARY).find("Parts.WithCapacitor")
    assert found is not None
    entry = found.components[0].modification.entries["element"]
    assert isinstance(entry.redeclare, ir.Component)
    assert entry.redeclare.type_name == "Capacitor"
    # Its own modification rides with it, rather than being split off as a sibling entry.
    assert entry.redeclare.modification.entries["C"].value == ir.RealLiteral(0.01)


def test_a_redeclaration_makes_a_modification_truthy() -> None:
    """``__bool__`` decides whether a modification is written at all, so it has to see one."""
    assert ir.Modification(redeclare=ir.Component("element", "Capacitor"))
    assert not ir.Modification()


def test_a_replaceable_class_is_a_short_definition_with_a_constraint() -> None:
    source = (
        "package P\n"
        "  model Water\n    Real rho;\n  end Water;\n\n"
        "  model Rig\n    replaceable package Medium = Water constrainedby Water;\n  end Rig;\n"
        "end P;\n"
    )
    found = parse(source).find("P.Rig")
    assert found is not None
    medium = found.classes[0]
    assert medium.replaceable
    assert medium.restriction == "package"
    assert medium.constraining == "Water"
    assert medium.extends == (ir.Extends("Water"),)


@pytest.mark.parametrize(
    "source",
    [
        LIBRARY,
        "model M\n  replaceable Real x constrainedby Real;\nend M;\n",
        "model M\n  Real y;\n  N n(redeclare Real y(start = 1));\nend M;\n",
    ],
    ids=["library", "constrained-scalar", "redeclare-with-attributes"],
)
def test_it_round_trips(source: str) -> None:
    program = parse(source)
    assert parse(unparse(program)) == program


def test_a_replaceable_class_still_prints_as_a_package() -> None:
    """The keyword is part of the declaration; printing it as ``type`` would change it.

    A nested class is printed with a blank line on either side whatever it is, so the line
    rather than the file is what this compares.
    """
    source = "model M\n  replaceable package Medium = Water constrainedby Water;\nend M;\n"
    printed = unparse(parse(source))
    assert "  replaceable package Medium = Water constrainedby Water;" in printed.splitlines()


def test_a_redeclare_element_in_a_class_body_is_refused_by_name() -> None:
    """Legal Modelica and a different feature: it modifies what an ``extends`` brought in."""
    with pytest.raises(ModelicaSyntaxError, match="'redeclare' element in a class body"):
        parse("model M\n  extends Base;\n  redeclare Capacitor element;\nend M;\n")
