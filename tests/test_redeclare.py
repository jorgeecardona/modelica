"""``replaceable`` and ``redeclare``: the language half.

A redeclaration says *which class* a component is, which is a question settled before any
equation exists. So it is not a value and not an entry -- it rides on the modification,
where Modelica writes it, and it carries a whole declaration rather than a type name,
because ``redeclare Capacitor element(C = 0.01)`` is one redeclaration and not a
redeclaration plus a binding.

What this file pins is that both halves survive the text: the declaration that offers the
slot (``replaceable``, with the ``constrainedby`` that says what fits) and the modification
that fills it. The round trip had to hold before flattening could apply anything: a redeclaration read
as something else would be applied to the wrong component. The second half of the file is
the application -- one slot, two classes, two different flat models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from modelica.ir.flat import FlatModel

from modelica import build, ir
from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse, unparse_equation

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


def test_redeclaring_a_class_as_an_element_is_refused_by_name() -> None:
    """The component spelling of this works; the class spelling is the other open issue."""
    with pytest.raises(ModelicaSyntaxError, match="redeclaring a class as an element"):
        parse("model M\n  extends Base;\n  redeclare package Medium = Water;\nend M;\n")


# --------------------------------------------------------------------------------------
# The other spelling: an element under an `extends`, meaning the same thing
# --------------------------------------------------------------------------------------

BY_ELEMENT = """\
model Specialised "The extends spelling of the same choice"
  extends Parts.Rig;
  redeclare Parts.Capacitor element(C = 0.01);
end Specialised;

model ByElement
  Specialised rig;
end ByElement;
"""

BY_MODIFICATION = """\
model ByModification
  Parts.Rig rig(redeclare Parts.Capacitor element(C = 0.01));
end ByModification;
"""


def test_an_element_redeclaration_is_kept_apart_from_the_components() -> None:
    """It declares nothing new, so filing it with the components would add a member."""
    specialised = parse(BY_ELEMENT).classes[0]
    assert specialised.components == ()
    assert [c.type_name for c in specialised.redeclarations] == ["Parts.Capacitor"]


def test_the_element_spelling_round_trips() -> None:
    program = parse(BY_ELEMENT)
    assert parse(unparse(program)) == program
    assert unparse(program) == BY_ELEMENT


def test_the_two_spellings_flatten_to_the_same_model() -> None:
    """Two spellings of one meaning, which is a claim that can be checked rather than said."""
    by_element = _flat(BY_ELEMENT, "ByElement")
    by_modification = _flat(BY_MODIFICATION, "ByModification")
    assert [v.name for v in by_element.variables] == [v.name for v in by_modification.variables]
    assert by_element.equations == by_modification.equations


def test_redeclaring_something_the_class_does_not_inherit_is_refused() -> None:
    """The mirror of the check the modification spelling already makes."""
    extra = "model X\n  extends Parts.Rig;\n  redeclare Parts.Capacitor absent;\nend X;\n"
    with pytest.raises(FlattenError, match="redeclares 'absent', which it does not inherit"):
        _flat(extra, "X")


def test_the_constraining_clause_is_checked_here_too() -> None:
    """One rule, reached from two places: both spellings end in the same check."""
    extra = "model Y\n  extends Parts.Rig;\n  redeclare Parts.Rig element;\nend Y;\n"
    with pytest.raises(FlattenError, match="does not satisfy 'Element'"):
        _flat(extra, "Y")


# --------------------------------------------------------------------------------------
# Flattening applies it: the same slot, two different models
# --------------------------------------------------------------------------------------

DEFAULT = "model A\n  Parts.Rig rig;\nend A;\n"
SWAPPED = "model B\n  Parts.Rig rig(redeclare Parts.Capacitor element(C = 0.01));\nend B;\n"


def _flat(extra: str, model: str) -> FlatModel:
    return flatten(parse(LIBRARY + extra), model)


def test_without_a_redeclaration_the_declared_class_is_used() -> None:
    flat = _flat(DEFAULT, "A")
    assert [v.name for v in flat.variables] == ["rig.element.v", "rig.element.i", "rig.element.R"]


def test_a_redeclaration_instantiates_the_replacement_instead() -> None:
    """Not a coefficient: a different class, so a different parameter and equation."""
    flat = _flat(SWAPPED, "B")
    assert [v.name for v in flat.variables] == ["rig.element.v", "rig.element.i", "rig.element.C"]
    assert (
        unparse_equation(flat.equations[0])[0]
        == "rig.element.i = rig.element.C * der(rig.element.v);"
    )


def test_the_replacement_brings_its_own_modification_and_not_the_old_one() -> None:
    """``R = 100`` describes a resistor; carrying it onto a capacitor would set nothing."""
    binding = {v.name: v.binding for v in _flat(SWAPPED, "B").variables}
    assert binding["rig.element.C"] == ir.RealLiteral(0.01)


def test_a_replacement_that_does_not_satisfy_the_constraint_names_both_classes() -> None:
    extra = "model D\n  Real q;\nend D;\n\nmodel C\n  Parts.Rig rig(redeclare D element);\nend C;\n"
    with pytest.raises(FlattenError, match="'D' does not satisfy 'Element'"):
        _flat(extra, "C")


def test_a_component_that_is_not_replaceable_cannot_be_redeclared() -> None:
    extra = (
        "model E\n  Parts.Resistor fixed;\nend E;\n\n"
        "model F\n  E e(redeclare Parts.Capacitor fixed);\nend F;\n"
    )
    with pytest.raises(FlattenError, match="not declared `replaceable`"):
        _flat(extra, "F")


def test_a_class_redeclaration_is_applied_and_lives_in_its_own_file() -> None:
    """It parses and reaches the IR here; what it *means* is `tests/test_redeclare_class.py`.

    The two spellings share this file's front-matter machinery and nothing else: a component
    redeclaration is a value on a member, and a class redeclaration is a binding on a scope.
    """
    source = (
        "package P\n"
        "  model Water\n    Real rho;\n  end Water;\n\n"
        "  model Rig\n    replaceable package Medium = Water constrainedby Water;\n"
        "    Real x;\n  equation\n    x = 1.0;\n  end Rig;\n"
        "end P;\n\n"
        "model M\n  P.Rig rig(redeclare package Medium = P.Water);\nend M;\n"
    )
    assert [v.name for v in flatten(parse(source), "M").variables] == ["rig.x"]


def test_the_python_front_end_writes_an_element_redeclaration() -> None:
    """``redeclared`` files a declaration under the class's redeclarations, not its components.

    Built here rather than as a class body, because the ``.mo`` twin's base lives in a
    ``.mo``-only package -- what is being checked is where the declaration lands, which is
    the part `_class_to_ir` decides.
    """
    declaration = build.redeclared(build.Real(start=1))
    assert declaration.redeclaring
    assert not build.Real(start=1).redeclaring
