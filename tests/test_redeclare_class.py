"""Redeclaring a *class*: one name, two meanings, in one program.

A component redeclaration changes one declaration, and whoever instantiates already holds
it as a value. A class redeclaration is a different kind of thing: it changes what a **name
means inside a whole instance**, at every depth of it, and the same class instantiated twice
with different replacements has to resolve that name two ways at once. A lexical scope chain
cannot say that, which is why `modelica.ir.flatten.Scope` carries per-instance bindings and
why they travel down with `inside()`.

The observation that shows it is working is the last test here: two instances of one class,
in one flat model, with different classes inside them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.unparse import unparse_equation

if TYPE_CHECKING:
    from modelica.ir.flat import FlatModel

PARTS = """\
package Parts
  partial model Element
    Real v;
    Real i;
  end Element;

  model Resistor
    extends Element;
    parameter Real R = 100;
  equation
    v = R * i;
  end Resistor;

  model Capacitor
    extends Element;
    parameter Real C = 0.01;
  equation
    i = C * der(v);
  end Capacitor;

  model Bare
    Real w;
  end Bare;
end Parts;

model Rig
  replaceable model Kind = Parts.Resistor constrainedby Parts.Element;
  Kind element;
  Real drive;
equation
  drive = element.v;
end Rig;
"""


def _flat(extra: str, model: str) -> FlatModel:
    return flatten(parse(PARTS + extra), model)


def _equations(extra: str, model: str) -> list[str]:
    return [line for e in _flat(extra, model).equations for line in unparse_equation(e)]


def test_without_a_redeclaration_the_default_class_is_used() -> None:
    assert "a.element.v = a.element.R * a.element.i;" in _equations(
        "model One\n  Rig a;\nend One;\n", "One"
    )


def test_a_redeclared_class_changes_what_a_component_inside_is() -> None:
    extra = "model One\n  Rig a(redeclare model Kind = Parts.Capacitor);\nend One;\n"
    assert "a.element.i = a.element.C * der(a.element.v);" in _equations(extra, "One")


def test_two_instances_of_one_class_hold_two_different_classes() -> None:
    """The observation the whole feature exists for, and the one a lexical scope cannot give.

    `a` and `b` are the same `Rig`. In the flat model one contains a resistor and the other a
    capacitor, which means the binding is per *instance* and not per class.
    """
    extra = "model Two\n  Rig a;\n  Rig b(redeclare model Kind = Parts.Capacitor);\nend Two;\n"
    flat = _flat(extra, "Two")
    names = [v.name for v in flat.variables]
    assert "a.element.R" in names
    assert "b.element.C" in names
    assert "a.element.C" not in names
    assert "b.element.R" not in names


def test_the_binding_reaches_every_depth_of_the_instance_and_no_further() -> None:
    """A rig inside a rig: the outer redeclaration does not leak into a sibling."""
    extra = (
        "model Pair\n  Rig inner_rig;\nend Pair;\n\n"
        "model Nested\n"
        "  Pair held(redeclare model Kind = Parts.Capacitor);\n"
        "  Rig loose;\n"
        "end Nested;\n"
    )
    # `Pair` has no `Kind` of its own to redeclare, so the modification names nothing.
    with pytest.raises(FlattenError, match="has no class 'Kind' to redeclare"):
        _flat(extra, "Nested")


def test_a_replacement_that_does_not_satisfy_the_constraint_names_both() -> None:
    extra = "model One\n  Rig a(redeclare model Kind = Parts.Bare);\nend One;\n"
    wanted = re.escape("'Bare' does not satisfy 'Parts.Element'")
    with pytest.raises(FlattenError, match=wanted):
        _flat(extra, "One")


def test_a_class_that_is_not_replaceable_cannot_be_redeclared() -> None:
    extra = (
        "model Fixed\n  model Kind = Parts.Resistor;\n  Kind e;\nend Fixed;\n\n"
        "model One\n  Fixed f(redeclare model Kind = Parts.Capacitor);\nend One;\n"
    )
    with pytest.raises(FlattenError, match="not declared `replaceable`"):
        _flat(extra, "One")


def test_redeclaring_a_name_the_class_does_not_declare_is_refused() -> None:
    extra = "model One\n  Rig a(redeclare model Absent = Parts.Capacitor);\nend One;\n"
    with pytest.raises(FlattenError, match="has no class 'Absent' to redeclare"):
        _flat(extra, "One")


def test_a_package_constant_resolves_to_its_value() -> None:
    """`Media.Water.rho` -- a constant declared in a package -- is read as its value.

    A medium is mostly constants, so this is the reference such a model reaches for. A dotted
    reference whose head is not a local name is looked up, and a constant with a value that
    evaluates to a number here is substituted, so nothing below the flattener sees the dotted name.
    (A value that names another constant, or a function that is not a builtin, is still left alone.)
    """
    source = (
        "package Media\n  package Water\n    constant Real rho = 1000.0;\n"
        "  end Water;\nend Media;\n\n"
        "model Tank\n  Real mass;\nequation\n  mass = Media.Water.rho * 2.0;\nend Tank;\n"
    )
    assert (
        unparse_equation(flatten(parse(source), "Tank").equations[0])[0] == "mass = 1000.0 * 2.0;"
    )
