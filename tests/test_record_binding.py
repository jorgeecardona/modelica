"""A whole-record binding ``a = b``: each of ``a``'s members set to ``b``'s.

The standard library's machines bind one parameter record to another --
``frictionParameters = dcpmData.frictionParameters`` -- rather than repeating every field, and the
flux-tube and fluid models write a constructor -- ``material = BaseData()`` for the defaults,
``R(a = 5)`` to set a field. A composite component given a value used to be refused outright; it is
now read as the member-wise binding it stands for. A *reference* copies each member (``dst.a`` gets
``src.a``, a ``final`` field comes out consistent because the fields it reads were copied too); a
*constructor* sets each named or positional argument and leaves the rest at the record's own default.
A constructor of a *different* record is a redeclaration, not a binding, and stays refused; so does a
value of any other shape.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse

LIB = (
    "package P\n"
    "  record R\n"
    "    parameter Real a = 1;\n"
    "    parameter Real w = 3;\n"
    "    parameter Real b = 2 * a;\n"
    "  end R;\n"
)


def test_a_whole_record_binding_copies_each_member() -> None:
    source = (
        LIB + "  model M\n"
        "    parameter R src(a = 5, w = 7);\n"
        "    parameter R dst = src;\n"
        "    Real x(start = 1);\n"
        "  equation\n"
        "    der(x) = -dst.a * x;\n"
        "  end M;\n"
        "end P;\n"
    )
    model = flatten([parse(source)], "P.M")
    bindings = {v.name: v.binding for v in model.variables}
    # Every member of dst reads the matching member of src, by the flat name src prefixes to.
    assert bindings["dst.a"] == ir.Ref(("src", "a"))
    assert bindings["dst.w"] == ir.Ref(("src", "w"))
    assert bindings["dst.b"] == ir.Ref(("src", "b"))
    # src keeps its own bindings; only dst is the copy.
    assert bindings["src.a"] == ir.IntLiteral(5)
    assert bindings["src.w"] == ir.IntLiteral(7)


def test_an_explicit_member_modification_wins_over_the_copy() -> None:
    """``dst(w = 9) = src`` takes 9 for ``w`` and ``src``'s value for the rest."""
    source = (
        LIB + "  model M\n"
        "    parameter R src(a = 5, w = 7);\n"
        "    parameter R dst(w = 9) = src;\n"
        "    Real x(start = 1);\n"
        "  equation\n"
        "    der(x) = -dst.a * x;\n"
        "  end M;\n"
        "end P;\n"
    )
    model = flatten([parse(source)], "P.M")
    bindings = {v.name: v.binding for v in model.variables}
    assert bindings["dst.w"] == ir.IntLiteral(9)
    assert bindings["dst.a"] == ir.Ref(("src", "a"))


def test_a_nested_record_binding_copies_through_the_levels() -> None:
    """A record member that is itself a record is copied by the same rule, one level down."""
    source = (
        "package P\n"
        "  record Sub\n    parameter Real p = 1;\n  end Sub;\n"
        "  record Outer\n    parameter Sub sub;\n    parameter Real q = 2;\n  end Outer;\n"
        "  model M\n"
        "    parameter Outer src(sub(p = 4), q = 8);\n"
        "    parameter Outer dst = src;\n"
        "    Real x(start = 1);\n"
        "  equation\n"
        "    der(x) = -dst.sub.p * x;\n"
        "  end M;\n"
        "end P;\n"
    )
    model = flatten([parse(source)], "P.M")
    bindings = {v.name: v.binding for v in model.variables}
    assert bindings["dst.q"] == ir.Ref(("src", "q"))
    # The nested record is copied whole, so its leaf reads the source leaf by the full flat name.
    assert bindings["dst.sub.p"] == ir.Ref(("src", "sub", "p"))


def test_a_constructor_binding_sets_each_named_field() -> None:
    """``dst = R(a = 5, w = 7)`` sets those fields; a field it does not name keeps its own default.

    ``b`` is the record's own ``2 * a``, so unlike the reference form it is not copied -- it recomputes
    from the copied ``a``, which is the cleaner reading.
    """
    source = (
        LIB + "  model M\n"
        "    parameter R dst = R(a = 5, w = 7);\n"
        "    Real x(start = 1);\n"
        "  equation\n"
        "    der(x) = -dst.a * x;\n"
        "  end M;\n"
        "end P;\n"
    )
    model = flatten([parse(source)], "P.M")
    bindings = {v.name: v.binding for v in model.variables}
    assert bindings["dst.a"] == ir.IntLiteral(5)
    assert bindings["dst.w"] == ir.IntLiteral(7)
    # b is not among the arguments, so it keeps the record's own formula, now over the flat name.
    assert bindings["dst.b"] == ir.Binary("*", ir.IntLiteral(2), ir.Ref(("dst", "a")))


def test_an_empty_constructor_takes_the_record_defaults() -> None:
    """Every member is the record's own default when no field is named.

    The fluid models write the cooling medium this way (``medium = Medium()``).
    """
    source = (
        LIB + "  model M\n"
        "    parameter R dst = R();\n"
        "    Real x(start = 1);\n"
        "  equation\n"
        "    der(x) = -dst.a * x;\n"
        "  end M;\n"
        "end P;\n"
    )
    model = flatten([parse(source)], "P.M")
    bindings = {v.name: v.binding for v in model.variables}
    assert bindings["dst.a"] == ir.IntLiteral(1)
    assert bindings["dst.w"] == ir.IntLiteral(3)


def test_a_constructor_matches_positional_arguments_in_order() -> None:
    """``dst = R(9, 10)`` matches its arguments to the members in declaration order."""
    source = (
        LIB + "  model M\n"
        "    parameter R dst = R(9, 10);\n"
        "    Real x(start = 1);\n"
        "  equation\n"
        "    der(x) = -dst.a * x;\n"
        "  end M;\n"
        "end P;\n"
    )
    model = flatten([parse(source)], "P.M")
    bindings = {v.name: v.binding for v in model.variables}
    assert bindings["dst.a"] == ir.IntLiteral(9)
    assert bindings["dst.w"] == ir.IntLiteral(10)


def test_a_constructor_of_a_different_type_is_refused() -> None:
    """A constructor of a different record is a redeclaration, refused as one.

    The flux-tube examples pick a specific magnetic material this way; changing the type is not a
    plain binding.
    """
    source = (
        LIB + "  record S\n    extends R;\n  end S;\n"
        "  model M\n"
        "    parameter R dst = S();\n"
        "    Real x(start = 1);\n"
        "  equation\n"
        "    der(x) = -dst.a * x;\n"
        "  end M;\n"
        "end P;\n"
    )
    with pytest.raises(FlattenError, match="a redeclaration, not supported yet"):
        flatten([parse(source)], "P.M")
