"""A whole-record binding ``a = b``: each of ``a``'s members set to ``b``'s.

The standard library's machines bind one parameter record to another --
``frictionParameters = dcpmData.frictionParameters`` -- rather than repeating every field. A
composite component given a value used to be refused outright; it is now read as the member-wise
binding it stands for, so ``dst.a`` gets ``src.a``, ``dst.b`` gets ``src.b``, and a field derived
from the others (a ``final`` in the record) comes out consistent because the fields it reads were
copied too. Only a reference to another record is expanded; a value of any other shape has no name
to read the members from and stays refused.
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


def test_a_composite_bound_to_a_constructor_is_refused_by_name() -> None:
    """A record constructor is not a name to read the members from, so it stays refused."""
    source = (
        LIB + "  model M\n"
        "    parameter R dst = R(a = 5);\n"
        "    Real x(start = 1);\n"
        "  equation\n"
        "    der(x) = -dst.a * x;\n"
        "  end M;\n"
        "end P;\n"
    )
    with pytest.raises(FlattenError, match="can only be bound to another by name"):
        flatten([parse(source)], "P.M")
