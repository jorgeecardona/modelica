"""``x[1:3]`` and ``A[:, 1]``: slices read where the grammar expects them.

The MSL measurement (docs/ideas/coverage.md) counted 60 example models whose first
refusal was a slice subscript. A range was already an expression the parser could read --
but only in the header of a ``for``; the subscript is the other place the grammar expects
one, and the bare ``:`` gets its own node (:class:`ir.Colon`) because it can only ever be
the whole subscript. Both are carried the way :class:`ir.Array` is, so the file that writes
one loads. A range subscript in an equation now flattens to the elements it selects; the
other uses (a bare colon, a slice handed to the evaluator) are still refused at the point of use.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.evaluate import EvaluationError, evaluate
from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import parse, parse_expression, unparse, unparse_equation, unparse_expr


def test_a_slice_subscript_is_read_and_round_trips() -> None:
    node = parse_expression("x[1:3]")
    assert node == ir.Subscript(ir.Ref(("x",)), (ir.Range(ir.IntLiteral(1), ir.IntLiteral(3)),))
    assert unparse_expr(node) == "x[1:3]"
    stepped = parse_expression("x[2:2:8]")
    assert stepped == ir.Subscript(
        ir.Ref(("x",)),
        (ir.Range(ir.IntLiteral(2), ir.IntLiteral(8), ir.IntLiteral(2)),),
    )


def test_a_bare_colon_is_the_whole_dimension() -> None:
    node = parse_expression("A[:, 1]")
    assert node == ir.Subscript(ir.Ref(("A",)), (ir.Colon(), ir.IntLiteral(1)))
    assert unparse_expr(node) == "A[:, 1]"


def test_using_a_slice_is_refused_with_the_stage_named() -> None:
    with pytest.raises(EvaluationError, match="not supported yet"):
        evaluate(parse_expression("x[1:3]"), {})


def test_a_sliced_equation_splits_into_one_equation_per_element() -> None:
    """``x[1:3] = {1, 2, 3}`` is three scalar equations: a range subscript is an array slice.

    A range in a subscript used to be refused (it read only in a ``for`` header); it now expands
    to the elements it selects, so the equation splits like any equation between two arrays.
    """
    source = "model M\n  Real x[3];\nequation\n  x[1:3] = {1, 2, 3};\nend M;\n"
    equations = [unparse_equation(e)[0].strip() for e in flatten(parse(source), "M").equations]
    assert equations == ["x[1] = 1;", "x[2] = 2;", "x[3] = 3;"]


def test_a_bare_colon_slice_is_still_refused() -> None:
    """A bare ``:`` is refused, not silently ignored.

    It needs the array's own extent, which the range handler does not read, so it stays refused.
    """
    source = "model M\n  Real x[3];\nequation\n  x[:] = {0, 0, 0};\nend M;\n"
    with pytest.raises(FlattenError):
        flatten(parse(source), "M")


def test_a_range_is_an_expression_wherever_one_is_expected() -> None:
    """``parameter Integer columns[:] = 2:size(table, 2)`` — Blocks/Tables.mo:33.

    The range tail reads in ``expression()`` itself (specification: simple_expression),
    so a binding, an argument, or a subscript all take one; the ``for`` header stopped
    being special the moment it could not be the only place.
    """
    node = parse_expression("2:size(table, 2)")
    assert isinstance(node, ir.Range)
    source = "model M\n  parameter Integer columns[:] = 2:size(table, 2);\nend M;\n"
    program = parse(source)
    assert parse(unparse(program)) == program


def test_a_comprehension_is_read_in_both_spellings() -> None:
    """``{x[i] for i in 1:n}`` and ``sum(x[i] for i in 1:n)`` are one node.

    The printer always writes the braces, and the braced form parses back to the same
    value, so the round trip holds structurally even where the text differs.
    """
    braced = parse_expression("{x[i] for i in 1:3}")
    assert isinstance(braced, ir.Comprehension)
    assert braced.iterators[0][0] == "i"
    reduction = parse_expression("sum(x[i] for i in 1:3)")
    assert reduction == ir.Call("sum", (braced,), ())
    assert parse_expression(unparse_expr(braced)) == braced
    assert parse_expression(unparse_expr(reduction)) == reduction


def test_a_comprehension_is_refused_at_the_point_of_use() -> None:
    with pytest.raises(EvaluationError, match="array comprehensions are not supported yet"):
        evaluate(parse_expression("{i for i in 1:3}"), {})
