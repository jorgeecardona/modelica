"""``x[1:3]`` and ``A[:, 1]``: slices read where the grammar expects them.

The MSL measurement (docs/ideas/coverage.md) counted 60 example models whose first
refusal was a slice subscript. A range was already an expression the parser could read --
but only in the header of a ``for``; the subscript is the other place the grammar expects
one, and the bare ``:`` gets its own node (:class:`ir.Colon`) because it can only ever be
the whole subscript. Both are carried the way :class:`ir.Array` is, so the file that
writes one loads, and the semantic layers refuse them beside it at the point of use.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.evaluate import EvaluationError, evaluate
from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import parse, parse_expression, unparse, unparse_expr


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


def test_a_sliced_equation_is_refused_by_name() -> None:
    source = "model M\n  Real x[3];\nequation\n  x[1:3] = {0, 0, 0};\nend M;\n"
    with pytest.raises(FlattenError, match="a range is read only in the header of a 'for'"):
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
