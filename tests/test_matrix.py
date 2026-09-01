"""``[1, 2; 3, 4]``: read, carried, and refused only where it would need a meaning.

The MSL measurement (docs/ideas/coverage.md) counts 66 example models whose first refusal
was a matrix expression -- almost always in a *dependency*, where one matrix in one file
kept the whole package from loading. The node follows :class:`ir.Array`'s design: it
survives parsing and flattening so the file that holds it loads, and the semantic layers
(evaluate, derive, emit) refuse it by name at the point of use.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.derive import DerivativeError, d_dx
from modelica.ir.evaluate import EvaluationError, evaluate
from modelica.ir.flatten import flatten
from modelica.lang import parse, parse_expression, unparse, unparse_expr

MATRIX = ir.Matrix(
    (
        (ir.IntLiteral(1), ir.IntLiteral(2)),
        (ir.IntLiteral(3), ir.IntLiteral(4)),
    ),
)


def test_a_matrix_is_read_and_round_trips() -> None:
    assert parse_expression("[1, 2; 3, 4]") == MATRIX
    assert unparse_expr(MATRIX) == "[1, 2; 3, 4]"
    assert parse_expression(unparse_expr(MATRIX)) == MATRIX


def test_a_single_row_needs_no_semicolon() -> None:
    assert parse_expression("[1, 2]") == ir.Matrix(((ir.IntLiteral(1), ir.IntLiteral(2)),))


def test_a_file_holding_a_matrix_loads_and_the_model_beside_it_flattens() -> None:
    source = """\
model Table
  parameter Real k[2, 2] = [1, 2; 3, 4];
end Table;
model M
  Real x;
equation
  x = 1;
end M;
"""
    program = parse(source)
    assert parse(unparse(program)) == program
    assert [v.name for v in flatten(program, "M").variables] == ["x"]


def test_using_a_matrix_is_refused_by_name() -> None:
    with pytest.raises(EvaluationError, match="matrix expressions are not supported yet"):
        evaluate(MATRIX, {})
    with pytest.raises(DerivativeError, match="matrix expressions are not supported yet"):
        d_dx(MATRIX, "x")
