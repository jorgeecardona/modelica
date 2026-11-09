"""`ArrayEquation`: a ``for`` loop kept as one equation, and its scalar form."""

from __future__ import annotations

import pytest

from modelica.ir import nodes as ir
from modelica.ir.flat import ArrayEquation, CellError, IndexSet


def _ref(name: str) -> ir.Ref:
    return ir.Ref((name,))


def _sub(base: str, *indices: ir.Expr) -> ir.Subscript:
    return ir.Subscript(_ref(base), indices)


def _der(x: ir.Expr) -> ir.Call:
    return ir.Call("der", (x,), ())


def test_a_periodic_stencil_scalarizes_like_the_unrolled_loop() -> None:
    """``der(x[i]) = x[mod(i - 2, 3) + 1] - x[i]`` over ``1:3``, cell by cell, in loop order."""
    i = _ref("i")
    left = ir.Binary(
        "+",
        ir.Call("mod", (ir.Binary("-", i, ir.IntLiteral(2)), ir.IntLiteral(3)), ()),
        ir.IntLiteral(1),
    )
    equation = ArrayEquation(
        (IndexSet("i", 1, 3),),
        ir.Equal(_der(_sub("x", i)), ir.Binary("-", _sub("x", left), _sub("x", i))),
        (("x", (3,)),),
    )
    expected = tuple(
        ir.Equal(
            _der(_ref(f"x[{k}]")), ir.Binary("-", _ref(f"x[{(k - 2) % 3 + 1}]"), _ref(f"x[{k}]"))
        )
        for k in (1, 2, 3)
    )
    assert equation.scalarize() == expected


def test_two_indices_run_outer_first_and_a_value_use_becomes_a_literal() -> None:
    """``for i, for j``: ``j`` varies fastest; ``i`` read as a value is an integer literal."""
    i, j = _ref("i"), _ref("j")
    equation = ArrayEquation(
        (IndexSet("i", 1, 2), IndexSet("j", 1, 2)),
        ir.Equal(_sub("T", i, j), ir.Binary("*", i, _ref("a"))),
        (("T", (2, 2)),),
    )
    names = [e.lhs for e in equation.scalarize()]
    assert names == [_ref("T[1, 1]"), _ref("T[1, 2]"), _ref("T[2, 1]"), _ref("T[2, 2]")]
    assert equation.scalarize()[2].rhs == ir.Binary("*", ir.IntLiteral(2), _ref("a"))


def test_a_read_past_the_last_cell_is_an_error_not_a_wrap() -> None:
    """``x[i + 1]`` over ``1:3`` reads ``x[4]`` at ``i = 3``."""
    i = _ref("i")
    equation = ArrayEquation(
        (IndexSet("i", 1, 3),),
        ir.Equal(_der(_sub("x", i)), _sub("x", ir.Binary("+", i, ir.IntLiteral(1)))),
        (("x", (3,)),),
    )
    with pytest.raises(CellError, match=r"x\[4\] is outside \(3,\) at i = 3"):
        equation.scalarize()
