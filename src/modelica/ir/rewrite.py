"""Walking and rewriting expressions: the three things every pass wants.

:func:`names` asks what an expression depends on -- the incidence information matching and
sorting are built from. :func:`substitute` replaces variables by expressions, which is what
alias elimination and symbolic solving both reduce to. :func:`solve_for` isolates one
variable, when the expression lets it.

They live here rather than in each pass because getting the traversal wrong is easy and
getting it wrong in three places is easier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelica.ir import nodes as ir
from modelica.ir.derive import ZERO, d_dx, div, neg, simplify, sub

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = ["contains", "names", "residual", "solve_for", "substitute"]


def residual(equation: ir.Equal) -> ir.Expr:
    """The equation as the expression that has to be zero."""
    return simplify(sub(equation.lhs, equation.rhs))


def names(node: ir.Expr) -> Iterator[str]:
    """Every variable name ``node`` reads, with repeats, in reading order."""
    match node:
        case ir.Ref():
            yield node.name
        case ir.Binary(lhs=lhs, rhs=rhs):
            yield from names(lhs)
            yield from names(rhs)
        case ir.Unary(operand=operand):
            yield from names(operand)
        case ir.Call(args=args):
            for arg in args:
                yield from names(arg)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            yield from names(condition)
            yield from names(then)
            yield from names(otherwise)
        case _:
            return


def contains(node: ir.Expr, name: str) -> bool:
    return any(found == name for found in names(node))


def substitute(node: ir.Expr, mapping: Mapping[str, ir.Expr]) -> ir.Expr:  # noqa: PLR0911 - one case per node type
    """Replace each variable named in ``mapping`` by its expression.

    One pass, not to a fixed point: a mapping that maps a name to an expression containing
    that same name would otherwise loop forever, and the callers here resolve their chains
    before substituting rather than relying on repetition to do it.
    """
    if not mapping:
        return node
    match node:
        case ir.Ref():
            return mapping.get(node.name, node)
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, substitute(lhs, mapping), substitute(rhs, mapping))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, substitute(operand, mapping))
        case ir.Call(name=name, args=args):
            return ir.Call(name, tuple(substitute(a, mapping) for a in args))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(
                substitute(condition, mapping),
                substitute(then, mapping),
                substitute(otherwise, mapping),
            )
        case _:
            return node


def solve_for(residual: ir.Expr, name: str) -> ir.Expr | None:
    """Solve ``residual = 0`` for ``name``, or return ``None`` if it does not come out.

    Only the linear case is attempted, which is the case that matters: a block the solver
    would otherwise have to iterate on becomes a single assignment. Linearity is decided by
    differentiating -- if ``d residual / d name`` still mentions ``name``, the residual was
    not linear in it -- and the answer is then the schoolbook ``-b/a`` with
    ``b = residual`` evaluated at ``name = 0``.
    """
    slope = simplify(d_dx(residual, name))
    if slope == ZERO or contains(slope, name):
        return None
    intercept = simplify(substitute(residual, {name: ZERO}))
    return simplify(neg(div(intercept, slope)))
