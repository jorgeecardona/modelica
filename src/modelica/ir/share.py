"""Common subexpressions, named once instead of written out twice.

Forward-mode differentiation duplicates by construction -- every product rule re-emits both
factors -- so a Jacobian built that way repeats itself far more than the right-hand side it
came from. Measured over the example corpus, as trees against the directed acyclic graphs
they would collapse to: ``rhs`` is 22% larger than it needs to be and ``jacobian`` 33%, with
the worst model at 47%.

This pass finds those repeats and returns them. It does **not** rewrite anything in place:
:func:`share` hands back an ordered list of named subexpressions plus the roots with those
parts replaced by references, which is a value the caller emits however it likes. Nothing
here knows about the emitter, and the emitter is not the only possible caller.

**A subexpression inside a conditional branch is never named.** That is the one rule worth
reading twice. Naming it would hoist it out of the branch and evaluate it unconditionally,
and a branch is exactly where a model puts the arithmetic that is only valid on one side of
a test -- ``if h > 0 then sqrt(h) else 0`` is the shape, and the corpus has it. Hoisting the
``sqrt`` there computes the root of a negative number on every step of the run and lets the
NaN escape into a trajectory that was previously fine. The *condition* is always evaluated,
so subexpressions inside it are fair game; the arms are not.

The other deliberate limit: this is not an optimiser. Nothing is reassociated, nothing is
folded beyond what :func:`modelica.ir.derive.simplify` already did, and no term is
rearranged to expose a repeat that is not already there literally. Reassociation changes
floating-point results, and the example corpus is checked against reference points at 1e-10
and tighter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelica.ir import nodes as ir

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

__all__ = ["Shared", "share", "substituted"]

#: Node types worth naming. A reference or a literal is already one token, so naming it
#: makes the source longer rather than shorter.
_WORTH_NAMING = (ir.Binary, ir.Unary, ir.Call, ir.IfExpr)


@dataclass(frozen=True)
class Shared:
    """The result of :func:`share`: what to name, and the expressions that use the names."""

    definitions: tuple[tuple[str, ir.Expr], ...]
    """``(name, expression)`` in dependency order -- each may refer to the names before it."""

    roots: tuple[ir.Expr, ...]
    """The expressions handed in, with every named part replaced by a reference to its name."""

    names: tuple[tuple[ir.Expr, ir.Ref], ...] = ()
    """Each original subexpression and the reference that now stands for it.

    What a caller needs to rewrite something the pass never saw -- an expression built after
    the plan was made, which is the ordinary case for an emitter that discovers what it
    prints as it prints it.
    """

    def __bool__(self) -> bool:
        return bool(self.definitions)


def _children(node: ir.Expr) -> Iterator[ir.Expr]:
    """The subexpressions that are evaluated whenever ``node`` is.

    An ``if`` yields its condition and not its arms, which is what keeps a named
    subexpression out of a branch. See this module's docstring for why that matters.
    """
    match node:
        case ir.Binary(lhs=lhs, rhs=rhs):
            yield lhs
            yield rhs
        case ir.Unary(operand=operand):
            yield operand
        case ir.Call(args=args, named=named):
            yield from args
            yield from (value for _, value in named)
        case ir.IfExpr(condition=condition):
            yield condition
        case _:
            return


def _count(node: ir.Expr, counts: Counter[ir.Expr]) -> None:
    counts[node] += 1
    # Counted once however often it occurs: a repeat inside a repeat is named by naming the
    # outer one, and counting it again would order the two wrongly.
    if counts[node] == 1:
        for child in _children(node):
            _count(child, counts)


def substituted(node: ir.Expr, table: Mapping[ir.Expr, ir.Ref]) -> ir.Expr:
    """``node`` with every named subexpression replaced by its reference.

    Public because the plan is not the end of the story: a caller that keeps emitting after
    the names are fixed has to apply them to expressions the pass never saw, and doing that
    by hand is how the two would drift.
    """
    if node in table:
        return table[node]

    def go(inner: ir.Expr) -> ir.Expr:
        return substituted(inner, table)

    match node:
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, go(lhs), go(rhs))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, go(operand))
        case ir.Call(name=name, args=args, named=named):
            return ir.Call(name, tuple(go(a) for a in args), tuple((k, go(v)) for k, v in named))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            # The arms are left alone, because nothing inside them was named.
            return ir.IfExpr(go(condition), then, otherwise)
        case _:
            return node


def _depth(node: ir.Expr) -> int:
    children = list(_children(node))
    return 1 + max((_depth(c) for c in children), default=0)


def share(roots: Sequence[ir.Expr], prefix: str = "_s") -> Shared:
    """Name every subexpression that ``roots`` evaluates more than once.

    The definitions come back shallowest first, so each one refers only to names already
    defined and a caller can emit them in the order given. Ties are broken by how the
    expression prints, so the same input always produces the same output -- the generated
    source is compared byte for byte by the documentation build, and a set's iteration
    order would make it differ between runs.
    """
    counts: Counter[ir.Expr] = Counter()
    for root in roots:
        _count(root, counts)

    repeated = [
        node
        for node, seen in counts.items()
        if seen > 1 and isinstance(node, _WORTH_NAMING)
    ]
    repeated.sort(key=lambda node: (_depth(node), repr(node)))

    table: dict[ir.Expr, ir.Ref] = {}
    definitions: list[tuple[str, ir.Expr]] = []
    for position, node in enumerate(repeated):
        name = f"{prefix}{position}"
        definitions.append((name, substituted(node, table)))
        table[node] = ir.Ref((name,))
    return Shared(
        tuple(definitions),
        tuple(substituted(root, table) for root in roots),
        tuple(table.items()),
    )
