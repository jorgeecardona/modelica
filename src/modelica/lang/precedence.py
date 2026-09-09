"""Operator precedence, shared by the printer and the parser.

One table, imported by both directions, so they cannot drift apart. If the printer thinks
``-a * b`` means one thing and the parser thinks it means another, every round trip is a
silent lie; keeping the numbers in a single place makes that failure impossible rather
than merely unlikely.

Levels are Modelica 3.x, loosest first. ``if`` expressions bind loosest of all, which is
why an ``if`` used as an operand always comes back out parenthesised.

!!! note "One deliberate deviation"
    The specification's grammar puts the unary sign at the *additive* level
    (``arithmetic_expression: [add_op] term {add_op term}``), so ``-a * b`` parses as
    ``-(a * b)``. Here it sits between multiplication and exponentiation, so the same text
    parses as ``(-a) * b``. The two differ only in tree shape -- ``(-a) * b`` and
    ``-(a * b)`` are equal in IEEE arithmetic, signed zeros included -- and the tighter
    binding is what lets a tree survive a print/parse round trip unchanged. Exponentiation
    is unaffected: the sign stays looser than ``^``, so ``-a ^ b`` is ``-(a ^ b)`` as the
    specification requires.
"""

from __future__ import annotations

from typing import Final

__all__ = ["BINARY", "IF", "NON_ASSOCIATIVE", "PRIMARY", "UNARY_SIGN"]

IF: Final = 0

BINARY: Final[dict[str, int]] = {
    "or": 1,
    "and": 2,
    "not": 3,
    "==": 4,
    "<>": 4,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    # The dotted forms are Modelica's elementwise operators: `.+`/`.-` at the additive level,
    # `.*`/`./` at the multiplicative one, `.^` at the power level, each beside its plain sibling.
    ".+": 5,
    ".-": 5,
    "*": 6,
    "/": 6,
    ".*": 6,
    "./": 6,
    "^": 8,
    ".^": 8,
}

UNARY_SIGN: Final = 7
PRIMARY: Final = 9

# `^`/`.^` and the relational operators are non-associative: both operands print one level
# tighter, so `a ^ b ^ c` comes back out parenthesised rather than silently reparsed, and
# the parser refuses to chain them at all.
NON_ASSOCIATIVE: Final = frozenset({"^", ".^", "==", "<>", "<", "<=", ">", ">="})
