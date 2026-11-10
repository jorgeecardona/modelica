"""How an expression is spelled in Python, kept apart from what `modelica.sim.emit` decides.

This module holds the spelling: operators, library function names, the conditional form and
the boolean literals. `emit` holds the decisions: which variable a block solves for, the block
order, the latched relations. `PYTHON` is the only dialect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["PYTHON", "Dialect"]

CONDITIONAL: Final = 0
"""Binds loosest. A conditional in an operand always takes brackets."""

NOT: Final = 3
SIGN: Final = 7
PRIMARY: Final = 9
"""Binds tightest: a name, a literal, a call. Never bracketed."""

NON_ASSOCIATIVE: Final = frozenset({"^", "==", "<>", "<", "<=", ">", ">="})
"""Operators whose left operand takes brackets at equal precedence.

``a - b - c`` is fine and ``a ^ b ^ c`` is not, which is a fact about the operators rather
than about either language, so it lives beside the levels and not in a dialect.
"""


def _numpy_entry(name: str, row: int, column: int, size: int) -> str:
    """A numpy matrix is indexed by a pair, so its row length is not needed to find an entry.

    C's is: a flat block indexed by ``row * n + column``, which is why ``size`` is in the
    signature at all.
    """
    del size
    return f"{name}[{row}, {column}]"


def _numpy_solve(a: str, b: str, x: str, size: int) -> str:
    """``np.linalg.solve`` takes the shape from the arrays, so the size is not needed here.

    A named function rather than a lambda that ignores an argument, because a reader seeing
    an unused parameter should be told why once rather than wonder every time.
    """
    del size
    return f"{x} = np.linalg.solve({a}, {b})"


@dataclass(frozen=True)
class Dialect:
    """One target language, as the handful of things code generation has to know about it."""

    name: str
    binary: Mapping[str, tuple[str, int]]
    """Modelica's operator to this language's spelling and its precedence level."""
    functions: Mapping[str, str]
    """Modelica's built-in to the name that computes it here."""
    true: str
    false: str
    negation: str
    """How ``not`` is written, trailing space included when the language wants one."""
    conditional: Callable[[str, str, str], str]
    """``condition, then, otherwise`` to the language's conditional expression."""
    negation_level: int = NOT
    """How tightly negation binds, which the two languages disagree about.

    Python's ``not`` binds *looser* than a comparison, so ``not a > 0`` already means
    ``not (a > 0)``. C's ``!`` binds tighter than everything but a primary, so the same
    output means ``(!a) > 0`` -- a different expression that compiles, runs, and is wrong.
    """
    power: Callable[[str, str], str] | None = None
    """``^`` when it is a call rather than an operator. ``None`` to use :attr:`binary`."""
    truncate: str = "np.trunc"
    """The function Modelica's integer division rounds with."""
    integer: Callable[[int], str] = str
    """How a whole number is written, which is not a formatting question in C.

    Modelica's ``/`` is always real division. C's ``/`` on two ints is integer division, so
    ``burner / 100`` with a literal ``100`` and an int-valued expression computes zero and
    compiles without a word. Every value in the generated workspace is a ``double`` anyway,
    so the C dialect writes whole numbers with a decimal point and the question does not
    arise. Found by `RoastingMachine`, whose gas schedule is a chain of whole percentages
    divided by a hundred, and which came out 23% wrong rather than failing.
    """
    preamble: tuple[str, ...] = ()
    """Lines a translation unit needs before anything this dialect emits will compile."""
    square: Callable[[str, int], str] = lambda name, size: f"{name} = np.zeros(({size}, {size}))"
    """An ``n``-by-``n`` matrix of zeros, ready to be filled entry by entry."""
    column: Callable[[str, int], str] = lambda name, size: f"{name} = np.empty({size})"
    """An ``n``-vector, every entry of which is written before it is read."""
    entry: Callable[[str, int, int, int], str] = _numpy_entry
    """Where one entry of a square matrix is, which is the two languages' real disagreement.

    Python indexes a matrix with a pair and C indexes a flat block with arithmetic, so the
    size has to reach here -- which is why it is a parameter rather than something the
    caller formats.
    """
    linear_solve: Callable[[str, str, str, int], str] = _numpy_solve
    """Solve ``A x = b``. One line, because both languages can say it in one."""
    clarify: Mapping[str, frozenset[str]] = field(default_factory=dict[str, "frozenset[str]"])
    """Operand operators this language wants bracketed even where precedence settles it.

    C's ``&&`` binds tighter than ``||``, so ``a && b || c`` is unambiguous -- and ``-Wall``
    warns about it anyway, which under ``-Werror`` means the generated unit does not build.
    A warning in generated code is a defect in the generator, so the brackets go in rather
    than the flag coming out.
    """


PYTHON: Final = Dialect(
    name="python",
    binary={
        "or": ("or", 1),
        "and": ("and", 2),
        "==": ("==", 4),
        "<>": ("!=", 4),
        "<": ("<", 4),
        "<=": ("<=", 4),
        ">": (">", 4),
        ">=": (">=", 4),
        "+": ("+", 5),
        "-": ("-", 5),
        "*": ("*", 6),
        "/": ("/", 6),
        "^": ("**", 8),
    },
    functions={
        "abs": "abs",
        "acos": "np.arccos",
        "asin": "np.arcsin",
        "atan": "np.arctan",
        "atan2": "np.arctan2",
        "ceil": "np.ceil",
        "cos": "np.cos",
        "cosh": "np.cosh",
        "exp": "np.exp",
        "floor": "np.floor",
        "integer": "np.floor",
        "log": "np.log",
        "log10": "np.log10",
        "max": "max",
        "min": "min",
        "mod": "np.mod",
        "rem": "np.fmod",
        "sign": "np.sign",
        "sin": "np.sin",
        "sinh": "np.sinh",
        "sqrt": "np.sqrt",
        "tan": "np.tan",
        "tanh": "np.tanh",
    },
    true="True",
    false="False",
    negation="not ",
    conditional=lambda condition, then, otherwise: f"{then} if {condition} else {otherwise}",
)
