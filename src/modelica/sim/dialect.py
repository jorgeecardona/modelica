"""How an expression is spelled, so the emitter can be pointed at a second language.

`modelica.sim.emit` says the design in its own first paragraph: "The same emitter, pointed
at a different set of templates, is what will produce C mode." This module is that set of
templates -- the part of code generation that is *dialect*, separated from the part that is
*decision*.

The split is worth stating, because the value of a second backend depends on it. What
belongs here: the spelling of an operator, the name of a library function, how a
conditional is written, what a boolean literal looks like. What deliberately does not: which
variable a block solves for, what order the blocks run in, which relations are latched,
where a Jacobian's rows come from. Those are the same in every language and they are decided
once, in `emit`, or the second backend is a second compiler rather than a second printer.

Two dialects live here. :data:`PYTHON` is what the emitter has always produced, written out
as data so that it is comparable with the other one; :data:`C` is the new one. Anything the
two disagree about is visible in one place, which is the point -- a difference that is not in
this file is a difference the emitter had to know about, and that is exactly what a template
was supposed to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["PYTHON", "C", "Dialect"]

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


#: Written into every C unit before anything else. Three of Modelica's built-ins are not
#: C's: ``mod`` rounds toward minus infinity where ``fmod`` truncates, and ``min``/``max``
#: are not functions in C at all. Emitting them as helpers rather than as inline expressions
#: keeps the generated arithmetic readable, which `docs/ideas/backends.md:8` makes the whole
#: argument for owning the compiler.
#:
#: ``static inline`` and not ``static``, and that is not style. A generated unit carries every
#: helper whether or not this particular model uses one, and a plain ``static`` function that
#: nothing calls is an error under ``-Werror=unused-function`` -- so a model with no ``mod``
#: in it would fail to compile because of a helper written for a model that has one. The same
#: applies to ``md_solve``, which most models never call.
C_PREAMBLE: Final = (
    "#include <math.h>",
    "",
    "static inline double md_mod(double a, double b) {",
    "    double r = fmod(a, b);",
    "    return (r != 0.0 && (r < 0.0) != (b < 0.0)) ? r + b : r;",
    "}",
    "",
    "static inline double md_min(double a, double b) { return a < b ? a : b; }",
    "static inline double md_max(double a, double b) { return a > b ? a : b; }",
    "static inline double md_sign(double a) { return (double)((a > 0.0) - (a < 0.0)); }",
    "",
    "/* Dense LU with partial pivoting: the linear blocks are two- and three-by-two, and a",
    "   BLAS dependency for that would be a build system in exchange for nothing. */",
    "static inline void md_solve(int n, double *a, double *b, double *x) {",
    "    for (int k = 0; k < n; ++k) {",
    "        int pivot = k;",
    "        for (int i = k + 1; i < n; ++i) {",
    "            if (fabs(a[i * n + k]) > fabs(a[pivot * n + k])) { pivot = i; }",
    "        }",
    "        if (pivot != k) {",
    "            for (int j = 0; j < n; ++j) {",
    "                double swap = a[k * n + j];",
    "                a[k * n + j] = a[pivot * n + j];",
    "                a[pivot * n + j] = swap;",
    "            }",
    "            double swap = b[k]; b[k] = b[pivot]; b[pivot] = swap;",
    "        }",
    "        for (int i = k + 1; i < n; ++i) {",
    "            double factor = a[i * n + k] / a[k * n + k];",
    "            a[i * n + k] = 0.0;",
    "            for (int j = k + 1; j < n; ++j) { a[i * n + j] -= factor * a[k * n + j]; }",
    "            b[i] -= factor * b[k];",
    "        }",
    "    }",
    "    for (int i = n - 1; i >= 0; --i) {",
    "        double total = b[i];",
    "        for (int j = i + 1; j < n; ++j) { total -= a[i * n + j] * x[j]; }",
    "        x[i] = total / a[i * n + i];",
    "    }",
    "}",
    "",
)


C: Final = Dialect(
    name="c",
    binary={
        # The levels are C's, and they differ from Python's where it matters: `&&` binds
        # tighter than `||`, and equality binds looser than the orderings. Taking Python's
        # table would emit valid C that means something else.
        "or": ("||", 1),
        "and": ("&&", 2),
        "==": ("==", 3),
        "<>": ("!=", 3),
        "<": ("<", 4),
        "<=": ("<=", 4),
        ">": (">", 4),
        ">=": (">=", 4),
        "+": ("+", 5),
        "-": ("-", 5),
        "*": ("*", 6),
        "/": ("/", 6),
        "^": ("", 8),  # never reached: `power` below turns it into a call
    },
    functions={
        "abs": "fabs",
        "acos": "acos",
        "asin": "asin",
        "atan": "atan",
        "atan2": "atan2",
        "ceil": "ceil",
        "cos": "cos",
        "cosh": "cosh",
        "exp": "exp",
        "floor": "floor",
        "log": "log",
        "log10": "log10",
        "max": "md_max",
        "min": "md_min",
        "mod": "md_mod",
        "rem": "fmod",
        "sign": "md_sign",
        "sin": "sin",
        "sinh": "sinh",
        "sqrt": "sqrt",
        "tan": "tan",
        "tanh": "tanh",
    },
    true="1",
    false="0",
    negation="!",
    conditional=lambda condition, then, otherwise: f"{condition} ? {then} : {otherwise}",
    negation_level=SIGN,
    integer=lambda value: f"{value}.0",
    power=lambda base, exponent: f"pow({base}, {exponent})",
    truncate="trunc",
    square=lambda name, size: f"double {name}[{size * size}] = {{0}}",
    column=lambda name, size: f"double {name}[{size}] = {{0}}",
    entry=lambda name, row, col, size: f"{name}[{row * size + col}]",
    linear_solve=lambda a, b, x, size: (
        f"double {x}[{size}] = {{0}}; md_solve({size}, {a}, {b}, {x})"
    ),
    preamble=C_PREAMBLE,
    clarify={"or": frozenset({"and"})},
)
