"""Numeric evaluation of an expression in an environment.

The environment is a flat mapping from names to values. Two synthetic name forms let the
same evaluator serve every stage without special cases at the call sites:

* ``der(x)`` -- the time derivative of ``x``, an unknown of its own during
  causalization and simulation;
* ``pre(x)`` -- the value ``x`` held just before the current event, supplied by the
  event handler while it applies a ``when`` body.

``evaluate`` never guesses: a name the environment does not answer raises
:exc:`EvaluationError` naming it, because a silent default of zero is how simulators
produce plausible garbage.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Final

from modelica.ir import nodes as ir

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["EvaluationError", "Value", "evaluate"]

type Value = float | int | bool | str


def at(where: ir.Span | None) -> str:
    """`` at Motor.mo:12:5``, or nothing at all when the value came from Python.

    A suffix rather than a prefix so that a message reads as a sentence with the location as
    an aside, and so that the message a caller already matches on is still its beginning.
    The empty case is not a degradation to apologise for: :mod:`modelica.build` builds the
    same IR from Python, where there is no line to point at, and an error there is no worse
    than it was.
    """
    return f" at {where}" if where is not None else ""


PAIR = 2
"""Arguments a two-argument built-in takes."""


class EvaluationError(Exception):
    """An expression could not be evaluated: an unknown name or function."""


_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    # Added when the C dialect landed: the emitter could translate these and the reference
    # tier could not evaluate them, so a model using `floor` had a compiled answer and no
    # interpreted one to check it against. `docs/ideas/backends.md:8` calls this tier the
    # reference, and a reference that knows fewer functions than the compiler is not one.
    "ceil": math.ceil,
    "floor": math.floor,
}

_PAIRS: dict[str, Callable[[float, float], float]] = {
    "atan2": math.atan2,
    "mod": lambda a, b: math.fmod(a, b) + b if math.fmod(a, b) and (a < 0) != (b < 0) else math.fmod(a, b),
    "rem": math.fmod,
}
"""The two-argument built-ins. ``mod`` floors and ``rem`` truncates, which is Modelica's
distinction and not Python's -- `math.fmod` is `rem`, and using it for both is the mistake
this table exists to avoid."""


# The numeric operators, as tables rather than a `match`. Both take two numbers and the
# dispatch carries no information, so spelling them out as branches would say nothing that
# the key does not already say.
_ARITHMETIC: Final[dict[str, Callable[[float, float], float]]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    # `math.pow` rather than `**`: a negative base with a fractional exponent is a domain
    # error, and `**` answers it with a complex number instead of raising.
    "^": math.pow,
}

_COMPARISON: Final[dict[str, Callable[[float, float], bool]]] = {  # `operator` types these
    "==": operator.eq,
    "<>": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _number(value: Value, origin: ir.Expr) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"expected a number, got {value!r} from {origin!r}")
    return float(value)


def _boolean(value: Value, origin: ir.Expr) -> bool:
    if not isinstance(value, bool):
        raise EvaluationError(f"expected a boolean, got {value!r} from {origin!r}")
    return value


def _lookup(name: str, env: Mapping[str, Value], where: ir.Span | None = None) -> Value:
    try:
        return env[name]
    except KeyError:
        raise EvaluationError(f"nothing in the environment answers {name!r}{at(where)}") from None


def _call(node: ir.Call, env: Mapping[str, Value]) -> Value:
    # `der`/`pre` are lookups of synthetic names, not computations: their values are
    # decided by causalization and event handling, never by the expression itself.
    if node.name in ("der", "pre"):
        (arg,) = node.args
        if not isinstance(arg, ir.Ref):
            raise EvaluationError(f"{node.name}() needs a variable, got {arg!r}")
        return _lookup(f"{node.name}({arg.name})", env, arg.span)
    if node.name == "noEvent":
        (arg,) = node.args
        return evaluate(arg, env)
    args = [_number(evaluate(a, env), a) for a in node.args]
    if node.name == "sign":
        (x,) = args
        return float((x > 0) - (x < 0))
    if node.name in ("min", "max"):
        return min(args) if node.name == "min" else max(args)
    if node.name in _FUNCTIONS and len(args) == 1:
        return _FUNCTIONS[node.name](args[0])
    if node.name in _PAIRS and len(args) == PAIR:
        return _PAIRS[node.name](args[0], args[1])
    raise EvaluationError(f"unknown function {node.name!r}")


def _binary(node: ir.Binary, env: Mapping[str, Value]) -> Value:
    if node.op in ("and", "or"):
        # Short-circuit, because Modelica guards depend on it: in `n <> 0 and 1 / n > 1`
        # the right operand is only well defined once the left one holds.
        left = _boolean(evaluate(node.lhs, env), node.lhs)
        if left is (node.op == "or"):
            return left
        return _boolean(evaluate(node.rhs, env), node.rhs)

    lhs = _number(evaluate(node.lhs, env), node.lhs)
    rhs = _number(evaluate(node.rhs, env), node.rhs)
    if node.op in _ARITHMETIC:
        return _ARITHMETIC[node.op](lhs, rhs)
    return _COMPARISON[node.op](lhs, rhs)


def _unary(node: ir.Unary, env: Mapping[str, Value]) -> Value:
    if node.op == "not":
        return not _boolean(evaluate(node.operand, env), node.operand)
    value = _number(evaluate(node.operand, env), node.operand)
    return -value if node.op == "-" else value


def evaluate(node: ir.Expr, env: Mapping[str, Value]) -> Value:
    """Evaluate ``node`` against ``env``. Fails loudly on names ``env`` cannot answer."""
    match node:
        case ir.Ref():
            return _lookup(node.name, env, node.span)
        case (
            ir.RealLiteral(value=value)
            | ir.IntLiteral(value=value)
            | ir.BoolLiteral(value=value)
            | ir.StringLiteral(value=value)
        ):
            return value
        case ir.Array() | ir.Subscript() | ir.Range():
            raise EvaluationError("array equations are not supported yet")
        case ir.Binary():
            return _binary(node, env)
        case ir.Unary():
            return _unary(node, env)
        case ir.Call():
            return _call(node, env)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            branch = then if _boolean(evaluate(condition, env), condition) else otherwise
            return evaluate(branch, env)
