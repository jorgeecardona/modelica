"""Symbolic differentiation, and the folding that keeps its output readable.

Two derivatives matter downstream and they differ only in what a leaf is worth:

* :func:`d_dt` -- with respect to time. Index reduction differentiates equations until a
  structurally singular system becomes solvable, so this is what makes the pendulum run.
* :func:`d_dx` -- with respect to one variable. This is the Jacobian, which every
  implicit solver wants and every algebraic loop needs.

Higher time derivatives are named, not nested: :func:`d_dt` of ``der(x)`` is
``der(der_x)``, where ``der_x`` is an ordinary variable the caller is expected to declare.
That is the same trick :mod:`modelica.ir.evaluate` uses for ``der``/``pre`` -- keep the IR
closed under differentiation by spelling new unknowns as names, rather than growing new
node types that every later pass would have to learn.

Everything is folded on the way out. Not for speed: an unfolded derivative of a chain of
products is unreadable, and unreadable output is unreviewable output. Folding is done by
the constructors themselves, so no pass can forget to call it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelica.ir import nodes as ir

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterator

__all__ = ["DerivativeError", "d_dt", "d_dx", "der_name", "simplify"]

ZERO = ir.RealLiteral(0.0)
ONE = ir.RealLiteral(1.0)


class DerivativeError(Exception):
    """An expression has no derivative: a comparison, or an unknown function."""


def der_name(name: str) -> str:
    """The variable standing for ``der(name)``: ``der_x``, ``der_body.v``.

    Flattened names already contain dots, so the prefix goes on the front and stays out of
    the way of the hierarchy the dots describe.
    """
    return f"der_{name}"


# --------------------------------------------------------------------------------------
# Folding constructors -- the only way this module builds a node
# --------------------------------------------------------------------------------------


def _number(node: ir.Expr) -> float | None:
    """The value of ``node`` if it is a numeric literal, else ``None``."""
    match node:
        case ir.RealLiteral(value=value):
            return value
        case ir.IntLiteral(value=value):
            return float(value)
        case _:
            return None


def _is(node: ir.Expr, value: float) -> bool:
    return _number(node) == value


def _refs(node: ir.Expr) -> Iterator[ir.Ref]:  # noqa: C901 -- one case per node type
    """Every variable reference in ``node``, for asking a subtree whether the target is even in it."""
    match node:
        case ir.Ref():
            yield node
        case ir.Binary(lhs=lhs, rhs=rhs):
            yield from _refs(lhs)
            yield from _refs(rhs)
        case ir.Unary(operand=operand):
            yield from _refs(operand)
        case ir.Call(args=args):
            for argument in args:
                yield from _refs(argument)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            yield from _refs(condition)
            yield from _refs(then)
            yield from _refs(otherwise)
        case ir.Subscript(base=base, indices=indices):
            yield from _refs(base)
            for index in indices:
                yield from _refs(index)
        case ir.Array(elements=elements):
            for element in elements:
                yield from _refs(element)
        case _:
            return  # a literal, a range, a matrix: no scalar variable reference to reach here


def _active(node: ir.Expr, leaf: Callable[[ir.Ref], ir.Expr]) -> bool:
    """Whether ``node``'s derivative under ``leaf`` is not identically zero.

    A relation or a boolean has no derivative rule, but its *value* only changes with the variables
    inside it. Asked for a Jacobian entry with respect to a variable that does not appear there, or a
    time derivative of a relation on constants, the derivative is zero and there is nothing to
    refuse. So the refusal is kept for the case that matters -- the variable being solved for actually
    sitting inside the relation -- and everything else it does not touch differentiates to zero.
    """
    return any(not _is(leaf(ref), 0) for ref in _refs(node))


def neg(node: ir.Expr) -> ir.Expr:
    number = _number(node)
    if number is not None:
        return ir.RealLiteral(-number)
    if isinstance(node, ir.Unary) and node.op == "-":
        return node.operand
    return ir.Unary("-", node)


def add(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
    if _is(lhs, 0):
        return rhs
    if _is(rhs, 0):
        return lhs
    a, b = _number(lhs), _number(rhs)
    if a is not None and b is not None:
        return ir.RealLiteral(a + b)
    if isinstance(rhs, ir.Unary) and rhs.op == "-":
        return sub(lhs, rhs.operand)
    return ir.Binary("+", lhs, rhs)


def sub(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
    if _is(rhs, 0):
        return lhs
    if _is(lhs, 0):
        return neg(rhs)
    a, b = _number(lhs), _number(rhs)
    if a is not None and b is not None:
        return ir.RealLiteral(a - b)
    return ir.Binary("-", lhs, rhs)


def _scale(factor: float, node: ir.Expr) -> ir.Expr:
    """A number times something that is not a number."""
    if factor == 0:
        return ZERO
    if factor == 1:
        return node
    if factor == -1:
        return neg(node)
    return ir.Binary("*", ir.RealLiteral(factor), node)


def mul(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
    # A numeric factor is moved to the front, so that repeated folding collapses
    # `2 * (3 * x)` rather than leaving the constants stranded on opposite sides.
    a, b = _number(lhs), _number(rhs)
    if a is not None and b is not None:
        return ir.RealLiteral(a * b)
    if a is not None:
        return _scale(a, rhs)
    if b is not None:
        return _scale(b, lhs)
    return ir.Binary("*", lhs, rhs)


def div(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
    if _is(lhs, 0):
        return ZERO
    if _is(rhs, 1):
        return lhs
    a, b = _number(lhs), _number(rhs)
    if a is not None and b is not None and b != 0:
        return ir.RealLiteral(a / b)
    return ir.Binary("/", lhs, rhs)


def pow_(base: ir.Expr, exponent: ir.Expr) -> ir.Expr:
    if _is(exponent, 0):
        return ONE
    if _is(exponent, 1):
        return base
    a, b = _number(base), _number(exponent)
    if a is not None and b is not None:
        return ir.RealLiteral(a**b)
    return ir.Binary("^", base, exponent)


_FOLD: dict[ir.BinaryOp, Callable[[ir.Expr, ir.Expr], ir.Expr]] = {
    "+": add,
    "-": sub,
    "*": mul,
    "/": div,
    "^": pow_,
    # A dotted operator that survives to here is on scalars (arrays are read out elementwise before
    # a Jacobian is taken), so it differentiates as its plain sibling.
    ".*": mul,
    "./": div,
    ".^": pow_,
}


def _simplify_unary(node: ir.Unary) -> ir.Expr:
    operand = simplify(node.operand)
    match node.op:
        case "-":
            return neg(operand)
        case "+":
            return operand
        case _:
            return ir.Unary(node.op, operand)


def simplify(node: ir.Expr) -> ir.Expr:
    """Rebuild ``node`` through the folding constructors, bottom up."""
    match node:
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            fold = _FOLD.get(op)
            left, right = simplify(lhs), simplify(rhs)
            return fold(left, right) if fold else ir.Binary(op, left, right)
        case ir.Unary():
            return _simplify_unary(node)
        case ir.Call(name=name, args=args):
            return ir.Call(name, tuple(simplify(a) for a in args))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            chosen, other = simplify(then), simplify(otherwise)
            if chosen == other:
                # A test whose two answers are the same is not a test. This one fold is
                # what keeps a *schedule* differentiable in practice: a chain of `n`
                # constant segments differentiates to `0.0 if c1 else 0.0 if c2 else ...`,
                # which is `0.0` and nothing else, and every later stage pays for it if it
                # is left standing.
                return chosen
            return ir.IfExpr(simplify(condition), chosen, other)
        case _:
            return node


# --------------------------------------------------------------------------------------
# Differentiation
# --------------------------------------------------------------------------------------

# d/du f(u), as a function of the already-differentiated argument's base. The chain rule
# multiplies each by du, which the caller supplies.
_CHAIN: dict[str, Callable[[ir.Expr], ir.Expr]] = {
    "sqrt": lambda u: div(ONE, mul(ir.RealLiteral(2.0), ir.Call("sqrt", (u,)))),
    "sin": lambda u: ir.Call("cos", (u,)),
    "cos": lambda u: neg(ir.Call("sin", (u,))),
    "tan": lambda u: add(ONE, pow_(ir.Call("tan", (u,)), ir.RealLiteral(2.0))),
    "exp": lambda u: ir.Call("exp", (u,)),
    "log": lambda u: div(ONE, u),
    "log10": lambda u: div(ONE, mul(u, ir.RealLiteral(2.302585092994046))),
    "sinh": lambda u: ir.Call("cosh", (u,)),
    "cosh": lambda u: ir.Call("sinh", (u,)),
    "tanh": lambda u: sub(ONE, pow_(ir.Call("tanh", (u,)), ir.RealLiteral(2.0))),
    "asin": lambda u: div(ONE, ir.Call("sqrt", (sub(ONE, mul(u, u)),))),
    "acos": lambda u: neg(div(ONE, ir.Call("sqrt", (sub(ONE, mul(u, u)),)))),
    "atan": lambda u: div(ONE, add(ONE, mul(u, u))),
}


def _multiary(node: ir.Call, leaf: Callable[[ir.Ref], ir.Expr]) -> ir.Expr | None:
    """Derivative rules for the functions that are not a single-argument chain.

    ``floor``/``ceil``/``integer``/``div`` (piecewise constant), ``min``/``max`` (the active
    argument), ``atan2``, and ``mod``/``rem`` (numerator minus an integer factor). Returns ``None``
    when the call is none of these, so the single-argument table gets its turn.
    """
    if node.name in ("floor", "ceil", "integer", "div"):
        # Piecewise constant: zero derivative between its steps, which is where it is smooth.
        # `integer(x)` is `floor(x)` typed Integer; `div(a, b)` is the truncated quotient, also an
        # integer, constant in both arguments.
        return ZERO
    if len(node.args) != 2:  # noqa: PLR2004 - the rules below are all the two-argument forms
        return None
    a, b = node.args
    if node.name in ("min", "max"):
        # The derivative of whichever argument is active -- `min` the smaller, `max` the larger.
        # At the crossing the two meet, the kink the solver steps over the way it does `abs`.
        picks_first = ir.Binary("<", a, b) if node.name == "min" else ir.Binary(">", a, b)
        return ir.IfExpr(picks_first, _derive(a, leaf), _derive(b, leaf))
    if node.name == "atan2":
        # d atan2(y, x) = (x dy - y dx) / (x^2 + y^2).
        return div(
            sub(mul(b, _derive(a, leaf)), mul(a, _derive(b, leaf))), add(mul(b, b), mul(a, a))
        )
    if node.name in ("mod", "rem"):
        # mod(a, b) = a - floor(a/b)*b and rem(a, b) = a - div(a, b)*b; the integer factor is
        # constant between its steps, so d = da - factor*db.
        factor = ir.Call("floor", (div(a, b),)) if node.name == "mod" else ir.Call("div", (a, b))
        return sub(_derive(a, leaf), mul(factor, _derive(b, leaf)))
    return None


def _call(node: ir.Call, leaf: Callable[[ir.Ref], ir.Expr]) -> ir.Expr:  # noqa: PLR0911 -- one rule per operator family
    if node.name in ("der", "pre"):
        # `der(x)` differentiates to the derivative of the *variable* `der_x`, so that
        # higher derivatives stay ordinary unknowns. `pre(x)` is constant between events.
        (arg,) = node.args
        if not isinstance(arg, ir.Ref):
            raise DerivativeError(f"{node.name}() needs a variable, got {arg!r}")
        return ZERO if node.name == "pre" else leaf(ir.Ref((der_name(arg.name),)))
    if node.name in ("noEvent", "smooth"):
        # Transparent wrappers: the derivative keeps the wrapper (and smooth's leading continuity
        # order) around the derivative of the expression it carries -- the sole argument for
        # `noEvent`, the last for `smooth(order, expr)`. Emit strips the wrapper from generated code.
        *rest, arg = node.args
        return ir.Call(node.name, (*rest, _derive(arg, leaf)))
    if node.name == "homotopy":
        # homotopy(actual, simplified) is the actual once initialisation is done, so its derivative
        # is the derivative of the actual -- the first argument -- and no wrapper is kept, since emit
        # strips homotopy from the generated code entirely (it is not a run-time operator).
        return _derive(node.args[0], leaf)
    if node.name in ("abs", "sign"):
        # Differentiable away from zero, which is where an event indicator should have
        # split the trajectory anyway; at zero the derivative is the solver's problem.
        (arg,) = node.args
        if node.name == "sign":
            return ZERO
        return mul(ir.Call("sign", (arg,)), _derive(arg, leaf))
    special = _multiary(node, leaf)
    if special is not None:
        return special
    name = node.name
    # `Modelica.Math.sin` is the builtin sin under its library path; the chain rule is keyed by the
    # simple name. Only the last segment is dropped, and only when it is a rule the chain has, the
    # same reading the emitter gives a qualified builtin call.
    if name not in _CHAIN and "." in name and name.rsplit(".", 1)[-1] in _CHAIN:
        name = name.rsplit(".", 1)[-1]
    rule = _CHAIN.get(name)
    if rule is None:
        raise DerivativeError(f"no derivative rule for {node.name!r}")
    (arg,) = node.args
    return mul(rule(arg), _derive(arg, leaf))


def _binary(node: ir.Binary, leaf: Callable[[ir.Ref], ir.Expr]) -> ir.Expr:
    lhs, rhs = node.lhs, node.rhs
    match node.op:
        case "+":
            return add(_derive(lhs, leaf), _derive(rhs, leaf))
        case "-":
            return sub(_derive(lhs, leaf), _derive(rhs, leaf))
        case "*":
            return add(mul(_derive(lhs, leaf), rhs), mul(lhs, _derive(rhs, leaf)))
        case "/":
            # (f/g)' = (f'g - fg')/g^2
            numerator = sub(mul(_derive(lhs, leaf), rhs), mul(lhs, _derive(rhs, leaf)))
            return div(numerator, pow_(rhs, ir.RealLiteral(2.0)))
        case "^":
            return _power(lhs, rhs, leaf)
        case _:
            # A relation or a logical operator (`>`, `or`) is not differentiable, but its value only
            # turns on the variables inside it: a Jacobian entry for a variable that does not appear
            # there is zero, not a refusal. This is why a thyristor's firing logic, `off or fire`, no
            # longer stops the rectifier it sits in from causalizing.
            if _active(node, leaf):
                raise DerivativeError(f"a {node.op!r} expression has no derivative")
            return ZERO


def _power(base: ir.Expr, exponent: ir.Expr, leaf: Callable[[ir.Ref], ir.Expr]) -> ir.Expr:
    d_exponent = _derive(exponent, leaf)
    d_base = _derive(base, leaf)
    # The common case by far -- a constant exponent -- keeps the plain power rule rather
    # than the general logarithmic form, which would put a log() in every Jacobian.
    if _is(d_exponent, 0):
        return mul(mul(exponent, pow_(base, sub(exponent, ONE))), d_base)
    general = add(mul(d_exponent, ir.Call("log", (base,))), div(mul(exponent, d_base), base))
    return mul(pow_(base, exponent), general)


def _derive(  # noqa: C901, PLR0911 -- one case per node type
    node: ir.Expr, leaf: Callable[[ir.Ref], ir.Expr]
) -> ir.Expr:
    match node:
        case ir.Ref():
            return leaf(node)
        case ir.RealLiteral() | ir.IntLiteral() | ir.BoolLiteral() | ir.StringLiteral():
            return ZERO
        case ir.Array() | ir.Subscript() | ir.Range() | ir.Colon() | ir.End():
            raise DerivativeError("array equations are not supported yet")
        case ir.Matrix():
            raise DerivativeError("matrix expressions are not supported yet")
        case ir.Comprehension():
            raise DerivativeError("array comprehensions are not supported yet")
        case ir.Binary():
            return _binary(node, leaf)
        case ir.Unary(op=op, operand=operand):
            if op == "not":
                if _active(node, leaf):
                    raise DerivativeError("a boolean expression has no derivative")
                return ZERO
            return neg(_derive(operand, leaf)) if op == "-" else _derive(operand, leaf)
        case ir.Call():
            return _call(node, leaf)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            # The condition is held fixed: within a branch the model is smooth, and the
            # moment it stops being so is an event, not a derivative.
            return ir.IfExpr(condition, _derive(then, leaf), _derive(otherwise, leaf))


def d_dt(node: ir.Expr, constants: Container[str] = frozenset()) -> ir.Expr:
    """Differentiate ``node`` with respect to time.

    ``constants`` names what does not move -- parameters and constants. Everything else is
    taken to be a function of time, so a variable ``x`` differentiates to ``der(x)``.
    """

    def leaf(ref: ir.Ref) -> ir.Expr:
        if ref.parts == ("time",):
            return ONE
        if ref.name in constants:
            return ZERO
        return ir.Call("der", (ref,))

    return _derive(node, leaf)


def d_dx(node: ir.Expr, name: str) -> ir.Expr:
    """Differentiate ``node`` with respect to the variable ``name``: one Jacobian entry.

    ``name`` may be a derivative name, so the derivative of a residual with respect to
    ``der(x)`` is asked for as ``d_dx(residual, "der_x")``. That works for free: ``der(x)``
    reaches the leaf rule already renamed to ``der_x``, so a derivative is just a variable
    here, which is exactly what an implicit solver treats it as.
    """

    def leaf(ref: ir.Ref) -> ir.Expr:
        return ONE if ref.name == name else ZERO

    return _derive(node, leaf)
