"""Differentiation, checked two ways.

The symbolic checks pin the *shape* of the answer, because a Jacobian that is correct but
unreadable is a Jacobian nobody will debug. The numeric checks pin the value: differentiate
symbolically, evaluate both sides at a point, and compare against a finite difference. The
second kind is what would catch a wrong rule; the first is what keeps the output small.
"""

from __future__ import annotations

import math

import pytest

from modelica.ir import nodes as ir
from modelica.ir.derive import DerivativeError, d_dt, d_dx, der_name, simplify
from modelica.ir.evaluate import EvaluationError, evaluate


def ref(name: str) -> ir.Ref:
    return ir.Ref(tuple(name.split(".")))


X = ref("x")
Y = ref("y")
TIME = ref("time")


def num(value: float) -> ir.RealLiteral:
    return ir.RealLiteral(value)


def mul(lhs: ir.Expr, rhs: ir.Expr) -> ir.Binary:
    return ir.Binary("*", lhs, rhs)


def der(name: str) -> ir.Call:
    return ir.Call("der", (ref(name),))


# --------------------------------------------------------------------------------------
# Folding
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (ir.Binary("+", X, num(0.0)), X),
        (ir.Binary("*", X, num(1.0)), X),
        (ir.Binary("*", X, num(0.0)), num(0.0)),
        (ir.Binary("*", num(-1.0), X), ir.Unary("-", X)),
        (ir.Binary("^", X, num(1.0)), X),
        (ir.Binary("^", X, num(0.0)), num(1.0)),
        (ir.Binary("+", num(2.0), num(3.0)), num(5.0)),
        (ir.Binary("/", num(3.0), num(4.0)), num(0.75)),
        (ir.Unary("-", ir.Unary("-", X)), X),
        (ir.Unary("+", X), X),
        # Folding is bottom-up, so an identity exposed by folding a subterm is also taken.
        (ir.Binary("*", X, ir.Binary("-", num(3.0), num(2.0))), X),
    ],
)
def test_simplify(expr: ir.Expr, expected: ir.Expr) -> None:
    assert simplify(expr) == expected


def test_simplify_leaves_division_by_zero_alone() -> None:
    """Folding must not invent a ZeroDivisionError the model would never have raised."""
    expr = ir.Binary("/", num(1.0), num(0.0))
    assert simplify(expr) == expr


def test_simplify_does_not_touch_comparisons() -> None:
    expr = ir.Binary("<", X, num(0.0))
    assert simplify(expr) == expr


# --------------------------------------------------------------------------------------
# Time derivatives
# --------------------------------------------------------------------------------------


def test_a_variable_differentiates_to_its_derivative() -> None:
    assert d_dt(X) == der("x")


def test_time_differentiates_to_one() -> None:
    assert d_dt(TIME) == num(1.0)


def test_a_parameter_differentiates_to_zero() -> None:
    assert d_dt(mul(ref("k"), X), constants={"k"}) == mul(ref("k"), der("x"))


def test_a_literal_differentiates_to_zero() -> None:
    assert d_dt(num(3.5)) == num(0.0)


def test_higher_derivatives_are_named_variables() -> None:
    """``der(der(x))`` is ``der(der_x)`` -- the second derivative of a new unknown.

    This is what lets index reduction differentiate an equation twice without the IR
    growing a node type for it, and what makes the extra unknown something the causalizer
    can match like any other.
    """
    assert d_dt(der("x")) == ir.Call("der", (ref(der_name("x")),))
    assert d_dt(d_dt(der("x"))) == ir.Call("der", (ref("der_der_x"),))


def test_pre_is_constant_between_events() -> None:
    assert d_dt(ir.Call("pre", (X,))) == num(0.0)


def test_smooth_is_transparent_to_value_and_derivative() -> None:
    """``smooth(order, expr)`` has the value of ``expr`` and differentiates through it.

    The order is a continuity hint for the solver's event detection, so it changes no number:
    the value is ``expr``, and the derivative keeps the wrapper around the derivative of ``expr``
    with the order carried through unchanged -- the same shape ``noEvent`` takes. Before this rule
    a stiff model that wrapped a term in ``smooth`` was refused at the Jacobian with
    "no derivative rule for 'smooth'", even though emit already treated it as transparent.
    """
    inner = mul(X, X)
    expr = ir.Call("smooth", (ir.IntLiteral(2), inner))
    assert evaluate(expr, {"x": 4.0}) == 16.0
    assert d_dt(expr) == ir.Call("smooth", (ir.IntLiteral(2), d_dt(inner)))
    assert evaluate(d_dt(expr), {"x": 4.0, "der(x)": 1.0}) == pytest.approx(8.0)


def test_product_rule_folds_away_constant_factors() -> None:
    # d/dt (k * x) with k a parameter is k * der(x), not 0 * x + k * der(x).
    assert d_dt(mul(ref("k"), X), constants={"k"}) == mul(ref("k"), der("x"))


def test_a_condition_is_held_fixed_across_a_branch() -> None:
    expr = ir.IfExpr(ir.Binary("<", X, num(0.0)), X, Y)
    assert d_dt(expr) == ir.IfExpr(ir.Binary("<", X, num(0.0)), der("x"), der("y"))


def test_a_comparison_has_no_derivative() -> None:
    with pytest.raises(DerivativeError):
        d_dt(ir.Binary("<", X, Y))


def test_a_comparison_derives_to_zero_for_a_variable_it_does_not_mention() -> None:
    """A Jacobian entry for a variable a relation or a logical operator does not contain is zero.

    ``d/dy (x > 0)`` is zero: the relation's value turns only on x, so asked about y it contributes
    nothing, and an algebraic block whose residual carries an unrelated relation can still take the
    slope it needs. A relation on the variable itself stays a refusal -- there is no derivative then.
    """
    assert d_dx(ir.Binary(">", X, num(0)), "y") == ir.RealLiteral(0.0)
    assert d_dx(ir.Binary("or", ir.Binary("<", X, num(0)), ref("b")), "y") == ir.RealLiteral(0.0)
    assert d_dx(ir.Unary("not", ir.Binary(">", X, num(0))), "y") == ir.RealLiteral(0.0)
    with pytest.raises(DerivativeError):
        d_dx(ir.Binary(">", X, num(0)), "x")


def test_an_unknown_function_has_no_derivative() -> None:
    with pytest.raises(DerivativeError):
        d_dt(ir.Call("mystery", (X,)))


def test_a_qualified_builtin_differentiates_like_the_builtin() -> None:
    """`Modelica.Math.sin` is the builtin sin under its library path, differentiated by the same rule.

    The standard library reaches the maths functions qualified where a call is not inlined; the chain
    rule reads the last segment when it is a rule it has, so the derivative is the bare one's.
    """
    assert d_dt(ir.Call("Modelica.Math.sin", (X,))) == d_dt(ir.Call("sin", (X,)))


def test_a_qualified_name_with_no_rule_still_has_no_derivative() -> None:
    """Only a known builtin is de-qualified; an unknown qualified call still has no rule."""
    with pytest.raises(DerivativeError, match=r"no derivative rule for 'Foo\.mystery'"):
        d_dt(ir.Call("Foo.mystery", (X,)))


def _at(expr: ir.Expr, **point: float) -> float:
    """The partial evaluated at a point -- a number, so a piecewise rule can be checked."""
    return float(evaluate(expr, point))


def test_min_and_max_differentiate_to_the_active_argument() -> None:
    lo = ir.Call("min", (X, num(0.5)))
    assert _at(d_dx(lo, "x"), x=0.3) == pytest.approx(1.0)  # x is smaller: d = dx
    assert _at(d_dx(lo, "x"), x=0.7) == pytest.approx(0.0)  # the constant is smaller: d = 0
    hi = ir.Call("max", (X, num(0.5)))
    assert _at(d_dx(hi, "x"), x=0.7) == pytest.approx(1.0)
    assert _at(d_dx(hi, "x"), x=0.3) == pytest.approx(0.0)


def test_atan2_uses_the_quotient_rule() -> None:
    # atan2(y, x): d/dy = x/(x^2+y^2), d/dx = -y/(x^2+y^2). Here the first arg is X, second Y.
    a = ir.Call("atan2", (X, Y))
    assert _at(d_dx(a, "x"), x=1.0, y=2.0) == pytest.approx(2.0 / 5.0)
    assert _at(d_dx(a, "y"), x=1.0, y=2.0) == pytest.approx(-1.0 / 5.0)


def test_floor_ceil_integer_and_div_have_zero_derivative() -> None:
    assert _at(d_dx(ir.Call("floor", (X,)), "x"), x=2.3) == pytest.approx(0.0)
    assert _at(d_dx(ir.Call("ceil", (X,)), "x"), x=2.3) == pytest.approx(0.0)
    assert _at(d_dx(ir.Call("integer", (X,)), "x"), x=2.3) == pytest.approx(0.0)
    assert _at(d_dx(ir.Call("div", (X, Y)), "x"), x=7.0, y=2.0) == pytest.approx(0.0)
    assert _at(d_dx(ir.Call("div", (X, Y)), "y"), x=7.0, y=2.0) == pytest.approx(0.0)


def test_mod_and_rem_differentiate_by_their_numerator() -> None:
    m = ir.Call("mod", (X, Y))
    assert _at(d_dx(m, "x"), x=1.3, y=2.0) == pytest.approx(1.0)  # d/dx mod = 1
    assert _at(d_dx(m, "y"), x=3.0, y=2.0) == pytest.approx(-1.0)  # -floor(3/2) = -1
    r = ir.Call("rem", (X, Y))
    assert _at(d_dx(r, "x"), x=1.3, y=2.0) == pytest.approx(1.0)  # d/dx rem = 1
    assert _at(d_dx(r, "y"), x=7.0, y=2.0) == pytest.approx(-3.0)  # -div(7, 2) = -3


# --------------------------------------------------------------------------------------
# Jacobian entries
# --------------------------------------------------------------------------------------


def test_d_dx_picks_out_one_variable() -> None:
    assert d_dx(ir.Binary("+", mul(num(3.0), X), Y), "x") == num(3.0)
    assert d_dx(ir.Binary("+", mul(num(3.0), X), Y), "y") == num(1.0)
    assert d_dx(ir.Binary("+", mul(num(3.0), X), Y), "z") == num(0.0)


def test_d_dx_treats_a_derivative_as_a_variable() -> None:
    """An implicit solver asks for the residual's derivative with respect to ``der(x)``."""
    residual = ir.Binary("-", der("x"), mul(ref("k"), X))
    assert d_dx(residual, der_name("x")) == num(1.0)
    assert d_dx(residual, "x") == ir.Unary("-", ref("k"))


def test_d_dx_of_a_qualified_name() -> None:
    assert d_dx(ref("body.v"), "body.v") == num(1.0)
    assert d_dx(ref("body.v"), "v") == num(0.0)


# --------------------------------------------------------------------------------------
# Numeric agreement -- the check that would catch a wrong rule
# --------------------------------------------------------------------------------------

POINT = {"x": 0.7, "y": 1.3, "k": 2.5}


def at(expr: ir.Expr, environment: dict[str, float]) -> float:
    """Evaluate to a number, or say which expression refused to be one."""
    value = evaluate(expr, environment)
    assert isinstance(value, float | int), f"{expr!r} evaluated to {value!r}"
    return float(value)


EXPRESSIONS = [
    ir.Binary("+", X, Y),
    ir.Binary("-", X, mul(num(3.0), Y)),
    mul(X, Y),
    ir.Binary("/", X, Y),
    ir.Binary("^", X, num(3.0)),
    ir.Binary("^", X, Y),
    ir.Call("sqrt", (X,)),
    ir.Call("sin", (mul(num(2.0), X),)),
    ir.Call("cos", (X,)),
    ir.Call("tan", (X,)),
    ir.Call("exp", (ir.Unary("-", X),)),
    ir.Call("log", (X,)),
    ir.Call("log10", (X,)),
    ir.Call("sinh", (X,)),
    ir.Call("cosh", (X,)),
    ir.Call("tanh", (X,)),
    ir.Call("asin", (X,)),
    ir.Call("acos", (X,)),
    ir.Call("atan", (X,)),
    ir.Call("abs", (X,)),
    mul(ir.Call("exp", (X,)), ir.Call("sin", (Y,))),
    ir.Binary("/", ir.Call("sin", (X,)), ir.Binary("+", num(1.0), mul(X, X))),
]


def _ids() -> list[str]:
    return [str(i) for i, _ in enumerate(EXPRESSIONS)]


@pytest.mark.parametrize("expr", EXPRESSIONS, ids=_ids())
@pytest.mark.parametrize("wrt", ["x", "y"])
def test_matches_a_finite_difference(expr: ir.Expr, wrt: str) -> None:
    step = 1e-6
    forward = dict(POINT, **{wrt: POINT[wrt] + step})
    backward = dict(POINT, **{wrt: POINT[wrt] - step})
    numeric = (at(expr, forward) - at(expr, backward)) / (2 * step)

    assert at(d_dx(expr, wrt), POINT) == pytest.approx(numeric, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("expr", EXPRESSIONS, ids=_ids())
def test_the_chain_rule_composes(expr: ir.Expr) -> None:
    """``d/dt f(x, y)`` must equal ``f_x * der(x) + f_y * der(y)`` evaluated at a point."""
    rates = {"der(x)": -0.9, "der(y)": 0.4}
    environment = dict(POINT, **rates)

    total = at(d_dt(expr, constants={"k"}), environment)
    expected = sum(at(d_dx(expr, name), POINT) * rates[f"der({name})"] for name in ("x", "y"))
    assert total == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_log10_uses_the_right_constant() -> None:
    value = at(d_dx(ir.Call("log10", (X,)), "x"), POINT)
    assert value == pytest.approx(1.0 / (POINT["x"] * math.log(10)))


def test_an_array_is_refused_by_every_stage_that_walks_an_expression() -> None:
    """``ir.Array`` reaches the IR from annotations only, and three passes must say so.

    An annotation is full of arrays -- every point, extent and colour is one -- so the node
    exists and a `.mo` file that carries graphics parses. None of that is an *equation*, and
    array equations are refused at the parser (`_UNSUPPORTED["for"]`). This asserts the three
    passes below the parser agree, and agree in the same words, because the failure mode
    worth preventing is one of them quietly inventing a meaning for a list of numbers.

    Delete this when arrays land. Its whole content is that they have not.
    """
    # Imported here rather than at the top: this file is about the IR, and `modelica.sim`
    # reaches the numerics through its package import.
    from modelica.sim.emit import EmitError, emit_expr  # noqa: PLC0415

    array = ir.Array((num(1.0), num(2.0)))

    with pytest.raises(DerivativeError, match="array equations are not supported yet"):
        d_dx(array, "x")
    with pytest.raises(EvaluationError, match="array equations are not supported yet"):
        evaluate(array, {})
    with pytest.raises(EmitError, match="array equations are not supported yet"):
        emit_expr(array, {})
