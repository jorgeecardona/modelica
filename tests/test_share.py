"""Common subexpressions: what gets a name, what deliberately does not.

The pass returns a value rather than rewriting a tree, so these tests read like the caller
does: hand it roots, get back definitions and rewritten roots, and check that substituting
one into the other gives the expression you started with.

The rule worth two tests is the conditional one. A subexpression inside an ``if`` arm is
never named, because naming it hoists it out of the arm and evaluates it on every step --
and an arm is exactly where a model puts arithmetic that is only valid on one side of a
test. ``DrainingTank`` has that shape today, and hoisting its ``sqrt`` would let a NaN out
of a branch the trajectory never takes.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.share import share
from modelica.lang import parse_expression
from modelica.lang.unparse import unparse_expr


def _shared(*sources: str) -> tuple[list[tuple[str, str]], list[str]]:
    plan = share([parse_expression(s) for s in sources])
    return (
        [(name, unparse_expr(node)) for name, node in plan.definitions],
        [unparse_expr(root) for root in plan.roots],
    )


def test_a_repeat_within_one_expression_is_named_once() -> None:
    definitions, roots = _shared("a * b + a * b")
    assert definitions == [("_s0", "a * b")]
    assert roots == ["_s0 + _s0"]


def test_a_repeat_across_two_expressions_is_named_once() -> None:
    """The pass takes the whole set, because that is the scope a caller emits at."""
    definitions, roots = _shared("a * b + c", "(a * b) * d")
    assert definitions == [("_s0", "a * b")]
    assert roots == ["_s0 + c", "_s0 * d"]


def test_something_written_once_is_left_alone() -> None:
    assert _shared("a * b + c") == ([], ["a * b + c"])


def test_a_reference_or_a_literal_is_never_named() -> None:
    """Naming a one-token expression makes the source longer, not shorter."""
    assert _shared("a + a", "2.0 * 2.0") == ([], ["a + a", "2.0 * 2.0"])


def test_a_nested_repeat_is_named_by_naming_the_outer_one() -> None:
    """``a * b`` occurs twice only because ``a * b + c`` does; one name covers both."""
    definitions, roots = _shared("(a * b + c) * (a * b + c)")
    assert definitions == [("_s0", "a * b + c")]
    assert roots == ["_s0 * _s0"]


def test_definitions_come_back_in_dependency_order() -> None:
    """Shallowest first, so a caller can emit them in the order given and never look ahead."""
    definitions, roots = _shared("sqrt(a * b) + sqrt(a * b)", "a * b")
    assert [name for name, _ in definitions] == ["_s0", "_s1"]
    assert definitions[0] == ("_s0", "a * b")
    assert definitions[1] == ("_s1", "sqrt(_s0)")
    assert roots == ["_s1 + _s1", "_s0"]


def test_the_arms_of_a_conditional_are_never_named() -> None:
    """The rule that keeps a guarded square root guarded."""
    definitions, roots = _shared("if h > 0.0 then sqrt(h) else 0.0", "sqrt(h) * 2.0")
    assert definitions == []
    assert roots == ["if h > 0.0 then sqrt(h) else 0.0", "sqrt(h) * 2.0"]


def test_a_repeat_inside_two_arms_is_still_not_named() -> None:
    """Even a repeat *within* the arms stays put: one arm runs, so it is not a repeat."""
    definitions, _ = _shared("if c then a * b else a * b")
    assert definitions == []


def test_a_conditions_own_subexpressions_are_fair_game() -> None:
    """A condition is always evaluated, so hoisting out of one changes nothing."""
    definitions, roots = _shared("if a * b > 0.0 then 1.0 else 2.0", "a * b")
    assert definitions == [("_s0", "a * b")]
    assert roots == ["if _s0 > 0.0 then 1.0 else 2.0", "_s0"]


def test_the_result_is_the_same_expression_when_the_names_are_put_back() -> None:
    """The correctness property, stated directly: naming changes nothing about the value."""
    sources = ["sqrt(a * b) + sqrt(a * b) * c", "a * b - c", "f(a * b, c) + f(a * b, c)"]
    plan = share([parse_expression(s) for s in sources])
    table = dict(plan.definitions)

    def expand(node: ir.Expr) -> ir.Expr:
        match node:
            case ir.Ref(parts=(name,)) if name in table:
                return expand(table[name])
            case ir.Binary(op=op, lhs=lhs, rhs=rhs):
                return ir.Binary(op, expand(lhs), expand(rhs))
            case ir.Unary(op=op, operand=operand):
                return ir.Unary(op, expand(operand))
            case ir.Call(name=name, args=args):
                return ir.Call(name, tuple(expand(a) for a in args))
            case ir.IfExpr(condition=c, then=t, otherwise=o):
                return ir.IfExpr(expand(c), expand(t), expand(o))
            case _:
                return node

    assert [expand(root) for root in plan.roots] == [parse_expression(s) for s in sources]


@pytest.mark.parametrize("prefix", ["_s", "_t"])
def test_the_prefix_is_the_callers(prefix: str) -> None:
    plan = share([parse_expression("a * b + a * b")], prefix)
    assert plan.definitions[0][0] == f"{prefix}0"


def test_it_is_deterministic() -> None:
    """The generated source is compared byte for byte, so the order cannot come from a set."""
    sources = ["a * b + c * d", "c * d + a * b", "sqrt(a * b) + sqrt(c * d)"]
    once = share([parse_expression(s) for s in sources])
    twice = share([parse_expression(s) for s in sources])
    assert once == twice
    assert len(once.definitions) == 2


def test_nothing_to_share_is_falsy() -> None:
    """So a caller can write ``if plan:`` rather than reaching into the tuple."""
    assert not share([parse_expression("a + b")])
    assert share([parse_expression("a * b + a * b")])
