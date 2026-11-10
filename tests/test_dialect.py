"""The Python dialect: the spelling the emitter writes, and the calls it refuses."""

from __future__ import annotations

import pytest

from modelica.ir import nodes as ir
from modelica.lang import parse_expression
from modelica.sim.dialect import PYTHON
from modelica.sim.emit import EmitError, emit_expr


def test_the_python_dialect_is_what_the_emitter_always_wrote() -> None:
    """The refactor's own guard: moving the tables into a value changed no output.

    Every other test in the suite would catch a change in the *arithmetic*; this catches a
    change in the *spelling*, which is what a reader of the generated source sees.
    """
    node = parse_expression("if a > 0.0 and b <= 1.0 then sqrt(c) ^ 2.0 else mod(d, e)")
    symbols = {name: name for name in "abcde"}
    assert emit_expr(node, symbols) == (
        "np.sqrt(c) ** 2.0 if a > 0.0 and b <= 1.0 else np.mod(d, e)"
    )
    assert emit_expr(node, symbols, dialect=PYTHON) == emit_expr(node, symbols)


def test_a_function_the_dialect_lacks_says_which_dialect() -> None:
    """An untranslatable call names the target, because the answer differs per target."""
    node = ir.Call("wobble", (ir.RealLiteral(1.0),))
    with pytest.raises(EmitError, match="'wobble' in python"):
        emit_expr(node, {}, dialect=PYTHON)


def test_a_qualified_builtin_call_reads_as_the_builtin() -> None:
    """`Modelica.Math.sin(x)` is the builtin sin under its library path.

    The standard library writes the maths functions this way where a `when` or an equation reaches
    them un-inlined, so the emitter reads a qualified call as the builtin the dialect knows by its
    simple name -- the same value as if it had been written bare.
    """
    qualified = parse_expression("Modelica.Math.sin(x)")
    assert emit_expr(qualified, {"x": "x"}) == "np.sin(x)"
    assert emit_expr(qualified, {"x": "x"}) == emit_expr(parse_expression("sin(x)"), {"x": "x"})


def test_a_qualified_name_the_dialect_still_lacks_is_refused() -> None:
    """Only a known builtin is de-qualified: an unknown qualified call still fails, by its full name."""
    with pytest.raises(EmitError, match=r"'Foo\.bar' in python"):
        emit_expr(parse_expression("Foo.bar(x)"), {"x": "x"})
