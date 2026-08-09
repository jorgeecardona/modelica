"""Two spellings of one expression, checked by running both.

The claim `docs/ideas/backends.md:88` makes is that the emitters are *consequences* of the
IR -- mechanical, and cheap to add. A second dialect is where that claim is either true or
found out, so these tests do not compare the two strings to each other. They compile the C,
run it, evaluate the same expression with the interpreter, and compare the **numbers**.

Comparing text would pin today's spelling and prove nothing about the arithmetic. Comparing
answers is what says a difference in operator precedence, in how ``mod`` rounds, or in what
``^`` means, has not quietly changed the value.

The C compiler is optional here, exactly as Numba is in `tests/test_jit.py`: no ``cc``, no
test, and the rest of the suite is unaffected.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from modelica.ir import nodes as ir
from modelica.ir.evaluate import evaluate
from modelica.lang import parse_expression
from modelica.sim.dialect import PYTHON, C
from modelica.sim.emit import EmitError, emit_expr

if TYPE_CHECKING:
    from pathlib import Path

COMPILER = shutil.which("cc") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(COMPILER is None, reason="no C compiler on this machine")

#: Every expression shape the emitter can produce, with the values to try it at. Chosen for
#: the places the two languages disagree rather than for coverage: precedence between `&&`
#: and `||`, `^` as a call instead of an operator, a conditional as a ternary, and the three
#: built-ins C does not have.
CASES: list[tuple[str, dict[str, float]]] = [
    ("a + b * c", {"a": 2.0, "b": 3.0, "c": 4.0}),
    ("(a + b) * c", {"a": 2.0, "b": 3.0, "c": 4.0}),
    ("a - b - c", {"a": 2.0, "b": 3.0, "c": 4.0}),
    ("a / b / c", {"a": 24.0, "b": 3.0, "c": 2.0}),
    ("a / (b / c)", {"a": 24.0, "b": 3.0, "c": 2.0}),
    ("a ^ b", {"a": 2.0, "b": 10.0}),
    ("a ^ (b ^ c)", {"a": 2.0, "b": 3.0, "c": 2.0}),
    ("-a ^ b", {"a": 3.0, "b": 2.0}),
    ("(-a) ^ b", {"a": 3.0, "b": 2.0}),
    ("a * (-b)", {"a": 3.0, "b": 2.0}),
    ("sqrt(a) + sin(b) * cos(c)", {"a": 2.0, "b": 0.5, "c": 1.5}),
    ("exp(a) - log(b)", {"a": 1.0, "b": 7.0}),
    ("atan2(a, b)", {"a": 1.0, "b": -2.0}),
    ("min(a, b) + max(a, b)", {"a": 3.0, "b": -4.0}),
    ("abs(a) + abs(b)", {"a": -3.0, "b": 4.0}),
    ("sign(a) + sign(b) + sign(c)", {"a": -3.0, "b": 0.0, "c": 4.0}),
    ("floor(a) + ceil(b)", {"a": 2.7, "b": 2.1}),
    # `mod` rounds toward minus infinity and `rem` truncates. C's `fmod` is `rem`, so a
    # dialect that mapped `mod` onto it would be wrong here and nowhere else.
    ("mod(a, b)", {"a": -7.0, "b": 3.0}),
    ("mod(a, b)", {"a": 7.0, "b": -3.0}),
    ("rem(a, b)", {"a": -7.0, "b": 3.0}),
    ("if a > b then c else d", {"a": 2.0, "b": 1.0, "c": 10.0, "d": 20.0}),
    ("if a > b then c else d", {"a": 0.0, "b": 1.0, "c": 10.0, "d": 20.0}),
    ("(if a > b then c else d) + e", {"a": 2.0, "b": 1.0, "c": 10.0, "d": 20.0, "e": 1.0}),
    # `&&` binds tighter than `||` in C, and `and` tighter than `or` in Python -- the same
    # order, which is why this passes; it is here because taking one table for both would
    # not have been checked otherwise.
    (
        "if a > 0.0 and b > 0.0 or c > 0.0 then d else e",
        {"a": -1.0, "b": 1.0, "c": 1.0, "d": 5.0, "e": 6.0},
    ),
    ("if not (a > 0.0) then d else e", {"a": -1.0, "d": 5.0, "e": 6.0}),
    ("if a > 0.0 and not (b > 0.0) then d else e", {"a": 1.0, "b": -1.0, "d": 5.0, "e": 6.0}),
]


def _in_c(source: str, values: dict[str, float], tmp_path: Path) -> float:
    """Compile one expression into a program that prints it, and run it."""
    node = parse_expression(source)
    symbols = {name: name for name in values}
    body = emit_expr(node, symbols, dialect=C)
    declarations = "".join(f"    double {name} = {value!r};\n" for name, value in values.items())
    program = "\n".join(
        [
            *C.preamble,
            "#include <stdio.h>",
            "",
            "int main(void) {",
            declarations.rstrip("\n"),
            f'    printf("%.17g\\n", (double)({body}));',
            "    return 0;",
            "}",
            "",
        ]
    )
    unit = tmp_path / "case.c"
    unit.write_text(program, encoding="utf-8")
    binary = tmp_path / "case"
    assert COMPILER is not None
    # `-Werror` is the point of the flag list: a warning in generated code is a bug in the
    # generator, and the only way it gets noticed is if it fails something.
    subprocess.run(  # noqa: S603 - the compiler is found on PATH and the input is generated here
        [COMPILER, "-std=c11", "-O2", "-Wall", "-Werror", str(unit), "-o", str(binary), "-lm"],
        check=True,
        capture_output=True,
    )
    done = subprocess.run([str(binary)], check=True, capture_output=True, text=True)  # noqa: S603
    return float(done.stdout.strip())


@needs_cc
@pytest.mark.parametrize(("source", "values"), CASES, ids=[f"{s}" for s, _ in CASES])
def test_the_c_dialect_computes_what_the_interpreter_computes(
    source: str, values: dict[str, float], tmp_path: Path
) -> None:
    wanted = evaluate(parse_expression(source), dict(values))
    assert isinstance(wanted, (int, float))
    assert _in_c(source, values, tmp_path) == pytest.approx(float(wanted), rel=1e-12, abs=1e-12)


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
    with pytest.raises(EmitError, match="'wobble' in c"):
        emit_expr(node, {}, dialect=C)
