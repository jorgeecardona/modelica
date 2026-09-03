"""The same expressions as `tests/test_dialect.py`, compiled and run as Rust.

`docs/ideas/backends.md:88` claims the emitters are consequences of the IR: mechanical, and
cheap to add. `test_dialect` is that claim for C; this file is the same claim for a third
dialect, and it is checked the same way. It does not compare the Rust text to anything. It
compiles the Rust, runs it, evaluates the same expression with the interpreter, and compares
the **numbers**.

The cases are shared with the C test on purpose. A dialect that got `mod`'s rounding, `^`'s
meaning, or the precedence between `&&` and `||` wrong would disagree with the interpreter on
exactly the cases that were chosen to find that, whichever language it emits.

`rustc` is optional here, exactly as `cc` is in `test_dialect` and Numba is in `test_jit`: no
compiler, no test, and the rest of the suite is unaffected.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from modelica.ir.evaluate import evaluate
from modelica.lang import parse_expression
from modelica.sim.dialect import RUST
from modelica.sim.emit import emit_expr

# The cases live with the C test; sharing them is what keeps the two dialects checked against
# one bar. `test_verify` already imports a sibling test module this way.
from test_dialect import CASES

if TYPE_CHECKING:
    from pathlib import Path

COMPILER = shutil.which("rustc")
needs_rustc = pytest.mark.skipif(COMPILER is None, reason="no rustc on this machine")


def _in_rust(source: str, values: dict[str, float], tmp_path: Path) -> float:
    """Compile one expression into a program that prints it, and run it."""
    node = parse_expression(source)
    symbols = {name: name for name in values}
    body = emit_expr(node, symbols, dialect=RUST)
    declarations = "".join(f"    let {name}: f64 = {value!r};\n" for name, value in values.items())
    program = "\n".join(
        [
            *RUST.preamble,
            "fn main() {",
            declarations.rstrip("\n"),
            f"    let result: f64 = {body};",
            '    println!("{}", result);',
            "}",
            "",
        ]
    )
    unit = tmp_path / "case.rs"
    unit.write_text(program, encoding="utf-8")
    binary = tmp_path / "case"
    assert COMPILER is not None
    # `-D warnings` is the point, the way `-Werror` is on the C side: a warning in generated
    # code is a bug in the generator, and denial is the only way it gets noticed.
    subprocess.run(  # noqa: S603 - the compiler is found on PATH and the input is generated here
        [COMPILER, "--edition", "2021", "-O", "-D", "warnings", str(unit), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    done = subprocess.run([str(binary)], check=True, capture_output=True, text=True)  # noqa: S603
    return float(done.stdout.strip())


@needs_rustc
@pytest.mark.parametrize(("source", "values"), CASES, ids=[f"{s}" for s, _ in CASES])
def test_the_rust_dialect_computes_what_the_interpreter_computes(
    source: str, values: dict[str, float], tmp_path: Path
) -> None:
    wanted = evaluate(parse_expression(source), dict(values))
    assert isinstance(wanted, (int, float))
    assert _in_rust(source, values, tmp_path) == pytest.approx(float(wanted), rel=1e-12, abs=1e-12)
