"""The Rust Radau runtime against a reference, one piece at a time.

The first piece is the complex dense LU (`modelica.sim.rust_radau.COMPLEX_SOLVE`), the twin of
``md_solve`` a Radau step needs for its complex eigenvalue. It is checked the way every backend
here is: compile it, run it, and compare numbers against a reference the rest of the ecosystem
trusts -- here `numpy.linalg.solve` on the same complex system, since the point is that the
hand-written factorisation agrees with LAPACK, not with itself.

`rustc` is optional exactly as `cc` is in `tests/test_cgen.py`: no compiler, no test.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from modelica.sim.rust_radau import COMPLEX_SOLVE, CONSTANTS

COMPILER = shutil.which("rustc")
needs_rustc = pytest.mark.skipif(COMPILER is None, reason="no rustc on this machine")


def _floats(values: object) -> str:
    return ", ".join(f"{float(v)!r}f64" for v in values)  # type: ignore[union-attr]


def _run(program: str) -> list[float]:
    assert COMPILER is not None
    with tempfile.TemporaryDirectory() as tmp:
        unit = Path(tmp) / "csolve.rs"
        unit.write_text(program, encoding="utf-8")
        binary = Path(tmp) / "csolve"
        subprocess.run(  # noqa: S603 - compiler from PATH, input generated here
            [COMPILER, "--edition", "2021", "-O", "-D", "warnings", str(unit), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        done = subprocess.run([str(binary)], check=True, capture_output=True, text=True)  # noqa: S603
    return [float(token) for token in done.stdout.split()]


@needs_rustc
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_md_csolve_answers_what_numpy_answers(n: int) -> None:
    """A random complex system solved in Rust matches ``numpy.linalg.solve`` to 1e-12.

    Random, and seeded so a failure is reproducible: the interleaved-versus-permuted swap bug
    `radau.py:252` records survived every corpus model and failed the first random 4x4, so a
    complex LU has to be shown a matrix that actually pivots, not only a diagonally dominant one.
    """
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(20260903 + n)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    wanted = numpy.linalg.solve(a, b)

    program = "\n".join(
        [
            *COMPLEX_SOLVE,
            "fn main() {",
            f"    let mut ar: [f64; {n * n}] = [{_floats(a.real.ravel())}];",
            f"    let mut ai: [f64; {n * n}] = [{_floats(a.imag.ravel())}];",
            f"    let mut br: [f64; {n}] = [{_floats(b.real)}];",
            f"    let mut bi: [f64; {n}] = [{_floats(b.imag)}];",
            f"    let mut xr = [0.0f64; {n}];",
            f"    let mut xi = [0.0f64; {n}];",
            f"    md_csolve({n}, &mut ar, &mut ai, &mut br, &mut bi, &mut xr, &mut xi);",
            "    let mut printed: Vec<String> = xr.iter().map(|v| v.to_string()).collect();",
            "    printed.extend(xi.iter().map(|v| v.to_string()));",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    flat = _run(program)
    found = numpy.array(flat[:n]) + 1j * numpy.array(flat[n:])
    numpy.testing.assert_allclose(found, wanted, rtol=1e-12, atol=1e-12)


@needs_rustc
def test_the_constants_match_radau() -> None:
    """The emitted Rust constants equal `radau.py`'s to full f64 precision.

    Generated from the same formulas rather than hand-typed, so this is a transcription check:
    a wrong digit in a transform matrix would make every step subtly wrong and nothing else
    would notice.
    """
    numpy = pytest.importorskip("numpy")
    radau = pytest.importorskip("modelica.sim.radau")
    program = "\n".join(
        [
            *CONSTANTS,
            "fn main() {",
            "    let mut v: Vec<f64> = Vec::new();",
            "    v.extend_from_slice(&C);",
            "    v.extend_from_slice(&E);",
            "    v.push(MU_REAL);",
            "    v.push(MU_COMPLEX_R);",
            "    v.push(MU_COMPLEX_I);",
            "    for r in T.iter() { v.extend_from_slice(r); }",
            "    for r in TI.iter() { v.extend_from_slice(r); }",
            "    for r in P.iter() { v.extend_from_slice(r); }",
            "    let printed: Vec<String> = v.iter().map(|x| x.to_string()).collect();",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    found = _run(program)
    wanted = [
        *radau.C.tolist(),
        *radau.E.tolist(),
        radau.MU_REAL,
        radau.MU_COMPLEX.real,
        radau.MU_COMPLEX.imag,
        *radau.T.ravel().tolist(),
        *radau.TI.ravel().tolist(),
        *radau.P.ravel().tolist(),
    ]
    numpy.testing.assert_allclose(found, wanted, rtol=1e-12, atol=1e-12)
