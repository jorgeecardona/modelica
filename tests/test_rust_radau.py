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

from modelica.sim.rust_radau import CONSTANTS, LINALG, STEP

COMPILER = shutil.which("rustc")
needs_rustc = pytest.mark.skipif(COMPILER is None, reason="no rustc on this machine")


def _floats(values: object) -> str:
    return ", ".join(f"{float(v)!r}f64" for v in values)  # type: ignore[union-attr]


def _run(program: str) -> list[float]:
    assert COMPILER is not None
    # These programs paste the whole runtime and a main that exercises one piece, so most of it
    # is unused per test; allow that at the crate root rather than per item. The real runtime,
    # compiled with a model that calls it, is where dead code would actually matter.
    program = "#![allow(dead_code)]\n" + program
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
def test_md_lu_factor_solve_answers_what_numpy_answers(n: int) -> None:
    """A real system factored once and solved for two right-hand sides matches numpy to 1e-12.

    Two right-hand sides against one factorisation, because that is how a step uses it -- every
    Newton iteration solves against the same factors. Random and seeded so a failure repeats:
    the swap bug `radau.py:252` records survived every corpus model and failed the first random
    4x4, so the LU must be shown a matrix that actually pivots, not only a diagonal-dominant one.
    """
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(20260903 + n)
    a = rng.standard_normal((n, n))
    b1 = rng.standard_normal(n)
    b2 = rng.standard_normal(n)

    program = "\n".join(
        [
            *LINALG,
            "fn main() {",
            f"    let mut a: [f64; {n * n}] = [{_floats(a.ravel())}];",
            f"    let mut piv: [usize; {n}] = [0; {n}];",
            f"    assert!(md_lu_factor({n}, &mut a, &mut piv));",
            f"    let mut b1: [f64; {n}] = [{_floats(b1)}];",
            f"    md_lu_solve({n}, &a, &piv, &mut b1);",
            f"    let mut b2: [f64; {n}] = [{_floats(b2)}];",
            f"    md_lu_solve({n}, &a, &piv, &mut b2);",
            "    let mut printed: Vec<String> = b1.iter().map(|v| v.to_string()).collect();",
            "    printed.extend(b2.iter().map(|v| v.to_string()));",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    flat = _run(program)
    numpy.testing.assert_allclose(flat[:n], numpy.linalg.solve(a, b1), rtol=1e-12, atol=1e-12)
    numpy.testing.assert_allclose(flat[n:], numpy.linalg.solve(a, b2), rtol=1e-12, atol=1e-12)


@needs_rustc
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_md_clu_factor_solve_answers_what_numpy_answers(n: int) -> None:
    """The complex twin: one factorisation, two right-hand sides, matched against numpy."""
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(20260904 + n)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    b1 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    b2 = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def system(b: object) -> list[str]:
        return [
            f"    let mut br: [f64; {n}] = [{_floats(b.real)}];",  # type: ignore[union-attr]
            f"    let mut bi: [f64; {n}] = [{_floats(b.imag)}];",  # type: ignore[union-attr]
            f"    md_clu_solve({n}, &ar, &ai, &piv, &mut br, &mut bi);",
            "    printed.extend(br.iter().map(|v| v.to_string()));",
            "    printed.extend(bi.iter().map(|v| v.to_string()));",
        ]

    program = "\n".join(
        [
            *LINALG,
            "fn main() {",
            f"    let mut ar: [f64; {n * n}] = [{_floats(a.real.ravel())}];",
            f"    let mut ai: [f64; {n * n}] = [{_floats(a.imag.ravel())}];",
            f"    let mut piv: [usize; {n}] = [0; {n}];",
            f"    assert!(md_clu_factor({n}, &mut ar, &mut ai, &mut piv));",
            "    let mut printed: Vec<String> = Vec::new();",
            *system(b1),
            *system(b2),
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    flat = _run(program)
    got1 = numpy.array(flat[:n]) + 1j * numpy.array(flat[n : 2 * n])
    got2 = numpy.array(flat[2 * n : 3 * n]) + 1j * numpy.array(flat[3 * n :])
    numpy.testing.assert_allclose(got1, numpy.linalg.solve(a, b1), rtol=1e-12, atol=1e-12)
    numpy.testing.assert_allclose(got2, numpy.linalg.solve(a, b2), rtol=1e-12, atol=1e-12)


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


@needs_rustc
def test_md_norm_matches_radau() -> None:
    """The scaled RMS norm equals `radau._norm`."""
    numpy = pytest.importorskip("numpy")
    radau = pytest.importorskip("modelica.sim.radau")
    x = numpy.random.default_rng(7).standard_normal(9)
    program = "\n".join(
        [
            *CONSTANTS,
            *LINALG,
            *STEP,
            "fn main() {",
            f"    let x: [f64; {len(x)}] = [{_floats(x)}];",
            '    println!("{}", md_norm(&x));',
            "}",
            "",
        ]
    )
    found = _run(program)
    assert found[0] == pytest.approx(float(radau._norm(x)), rel=1e-12, abs=1e-12)  # noqa: SLF001


@needs_rustc
def test_md_predict_matches_radau() -> None:
    """The step-size predictor equals `radau._predict` across the cases it branches on."""
    numpy = pytest.importorskip("numpy")
    radau = pytest.importorskip("modelica.sim.radau")
    cases = [
        (0.1, -1.0, 0.5, -1.0),  # first step, error_old unknown
        (0.1, 0.2, 0.5, 0.8),  # both known
        (0.1, 0.2, 2.0, 0.3),  # error > 1
        (0.1, 0.2, 0.0, 0.3),  # error == 0 -> MAX_FACTOR
    ]
    calls = "\n".join(
        f"    v.push(md_predict({h!r}f64, {ho!r}f64, {e!r}f64, {eo!r}f64));"
        for h, ho, e, eo in cases
    )
    program = "\n".join(
        [
            *CONSTANTS,
            *LINALG,
            *STEP,
            "fn main() {",
            "    let mut v: Vec<f64> = Vec::new();",
            calls,
            "    let printed: Vec<String> = v.iter().map(|x| x.to_string()).collect();",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    found = _run(program)
    wanted = [radau._predict(*case) for case in cases]  # noqa: SLF001
    numpy.testing.assert_allclose(found, wanted, rtol=1e-12, atol=1e-12)


@needs_rustc
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_md_factor_pair_matches_radau(n: int) -> None:
    """Building mu/h*I - J and factoring both eigen-systems, then solving, matches radau.

    The whole linear-algebra path a step takes before Newton: `md_factor_pair` composes the
    constants and the factored LU, and a solve against each factorisation must reproduce
    ``numpy.linalg.solve`` on the same two matrices radau builds.
    """
    numpy = pytest.importorskip("numpy")
    radau = pytest.importorskip("modelica.sim.radau")
    rng = numpy.random.default_rng(100 + n)
    jac = rng.standard_normal((n, n))
    h = 0.037
    br = rng.standard_normal(n)
    bcr = rng.standard_normal(n)
    bci = rng.standard_normal(n)

    program = "\n".join(
        [
            *CONSTANTS,
            *LINALG,
            *STEP,
            "fn main() {",
            f"    let jac: [f64; {n * n}] = [{_floats(jac.ravel())}];",
            f"    let mut ar = [0.0f64; {n * n}];",
            f"    let mut pr = [0usize; {n}];",
            f"    let mut acr = [0.0f64; {n * n}];",
            f"    let mut aci = [0.0f64; {n * n}];",
            f"    let mut pc = [0usize; {n}];",
            f"    assert!(md_factor_pair({n}, &jac, {h!r}f64, &mut ar, &mut pr, &mut acr, &mut aci, &mut pc));",
            f"    let mut br: [f64; {n}] = [{_floats(br)}];",
            f"    md_lu_solve({n}, &ar, &pr, &mut br);",
            f"    let mut bcr: [f64; {n}] = [{_floats(bcr)}];",
            f"    let mut bci: [f64; {n}] = [{_floats(bci)}];",
            f"    md_clu_solve({n}, &acr, &aci, &pc, &mut bcr, &mut bci);",
            "    let mut printed: Vec<String> = br.iter().map(|v| v.to_string()).collect();",
            "    printed.extend(bcr.iter().map(|v| v.to_string()));",
            "    printed.extend(bci.iter().map(|v| v.to_string()));",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    found = _run(program)
    mu_real = radau.MU_REAL / h
    mu_complex = radau.MU_COMPLEX / h
    x_real = numpy.linalg.solve(mu_real * numpy.eye(n) - jac, br)
    x_complex = numpy.linalg.solve(mu_complex * numpy.eye(n) - jac, bcr + 1j * bci)
    numpy.testing.assert_allclose(found[:n], x_real, rtol=1e-12, atol=1e-12)
    got_complex = numpy.array(found[n : 2 * n]) + 1j * numpy.array(found[2 * n :])
    numpy.testing.assert_allclose(got_complex, x_complex, rtol=1e-12, atol=1e-12)
