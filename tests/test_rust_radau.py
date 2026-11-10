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
from typing import Any

import pytest

from modelica.sim.rust_radau import CONSTANTS, FIRST_STEP, LINALG, NEWTON, STEP

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
            "    v.extend_from_slice(&MD_C);",
            "    v.extend_from_slice(&MD_E);",
            "    v.push(MD_MU_REAL);",
            "    v.push(MD_MU_COMPLEX_R);",
            "    v.push(MD_MU_COMPLEX_I);",
            "    for r in MD_T.iter() { v.extend_from_slice(r); }",
            "    for r in MD_TI.iter() { v.extend_from_slice(r); }",
            "    for r in MD_P.iter() { v.extend_from_slice(r); }",
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
            f"    assert!(md_factor_pair({n}, &jac, {h!r}f64, &mut ar, &mut pr, &mut acr, &mut aci, &mut pc, &[]));",
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


@needs_rustc
def test_md_newton_matches_radau() -> None:
    """One Newton solve of the collocation stages matches `radau._newton`.

    A linear right-hand side ``y' = A y`` makes the collocation system linear, so simplified
    Newton with the exact factored Jacobian lands the stage increments in one iteration; both
    tiers have to agree on those increments and on the convergence verdict. The rhs is defined
    in the program because the integrator and the model compile together -- md_newton calls it
    by name, not through a pointer.
    """
    numpy = pytest.importorskip("numpy")
    radau = pytest.importorskip("modelica.sim.radau")

    n = 2
    a = numpy.array([[-2.0, 1.0], [1.0, -2.0]])
    h = 0.1
    y = numpy.array([1.0, 0.0])
    scale = numpy.array([1.0, 1.0])
    tol = 1e-8

    def rhs(_t: float, yy: object, _p: object, _w: object, out: object) -> None:
        out[:] = a @ yy  # type: ignore[operator, index]

    ar = numpy.empty((n, n))
    pr = numpy.empty(n, dtype=numpy.int64)
    ac = numpy.empty((n, n), dtype=numpy.complex128)
    pc = numpy.empty(n, dtype=numpy.int64)
    radau._factor_pair(a.copy(), h, ar, pr, ac, pc)  # noqa: SLF001
    z_py = numpy.zeros((3, n))
    conv, iters, _rate, _calls = radau._newton(  # noqa: SLF001
        rhs, 0.0, y, h, numpy.zeros(0), numpy.zeros(0), z_py, scale, tol, ar, pr, ac, pc
    )

    program = "\n".join(
        [
            *CONSTANTS,
            *LINALG,
            *STEP,
            *NEWTON,
            "fn rhs(_t: f64, y: &[f64], _p: &[f64], _w: &mut [f64], dy: &mut [f64]) {",
            "    let a = [[-2.0f64, 1.0], [1.0, -2.0]];",
            "    for i in 0..y.len() {",
            "        let mut s = 0.0;",
            "        for j in 0..y.len() { s += a[i][j] * y[j]; }",
            "        dy[i] = s;",
            "    }",
            "}",
            "fn main() {",
            "    let y = [1.0f64, 0.0];",
            "    let p: [f64; 0] = [];",
            "    let mut w: [f64; 0] = [];",
            "    let jac = [-2.0f64, 1.0, 1.0, -2.0];",
            f"    let h = {h!r}f64;",
            "    let mut lin = MdLinear::new(2, &[(0, 0), (0, 1), (1, 0), (1, 1)], &[], &[], false);",
            "    assert!(lin.factor(&jac, h));",
            "    let mut z = [0.0f64; 6];",
            "    let scale = [1.0f64, 1.0];",
            f"    let (conv, iters, _rate, _calls) = md_newton(0.0, &y, h, &p, &mut w, &mut z, &scale, {tol!r}f64, &mut lin, &[], &mut MdNewtonWork::new(2));",
            "    let mut printed: Vec<String> = z.iter().map(|v| v.to_string()).collect();",
            "    printed.push(if conv { 1.0 } else { 0.0 }.to_string());",
            "    printed.push((iters as f64).to_string());",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    found = _run(program)
    numpy.testing.assert_allclose(found[:6], z_py.ravel(), rtol=1e-10, atol=1e-12)
    assert int(found[6]) == int(conv)
    assert int(found[7]) == iters


@needs_rustc
def test_md_first_step_matches_radau() -> None:
    """The starting-step size equals `radau._first_step` on a linear system."""
    numpy = pytest.importorskip("numpy")
    radau = pytest.importorskip("modelica.sim.radau")

    a = numpy.array([[-2.0, 1.0], [1.0, -2.0]])
    y = numpy.array([1.0, 0.0])
    f0 = a @ y
    rtol, atol, span, t0 = 1e-6, 1e-6, 10.0, 0.0

    def rhs(_t: float, yy: object, _p: object, _w: object, out: object) -> None:
        out[:] = a @ yy  # type: ignore[operator, index]

    wanted, _probes = radau._first_step(  # noqa: SLF001
        rhs, t0, y, numpy.zeros(0), numpy.zeros(0), f0, rtol, atol, span
    )

    program = "\n".join(
        [
            *CONSTANTS,
            *LINALG,
            *STEP,
            *FIRST_STEP,
            "fn rhs(_t: f64, y: &[f64], _p: &[f64], _w: &mut [f64], dy: &mut [f64]) {",
            "    let a = [[-2.0f64, 1.0], [1.0, -2.0]];",
            "    for i in 0..y.len() {",
            "        let mut s = 0.0;",
            "        for j in 0..y.len() { s += a[i][j] * y[j]; }",
            "        dy[i] = s;",
            "    }",
            "}",
            "fn main() {",
            "    let y = [1.0f64, 0.0];",
            "    let p: [f64; 0] = [];",
            "    let mut w: [f64; 0] = [];",
            f"    let f0 = [{_floats(f0)}];",
            f"    let h = md_first_step(0.0, &y, &p, &mut w, &f0, {rtol!r}f64, {atol!r}f64, {span!r}f64);",
            '    println!("{}", h);',
            "}",
            "",
        ]
    )
    found = _run(program)
    assert found[0] == pytest.approx(float(wanted), rel=1e-12, abs=1e-12)


@needs_rustc
@pytest.mark.parametrize(("n", "lower", "upper"), [(6, 1, 1), (9, 2, 3), (8, 3, 0), (5, 4, 4)])
def test_md_gb_factor_solve_answers_what_numpy_answers(n: int, lower: int, upper: int) -> None:
    """The band LU, real and complex, solves a random banded system as numpy does, to 1e-12.

    The matrix fills its whole band and is not diagonally dominant, so the factorisation
    pivots and the swapped rows reach into the extra ``lower`` columns of band storage.
    """
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(20261110 + n + 10 * lower + 100 * upper)
    inside = numpy.array(
        [[1.0 if -lower <= j - i <= upper else 0.0 for j in range(n)] for i in range(n)]
    )
    a = rng.standard_normal((n, n)) * inside
    c = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * inside
    b = rng.standard_normal(n)
    d = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    width = 2 * lower + upper + 1

    def band(m: Any) -> Any:
        out = numpy.zeros(n * width, dtype=m.dtype)
        for i in range(n):
            for j in range(n):
                if inside[i, j]:
                    out[i * width + j + lower - i] = m[i, j]
        return out

    program = "\n".join(
        [
            *LINALG,
            "fn main() {",
            f"    let mut a: [f64; {n * width}] = [{_floats(band(a))}];",
            f"    let mut piv = [0usize; {n + 2}];",
            f"    assert!(md_gb_factor({n}, &mut a, &mut piv, {lower}, {upper}));",
            f"    let mut b: [f64; {n}] = [{_floats(b)}];",
            f"    md_gb_solve({n}, &a, &piv, &mut b);",
            f"    let mut cr: [f64; {n * width}] = [{_floats(band(c).real)}];",
            f"    let mut ci: [f64; {n * width}] = [{_floats(band(c).imag)}];",
            f"    let mut cpiv = [0usize; {n + 2}];",
            f"    assert!(md_cgb_factor({n}, &mut cr, &mut ci, &mut cpiv, {lower}, {upper}));",
            f"    let mut dr: [f64; {n}] = [{_floats(d.real)}];",
            f"    let mut di: [f64; {n}] = [{_floats(d.imag)}];",
            f"    md_cgb_solve({n}, &cr, &ci, &cpiv, &mut dr, &mut di);",
            "    let mut printed: Vec<String> = b.iter().map(|v| v.to_string()).collect();",
            "    printed.extend(dr.iter().map(|v| v.to_string()));",
            "    printed.extend(di.iter().map(|v| v.to_string()));",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    flat = _run(program)
    numpy.testing.assert_allclose(flat[:n], numpy.linalg.solve(a, b), rtol=1e-12, atol=1e-12)
    complex_solution = numpy.linalg.solve(c, d)
    numpy.testing.assert_allclose(flat[n : 2 * n], complex_solution.real, rtol=1e-11, atol=1e-12)
    numpy.testing.assert_allclose(flat[2 * n :], complex_solution.imag, rtol=1e-11, atol=1e-12)


@needs_rustc
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_md_bicgstab_with_dilu_answers_what_numpy_answers(kind: str) -> None:  # noqa: C901 - builds the CSR arrays inline
    """BiCGStab with DILU solves a 2-D grid system as numpy does, real or complex symmetric.

    The pattern is a 6 x 6 five-point grid. The real matrix is not symmetric; the complex one is
    complex symmetric (``A = A^T``), as Radau's ``(alpha + i beta)/h - J`` is for a symmetric
    ``J``. Both are diagonally dominant, so the iteration converges.
    """
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(20261110)
    side = 6
    n = side * side
    a = numpy.zeros((n, n), dtype=complex)
    for i in range(side):
        for j in range(side):
            c = i + side * j
            for di, dj in ((1, 0), (0, 1)):
                if i + di < side and j + dj < side:
                    d = i + di + side * (j + dj)
                    if kind == "real":
                        a[c, d], a[d, c] = rng.uniform(-1, 0, 2)
                    else:
                        a[c, d] = a[d, c] = complex(*rng.uniform(-1, 0, 2))
    for c in range(n):
        a[c, c] = 4.5 + (0.0 if kind == "real" else 2.0j)
    if kind == "real":
        a = a.real
    b = rng.standard_normal(n) + (0.0 if kind == "real" else 1j * rng.standard_normal(n))
    entries = [(r, c) for r in range(n) for c in range(n) if a[r, c] != 0]
    start = [0] * (n + 1)
    for r, _ in entries:
        start[r + 1] += 1
    for r in range(n):
        start[r + 1] += start[r]
    where = {entry: k for k, entry in enumerate(entries)}
    mirror = [where.get((c, r), 2**64 - 1) for r, c in entries]
    diagonal = [where[r, r] for r in range(n)]

    def value(x: complex) -> str:
        if kind == "real":
            return f"{float(x.real)!r}f64"
        return f"MdC {{ re: {float(x.real)!r}, im: {float(x.imag)!r} }}"

    kind_type = "f64" if kind == "real" else "MdC"
    program = "\n".join(
        [
            *LINALG,
            "fn main() {",
            f"    let start: [usize; {n + 1}] = [{', '.join(map(str, start))}];",
            f"    let columns: [usize; {len(entries)}] = [{', '.join(str(c) for _, c in entries)}];",
            f"    let values: [{kind_type}; {len(entries)}] = [{', '.join(value(a[r, c]) for r, c in entries)}];",
            f"    let mirror: [usize; {len(entries)}] = [{', '.join(map(str, mirror))}];",
            f"    let diagonal: [usize; {n}] = [{', '.join(map(str, diagonal))}];",
            f"    let b: [{kind_type}; {n}] = [{', '.join(value(x) for x in b)}];",
            f"    let mut rd = [<{kind_type} as MdNum>::zero(); {n}];",
            f"    assert!(md_dilu({n}, &start, &columns, &values, &mirror, &diagonal, &mut rd));",
            f"    let mut x = [<{kind_type} as MdNum>::zero(); {n}];",
            f"    let mut work = [<{kind_type} as MdNum>::zero(); {7 * n}];",
            f"    assert!(md_bicgstab({n}, &start, &columns, &values, &rd, &b, &mut x, 1e-14, 200, &mut work));",
            "    let printed: Vec<String> = x.iter().map(|v| v.to_string()).collect();"
            if kind == "real"
            else "    let printed: Vec<String> = x.iter().flat_map(|v| [v.re.to_string(), v.im.to_string()]).collect();",
            '    println!("{}", printed.join(" "));',
            "}",
            "",
        ]
    )
    flat = numpy.array(_run(program))
    got = flat if kind == "real" else flat[0::2] + 1j * flat[1::2]
    numpy.testing.assert_allclose(got, numpy.linalg.solve(a, b), rtol=1e-10, atol=1e-12)
