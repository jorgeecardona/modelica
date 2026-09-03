"""The Rust Radau IIA runtime, ported from `modelica.sim.radau`.

`docs/issues/03-09-2026-a-rust-backend-is-a-third-dialect-plus-a-port-of-radau.md` names this
as step 2: once `rustgen` emits a model's kernel (done for the supported subset), the
integrator is a port of `radau.py` -- three-stage Radau IIA, order five, RADAU5 after Hairer
and Wanner IV.8 -- with the linear algebra SciPy cannot lend a compiled caller written out
here instead. `radau.py:19` says the same for the numba path.

A Radau step factorises two matrices: one real ``n``-by-``n`` for the method's real
eigenvalue, and one complex ``n``-by-``n`` for its conjugate pair. The real one is
`modelica.sim.dialect.RUST`'s ``md_solve``, already written. This module begins with the
complex one, because it is the piece the issue calls out as still to write and it is testable
on its own before any of the step exists.

The module holds Rust source as text, the way `dialect.C_PREAMBLE` does, because the artifact
is compiled by `rustc`, not imported by Python.
"""

from __future__ import annotations

from typing import Final

#: The dense linear algebra a Radau step needs, ported from `radau.py`'s ``_factor``/``_solve``.
#:
#: A step factorises ``mu/h * I - J`` once and solves against it several times -- every Newton
#: iteration and every rejected retry at the same ``h`` -- so the factorisation is separate from
#: the solve, unlike the one-shot ``md_solve`` the emitted right-hand side uses for its own
#: blocks. ``md_lu_factor`` overwrites ``a`` with its LU factors and records the pivots; a later
#: ``md_lu_solve`` applies them to a right-hand side in place. The complex twin carries a complex
#: value as a pair of ``f64`` -- Rust has no complex type and a crate for one is the dependency
#: this backend avoids -- with the matrix as two flat ``n*n`` slices and each vector as a pair.
#: Partial pivoting compares magnitude: ``abs`` in the real case, squared magnitude
#: ``re*re + im*im`` in the complex one, which orders the same and needs no ``sqrt``. The pivot
#: swaps are applied to the right-hand side all at once before substitution, the ordering
#: `radau.py:244` explains is not optional.
LINALG: Final = (
    "// Real LU with partial pivoting: overwrite a with its factors, record the pivots in piv.",
    "// False when a column is exactly zero (a singular matrix).",
    "#[allow(dead_code)]",
    "fn md_lu_factor(n: usize, a: &mut [f64], piv: &mut [usize]) -> bool {",
    "    for k in 0..n {",
    "        let mut big = 0.0;",
    "        let mut row = k;",
    "        for i in k..n {",
    "            let size = a[i * n + k].abs();",
    "            if size > big { big = size; row = i; }",
    "        }",
    "        piv[k] = row;",
    "        if big == 0.0 { return false; }",
    "        if row != k {",
    "            for j in 0..n { a.swap(k * n + j, row * n + j); }",
    "        }",
    "        let pivot = a[k * n + k];",
    "        for i in (k + 1)..n {",
    "            let share = a[i * n + k] / pivot;",
    "            a[i * n + k] = share;",
    "            if share != 0.0 {",
    "                for j in (k + 1)..n { a[i * n + j] -= share * a[k * n + j]; }",
    "            }",
    "        }",
    "    }",
    "    true",
    "}",
    "",
    "// Forward and back substitution against a factorisation, in place on b. Every pivot swap",
    "// first, then the substitution -- see radau.py's _solve for why interleaving is wrong.",
    "#[allow(dead_code)]",
    "fn md_lu_solve(n: usize, a: &[f64], piv: &[usize], b: &mut [f64]) {",
    "    for k in 0..n {",
    "        let row = piv[k];",
    "        if row != k { b.swap(k, row); }",
    "    }",
    "    for k in 0..n {",
    "        for i in (k + 1)..n { b[i] -= a[i * n + k] * b[k]; }",
    "    }",
    "    for k in (0..n).rev() {",
    "        for j in (k + 1)..n { b[k] -= a[k * n + j] * b[j]; }",
    "        b[k] /= a[k * n + k];",
    "    }",
    "}",
    "",
    "// Complex LU with partial pivoting: the matrix is (ar, ai) row-major, a complex value a",
    "// pair of f64. Pivots by squared magnitude. Same structure as md_lu_factor.",
    "#[allow(dead_code)]",
    "fn md_clu_factor(n: usize, ar: &mut [f64], ai: &mut [f64], piv: &mut [usize]) -> bool {",
    "    for k in 0..n {",
    "        let mut big = 0.0;",
    "        let mut row = k;",
    "        for i in k..n {",
    "            let size = ar[i * n + k] * ar[i * n + k] + ai[i * n + k] * ai[i * n + k];",
    "            if size > big { big = size; row = i; }",
    "        }",
    "        piv[k] = row;",
    "        if big == 0.0 { return false; }",
    "        if row != k {",
    "            for j in 0..n {",
    "                ar.swap(k * n + j, row * n + j);",
    "                ai.swap(k * n + j, row * n + j);",
    "            }",
    "        }",
    "        let pr = ar[k * n + k];",
    "        let pi = ai[k * n + k];",
    "        let den = pr * pr + pi * pi;",
    "        for i in (k + 1)..n {",
    "            // share = a[i][k] / a[k][k]",
    "            let nr = ar[i * n + k];",
    "            let ni = ai[i * n + k];",
    "            let sr = (nr * pr + ni * pi) / den;",
    "            let si = (ni * pr - nr * pi) / den;",
    "            ar[i * n + k] = sr;",
    "            ai[i * n + k] = si;",
    "            if sr != 0.0 || si != 0.0 {",
    "                for j in (k + 1)..n {",
    "                    // a[i][j] -= share * a[k][j]",
    "                    let akr = ar[k * n + j];",
    "                    let aki = ai[k * n + j];",
    "                    ar[i * n + j] -= sr * akr - si * aki;",
    "                    ai[i * n + j] -= sr * aki + si * akr;",
    "                }",
    "            }",
    "        }",
    "    }",
    "    true",
    "}",
    "",
    "// Complex forward and back substitution against a factorisation, in place on (br, bi).",
    "#[allow(dead_code)]",
    "fn md_clu_solve(n: usize, ar: &[f64], ai: &[f64], piv: &[usize], br: &mut [f64], bi: &mut [f64]) {",
    "    for k in 0..n {",
    "        let row = piv[k];",
    "        if row != k { br.swap(k, row); bi.swap(k, row); }",
    "    }",
    "    for k in 0..n {",
    "        for i in (k + 1)..n {",
    "            // b[i] -= a[i][k] * b[k]",
    "            let akr = ar[i * n + k];",
    "            let aki = ai[i * n + k];",
    "            let bkr = br[k];",
    "            let bki = bi[k];",
    "            br[i] -= akr * bkr - aki * bki;",
    "            bi[i] -= akr * bki + aki * bkr;",
    "        }",
    "    }",
    "    for k in (0..n).rev() {",
    "        for j in (k + 1)..n {",
    "            // b[k] -= a[k][j] * b[j]",
    "            let akr = ar[k * n + j];",
    "            let aki = ai[k * n + j];",
    "            let bjr = br[j];",
    "            let bji = bi[j];",
    "            br[k] -= akr * bjr - aki * bji;",
    "            bi[k] -= akr * bji + aki * bjr;",
    "        }",
    "        // b[k] /= a[k][k]",
    "        let dr = ar[k * n + k];",
    "        let di = ai[k * n + k];",
    "        let den = dr * dr + di * di;",
    "        let nr = br[k];",
    "        let ni = bi[k];",
    "        br[k] = (nr * dr + ni * di) / den;",
    "        bi[k] = (ni * dr - nr * di) / den;",
    "    }",
    "}",
    "",
)


# The method's constants, computed here from the same formulas `radau.py` uses so a
# transcription error cannot creep in, then emitted as Rust literals. Pure Python (no numpy),
# so this module stays importable without the `sim` extra, the way `cgen` does.
_S6 = 6.0**0.5
_C = [(4.0 - _S6) / 10.0, (4.0 + _S6) / 10.0, 1.0]
_E = [(-13.0 - 7.0 * _S6) / 3.0, (-13.0 + 7.0 * _S6) / 3.0, -1.0 / 3.0]
_MU_REAL = 3.0 + 3.0 ** (2.0 / 3.0) - 3.0 ** (1.0 / 3.0)
_MU_COMPLEX_R = 3.0 + 0.5 * (3.0 ** (1.0 / 3.0) - 3.0 ** (2.0 / 3.0))
_MU_COMPLEX_I = -0.5 * (3.0 ** (5.0 / 6.0) + 3.0 ** (7.0 / 6.0))
# T and TI are the published diagonalising transform `A = T L T^-1`, copied verbatim from
# `radau.py` (Hairer and Wanner's tabulated values), not recomputed.
_T = [
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1.0, 1.0, 0.0],
]
_TI = [
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492],
]
_P = [
    [13.0 / 3.0 + 7.0 * _S6 / 3.0, -23.0 / 3.0 - 22.0 * _S6 / 3.0, 10.0 / 3.0 + 5.0 * _S6],
    [13.0 / 3.0 - 7.0 * _S6 / 3.0, -23.0 / 3.0 + 22.0 * _S6 / 3.0, 10.0 / 3.0 - 5.0 * _S6],
    [1.0 / 3.0, -8.0 / 3.0, 10.0 / 3.0],
]


def _lit(value: float) -> str:
    """A Rust ``f64`` literal that round-trips the exact bits of ``value``."""
    return f"{float(value)!r}f64"


def _rrow(values: list[float]) -> str:
    return "[" + ", ".join(_lit(v) for v in values) + "]"


def _rmat(rows: list[list[float]]) -> str:
    return "[" + ", ".join(_rrow(row) for row in rows) + "]"


#: The Radau IIA constants as Rust ``const`` declarations: the collocation nodes ``C``, the
#: embedded error weights ``E``, the real and complex inverse-Butcher eigenvalues, the
#: diagonalising transforms ``T``/``TI``, and the dense-output polynomial ``P``. The step and
#: the controller read these; the test compiles them and checks each against `radau.py` to full
#: ``f64`` precision, which is the whole point of generating rather than hand-typing them.
CONSTANTS: Final = (
    f"const C: [f64; 3] = {_rrow(_C)};",
    f"const E: [f64; 3] = {_rrow(_E)};",
    f"const MU_REAL: f64 = {_lit(_MU_REAL)};",
    f"const MU_COMPLEX_R: f64 = {_lit(_MU_COMPLEX_R)};",
    f"const MU_COMPLEX_I: f64 = {_lit(_MU_COMPLEX_I)};",
    f"const T: [[f64; 3]; 3] = {_rmat(_T)};",
    f"const TI: [[f64; 3]; 3] = {_rmat(_TI)};",
    f"const P: [[f64; 3]; 3] = {_rmat(_P)};",
    "",
)


#: The step's surrounding helpers, ported from `radau.py`: the scaled RMS norm the error test
#: reads, the Gustafsson two-step step-size predictor, and the pair-factorisation that builds
#: ``mu/h * I - J`` for the real and the complex eigenvalue and factors both. `md_factor_pair`
#: composes the constants (`MU_REAL`, `MU_COMPLEX_*`) and the factored LU (`LINALG`), so a test
#: for it exercises the whole linear-algebra path a step takes before the Newton iteration.
STEP: Final = (
    "const MAX_FACTOR: f64 = 10.0;",
    "const NEWTON_MAXITER: usize = 6;",
    "",
    "// Scaled root-mean-square norm: sqrt(mean(x*x)). radau.py:273.",
    "#[allow(dead_code)]",
    "fn md_norm(x: &[f64]) -> f64 {",
    "    let mut s = 0.0;",
    "    for &v in x { s += v * v; }",
    "    (s / x.len() as f64).sqrt()",
    "}",
    "",
    "// Gustafsson's two-step step-size factor, from this error and the previous. radau.py:454.",
    "#[allow(dead_code)]",
    "fn md_predict(h: f64, h_old: f64, error: f64, error_old: f64) -> f64 {",
    "    if error == 0.0 { return MAX_FACTOR; }",
    "    let unknown = error_old < 0.0 || h_old < 0.0;",
    "    let multiplier = if unknown { 1.0 } else { h / h_old * (error_old / error).powf(0.25) };",
    "    multiplier.min(1.0) * error.powf(-0.25)",
    "}",
    "",
    "// Build mu/h*I - J for the real and complex eigenvalues and factor both. radau.py:432.",
    "#[allow(dead_code)]",
    "fn md_factor_pair(",
    "    n: usize,",
    "    jac: &[f64],",
    "    h: f64,",
    "    ar: &mut [f64],",
    "    pr: &mut [usize],",
    "    acr: &mut [f64],",
    "    aci: &mut [f64],",
    "    pc: &mut [usize],",
    ") -> bool {",
    "    let mu_real = MU_REAL / h;",
    "    let mu_complex_r = MU_COMPLEX_R / h;",
    "    let mu_complex_i = MU_COMPLEX_I / h;",
    "    for i in 0..n {",
    "        for j in 0..n {",
    "            ar[i * n + j] = -jac[i * n + j];",
    "            acr[i * n + j] = -jac[i * n + j];",
    "            aci[i * n + j] = 0.0;",
    "        }",
    "        ar[i * n + i] += mu_real;",
    "        acr[i * n + i] += mu_complex_r;",
    "        aci[i * n + i] += mu_complex_i;",
    "    }",
    "    md_lu_factor(n, ar, pr) && md_clu_factor(n, acr, aci, pc)",
    "}",
    "",
)


#: The simplified-Newton stage iteration, ported from `radau.py`'s `_newton`. It calls the
#: model's ``rhs`` by name -- the integrator and the model compile together, so `rustc` inlines
#: it, which is the whole reason `docs/ideas/backends.md` owns the compiler -- so this block is
#: kept separate from `STEP`: a program that pastes it must define ``rhs``, and the rhs-free
#: tests (norm, predict, factor_pair) do not have to.
#:
#: Simplified: the Jacobian is the one already factored in ``ar``/``ac``, reused across
#: iterations, so an iteration costs three right-hand sides and two substitutions. The loop
#: works in the eigenbasis of the Butcher matrix (``TI`` in, ``T`` out) so the ``3n`` collocation
#: system splits into one real ``n`` system and one complex ``n`` system, and it watches the
#: contraction rate, not only the residual, because a slow rate is a stale Jacobian.
NEWTON: Final = (
    "// Returns (converged, iterations, rate, rhs_calls). z holds the stage increments, 3*n flat,",
    "// and is updated in place. radau.py:286.",
    "#[allow(dead_code)]",
    "fn md_newton(",
    "    t: f64,",
    "    y: &[f64],",
    "    h: f64,",
    "    p: &[f64],",
    "    w: &mut [f64],",
    "    z: &mut [f64],",
    "    scale: &[f64],",
    "    tol: f64,",
    "    ar: &[f64],",
    "    pr: &[usize],",
    "    acr: &[f64],",
    "    aci: &[f64],",
    "    pc: &[usize],",
    ") -> (bool, usize, f64, usize) {",
    "    let n = y.len();",
    "    let mu_real = MU_REAL / h;",
    "    let mu_complex_r = MU_COMPLEX_R / h;",
    "    let mu_complex_i = MU_COMPLEX_I / h;",
    "    let mut stages = vec![0.0f64; 3 * n];",
    "    let mut point = vec![0.0f64; n];",
    "    let mut wm = vec![0.0f64; 3 * n];",
    "    let mut dw = vec![0.0f64; 3 * n];",
    "    let mut br = vec![0.0f64; n];",
    "    let mut bcr = vec![0.0f64; n];",
    "    let mut bci = vec![0.0f64; n];",
    "    // wm[i] = sum_j TI[i][j] * z[j]",
    "    for i in 0..3 {",
    "        for q in 0..n {",
    "            let mut total = 0.0;",
    "            for j in 0..3 { total += TI[i][j] * z[j * n + q]; }",
    "            wm[i * n + q] = total;",
    "        }",
    "    }",
    "    let mut calls = 0usize;",
    "    let mut rate = -1.0;",
    "    let mut previous = -1.0;",
    "    let mut converged = false;",
    "    let mut iterations = 0usize;",
    "    for k in 0..NEWTON_MAXITER {",
    "        iterations = k + 1;",
    "        for i in 0..3 {",
    "            for q in 0..n { point[q] = y[q] + z[i * n + q]; }",
    "            rhs(t + h * C[i], &point, p, w, &mut stages[i * n..(i + 1) * n]);",
    "        }",
    "        calls += 3;",
    "        let mut finite = true;",
    "        for &v in &stages { if !v.is_finite() { finite = false; break; } }",
    "        if !finite { break; }",
    "        for q in 0..n {",
    "            let mut real = 0.0;",
    "            let mut imag_r = 0.0;",
    "            let mut imag_i = 0.0;",
    "            for i in 0..3 {",
    "                real += TI[0][i] * stages[i * n + q];",
    "                imag_r += TI[1][i] * stages[i * n + q];",
    "                imag_i += TI[2][i] * stages[i * n + q];",
    "            }",
    "            br[q] = real - mu_real * wm[q];",
    "            bcr[q] = imag_r - (mu_complex_r * wm[n + q] - mu_complex_i * wm[2 * n + q]);",
    "            bci[q] = imag_i - (mu_complex_r * wm[2 * n + q] + mu_complex_i * wm[n + q]);",
    "        }",
    "        md_lu_solve(n, ar, pr, &mut br);",
    "        md_clu_solve(n, acr, aci, pc, &mut bcr, &mut bci);",
    "        for q in 0..n {",
    "            dw[q] = br[q];",
    "            dw[n + q] = bcr[q];",
    "            dw[2 * n + q] = bci[q];",
    "        }",
    "        let mut s = 0.0;",
    "        for i in 0..3 {",
    "            for q in 0..n { let e = dw[i * n + q] / scale[q]; s += e * e; }",
    "        }",
    "        let size = (s / (3.0 * n as f64)).sqrt();",
    "        if previous > 0.0 { rate = size / previous; }",
    "        if rate >= 1.0 { break; }",
    "        if rate >= 0.0 && rate.powi((NEWTON_MAXITER - k) as i32) / (1.0 - rate) * size > tol {",
    "            break;",
    "        }",
    "        for i in 0..3 { for q in 0..n { wm[i * n + q] += dw[i * n + q]; } }",
    "        for i in 0..3 {",
    "            for q in 0..n {",
    "                let mut total = 0.0;",
    "                for j in 0..3 { total += T[i][j] * wm[j * n + q]; }",
    "                z[i * n + q] = total;",
    "            }",
    "        }",
    "        if size == 0.0 || (rate >= 0.0 && rate / (1.0 - rate) * size < tol) {",
    "            converged = true;",
    "            break;",
    "        }",
    "        previous = size;",
    "    }",
    "    (converged, iterations, rate, calls)",
    "}",
    "",
)
