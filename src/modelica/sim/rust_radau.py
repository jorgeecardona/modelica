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
    f"const MD_C: [f64; 3] = {_rrow(_C)};",
    f"const MD_E: [f64; 3] = {_rrow(_E)};",
    f"const MD_MU_REAL: f64 = {_lit(_MU_REAL)};",
    f"const MD_MU_COMPLEX_R: f64 = {_lit(_MU_COMPLEX_R)};",
    f"const MD_MU_COMPLEX_I: f64 = {_lit(_MU_COMPLEX_I)};",
    f"const MD_T: [[f64; 3]; 3] = {_rmat(_T)};",
    f"const MD_TI: [[f64; 3]; 3] = {_rmat(_TI)};",
    f"const MD_P: [[f64; 3]; 3] = {_rmat(_P)};",
    # The nudge that holds a fired crossing off zero for the segment after it, so the root
    # finder does not report the same instant forever. `solve.py`'s `_REARM`, same value. The
    # side depends on where the crossing settled: a latch-free `when` restarts on the true side
    # (+this), a latched relation on whichever side its latch was left holding (see `_offset`).
    "const MD_REARM: f64 = 1e-9;",
    # A latch slot in `w` reads true above this and false below, the `HELD` threshold `emit`
    # writes: a latch is a float that carries a boolean, so it is compared against a half.
    "const MD_HELD: f64 = 0.5;",
    "",
)


#: The step's surrounding helpers, ported from `radau.py`: the scaled RMS norm the error test
#: reads, the Gustafsson two-step step-size predictor, and the pair-factorisation that builds
#: ``mu/h * I - J`` for the real and the complex eigenvalue and factors both. `md_factor_pair`
#: composes the constants (`MU_REAL`, `MU_COMPLEX_*`) and the factored LU (`LINALG`), so a test
#: for it exercises the whole linear-algebra path a step takes before the Newton iteration.
STEP: Final = (
    "const MAX_FACTOR: f64 = 10.0;",
    "const MIN_FACTOR: f64 = 0.2;",
    "const NEWTON_MAXITER: usize = 6;",
    "const NEWTON_ORDER: f64 = 3.0;",
    "const TRIVIAL: f64 = 1e-5;",
    "const NEGLIGIBLE: f64 = 1e-15;",
    "const STALE_ITERATIONS: usize = 2;",
    "const STALE_RATE: f64 = 1e-3;",
    "const WORTH_CHANGING: f64 = 1.2;",
    "const TOO_SMALL: i32 = 1;",
    "const NO_CONVERGENCE: i32 = 2;",
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
    "    let mu_real = MD_MU_REAL / h;",
    "    let mu_complex_r = MD_MU_COMPLEX_R / h;",
    "    let mu_complex_i = MD_MU_COMPLEX_I / h;",
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
    "    let mu_real = MD_MU_REAL / h;",
    "    let mu_complex_r = MD_MU_COMPLEX_R / h;",
    "    let mu_complex_i = MD_MU_COMPLEX_I / h;",
    "    let mut stages = vec![0.0f64; 3 * n];",
    "    let mut point = vec![0.0f64; n];",
    "    let mut wm = vec![0.0f64; 3 * n];",
    "    let mut dw = vec![0.0f64; 3 * n];",
    "    let mut br = vec![0.0f64; n];",
    "    let mut bcr = vec![0.0f64; n];",
    "    let mut bci = vec![0.0f64; n];",
    "    // wm[i] = sum_j MD_TI[i][j] * z[j]",
    "    for i in 0..3 {",
    "        for q in 0..n {",
    "            let mut total = 0.0;",
    "            for j in 0..3 { total += MD_TI[i][j] * z[j * n + q]; }",
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
    "            rhs(t + h * MD_C[i], &point, p, w, &mut stages[i * n..(i + 1) * n]);",
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
    "                real += MD_TI[0][i] * stages[i * n + q];",
    "                imag_r += MD_TI[1][i] * stages[i * n + q];",
    "                imag_i += MD_TI[2][i] * stages[i * n + q];",
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
    "                for j in 0..3 { total += MD_T[i][j] * wm[j * n + q]; }",
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


#: Hairer and Wanner's starting-step heuristic, ported from `radau.py`'s `_first_step`: one
#: explicit-Euler probe, then a curvature guess. Calls ``rhs`` once, so it is rhs-dependent like
#: `NEWTON`. Getting it wrong is not fatal -- the controller recovers -- but a first step far too
#: large is rejected repeatedly where no Jacobian has been reused yet, the most expensive place.
FIRST_STEP: Final = (
    "// The starting step size. radau.py:393.",
    "#[allow(dead_code)]",
    "fn md_first_step(",
    "    t0: f64,",
    "    y: &[f64],",
    "    p: &[f64],",
    "    w: &mut [f64],",
    "    f0: &[f64],",
    "    rtol: f64,",
    "    atol: f64,",
    "    span: f64,",
    ") -> f64 {",
    "    let n = y.len();",
    "    let mut scale = vec![0.0f64; n];",
    "    for q in 0..n { scale[q] = atol + rtol * y[q].abs(); }",
    "    let mut tmp = vec![0.0f64; n];",
    "    for q in 0..n { tmp[q] = y[q] / scale[q]; }",
    "    let d0 = md_norm(&tmp);",
    "    for q in 0..n { tmp[q] = f0[q] / scale[q]; }",
    "    let d1 = md_norm(&tmp);",
    "    let h0 = if d0 < TRIVIAL || d1 < TRIVIAL { 1e-6 } else { 0.01 * d0 / d1 };",
    "    let mut probe = vec![0.0f64; n];",
    "    for q in 0..n { probe[q] = y[q] + h0 * f0[q]; }",
    "    let mut slope = vec![0.0f64; n];",
    "    rhs(t0 + h0, &probe, p, w, &mut slope);",
    "    for q in 0..n { tmp[q] = (slope[q] - f0[q]) / scale[q]; }",
    "    let d2 = md_norm(&tmp) / h0;",
    "    let biggest = d1.max(d2);",
    "    let h1 = if biggest <= NEGLIGIBLE {",
    "        (1e-6f64).max(h0 * 1e-3)",
    "    } else {",
    "        (0.01 / biggest).powf(1.0 / (NEWTON_ORDER + 1.0))",
    "    };",
    "    (100.0 * h0).min(h1).min(span)",
    "}",
    "",
)


#: The Radau IIA driver loop, ported from `radau.py`'s `integrate`, with the event handling
#: `solve.py`'s `_Run.run` wraps around SciPy folded in. It calls ``rhs``, ``jacobian``,
#: ``indicators`` and ``transition`` by name -- the integrator and the model compile together,
#: which is the whole point -- and ties together every piece above: the starting step, the pair
#: factorisation, the Newton solve, the embedded error estimate, Gustafsson step control,
#: stale-Jacobian detection, and dense output through ``P``. Over each accepted step it also
#: watches the indicators for a crossing, locates the earliest with `md_locate`, applies the
#: model's transition (iterating over any latch it makes stale, `solve.py`'s `_settle`) and
#: restarts. One long function on purpose, as `radau.py:489` argues: splitting the controller
#: from the step would put a call on the hot path for every quantity they pass. It handles a
#: single state set -- the subset `rustgen` emits a transition for -- with ``when`` branches and
#: latched relations both, so there is no mode switch to rebase into; the rearm nudge is
#: +``MD_REARM`` for a `when` and the side the latch settled on for a relation. It fills ``out``
#: (grid.len() by n, row-major) and returns the run's counters, events included.
_INTEGRATE_SRC = """\
// Locate, along one accepted step's dense polynomial, where indicator `k` crosses zero.
// The step is y(x) = y + x*(q0 + x*(q1 + x*q2)) for x in [0, 1], and g(x) = z_k(y(x)) + nudge
// is the crossing function the loop watches. Called only when the segment ends bracket a
// directional crossing, so g(0) and g(1) straddle zero and plain bisection converges to the
// root -- the same instant `solve.py` gets from SciPy's brentq, to f64 precision in 60 halvings.
#[allow(dead_code)]
fn md_locate(
    k: usize,
    t: f64,
    h: f64,
    y: &[f64],
    quad: &[f64],
    p: &[f64],
    w: &mut [f64],
    nudge: f64,
    g0: f64,
    ev: &mut [f64],
    z: &mut [f64],
) -> f64 {
    if g0 == 0.0 {
        return 0.0;
    }
    let n = y.len();
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        for q in 0..n {
            ev[q] = y[q] + mid * (quad[q * 3] + mid * (quad[q * 3 + 1] + mid * quad[q * 3 + 2]));
        }
        indicators(t + mid * h, ev, p, w, z);
        let gm = z[k] + nudge;
        if (gm < 0.0) == (g0 < 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[allow(dead_code)]
fn md_integrate(
    t0: f64,
    t1: f64,
    y0: &[f64],
    p: &[f64],
    w: &mut [f64],
    rtol: f64,
    atol: f64,
    directions: &[f64],
    latches: &[i64],
    grid: &[f64],
    out: &mut [f64],
) -> (i32, usize, usize, usize, usize, usize, usize, f64) {
    let n = y0.len();
    let mut y = y0.to_vec();
    let mut t = t0;
    let mut f = vec![0.0f64; n];
    rhs(t, &y, p, w, &mut f);
    let mut evaluations = 1usize;

    let mut jac = vec![0.0f64; n * n];
    jacobian(t, &y, p, w, &mut jac);
    let mut jacobians = 1usize;
    let mut fresh = true;

    let mut ar = vec![0.0f64; n * n];
    let mut acr = vec![0.0f64; n * n];
    let mut aci = vec![0.0f64; n * n];
    let mut pr = vec![0usize; n];
    let mut pc = vec![0usize; n];
    let mut factored = false;
    let mut factorisations = 0usize;

    let mut z = vec![0.0f64; 3 * n];
    let mut quad = vec![0.0f64; n * 3];
    let mut y_poly = vec![0.0f64; n];
    let mut t_poly = t0;
    let mut h_poly = 1.0;
    let mut extrapolate = false;
    let mut y_new = vec![0.0f64; n];
    let mut f_new = vec![0.0f64; n];
    let mut error = vec![0.0f64; n];
    let mut combined = vec![0.0f64; n];
    let mut scale = vec![0.0f64; n];
    let mut scaled = vec![0.0f64; n];

    // Event bookkeeping: the crossing values at the two ends of a step, a scratch state and
    // indicator vector for the bisection, the per-event rearm nudge, and the count. Empty for a
    // model with no indicators, so its stepping stays exactly what it was before events landed.
    let nevents = directions.len();
    let mut z0 = vec![0.0f64; nevents];
    let mut z1 = vec![0.0f64; nevents];
    let mut z_ev = vec![0.0f64; nevents];
    let mut ev_y = vec![0.0f64; n];
    let mut rearm = vec![0.0f64; nevents];
    let mut fired_flags = vec![false; nevents];
    let mut events_total = 0usize;

    let points = grid.len();
    let mut cursor = 0usize;
    while cursor < points && grid[cursor] <= t0 {
        for q in 0..n { out[cursor * n + q] = y[q]; }
        cursor += 1;
    }

    let tol = (10.0 * f64::EPSILON / rtol).max((0.03f64).min(rtol.sqrt()));
    let mut h_prop = md_first_step(t0, &y, p, w, &f, rtol, atol, t1 - t0);
    evaluations += 1;
    let mut h_old = -1.0;
    let mut error_old = -1.0;

    let mut steps = 0usize;
    let mut rejected_total = 0usize;
    let mut status = 0i32;
    while t < t1 {
        let floor = 10.0 * (t.next_up() - t).abs();
        let mut h_abs = h_prop;
        let mut h_previous = h_old;
        let mut error_previous = error_old;
        if h_abs < floor {
            h_abs = floor;
            h_previous = -1.0;
            error_previous = -1.0;
        }

        let mut rejected = false;
        let mut accepted = false;
        let mut iterations = 0usize;
        let mut rate = -1.0;
        let mut error_norm = 0.0;
        let mut h = h_abs;
        let mut t_new = t;
        while !accepted {
            if h_abs < floor { status = TOO_SMALL; break; }
            h = h_abs;
            t_new = (t + h).min(t1);
            h = t_new - t;
            h_abs = h;

            for i in 0..3 {
                if extrapolate {
                    let x = (t + h * MD_C[i] - t_poly) / h_poly;
                    for q in 0..n {
                        z[i * n + q] = y_poly[q]
                            + x * (quad[q * 3] + x * (quad[q * 3 + 1] + x * quad[q * 3 + 2]))
                            - y[q];
                    }
                } else {
                    for q in 0..n { z[i * n + q] = 0.0; }
                }
            }
            for q in 0..n { scale[q] = atol + rtol * y[q].abs(); }

            let mut converged = false;
            while !converged {
                if !factored {
                    if !md_factor_pair(n, &jac, h, &mut ar, &mut pr, &mut acr, &mut aci, &mut pc) {
                        status = NO_CONVERGENCE;
                        break;
                    }
                    factored = true;
                    factorisations += 1;
                }
                let result =
                    md_newton(t, &y, h, p, w, &mut z, &scale, tol, &ar, &pr, &acr, &aci, &pc);
                converged = result.0;
                iterations = result.1;
                rate = result.2;
                evaluations += result.3;
                if !converged {
                    if fresh { break; }
                    for e in jac.iter_mut() { *e = 0.0; }
                    jacobian(t, &y, p, w, &mut jac);
                    jacobians += 1;
                    fresh = true;
                    factored = false;
                }
            }
            if status != 0 { break; }
            if !converged {
                h_abs *= 0.5;
                factored = false;
                rejected = true;
                continue;
            }

            for q in 0..n {
                y_new[q] = y[q] + z[2 * n + q];
                combined[q] = f[q] + (MD_E[0] * z[q] + MD_E[1] * z[n + q] + MD_E[2] * z[2 * n + q]) / h;
                error[q] = combined[q];
            }
            md_lu_solve(n, &ar, &pr, &mut error);
            for q in 0..n { scale[q] = atol + rtol * y[q].abs().max(y_new[q].abs()); }
            for q in 0..n { scaled[q] = error[q] / scale[q]; }
            error_norm = md_norm(&scaled);
            let safety =
                0.9 * (2 * NEWTON_MAXITER + 1) as f64 / (2 * NEWTON_MAXITER + iterations) as f64;

            if rejected && error_norm > 1.0 {
                for q in 0..n { y_new[q] = y[q] + error[q]; }
                rhs(t, &y_new, p, w, &mut f_new);
                evaluations += 1;
                for q in 0..n { error[q] = f_new[q] + combined[q] - f[q]; }
                md_lu_solve(n, &ar, &pr, &mut error);
                for q in 0..n { scaled[q] = error[q] / scale[q]; }
                error_norm = md_norm(&scaled);
                for q in 0..n { y_new[q] = y[q] + z[2 * n + q]; }
            }

            if error_norm > 1.0 {
                let factor = md_predict(h_abs, h_previous, error_norm, error_previous);
                h_abs *= MIN_FACTOR.max(safety * factor);
                factored = false;
                rejected = true;
                rejected_total += 1;
            } else {
                accepted = true;
            }
        }
        if !accepted { break; }

        steps += 1;
        for q in 0..n {
            for j in 0..3 {
                let mut total = 0.0;
                for i in 0..3 { total += z[i * n + q] * MD_P[i][j]; }
                quad[q * 3 + j] = total;
            }
        }

        // Did an indicator cross within this step? Compare the crossing functions at the two
        // ends (x = 0 is y, x = 1 is y_new), in each event's own direction, and locate the
        // earliest root. `find_active_events` in SciPy, spelled out: an up-crossing is watched
        // by a rising event, a down-crossing by a falling one, either by an equation relation,
        // and an indicator that is zero at both ends is not a crossing at all.
        let mut fired: isize = -1;
        let mut fired_x = 1.0f64;
        if nevents > 0 {
            indicators(t, &y, p, w, &mut z0);
            indicators(t_new, &y_new, p, w, &mut z1);
            for k in 0..nevents {
                let g0 = z0[k] + rearm[k];
                let g1 = z1[k] + rearm[k];
                let up = g0 <= 0.0 && g1 >= 0.0;
                let down = g0 >= 0.0 && g1 <= 0.0;
                let active = ((directions[k] > 0.0 && up)
                    || (directions[k] < 0.0 && down)
                    || (directions[k] == 0.0 && (up || down)))
                    && !(g0 == 0.0 && g1 == 0.0);
                if !active {
                    continue;
                }
                let x = md_locate(k, t, h, &y, &quad, p, w, rearm[k], g0, &mut ev_y, &mut z_ev);
                if x < fired_x {
                    fired_x = x;
                    fired = k as isize;
                }
            }
        }

        if fired >= 0 {
            // The instant one crossing fires. Sample the grid up to it from this segment's own
            // polynomial (those points belong to the pre-event trajectory), take the state at
            // the root, apply the model's transition, hold the fired crossing off its zero, and
            // restart the stepper there: state moved discontinuously, so the old Jacobian, the
            // extrapolation polynomial and the step-size history are all stale.
            let te = t + fired_x * h;
            while cursor < points && grid[cursor] <= te {
                let x = (grid[cursor] - t) / h;
                for q in 0..n {
                    out[cursor * n + q] =
                        y[q] + x * (quad[q * 3] + x * (quad[q * 3 + 1] + x * quad[q * 3 + 2]));
                }
                cursor += 1;
            }
            for q in 0..n {
                ev_y[q] = y[q]
                    + fired_x * (quad[q * 3] + fired_x * (quad[q * 3 + 1] + fired_x * quad[q * 3 + 2]));
            }

            // Modelica's event iteration (`solve.py`'s `_settle`): apply the crossing that
            // fired, then everything its latch flip made stale. A latched relation holds its
            // value between events, so flipping one latch can leave another on the wrong side of
            // its own -- and the integrator would never catch that second one, since on the next
            // segment its indicator starts already past zero instead of crossing it. So the
            // instant is not over until no latch disagrees with the relation it holds. An event
            // acts at most once per instant (`fired_flags`), which is what makes this terminate.
            for flag in fired_flags.iter_mut() { *flag = false; }
            let mut pending: isize = fired;
            while pending >= 0 {
                let e = pending as usize;
                transition(e, te, &ev_y, p, w, &mut y_new);
                for q in 0..n { ev_y[q] = y_new[q]; }
                fired_flags[e] = true;
                events_total += 1;
                pending = -1;
                indicators(te, &ev_y, p, w, &mut z_ev);
                for j in 0..nevents {
                    if latches[j] < 0 || fired_flags[j] {
                        continue;
                    }
                    let held = w[latches[j] as usize] > MD_HELD;
                    if (z_ev[j] > 0.0) != held {
                        pending = j as isize;
                        break;
                    }
                }
            }
            // Rearm every crossing that fired, on the side it settled: a latch-free `when` on the
            // true side (+), a latched relation on whichever side its latch was left holding.
            for e in rearm.iter_mut() { *e = 0.0; }
            for j in 0..nevents {
                if !fired_flags[j] {
                    continue;
                }
                rearm[j] = if latches[j] < 0 || w[latches[j] as usize] > MD_HELD {
                    MD_REARM
                } else {
                    -MD_REARM
                };
            }
            t = te;
            for q in 0..n { y[q] = ev_y[q]; }
            rhs(t, &y, p, w, &mut f);
            evaluations += 1;
            for e in jac.iter_mut() { *e = 0.0; }
            jacobian(t, &y, p, w, &mut jac);
            jacobians += 1;
            fresh = true;
            factored = false;
            extrapolate = false;
            h_prop = md_first_step(t, &y, p, w, &f, rtol, atol, t1 - t);
            evaluations += 1;
            h_old = -1.0;
            error_old = -1.0;
            continue;
        }

        while cursor < points && grid[cursor] <= t_new {
            let x = (grid[cursor] - t) / h;
            for q in 0..n {
                out[cursor * n + q] =
                    y[q] + x * (quad[q * 3] + x * (quad[q * 3 + 1] + x * quad[q * 3 + 2]));
            }
            cursor += 1;
        }

        let safety =
            0.9 * (2 * NEWTON_MAXITER + 1) as f64 / (2 * NEWTON_MAXITER + iterations) as f64;
        let again = iterations > STALE_ITERATIONS && rate > STALE_RATE;
        let mut factor =
            MAX_FACTOR.min(safety * md_predict(h_abs, h_previous, error_norm, error_previous));
        if !again && factor < WORTH_CHANGING {
            factor = 1.0;
        } else {
            factored = false;
        }

        rhs(t_new, &y_new, p, w, &mut f_new);
        evaluations += 1;
        if again {
            for e in jac.iter_mut() { *e = 0.0; }
            jacobian(t_new, &y_new, p, w, &mut jac);
            jacobians += 1;
            fresh = true;
        } else {
            fresh = false;
        }

        h_old = h_prop;
        error_old = error_norm;
        h_prop = h_abs * factor;
        for q in 0..n { y_poly[q] = y[q]; }
        t_poly = t;
        h_poly = h;
        extrapolate = true;
        t = t_new;
        for q in 0..n { y[q] = y_new[q]; f[q] = f_new[q]; }
    }

    while cursor < points {
        for q in 0..n { out[cursor * n + q] = y[q]; }
        cursor += 1;
    }
    (status, steps, rejected_total, evaluations, jacobians, factorisations, events_total, t)
}
"""

INTEGRATE: Final = (*_INTEGRATE_SRC.splitlines(), "")
