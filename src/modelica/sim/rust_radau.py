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
