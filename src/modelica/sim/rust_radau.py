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

#: A dense complex LU solve with partial pivoting, the complex twin of ``md_solve``.
#:
#: Rust's standard library has no complex type, and a crate for one would be the external
#: dependency this backend exists to avoid, so a complex value is a pair of ``f64`` and the
#: arithmetic is written out. The matrix is two flat ``n*n`` slices (real and imaginary parts,
#: row-major) and each vector is a pair of ``n`` slices; the shape mirrors ``md_solve`` exactly
#: -- the same pivoting, the same one-pass elimination, the same back-substitution -- so the
#: two read as one routine in two number systems. Partial pivoting compares squared magnitude
#: ``re*re + im*im``, which orders the same as ``abs`` and needs no ``sqrt``.
COMPLEX_SOLVE: Final = (
    "// Dense complex LU with partial pivoting: the complex twin of md_solve, with a complex",
    "// value carried as a pair of f64 so no crate is needed. a is (ar, ai) row-major n*n; the",
    "// right-hand side (br, bi) is overwritten during elimination; the answer lands in (xr, xi).",
    "#[allow(dead_code)]",
    "fn md_csolve(",
    "    n: usize,",
    "    ar: &mut [f64],",
    "    ai: &mut [f64],",
    "    br: &mut [f64],",
    "    bi: &mut [f64],",
    "    xr: &mut [f64],",
    "    xi: &mut [f64],",
    ") {",
    "    for k in 0..n {",
    "        let mut pivot = k;",
    "        let mut best = ar[k * n + k] * ar[k * n + k] + ai[k * n + k] * ai[k * n + k];",
    "        for i in (k + 1)..n {",
    "            let m = ar[i * n + k] * ar[i * n + k] + ai[i * n + k] * ai[i * n + k];",
    "            if m > best { best = m; pivot = i; }",
    "        }",
    "        if pivot != k {",
    "            for j in 0..n {",
    "                ar.swap(k * n + j, pivot * n + j);",
    "                ai.swap(k * n + j, pivot * n + j);",
    "            }",
    "            br.swap(k, pivot);",
    "            bi.swap(k, pivot);",
    "        }",
    "        let dr = ar[k * n + k];",
    "        let di = ai[k * n + k];",
    "        let den = dr * dr + di * di;",
    "        for i in (k + 1)..n {",
    "            // factor = a[i][k] / a[k][k]",
    "            let nr = ar[i * n + k];",
    "            let ni = ai[i * n + k];",
    "            let fr = (nr * dr + ni * di) / den;",
    "            let fi = (ni * dr - nr * di) / den;",
    "            ar[i * n + k] = 0.0;",
    "            ai[i * n + k] = 0.0;",
    "            for j in (k + 1)..n {",
    "                // a[i][j] -= factor * a[k][j]",
    "                let akr = ar[k * n + j];",
    "                let aki = ai[k * n + j];",
    "                ar[i * n + j] -= fr * akr - fi * aki;",
    "                ai[i * n + j] -= fr * aki + fi * akr;",
    "            }",
    "            // b[i] -= factor * b[k]",
    "            let bkr = br[k];",
    "            let bki = bi[k];",
    "            br[i] -= fr * bkr - fi * bki;",
    "            bi[i] -= fr * bki + fi * bkr;",
    "        }",
    "    }",
    "    for i in (0..n).rev() {",
    "        let mut sr = br[i];",
    "        let mut si = bi[i];",
    "        for j in (i + 1)..n {",
    "            // s -= a[i][j] * x[j]",
    "            let ajr = ar[i * n + j];",
    "            let aji = ai[i * n + j];",
    "            let xjr = xr[j];",
    "            let xji = xi[j];",
    "            sr -= ajr * xjr - aji * xji;",
    "            si -= ajr * xji + aji * xjr;",
    "        }",
    "        // x[i] = s / a[i][i]",
    "        let dr = ar[i * n + i];",
    "        let di = ai[i * n + i];",
    "        let den = dr * dr + di * di;",
    "        xr[i] = (sr * dr + si * di) / den;",
    "        xi[i] = (si * dr - sr * di) / den;",
    "    }",
    "}",
    "",
)
