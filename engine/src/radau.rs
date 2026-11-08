// Generated from modelica.sim.rust_radau by vm.write_runtime(). Do not edit.
const MD_C: [f64; 3] = [0.15505102572168222f64, 0.6449489742783178f64, 1.0f64];
const MD_E: [f64; 3] = [-10.048809399827414f64, 1.382142733160748f64, -0.3333333333333333f64];
const MD_MU_REAL: f64 = 3.637834252744496f64;
const MD_MU_COMPLEX_R: f64 = 2.6810828736277523f64;
const MD_MU_COMPLEX_I: f64 = -3.050430199247411f64;
const MD_T: [[f64; 3]; 3] = [[0.09443876248897524f64, -0.1412552950209542f64, 0.03002919410514742f64], [0.2502131229653333f64, 0.20412935229379994f64, -0.3829421127572619f64], [1.0f64, 1.0f64, 0.0f64]];
const MD_TI: [[f64; 3]; 3] = [[4.178718591551904f64, 0.32768282076106237f64, 0.5233764454994495f64], [-4.178718591551904f64, -0.32768282076106237f64, 0.47662355450055044f64], [0.5028726349457868f64, -2.571926949855605f64, 0.5960392048282249f64]];
const MD_P: [[f64; 3]; 3] = [[10.048809399827414f64, -25.62959144707664f64, 15.580782047249224f64], [-1.382142733160748f64, 10.296258113743303f64, -8.914115380582556f64], [0.3333333333333333f64, -2.6666666666666665f64, 3.3333333333333335f64]];
const MD_REARM: f64 = 1e-9;
const MD_HELD: f64 = 0.5;
const MD_STANDSTILL: f64 = 1e-4;
const MD_CHATTER: usize = 100;

// Real LU with partial pivoting: overwrite a with its factors, record the pivots in piv.
// False when a column is exactly zero (a singular matrix).
#[allow(dead_code)]
fn md_lu_factor(n: usize, a: &mut [f64], piv: &mut [usize]) -> bool {
    for k in 0..n {
        let mut big = 0.0;
        let mut row = k;
        for i in k..n {
            let size = a[i * n + k].abs();
            if size > big { big = size; row = i; }
        }
        piv[k] = row;
        if big == 0.0 { return false; }
        if row != k {
            for j in 0..n { a.swap(k * n + j, row * n + j); }
        }
        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let share = a[i * n + k] / pivot;
            a[i * n + k] = share;
            if share != 0.0 {
                for j in (k + 1)..n { a[i * n + j] -= share * a[k * n + j]; }
            }
        }
    }
    true
}

// Forward and back substitution against a factorisation, in place on b. Every pivot swap
// first, then the substitution -- see radau.py's _solve for why interleaving is wrong.
#[allow(dead_code)]
fn md_lu_solve(n: usize, a: &[f64], piv: &[usize], b: &mut [f64]) {
    for k in 0..n {
        let row = piv[k];
        if row != k { b.swap(k, row); }
    }
    for k in 0..n {
        for i in (k + 1)..n { b[i] -= a[i * n + k] * b[k]; }
    }
    for k in (0..n).rev() {
        for j in (k + 1)..n { b[k] -= a[k * n + j] * b[j]; }
        b[k] /= a[k * n + k];
    }
}

// Complex LU with partial pivoting: the matrix is (ar, ai) row-major, a complex value a
// pair of f64. Pivots by squared magnitude. Same structure as md_lu_factor.
#[allow(dead_code)]
fn md_clu_factor(n: usize, ar: &mut [f64], ai: &mut [f64], piv: &mut [usize]) -> bool {
    for k in 0..n {
        let mut big = 0.0;
        let mut row = k;
        for i in k..n {
            let size = ar[i * n + k] * ar[i * n + k] + ai[i * n + k] * ai[i * n + k];
            if size > big { big = size; row = i; }
        }
        piv[k] = row;
        if big == 0.0 { return false; }
        if row != k {
            for j in 0..n {
                ar.swap(k * n + j, row * n + j);
                ai.swap(k * n + j, row * n + j);
            }
        }
        let pr = ar[k * n + k];
        let pi = ai[k * n + k];
        let den = pr * pr + pi * pi;
        for i in (k + 1)..n {
            // share = a[i][k] / a[k][k]
            let nr = ar[i * n + k];
            let ni = ai[i * n + k];
            let sr = (nr * pr + ni * pi) / den;
            let si = (ni * pr - nr * pi) / den;
            ar[i * n + k] = sr;
            ai[i * n + k] = si;
            if sr != 0.0 || si != 0.0 {
                for j in (k + 1)..n {
                    // a[i][j] -= share * a[k][j]
                    let akr = ar[k * n + j];
                    let aki = ai[k * n + j];
                    ar[i * n + j] -= sr * akr - si * aki;
                    ai[i * n + j] -= sr * aki + si * akr;
                }
            }
        }
    }
    true
}

// Complex forward and back substitution against a factorisation, in place on (br, bi).
#[allow(dead_code)]
fn md_clu_solve(n: usize, ar: &[f64], ai: &[f64], piv: &[usize], br: &mut [f64], bi: &mut [f64]) {
    for k in 0..n {
        let row = piv[k];
        if row != k { br.swap(k, row); bi.swap(k, row); }
    }
    for k in 0..n {
        for i in (k + 1)..n {
            // b[i] -= a[i][k] * b[k]
            let akr = ar[i * n + k];
            let aki = ai[i * n + k];
            let bkr = br[k];
            let bki = bi[k];
            br[i] -= akr * bkr - aki * bki;
            bi[i] -= akr * bki + aki * bkr;
        }
    }
    for k in (0..n).rev() {
        for j in (k + 1)..n {
            // b[k] -= a[k][j] * b[j]
            let akr = ar[k * n + j];
            let aki = ai[k * n + j];
            let bjr = br[j];
            let bji = bi[j];
            br[k] -= akr * bjr - aki * bji;
            bi[k] -= akr * bji + aki * bjr;
        }
        // b[k] /= a[k][k]
        let dr = ar[k * n + k];
        let di = ai[k * n + k];
        let den = dr * dr + di * di;
        let nr = br[k];
        let ni = bi[k];
        br[k] = (nr * dr + ni * di) / den;
        bi[k] = (ni * dr - nr * di) / den;
    }
}

const MAX_FACTOR: f64 = 10.0;
const MIN_FACTOR: f64 = 0.2;
const NEWTON_MAXITER: usize = 6;
const NEWTON_ORDER: f64 = 3.0;
const TRIVIAL: f64 = 1e-5;
const NEGLIGIBLE: f64 = 1e-15;
const STALE_ITERATIONS: usize = 2;
const STALE_RATE: f64 = 1e-3;
const WORTH_CHANGING: f64 = 1.2;
const TOO_SMALL: i32 = 1;
const NO_CONVERGENCE: i32 = 2;
const CHATTERING: i32 = 3;
const TERMINATED: i32 = 4;

// Scaled root-mean-square norm: sqrt(mean(x*x)). radau.py:273.
#[allow(dead_code)]
fn md_norm(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for &v in x { s += v * v; }
    (s / x.len() as f64).sqrt()
}

// Gustafsson's two-step step-size factor, from this error and the previous. radau.py:454.
#[allow(dead_code)]
fn md_predict(h: f64, h_old: f64, error: f64, error_old: f64) -> f64 {
    if error == 0.0 { return MAX_FACTOR; }
    let unknown = error_old < 0.0 || h_old < 0.0;
    let multiplier = if unknown { 1.0 } else { h / h_old * (error_old / error).powf(0.25) };
    multiplier.min(1.0) * error.powf(-0.25)
}

// Build mu/h*I - J for the real and complex eigenvalues and factor both. radau.py:432.
#[allow(dead_code)]
fn md_factor_pair(
    n: usize,
    jac: &[f64],
    h: f64,
    ar: &mut [f64],
    pr: &mut [usize],
    acr: &mut [f64],
    aci: &mut [f64],
    pc: &mut [usize],
) -> bool {
    let mu_real = MD_MU_REAL / h;
    let mu_complex_r = MD_MU_COMPLEX_R / h;
    let mu_complex_i = MD_MU_COMPLEX_I / h;
    for i in 0..n {
        for j in 0..n {
            ar[i * n + j] = -jac[i * n + j];
            acr[i * n + j] = -jac[i * n + j];
            aci[i * n + j] = 0.0;
        }
        ar[i * n + i] += mu_real;
        acr[i * n + i] += mu_complex_r;
        aci[i * n + i] += mu_complex_i;
    }
    md_lu_factor(n, ar, pr) && md_clu_factor(n, acr, aci, pc)
}

// Returns (converged, iterations, rate, rhs_calls). z holds the stage increments, 3*n flat,
// and is updated in place. radau.py:286.
#[allow(dead_code)]
fn md_newton(
    t: f64,
    y: &[f64],
    h: f64,
    p: &[f64],
    w: &mut [f64],
    z: &mut [f64],
    scale: &[f64],
    tol: f64,
    ar: &[f64],
    pr: &[usize],
    acr: &[f64],
    aci: &[f64],
    pc: &[usize],
) -> (bool, usize, f64, usize) {
    let n = y.len();
    let mu_real = MD_MU_REAL / h;
    let mu_complex_r = MD_MU_COMPLEX_R / h;
    let mu_complex_i = MD_MU_COMPLEX_I / h;
    let mut stages = vec![0.0f64; 3 * n];
    let mut point = vec![0.0f64; n];
    let mut wm = vec![0.0f64; 3 * n];
    let mut dw = vec![0.0f64; 3 * n];
    let mut br = vec![0.0f64; n];
    let mut bcr = vec![0.0f64; n];
    let mut bci = vec![0.0f64; n];
    // wm[i] = sum_j MD_TI[i][j] * z[j]
    for i in 0..3 {
        for q in 0..n {
            let mut total = 0.0;
            for j in 0..3 { total += MD_TI[i][j] * z[j * n + q]; }
            wm[i * n + q] = total;
        }
    }
    let mut calls = 0usize;
    let mut rate = -1.0;
    let mut previous = -1.0;
    let mut converged = false;
    let mut iterations = 0usize;
    for k in 0..NEWTON_MAXITER {
        iterations = k + 1;
        for i in 0..3 {
            for q in 0..n { point[q] = y[q] + z[i * n + q]; }
            rhs(t + h * MD_C[i], &point, p, w, &mut stages[i * n..(i + 1) * n]);
        }
        calls += 3;
        let mut finite = true;
        for &v in &stages { if !v.is_finite() { finite = false; break; } }
        if !finite { break; }
        for q in 0..n {
            let mut real = 0.0;
            let mut imag_r = 0.0;
            let mut imag_i = 0.0;
            for i in 0..3 {
                real += MD_TI[0][i] * stages[i * n + q];
                imag_r += MD_TI[1][i] * stages[i * n + q];
                imag_i += MD_TI[2][i] * stages[i * n + q];
            }
            br[q] = real - mu_real * wm[q];
            bcr[q] = imag_r - (mu_complex_r * wm[n + q] - mu_complex_i * wm[2 * n + q]);
            bci[q] = imag_i - (mu_complex_r * wm[2 * n + q] + mu_complex_i * wm[n + q]);
        }
        md_lu_solve(n, ar, pr, &mut br);
        md_clu_solve(n, acr, aci, pc, &mut bcr, &mut bci);
        for q in 0..n {
            dw[q] = br[q];
            dw[n + q] = bcr[q];
            dw[2 * n + q] = bci[q];
        }
        let mut s = 0.0;
        for i in 0..3 {
            for q in 0..n { let e = dw[i * n + q] / scale[q]; s += e * e; }
        }
        let size = (s / (3.0 * n as f64)).sqrt();
        if previous > 0.0 { rate = size / previous; }
        if rate >= 1.0 { break; }
        if rate >= 0.0 && rate.powi((NEWTON_MAXITER - k) as i32) / (1.0 - rate) * size > tol {
            break;
        }
        for i in 0..3 { for q in 0..n { wm[i * n + q] += dw[i * n + q]; } }
        for i in 0..3 {
            for q in 0..n {
                let mut total = 0.0;
                for j in 0..3 { total += MD_T[i][j] * wm[j * n + q]; }
                z[i * n + q] = total;
            }
        }
        if size == 0.0 || (rate >= 0.0 && rate / (1.0 - rate) * size < tol) {
            converged = true;
            break;
        }
        previous = size;
    }
    (converged, iterations, rate, calls)
}

// The starting step size. radau.py:393.
#[allow(dead_code)]
fn md_first_step(
    t0: f64,
    y: &[f64],
    p: &[f64],
    w: &mut [f64],
    f0: &[f64],
    rtol: f64,
    atol: f64,
    span: f64,
) -> f64 {
    let n = y.len();
    let mut scale = vec![0.0f64; n];
    for q in 0..n { scale[q] = atol + rtol * y[q].abs(); }
    let mut tmp = vec![0.0f64; n];
    for q in 0..n { tmp[q] = y[q] / scale[q]; }
    let d0 = md_norm(&tmp);
    for q in 0..n { tmp[q] = f0[q] / scale[q]; }
    let d1 = md_norm(&tmp);
    let h0 = if d0 < TRIVIAL || d1 < TRIVIAL { 1e-6 } else { 0.01 * d0 / d1 };
    let mut probe = vec![0.0f64; n];
    for q in 0..n { probe[q] = y[q] + h0 * f0[q]; }
    let mut slope = vec![0.0f64; n];
    rhs(t0 + h0, &probe, p, w, &mut slope);
    for q in 0..n { tmp[q] = (slope[q] - f0[q]) / scale[q]; }
    let d2 = md_norm(&tmp) / h0;
    let biggest = d1.max(d2);
    let h1 = if biggest <= NEGLIGIBLE {
        (1e-6f64).max(h0 * 1e-3)
    } else {
        (0.01 / biggest).powf(1.0 / (NEWTON_ORDER + 1.0))
    };
    (100.0 * h0).min(h1).min(span)
}

// Locate, along one accepted step's dense polynomial, where indicator `k` crosses zero.
// The step is y(x) = y + x*(q0 + x*(q1 + x*q2)) for x in [0, 1], and g(x) = z_k(y(x)) + nudge
// is the crossing function the loop watches. Called only when the segment ends bracket a
// directional crossing, so g(0) and g(1) straddle zero and plain bisection converges to the
// root -- the same instant `solve.py` gets from SciPy's brentq, to f64 precision in 60 halvings
// -- and returns the end of the final bracket that lies past the crossing.
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
    // The far end of the bracket, not its midpoint: `hi` is the first instant at which the
    // indicator has the sign it crossed to, so the state handed to the transition has actually
    // crossed. The event iteration judges every *other* latched relation at that state, and a
    // relation sharing the threshold -- PhaseChange's `T > Ms` and `T < Ms` -- is only seen to
    // disagree with its latch if the crossing has happened there. Landing a hair before it left
    // the pair agreeing, and each then re-crossed its own rearm nudge on the next segments: four
    // events at one instant where the interpreter, whose root lands past the crossing, fires two.
    hi
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
    terminates: &[i64],
    grid: &[f64],
    out: &mut [f64],
    obs: &mut [f64],
    events_out: &mut [f64],
    ev_index_out: &mut [f64],
) -> (i32, usize, usize, usize, usize, usize, usize, f64) {
    let n = y0.len();
    // Observed variables at every grid point, grid.len() by nobs, alongside the states in
    // `out`. Empty `obs` asks for none. A model that switches state sets mid-run makes `out`
    // hard to read -- `y` means one thing before the switch and another after -- while an
    // observed variable means the same in every mode, so this is what a run is compared on.
    let points = grid.len();
    let nobs = if points > 0 { obs.len() / points } else { 0 };
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
    // The standstill counter: how many events in a row have failed to move the run forward,
    // and where it last did move (`solve.py`'s `chatter` and `stalled_at`).
    let mut chatter = 0usize;
    let mut stalled_at = t0;

    let mut cursor = 0usize;
    while cursor < points && grid[cursor] <= t0 {
        for q in 0..n { out[cursor * n + q] = y[q]; }
        if nobs > 0 { observe(grid[cursor], &y, p, w, &mut obs[cursor * nobs..(cursor + 1) * nobs]); }
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
                if nobs > 0 {
                    // Before the transition, with the workspace as it still is: these points
                    // belong to the segment that is ending, in the state set it ran in.
                    let (states, rest) = out.split_at_mut((cursor + 1) * n);
                    let _ = rest;
                    observe(grid[cursor], &states[cursor * n..], p, w, &mut obs[cursor * nobs..(cursor + 1) * nobs]);
                }
                cursor += 1;
            }
            for q in 0..n {
                ev_y[q] = y[q]
                    + fired_x * (quad[q * 3] + fired_x * (quad[q * 3 + 1] + fired_x * quad[q * 3 + 2]));
            }

            if terminates[fired as usize] != 0 {
                // A `terminate(...)` fired: the model is defined only up to this crossing. Record
                // the instant and its index (the caller names the reason), move to it, and stop --
                // there is no transition to apply and nothing meaningful past it. `solve.py` breaks
                // its loop here for the same reason.
                if events_total < events_out.len() { events_out[events_total] = te; }
                if events_total < ev_index_out.len() { ev_index_out[events_total] = fired as f64; }
                events_total += 1;
                t = te;
                for q in 0..n { y[q] = ev_y[q]; }
                status = TERMINATED;
                break;
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
                // Carry the instant and the crossing's index back, one per fired crossing: several
                // may settle at one `te`, and each gets that same time with its own index, so the
                // caller can name the cause. A buffer too small keeps the earliest; the count still
                // tallies.
                if events_total < events_out.len() { events_out[events_total] = te; }
                if events_total < ev_index_out.len() { ev_index_out[events_total] = e as f64; }
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
            // Measured from where the run last moved on, not from the previous event: a
            // chattering run advances by MD_REARM over the indicator's slope every time, so a
            // per-event test would reset at every crossing and never count to one.
            if te - stalled_at >= MD_STANDSTILL * te.abs().max(1.0) {
                chatter = 0;
                stalled_at = te;
            } else {
                chatter += 1;
            }
            t = te;
            for q in 0..n { y[q] = ev_y[q]; }
            if chatter > MD_CHATTER {
                status = CHATTERING;
                break;
            }
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
            if nobs > 0 {
                let (states, rest) = out.split_at_mut((cursor + 1) * n);
                let _ = rest;
                observe(grid[cursor], &states[cursor * n..], p, w, &mut obs[cursor * nobs..(cursor + 1) * nobs]);
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
        if nobs > 0 { observe(grid[cursor], &y, p, w, &mut obs[cursor * nobs..(cursor + 1) * nobs]); }
        cursor += 1;
    }
    (status, steps, rejected_total, evaluations, jacobians, factorisations, events_total, t)
}
