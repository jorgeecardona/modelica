//! The pymodelica engine: runs a model handed over as a register tape (`modelica.sim.vm`).
//!
//! The tape is built in Python by `vm.lower` and packed by `vm.encode` into five flat arrays.
//! This crate has no dependencies, so `rustc --crate-type cdylib` builds it without cargo; the
//! C entry points below are what `ctypes` calls. The opcode numbers are `vm.Op` and `vm.Builtin`.

use std::cell::Cell;
use std::slice;

// The Radau IIA runtime `rustsim` compiles, generated from `modelica.sim.rust_radau`. It calls
// the free functions `rhs`, `jacobian`, `observe`, `indicators` and `transition`, which are
// defined below over the program `md_vm_run` installed for the current thread.
include!("radau.rs");

const ITERATIONS: usize = 50;
const TOLERANCE: f64 = 1e-12;
const INSTR: usize = 5;
const FUNCTION: usize = 4;
const NEWTON: usize = 7;

/// One Newton block: indices into the program's arrays, as `vm.NewtonBlock`.
struct Newton {
    unknowns: Vec<usize>,
    body: (usize, usize),
    matrix: usize,
    right: usize,
    step: usize,
}

/// One function: a range of instructions and the Newton blocks its `NEWTON` opcodes name.
struct Function {
    code: (usize, usize),
    newton: Vec<Newton>,
}

/// A decoded program: owned copies of the arrays `vm.encode` produced.
pub struct Program {
    registers: usize,
    modes: usize,
    events: usize,
    switching: Option<usize>,
    constants: Vec<f64>,
    code: Vec<i32>,
    functions: Vec<Function>,
}

/// The inputs and outputs of one call.
struct Frame<'a> {
    t: f64,
    y: &'a [f64],
    p: &'a [f64],
    w: &'a mut [f64],
    out: &'a mut [f64],
    r: Vec<f64>,
}

fn md_min(a: f64, b: f64) -> f64 { if a < b { a } else { b } }
fn md_max(a: f64, b: f64) -> f64 { if a > b { a } else { b } }
fn md_sign(a: f64) -> f64 { ((a > 0.0) as i32 - (a < 0.0) as i32) as f64 }
fn md_mod(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && (r < 0.0) != (b < 0.0) { r + b } else { r }
}
fn truth(x: bool) -> f64 { if x { 1.0 } else { 0.0 } }

fn unary(builtin: i32, a: f64) -> f64 {
    match builtin {
        0 => a.abs(),
        1 => a.acos(),
        2 => a.asin(),
        3 => a.atan(),
        4 => a.ceil(),
        5 => a.cos(),
        6 => a.cosh(),
        7 => a.exp(),
        8 => a.floor(),
        9 => a.ln(),
        10 => a.log10(),
        11 => md_sign(a),
        12 => a.sin(),
        13 => a.sinh(),
        14 => a.sqrt(),
        15 => a.tan(),
        16 => a.tanh(),
        17 => a.trunc(),
        _ => f64::NAN,
    }
}

fn binary(builtin: i32, a: f64, b: f64) -> f64 {
    match builtin {
        32 => a.atan2(b),
        33 => md_max(a, b),
        34 => md_min(a, b),
        35 => md_mod(a, b),
        36 => a % b,
        _ => f64::NAN,
    }
}

/// Dense LU with partial pivoting on a copy of `A` and `b`, as `md_solve` in `dialect.RUST`.
fn solve(r: &mut [f64], n: usize, matrix: usize, right: usize, step: usize) {
    let mut a = r[matrix..matrix + n * n].to_vec();
    let mut b = r[right..right + n].to_vec();
    for k in 0..n {
        let mut pivot = k;
        for i in (k + 1)..n {
            if a[i * n + k].abs() > a[pivot * n + k].abs() { pivot = i; }
        }
        if pivot != k {
            for j in 0..n { a.swap(k * n + j, pivot * n + j); }
            b.swap(k, pivot);
        }
        for i in (k + 1)..n {
            let factor = a[i * n + k] / a[k * n + k];
            a[i * n + k] = 0.0;
            for j in (k + 1)..n { a[i * n + j] -= factor * a[k * n + j]; }
            b[i] -= factor * b[k];
        }
    }
    for i in (0..n).rev() {
        let mut total = b[i];
        for j in (i + 1)..n { total -= a[i * n + j] * r[step + j]; }
        r[step + i] = total / a[i * n + i];
    }
}

impl Program {
    fn run(&self, function: &Function, range: (usize, usize), f: &mut Frame) {
        let code = &self.code[range.0 * INSTR..(range.0 + range.1) * INSTR];
        for i in code.chunks_exact(INSTR) {
            let (op, dst, a, b, c) = (i[0], i[1] as usize, i[2] as usize, i[3] as usize, i[4]);
            let r = &mut f.r;
            match op {
                0 => r[dst] = self.constants[a],
                1 => r[dst] = f.t,
                2 => r[dst] = f.y[a],
                3 => r[dst] = f.p[a],
                4 => r[dst] = f.w[a],
                5 => f.out[a] = r[b],
                6 => f.w[a] = r[b],
                7 => r[dst] = r[a],
                8 => r[dst] = r[a] + r[b],
                9 => r[dst] = r[a] - r[b],
                10 => r[dst] = r[a] * r[b],
                11 => r[dst] = r[a] / r[b],
                12 => r[dst] = r[a].powf(r[b]),
                13 => r[dst] = truth(r[a] < r[b]),
                14 => r[dst] = truth(r[a] <= r[b]),
                15 => r[dst] = truth(r[a] > r[b]),
                16 => r[dst] = truth(r[a] >= r[b]),
                17 => r[dst] = truth(r[a] == r[b]),
                18 => r[dst] = truth(r[a] != r[b]),
                19 => r[dst] = truth(r[a] != 0.0 && r[b] != 0.0),
                20 => r[dst] = truth(r[a] != 0.0 || r[b] != 0.0),
                21 => r[dst] = -r[a],
                22 => r[dst] = truth(r[a] == 0.0),
                23 => r[dst] = unary(c, r[a]),
                24 => r[dst] = binary(c, r[a], r[b]),
                25 => r[dst] = if r[a] != 0.0 { r[b] } else { r[c as usize] },
                26 => solve(r, c as usize, a, b, dst),
                27 => self.newton(function, &function.newton[a], f),
                28 => {
                    let n = f.y.len();
                    f.out[..n].copy_from_slice(f.y);
                }
                29 => f.out[..a].fill(0.0),
                _ => unreachable!("opcode {op} is not in vm.Op"),
            }
        }
    }

    fn newton(&self, function: &Function, block: &Newton, f: &mut Frame) {
        let n = block.unknowns.len();
        for _ in 0..ITERATIONS {
            self.run(function, block.body, f);
            solve(&mut f.r, n, block.matrix, block.right, block.step);
            let mut largest: f64 = 0.0;
            for (k, &u) in block.unknowns.iter().enumerate() {
                let x = f.r[block.step + k];
                f.r[u] += x;
                largest = largest.max(x.abs());
            }
            if largest < TOLERANCE { break; }
        }
    }

    /// The mode `w` says is in use, as the emitted picker decides it.
    fn mode(&self, w: &[f64]) -> usize {
        match self.switching {
            None => 0,
            Some(slot) => {
                for k in 0..self.modes - 1 {
                    if w[slot] < k as f64 + 0.5 { return k; }
                }
                self.modes - 1
            }
        }
    }

    /// Run function `index`, then `then` if it names one: the event loop's `transition`.
    pub fn call(&self, index: usize, then: Option<usize>, t: f64, y: &[f64], p: &[f64], w: &mut [f64], out: &mut [f64]) {
        let mut frame = Frame { t, y, p, w, out, r: vec![0.0; self.registers] };
        let first = &self.functions[index];
        self.run(first, first.code, &mut frame);
        if let Some(k) = then {
            let second = &self.functions[k];
            self.run(second, second.code, &mut frame);
        }
    }
}

/// Decode the arrays from `vm.encode`. Returns null on a malformed table.
///
/// # Safety
/// Every pointer must be valid for the length beside it.
#[no_mangle]
pub unsafe extern "C" fn md_vm_new(
    registers: usize,
    modes: usize,
    events: usize,
    switching: i64,
    constants: *const f64,
    nconstants: usize,
    code: *const i32,
    ncode: usize,
    functions: *const i32,
    nfunctions: usize,
    newton: *const i32,
    nnewton: usize,
    unknowns: *const i32,
    nunknowns: usize,
) -> *mut Program {
    let view = |ptr: *const i32, n: usize| if n == 0 { &[][..] } else { slice::from_raw_parts(ptr, n) };
    let constants = if nconstants == 0 { Vec::new() } else { slice::from_raw_parts(constants, nconstants).to_vec() };
    let code = view(code, ncode).to_vec();
    let table = view(functions, nfunctions * FUNCTION);
    let blocks = view(newton, nnewton * NEWTON);
    let unknowns = view(unknowns, nunknowns);
    let mut decoded = Vec::with_capacity(nfunctions);
    for row in table.chunks_exact(FUNCTION) {
        let (start, len, first, count) = (row[0] as usize, row[1] as usize, row[2] as usize, row[3] as usize);
        let mut own = Vec::with_capacity(count);
        for b in blocks.get(first * NEWTON..(first + count) * NEWTON).unwrap_or(&[]).chunks_exact(NEWTON) {
            let (us, un) = (b[0] as usize, b[1] as usize);
            own.push(Newton {
                unknowns: unknowns[us..us + un].iter().map(|&u| u as usize).collect(),
                body: (b[2] as usize, b[3] as usize),
                matrix: b[4] as usize,
                right: b[5] as usize,
                step: b[6] as usize,
            });
        }
        decoded.push(Function { code: (start, len), newton: own });
    }
    let switching = if switching < 0 { None } else { Some(switching as usize) };
    Box::into_raw(Box::new(Program { registers, modes, events, switching, constants, code, functions: decoded }))
}

/// Run one function. `then < 0` means none. Returns 0.
///
/// # Safety
/// `program` comes from `md_vm_new`; every array is valid for the length beside it.
#[no_mangle]
pub unsafe extern "C" fn md_vm_call(
    program: *const Program,
    index: usize,
    then: i64,
    t: f64,
    y: *const f64,
    ny: usize,
    p: *const f64,
    np: usize,
    w: *mut f64,
    nw: usize,
    out: *mut f64,
    nout: usize,
) -> i32 {
    let program = &*program;
    let y = if ny == 0 { &[][..] } else { slice::from_raw_parts(y, ny) };
    let p = if np == 0 { &[][..] } else { slice::from_raw_parts(p, np) };
    let w = if nw == 0 { &mut [][..] } else { slice::from_raw_parts_mut(w, nw) };
    let out = if nout == 0 { &mut [][..] } else { slice::from_raw_parts_mut(out, nout) };
    let then = if then < 0 { None } else { Some(then as usize) };
    program.call(index, then, t, y, p, w, out);
    0
}

/// Free a program from `md_vm_new`.
///
/// # Safety
/// `program` comes from `md_vm_new` and is not used again.
#[no_mangle]
pub unsafe extern "C" fn md_vm_free(program: *mut Program) {
    if !program.is_null() { drop(Box::from_raw(program)); }
}

thread_local! {
    static CURRENT: Cell<*const Program> = const { Cell::new(std::ptr::null()) };
}

fn current<'a>() -> &'a Program {
    // Set by `md_vm_run` for the whole of one integration on this thread, and only read inside it.
    CURRENT.with(|c| unsafe { &*c.get() })
}

fn rhs(t: f64, y: &[f64], p: &[f64], w: &mut [f64], dy: &mut [f64]) {
    let m = current();
    let k = m.mode(w);
    m.call(k, None, t, y, p, w, dy);
}

fn observe(t: f64, y: &[f64], p: &[f64], w: &mut [f64], out: &mut [f64]) {
    let m = current();
    let k = m.mode(w);
    m.call(m.modes + k, None, t, y, p, w, out);
}

fn indicators(t: f64, y: &[f64], p: &[f64], w: &mut [f64], z: &mut [f64]) {
    let m = current();
    let k = m.mode(w);
    m.call(2 * m.modes + k, None, t, y, p, w, z);
}

fn jacobian(t: f64, y: &[f64], p: &[f64], w: &mut [f64], out: &mut [f64]) {
    let m = current();
    let k = m.mode(w);
    m.call(4 * m.modes + k, None, t, y, p, w, out);
}

fn transition(event: usize, t: f64, y: &[f64], p: &[f64], w: &mut [f64], out: &mut [f64]) {
    let m = current();
    let k = m.mode(w);
    m.call(3 * m.modes + k, Some(5 * m.modes + k * m.events + event), t, y, p, w, out);
}

/// Integrate with `program` as the model: the same arguments and result as `rustsim`'s `md_run`.
///
/// # Safety
/// `program` comes from `md_vm_new`; every array is valid for the length beside it.
#[no_mangle]
pub unsafe extern "C" fn md_vm_run(
    program: *const Program,
    t0: f64,
    t1: f64,
    y0: *const f64,
    n: usize,
    p: *const f64,
    np: usize,
    w: *mut f64,
    nw: usize,
    rtol: f64,
    atol: f64,
    dirs: *const f64,
    lat: *const i64,
    term: *const i64,
    ne: usize,
    grid: *const f64,
    npts: usize,
    out: *mut f64,
    obs: *mut f64,
    nobs: usize,
    events_out: *mut f64,
    ev_index_out: *mut f64,
    nev_out: usize,
    counters: *mut f64,
) -> i32 {
    unsafe fn view<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
        if len == 0 { &[] } else { slice::from_raw_parts(ptr, len) }
    }
    unsafe fn view_mut<'a>(ptr: *mut f64, len: usize) -> &'a mut [f64] {
        if len == 0 { &mut [] } else { slice::from_raw_parts_mut(ptr, len) }
    }
    let previous = CURRENT.with(|c| c.replace(program));
    let r = md_integrate(
        t0, t1, view(y0, n), view(p, np), view_mut(w, nw), rtol, atol,
        view(dirs, ne), view(lat, ne), view(term, ne), view(grid, npts),
        view_mut(out, npts * n), view_mut(obs, npts * nobs),
        view_mut(events_out, nev_out), view_mut(ev_index_out, nev_out),
    );
    CURRENT.with(|c| c.set(previous));
    let counters = view_mut(counters, 8);
    counters[0] = r.0 as f64;
    counters[1] = r.1 as f64;
    counters[2] = r.2 as f64;
    counters[3] = r.3 as f64;
    counters[4] = r.4 as f64;
    counters[5] = r.5 as f64;
    counters[6] = r.6 as f64;
    counters[7] = r.7;
    r.0
}
