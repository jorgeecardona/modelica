//! The pymodelica engine: runs a model handed over as a register tape (`modelica.sim.vm`).
//!
//! The tape is built in Python by `vm.lower` and packed by `vm.encode` into five flat arrays.
//! This crate has no dependencies, so `rustc --crate-type cdylib` builds it without cargo; the
//! C entry points below are what `ctypes` calls. The opcode numbers are `vm.Op` and `vm.Builtin`.

use std::cell::{Cell, RefCell};
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
#[derive(Clone)]
struct Newton {
    unknowns: Vec<usize>,
    body: (usize, usize),
    matrix: usize,
    right: usize,
    step: usize,
}

/// One function: a range of instructions and the Newton blocks its `NEWTON` opcodes name.
#[derive(Clone)]
struct Function {
    code: (usize, usize),
    newton: Vec<Newton>,
    /// Where `HOIST` is in `code`, counted from its start: everything before it reads only
    /// `time` and constants. Skipped when a cache slot holds the call's `t`.
    hoist: Option<usize>,
    /// The registers written before `HOIST` and read after it: what a cache slot saves.
    live: Vec<usize>,
}

/// A decoded program: owned copies of the arrays `vm.encode` produced.
#[derive(Clone)]
pub struct Program {
    modes: usize,
    events: usize,
    switching: Option<usize>,
    constants: Vec<f64>,
    code: Vec<i32>,
    functions: Vec<Function>,
    scratch: RefCell<Scratch>,
    /// The state order Radau factors its matrices in (empty: the natural order), set by
    /// `md_vm_order`. A good order makes `mu/h I - J` banded.
    order: Vec<usize>,
}

/// Memory one call works in, kept between calls so a call allocates nothing: the register
/// file, and the copies of `A` and `b` that `solve` factors.
#[derive(Clone)]
struct Scratch {
    r: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    /// Per function, the time-only values of the last `SLOTS` distinct `t` (`Slots`).
    cache: Vec<Slots>,
}

/// How many distinct `t` a function remembers. Radau calls `rhs` at three stage times in turn,
/// then at the step's start for the error estimate: four.
const SLOTS: usize = 4;

/// One function's cache: slot `k` holds `live` at time `t[k]` (NaN: empty) in
/// `values[k * live.len()..]`. `next` is the slot the next miss replaces.
#[derive(Clone)]
struct Slots {
    t: [f64; SLOTS],
    values: Vec<f64>,
    next: usize,
}

/// The inputs and outputs of one call.
struct Frame<'a> {
    t: f64,
    y: &'a [f64],
    p: &'a [f64],
    w: &'a mut [f64],
    out: &'a mut [f64],
    r: &'a mut [f64],
    a: &'a mut Vec<f64>,
    b: &'a mut Vec<f64>,
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
fn solve(r: &mut [f64], a: &mut Vec<f64>, b: &mut Vec<f64>, n: usize, matrix: usize, right: usize, step: usize) {
    a.clear();
    a.extend_from_slice(&r[matrix..matrix + n * n]);
    b.clear();
    b.extend_from_slice(&r[right..right + n]);
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

/// Whether every register an instruction in `code[range]` names is below `registers`, every
/// constant index is in the pool and every `NEWTON` names a block of its function. `run` reads
/// registers without bounds checks on the strength of this.
fn valid(code: &[i32], range: (usize, usize), registers: usize, constants: usize, blocks: usize) -> bool {
    let Some(slice) = code.get(range.0 * INSTR..(range.0 + range.1) * INSTR) else { return false };
    let fits = |x: i64| x >= 0 && (x as usize) < registers;
    let span = |start: i64, n: i64| start >= 0 && n >= 0 && (start + n) as usize <= registers;
    slice.chunks_exact(INSTR).all(|i| {
        let (op, dst, a, b, c) = (i[0], i[1] as i64, i[2] as i64, i[3] as i64, i[4] as i64);
        match op {
            0 => fits(dst) && a >= 0 && (a as usize) < constants,
            1..=4 => fits(dst),
            5 | 6 => fits(b),
            7 | 21 | 22 | 23 => fits(dst) && fits(a),
            8..=20 | 24 => fits(dst) && fits(a) && fits(b),
            25 => fits(dst) && fits(a) && fits(b) && fits(c),
            26 => span(a, c * c) && span(b, c) && span(dst, c),
            27 => a >= 0 && (a as usize) < blocks,
            28 | 29 | 30 => true,
            _ => false,
        }
    })
}

impl Program {
    fn run(&self, function: &Function, range: (usize, usize), f: &mut Frame) {
        let code = &self.code[range.0 * INSTR..(range.0 + range.1) * INSTR];
        for i in code.chunks_exact(INSTR) {
            let (op, dst, a, b, c) = (i[0], i[1] as usize, i[2] as usize, i[3] as usize, i[4]);
            // SAFETY: `valid` checked every register below against the length of `f.r`, which
            // is the program's register count (`Scratch::r`).
            let reg = f.r.as_mut_ptr();
            let get = |k: usize| unsafe { *reg.add(k) };
            let set = |k: usize, v: f64| unsafe { *reg.add(k) = v };
            match op {
                0 => set(dst, self.constants[a]),
                1 => set(dst, f.t),
                2 => set(dst, f.y[a]),
                3 => set(dst, f.p[a]),
                4 => set(dst, f.w[a]),
                5 => f.out[a] = get(b),
                6 => f.w[a] = get(b),
                7 => set(dst, get(a)),
                8 => set(dst, get(a) + get(b)),
                9 => set(dst, get(a) - get(b)),
                10 => set(dst, get(a) * get(b)),
                11 => set(dst, get(a) / get(b)),
                12 => set(dst, get(a).powf(get(b))),
                13 => set(dst, truth(get(a) < get(b))),
                14 => set(dst, truth(get(a) <= get(b))),
                15 => set(dst, truth(get(a) > get(b))),
                16 => set(dst, truth(get(a) >= get(b))),
                17 => set(dst, truth(get(a) == get(b))),
                18 => set(dst, truth(get(a) != get(b))),
                19 => set(dst, truth(get(a) != 0.0 && get(b) != 0.0)),
                20 => set(dst, truth(get(a) != 0.0 || get(b) != 0.0)),
                21 => set(dst, -get(a)),
                22 => set(dst, truth(get(a) == 0.0)),
                23 => set(dst, unary(c, get(a))),
                24 => set(dst, binary(c, get(a), get(b))),
                25 => set(dst, if get(a) != 0.0 { get(b) } else { get(c as usize) }),
                26 => solve(f.r, f.a, f.b, c as usize, a, b, dst),
                27 => self.newton(function, &function.newton[a], f),
                28 => {
                    let n = f.y.len();
                    f.out[..n].copy_from_slice(f.y);
                }
                29 => f.out[..a].fill(0.0),
                30 => {}
                _ => unreachable!("opcode {op} is not in vm.Op"),
            }
        }
    }

    fn newton(&self, function: &Function, block: &Newton, f: &mut Frame) {
        let n = block.unknowns.len();
        for _ in 0..ITERATIONS {
            self.run(function, block.body, f);
            solve(f.r, f.a, f.b, n, block.matrix, block.right, block.step);
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
        let mut scratch = self.scratch.borrow_mut();
        let Scratch { r, a, b, cache } = &mut *scratch;
        let mut frame = Frame { t, y, p, w, out, r, a, b };
        self.enter(index, cache, &mut frame);
        if let Some(k) = then {
            self.enter(k, cache, &mut frame);
        }
    }

    /// Run function `index`. When a cache slot holds `t`, put its values back and start after
    /// `HOIST`; otherwise run all of it and save the values in a slot.
    fn enter(&self, index: usize, cache: &mut [Slots], f: &mut Frame) {
        let function = &self.functions[index];
        let Some(split) = function.hoist else {
            self.run(function, function.code, f);
            return;
        };
        let n = function.live.len();
        let slots = &mut cache[index];
        if let Some(k) = slots.t.iter().position(|&t| t == f.t) {
            for (q, &reg) in function.live.iter().enumerate() {
                f.r[reg] = slots.values[k * n + q];
            }
            self.run(function, (function.code.0 + split + 1, function.code.1 - split - 1), f);
            return;
        }
        self.run(function, (function.code.0, split), f);
        let k = slots.next;
        slots.next = (k + 1) % SLOTS;
        slots.t[k] = f.t;
        for (q, &reg) in function.live.iter().enumerate() {
            slots.values[k * n + q] = f.r[reg];
        }
        self.run(function, (function.code.0 + split + 1, function.code.1 - split - 1), f);
    }
}

/// The registers the instructions in `code[range]` write, a solve's and a Newton block's too.
fn writes(code: &[i32], range: (usize, usize), function: &Function, out: &mut Vec<usize>) {
    for i in code[range.0 * INSTR..(range.0 + range.1) * INSTR].chunks_exact(INSTR) {
        let (op, dst, a, c) = (i[0], i[1] as usize, i[2] as usize, i[4] as usize);
        match op {
            0..=4 | 7..=25 => out.push(dst),
            26 => out.extend(dst..dst + c),
            27 => {
                let block = &function.newton[a];
                out.extend(&block.unknowns);
                out.extend(block.step..block.step + block.unknowns.len());
                writes(code, block.body, function, out);
            }
            _ => {}
        }
    }
}

/// The registers the instructions in `code[range]` read, a solve's and a Newton block's too.
fn reads(code: &[i32], range: (usize, usize), function: &Function, out: &mut Vec<usize>) {
    for i in code[range.0 * INSTR..(range.0 + range.1) * INSTR].chunks_exact(INSTR) {
        let (op, a, b, c) = (i[0], i[2] as usize, i[3] as usize, i[4] as usize);
        match op {
            5 | 6 => out.push(b),
            7 | 21 | 22 | 23 => out.push(a),
            8..=20 | 24 => out.extend([a, b]),
            25 => out.extend([a, b, c]),
            26 => {
                out.extend(a..a + c * c);
                out.extend(b..b + c);
            }
            27 => {
                let block = &function.newton[a];
                let n = block.unknowns.len();
                out.extend(&block.unknowns);
                out.extend(block.matrix..block.matrix + n * n);
                out.extend(block.right..block.right + n);
                reads(code, block.body, function, out);
            }
            _ => {}
        }
    }
}

/// The registers written before `split` in `function` and read after it.
fn live(code: &[i32], function: &Function, split: usize) -> Vec<usize> {
    let mut before = Vec::new();
    writes(code, (function.code.0, split), function, &mut before);
    let mut after = Vec::new();
    reads(code, (function.code.0 + split + 1, function.code.1 - split - 1), function, &mut after);
    let after: std::collections::HashSet<usize> = after.into_iter().collect();
    let mut out: Vec<usize> = before.into_iter().filter(|r| after.contains(r)).collect();
    out.sort_unstable();
    out.dedup();
    out
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
            let Some(names) = unknowns.get(us..us + un) else { return std::ptr::null_mut() };
            let block = Newton {
                unknowns: names.iter().map(|&u| u as usize).collect(),
                body: (b[2] as usize, b[3] as usize),
                matrix: b[4] as usize,
                right: b[5] as usize,
                step: b[6] as usize,
            };
            let n = block.unknowns.len();
            let inside = block.unknowns.iter().all(|&u| u < registers)
                && block.matrix + n * n <= registers
                && block.right + n <= registers
                && block.step + n <= registers;
            if !inside || !valid(&code, block.body, registers, constants.len(), count) {
                return std::ptr::null_mut();
            }
            own.push(block);
        }
        if !valid(&code, (start, len), registers, constants.len(), count) {
            return std::ptr::null_mut();
        }
        let hoist = code[start * INSTR..(start + len) * INSTR]
            .chunks_exact(INSTR)
            .position(|i| i[0] == 30);
        let mut function = Function { code: (start, len), newton: own, hoist, live: Vec::new() };
        if let Some(split) = hoist {
            function.live = live(&code, &function, split);
        }
        decoded.push(function);
    }
    let cache = decoded.iter().map(|f| Slots {
        t: [f64::NAN; SLOTS], values: vec![0.0; SLOTS * f.live.len()], next: 0,
    }).collect();
    let switching = if switching < 0 { None } else { Some(switching as usize) };
    // The last registers hold the constants (`vm._preload`); nothing writes them after this.
    if constants.len() > registers {
        return std::ptr::null_mut();
    }
    let mut preloaded = vec![0.0; registers];
    preloaded[registers - constants.len()..].copy_from_slice(&constants);
    Box::into_raw(Box::new(Program {
        modes, events, switching, constants, code, functions: decoded,
        scratch: RefCell::new(Scratch {
            r: preloaded, a: Vec::new(), b: Vec::new(), cache,
        }),
        order: Vec::new(),
    }))
}

/// Set the state order Radau factors in: `order[i]` is the state at row and column `i`.
/// Returns 0, or -1 (and changes nothing) when `order` is not a permutation of `0..n`.
///
/// # Safety
/// `program` comes from `md_vm_new`; `order` is valid for `n` values.
#[no_mangle]
pub unsafe extern "C" fn md_vm_order(program: *mut Program, order: *const u64, n: usize) -> i32 {
    let given = if n == 0 { &[][..] } else { slice::from_raw_parts(order, n) };
    let mut seen = vec![false; n];
    for &k in given {
        let k = k as usize;
        if k >= n || seen[k] { return -1; }
        seen[k] = true;
    }
    (*program).order = given.iter().map(|&k| k as usize).collect();
    0
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
    every: i32,
    t_out: *mut f64,
    cap: usize,
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
        view(dirs, ne), view(lat, ne), view(term, ne), view(grid, npts), &(*program).order, every != 0,
        view_mut(t_out, cap), view_mut(out, cap * n), view_mut(obs, cap * nobs),
        view_mut(events_out, nev_out), view_mut(ev_index_out, nev_out),
    );
    CURRENT.with(|c| c.set(previous));
    let counters = view_mut(counters, 9);
    counters[0] = r.0 as f64;
    counters[1] = r.1 as f64;
    counters[2] = r.2 as f64;
    counters[3] = r.3 as f64;
    counters[4] = r.4 as f64;
    counters[5] = r.5 as f64;
    counters[6] = r.6 as f64;
    counters[7] = r.7;
    counters[8] = r.8 as f64;
    r.0
}


/// Run `points` integrations of `program` at once, one thread per core (`threads = 0`) or
/// `threads` threads. Point `k` reads and writes the `k`-th block of every stacked array; the
/// shared arrays (`dirs`, `lat`, `term`, `grid`) are the same for every point. `status[k]` is
/// point `k`'s status and `counters[9 k..9 k + 9]` its counters, as `md_vm_run` returns them.
///
/// Each thread runs on its own copy of the program, so the register file is never shared.
///
/// # Safety
/// `program` comes from `md_vm_new`; every array is valid for `points` blocks of the length
/// given beside it.
#[no_mangle]
pub unsafe extern "C" fn md_vm_run_many(
    program: *const Program,
    threads: usize,
    points: usize,
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
    every: i32,
    t_out: *mut f64,
    cap: usize,
    out: *mut f64,
    obs: *mut f64,
    nobs: usize,
    events_out: *mut f64,
    ev_index_out: *mut f64,
    nev_out: usize,
    counters: *mut f64,
    status: *mut i32,
) {
    unsafe fn view<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
        if len == 0 { &[] } else { slice::from_raw_parts(ptr, len) }
    }
    unsafe fn view_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
        if len == 0 { &mut [] } else { slice::from_raw_parts_mut(ptr, len) }
    }
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |k| k.get())
    } else {
        threads
    }
    .min(points.max(1));
    // Raw pointers are not `Send`; the addresses are, and each point's block is disjoint.
    let addr = |x: *const f64| x as usize;
    let (y0, p, w) = (addr(y0), addr(p), w as usize);
    let (dirs, lat, term, grid) = (addr(dirs), lat as usize, term as usize, addr(grid));
    let (t_out, out, obs) = (t_out as usize, out as usize, obs as usize);
    let (events_out, ev_index_out) = (events_out as usize, ev_index_out as usize);
    let (counters, status) = (counters as usize, status as usize);
    // Cloned here, not in the threads: the register file sits in a `RefCell`, which a thread
    // may own but not share.
    let copies: Vec<Program> = (0..threads).map(|_| (*program).clone()).collect();
    std::thread::scope(|scope| {
        for (first, local) in copies.into_iter().enumerate() {
            scope.spawn(move || {
                let previous = CURRENT.with(|c| c.replace(&local));
                let mut k = first;
                while k < points {
                    let r = md_integrate(
                        t0, t1,
                        view((y0 as *const f64).add(k * n), n),
                        view((p as *const f64).add(k * np), np),
                        view_mut((w as *mut f64).add(k * nw), nw),
                        rtol, atol,
                        view(dirs as *const f64, ne), view(lat as *const i64, ne),
                        view(term as *const i64, ne), view(grid as *const f64, npts), &local.order,
                        every != 0,
                        view_mut((t_out as *mut f64).add(k * cap), cap),
                        view_mut((out as *mut f64).add(k * cap * n), cap * n),
                        view_mut((obs as *mut f64).add(k * cap * nobs), cap * nobs),
                        view_mut((events_out as *mut f64).add(k * nev_out), nev_out),
                        view_mut((ev_index_out as *mut f64).add(k * nev_out), nev_out),
                    );
                    let c = view_mut((counters as *mut f64).add(9 * k), 9);
                    c[0] = r.0 as f64;
                    c[1] = r.1 as f64;
                    c[2] = r.2 as f64;
                    c[3] = r.3 as f64;
                    c[4] = r.4 as f64;
                    c[5] = r.5 as f64;
                    c[6] = r.6 as f64;
                    c[7] = r.7;
                    c[8] = r.8 as f64;
                    *(status as *mut i32).add(k) = r.0;
                    k += threads;
                }
                CURRENT.with(|c| c.set(previous));
            });
        }
    });
}
