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
/// The tape format `vm.encode` writes and `md_vm_new` reads. 2 added shaped views.
const FORMAT: u32 = 3;
/// The most axes a view may have.
const DIMS: usize = 8;

/// An array as registers, as `vm.View`: element `index` is `base + sum(index[k] * strides[k])`.
#[derive(Clone)]
struct View {
    base: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl View {
    fn at(&self, index: &[usize]) -> usize {
        self.base + index.iter().zip(&self.strides).map(|(i, s)| i * s).sum::<usize>()
    }
    /// The last register an element can name, or `None` for an empty view.
    fn last(&self) -> Option<usize> {
        if self.shape.contains(&0) { return None; }
        Some(self.base + self.shape.iter().zip(&self.strides).map(|(n, s)| (n - 1) * s).sum::<usize>())
    }
    /// Call `f` with every index of `shape` in row-major order, the last axis fastest.
    fn each(shape: &[usize], mut f: impl FnMut(&[usize])) {
        if shape.contains(&0) { return; }
        let mut index = [0usize; DIMS];
        let n = shape.len();
        loop {
            f(&index[..n]);
            let mut k = n;
            loop {
                if k == 0 { return; }
                k -= 1;
                index[k] += 1;
                if index[k] < shape[k] { break; }
                index[k] = 0;
            }
        }
    }
    fn elements(&self, out: &mut Vec<usize>) {
        View::each(&self.shape, |i| out.push(self.at(i)));
    }
}

/// Decode `[base, ndim, shape.., strides..]` per view. `None` when a view names a register at
/// or past `registers`, has more than `DIMS` axes, or a field is negative.
fn decode_views(flat: &[i32], registers: usize) -> Option<Vec<View>> {
    let mut views = Vec::new();
    let mut k = 0;
    while k < flat.len() {
        let base = usize::try_from(*flat.get(k)?).ok()?;
        let nd = usize::try_from(*flat.get(k + 1)?).ok()?;
        if nd > DIMS { return None; }
        let field = |j: usize| flat.get(j).and_then(|&x| usize::try_from(x).ok());
        let shape = (0..nd).map(|j| field(k + 2 + j)).collect::<Option<Vec<_>>>()?;
        let strides = (0..nd).map(|j| field(k + 2 + nd + j)).collect::<Option<Vec<_>>>()?;
        let view = View { base, shape, strides };
        if view.last().is_some_and(|last| last >= registers) || base >= registers {
            return None;
        }
        views.push(view);
        k += 2 + 2 * nd;
    }
    Some(views)
}

/// Whether a shaped instruction names views that exist with shapes that fit its opcode.
fn shaped(op: i32, dst: i64, a: i64, b: i64, c: i64, views: &[View]) -> bool {
    let get = |k: i64| usize::try_from(k).ok().and_then(|k| views.get(k));
    let Some(d) = get(dst) else { return false };
    let same = |k: i64| get(k).is_some_and(|v| v.shape == d.shape);
    match op {
        43 => same(a) && same(b) && (8..=20).contains(&c),
        44 => same(a) && same(b) && (32..=36).contains(&c),
        45 => same(a) && (c == 21 || c == 22 || (c == 23 && (0..=17).contains(&b))),
        46 => same(a) && same(b) && same(c),
        47 => get(a).is_some_and(|v| {
            let axis = b as usize;
            b >= 0 && axis < v.shape.len() && (0..=3).contains(&c) && v.shape[axis] > 0
                && d.shape.len() + 1 == v.shape.len()
                && d.shape.iter().eq(v.shape.iter().enumerate().filter(|&(k, _)| k != axis).map(|(_, n)| n))
        }),
        48 => a >= 0 && (a as usize) < d.shape.len(),
        49 | 50 => get(a).is_some_and(|v| {
            // The destination must not overlap the operand: compared by their register hulls.
            let apart = match (d.last(), v.last()) {
                (Some(dl), Some(vl)) => dl < v.base || vl < d.base,
                _ => true,
            };
            v.shape == d.shape && b >= 0 && (b as usize) < d.shape.len() && apart
        }),
        _ => false,
    }
}

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
    /// Per event, the workspace slot of its time-event instant, or -1 (`md_vm_times`).
    times: Vec<i64>,
    /// The Jacobian's entries `(row, column)`, one per value the jacobian tape writes, in
    /// slot order (`md_vm_pattern`). Empty: the tape writes the dense `n * n` matrix.
    pattern: Vec<(usize, usize)>,
    /// The diagonal of the mass matrix in `M y' = f`, one per state (`md_vm_mass`); 0 marks an
    /// algebraic row. Empty: `M = I`.
    mass: Vec<f64>,
    /// Solve Radau's linear systems by BiCGStab with DILU instead of the band LU (`md_vm_solver`).
    iterative: bool,
    /// The gather tables (`vm.Function.tables`, every function's in one pool): register numbers.
    tables: Vec<usize>,
    /// The arrays the shaped opcodes name (`vm.Program.views`).
    views: Vec<View>,
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
fn valid(code: &[i32], range: (usize, usize), registers: usize, constants: usize, blocks: usize, tables: &[i32], views: &[View]) -> bool {
    let Some(slice) = code.get(range.0 * INSTR..(range.0 + range.1) * INSTR) else { return false };
    let fits = |x: i64| x >= 0 && (x as usize) < registers;
    let span = |start: i64, n: i64| start >= 0 && n >= 0 && (start + n) as usize <= registers;
    // A table is `n` register numbers in the pool from `start`, every one a register.
    let table = |start: i64, n: i64| {
        start >= 0
            && n >= 0
            && tables
                .get(start as usize..(start + n) as usize)
                .is_some_and(|t| t.iter().all(|&r| fits(r as i64)))
    };
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
            31 => span(dst, c) && table(a, c),
            32 => span(b, c) && table(a, c),
            33 => span(dst, c) && fits(a),
            34..=37 => span(dst, c) && span(a, c) && span(b, c),
            38 => span(dst, c) && span(a, c),
            39 => span(dst, b) && span(a, b),
            40 => span(dst, c) && a >= 0,
            42 => span(b, c) && table(a, c),
            41 => span(b, c) && a >= 0,
            43..=50 => shaped(op, dst, a, b, c, views),
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
                31 => {
                    for k in 0..c as usize {
                        set(dst + k, get(self.tables[a + k]));
                    }
                }
                32 => {
                    for k in 0..c as usize {
                        set(self.tables[a + k], get(b + k));
                    }
                }
                33 => {
                    let v = get(a);
                    for k in 0..c as usize {
                        set(dst + k, v);
                    }
                }
                34 => for k in 0..c as usize { set(dst + k, get(a + k) + get(b + k)); },
                35 => for k in 0..c as usize { set(dst + k, get(a + k) - get(b + k)); },
                36 => for k in 0..c as usize { set(dst + k, get(a + k) * get(b + k)); },
                37 => for k in 0..c as usize { set(dst + k, get(a + k) / get(b + k)); },
                38 => for k in 0..c as usize { set(dst + k, -get(a + k)); },
                39 => for k in 0..b { set(dst + k, unary(c, get(a + k))); },
                42 => {
                    for k in 0..c as usize {
                        let t = self.tables[a + k];
                        set(t, get(t) + get(b + k));
                    }
                }
                40 => f.r[dst..dst + c as usize].copy_from_slice(&f.y[a..a + c as usize]),
                41 => f.out[a..a + c as usize].copy_from_slice(&f.r[b..b + c as usize]),
                43..=50 => self.shaped(op, dst, a, b, c, f.r),
                _ => unreachable!("opcode {op} is not in vm.Op"),
            }
        }
    }

    /// One shaped opcode over its destination view, in row-major order, as `vm._Machine._shaped`.
    fn shaped(&self, op: i32, dst: usize, a: usize, b: usize, c: i32, r: &mut [f64]) {
        let views = &self.views;
        let d = &views[dst];
        // SAFETY: `valid` checked every view's last register against the register count.
        let reg = r.as_mut_ptr();
        let get = |k: usize| unsafe { *reg.add(k) };
        let set = |k: usize, v: f64| unsafe { *reg.add(k) = v };
        match op {
            48 => View::each(&d.shape, |i| set(d.at(i), (i[a] + 1) as f64)),
            47 => {
                let v = &views[a];
                let mut full = [0usize; DIMS];
                View::each(&d.shape, |i| {
                    full[..b].copy_from_slice(&i[..b]);
                    full[b + 1..=i.len()].copy_from_slice(&i[b..]);
                    let mut total = 0.0;
                    for k in 0..v.shape[b] {
                        full[b] = k;
                        let x = get(v.at(&full[..=i.len()]));
                        total = if k == 0 { x } else {
                            match c { 0 => total + x, 1 => total * x, 2 => md_min(total, x), _ => md_max(total, x) }
                        };
                    }
                    set(d.at(i), total);
                });
            }
            49 => {
                let v = &views[a];
                let n = v.shape[b] as i64;
                let mut source = [0usize; DIMS];
                View::each(&d.shape, |i| {
                    source[..i.len()].copy_from_slice(i);
                    source[b] = (i[b] as i64 - c as i64).rem_euclid(n) as usize;
                    set(d.at(i), get(v.at(&source[..i.len()])));
                });
            }
            50 => {
                // A clamped read: the edge element repeats (`T[max(i - 1, 1)]`).
                let v = &views[a];
                let last = v.shape[b] as i64 - 1;
                let mut source = [0usize; DIMS];
                View::each(&d.shape, |i| {
                    source[..i.len()].copy_from_slice(i);
                    source[b] = (i[b] as i64 - c as i64).clamp(0, last) as usize;
                    set(d.at(i), get(v.at(&source[..i.len()])));
                });
            }
            45 => {
                let v = &views[a];
                View::each(&d.shape, |i| {
                    let x = get(v.at(i));
                    let value = match c { 21 => -x, 22 => truth(x == 0.0), _ => unary(b as i32, x) };
                    set(d.at(i), value);
                });
            }
            46 => {
                let (cond, then, other) = (&views[a], &views[b], &views[c as usize]);
                View::each(&d.shape, |i| {
                    set(d.at(i), if get(cond.at(i)) != 0.0 { get(then.at(i)) } else { get(other.at(i)) });
                });
            }
            _ => {
                let (u, v) = (&views[a], &views[b]);
                View::each(&d.shape, |i| {
                    let (x, z) = (get(u.at(i)), get(v.at(i)));
                    let value = if op == 44 { binary(c, x, z) } else {
                        match c {
                            8 => x + z, 9 => x - z, 10 => x * z, 11 => x / z, 12 => x.powf(z),
                            13 => truth(x < z), 14 => truth(x <= z), 15 => truth(x > z),
                            16 => truth(x >= z), 17 => truth(x == z), 18 => truth(x != z),
                            19 => truth(x != 0.0 && z != 0.0), _ => truth(x != 0.0 || z != 0.0),
                        }
                    };
                    set(d.at(i), value);
                });
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
fn writes(code: &[i32], range: (usize, usize), function: &Function, tables: &[i32], views: &[View], out: &mut Vec<usize>) {
    for i in code[range.0 * INSTR..(range.0 + range.1) * INSTR].chunks_exact(INSTR) {
        let (op, dst, a, b, c) = (i[0], i[1] as usize, i[2] as usize, i[3] as usize, i[4] as usize);
        match op {
            0..=4 | 7..=25 => out.push(dst),
            43..=50 => views[dst].elements(out),
            31 | 33..=38 => out.extend(dst..dst + c),
            32 | 42 => out.extend(tables[a..a + c].iter().map(|&r| r as usize)),
            39 => out.extend(dst..dst + b),
            40 => out.extend(dst..dst + c),
            26 => out.extend(dst..dst + c),
            27 => {
                let block = &function.newton[a];
                out.extend(&block.unknowns);
                out.extend(block.step..block.step + block.unknowns.len());
                writes(code, block.body, function, tables, views, out);
            }
            _ => {}
        }
    }
}

/// The registers the instructions in `code[range]` read, a solve's and a Newton block's too.
fn reads(code: &[i32], range: (usize, usize), function: &Function, tables: &[i32], views: &[View], out: &mut Vec<usize>) {
    for i in code[range.0 * INSTR..(range.0 + range.1) * INSTR].chunks_exact(INSTR) {
        let (op, a, b, c) = (i[0], i[2] as usize, i[3] as usize, i[4] as usize);
        match op {
            31 => out.extend(tables[a..a + c].iter().map(|&r| r as usize)),
            32 => out.extend(b..b + c),
            43 | 44 => {
                views[a].elements(out);
                views[b].elements(out);
            }
            45 | 47 | 49 | 50 => views[a].elements(out),
            46 => {
                views[a].elements(out);
                views[b].elements(out);
                views[c].elements(out);
            }
            33 => out.push(a),
            34..=37 => {
                out.extend(a..a + c);
                out.extend(b..b + c);
            }
            38 => out.extend(a..a + c),
            39 => out.extend(a..a + b),
            41 => out.extend(b..b + c),
            42 => {
                out.extend(b..b + c);
                out.extend(tables[a..a + c].iter().map(|&r| r as usize));
            }
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
                reads(code, block.body, function, tables, views, out);
            }
            _ => {}
        }
    }
}

/// The registers written before `split` in `function` and read after it.
fn live(code: &[i32], function: &Function, split: usize, tables: &[i32], views: &[View]) -> Vec<usize> {
    let mut before = Vec::new();
    writes(code, (function.code.0, split), function, tables, views, &mut before);
    let mut after = Vec::new();
    reads(code, (function.code.0 + split + 1, function.code.1 - split - 1), function, tables, views, &mut after);
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
    views: *const i32,
    nviews: usize,
) -> *mut Program {
    let view = |ptr: *const i32, n: usize| if n == 0 { &[][..] } else { slice::from_raw_parts(ptr, n) };
    let constants = if nconstants == 0 { Vec::new() } else { slice::from_raw_parts(constants, nconstants).to_vec() };
    let code = view(code, ncode).to_vec();
    let table = view(functions, nfunctions * FUNCTION);
    let blocks = view(newton, nnewton * NEWTON);
    let unknowns = view(unknowns, nunknowns);
    let Some(views) = decode_views(view(views, nviews), registers) else { return std::ptr::null_mut() };
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
            if !inside || !valid(&code, block.body, registers, constants.len(), count, unknowns, &views) {
                return std::ptr::null_mut();
            }
            own.push(block);
        }
        if !valid(&code, (start, len), registers, constants.len(), count, unknowns, &views) {
            return std::ptr::null_mut();
        }
        let hoist = code[start * INSTR..(start + len) * INSTR]
            .chunks_exact(INSTR)
            .position(|i| i[0] == 30);
        let mut function = Function { code: (start, len), newton: own, hoist, live: Vec::new() };
        if let Some(split) = hoist {
            function.live = live(&code, &function, split, unknowns, &views);
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
        times: Vec::new(),
        pattern: Vec::new(),
        mass: Vec::new(),
        iterative: false,
        tables: unknowns.iter().map(|&r| r as usize).collect(),
        views,
    }))
}

/// The tape format this engine reads (`FORMAT`). `modelica.sim.engine` refuses a library
/// whose format differs from `vm.FORMAT`.
#[no_mangle]
pub extern "C" fn md_vm_format() -> u32 {
    FORMAT
}

/// Set, per event, the workspace slot that holds its time-event instant (-1: a state event).
/// Returns 0, or -1 (and changes nothing) when `n` is not the event count.
///
/// # Safety
/// `program` comes from `md_vm_new`; `times` is valid for `n` values.
#[no_mangle]
pub unsafe extern "C" fn md_vm_times(program: *mut Program, times: *const i64, n: usize) -> i32 {
    if n != (*program).events {
        return -1;
    }
    let given = if n == 0 { &[][..] } else { slice::from_raw_parts(times, n) };
    (*program).times = given.to_vec();
    0
}

/// Set the Jacobian's sparsity pattern: entry `k` of the jacobian tape's output is row
/// `rows[k]`, column `cols[k]`. Returns 0, or -1 (and changes nothing) when an entry is
/// outside `n x n` or appears twice.
///
/// # Safety
/// `program` comes from `md_vm_new`; `rows` and `cols` are valid for `nnz` values.
#[no_mangle]
pub unsafe extern "C" fn md_vm_pattern(program: *mut Program, rows: *const u64, cols: *const u64, nnz: usize, n: usize) -> i32 {
    let rows = if nnz == 0 { &[][..] } else { slice::from_raw_parts(rows, nnz) };
    let cols = if nnz == 0 { &[][..] } else { slice::from_raw_parts(cols, nnz) };
    let mut seen = std::collections::HashSet::with_capacity(nnz);
    let mut pattern = Vec::with_capacity(nnz);
    for (&i, &j) in rows.iter().zip(cols) {
        let (i, j) = (i as usize, j as usize);
        if i >= n || j >= n || !seen.insert((i, j)) {
            return -1;
        }
        pattern.push((i, j));
    }
    (*program).pattern = pattern;
    0
}

/// Set the diagonal mass matrix of `M y' = f`: one value per state, 0 for an algebraic row.
///
/// # Safety
/// `program` comes from `md_vm_new`; `mass` is valid for `n` values.
#[no_mangle]
pub unsafe extern "C" fn md_vm_mass(program: *mut Program, mass: *const f64, n: usize) -> i32 {
    let given = if n == 0 { &[][..] } else { slice::from_raw_parts(mass, n) };
    if given.iter().any(|m| !m.is_finite()) {
        return -1;
    }
    (*program).mass = given.to_vec();
    0
}

/// Choose how Radau solves its linear systems: 1 for BiCGStab with DILU, 0 for the band LU.
///
/// # Safety
/// `program` comes from `md_vm_new`.
#[no_mangle]
pub unsafe extern "C" fn md_vm_solver(program: *mut Program, iterative: i32) -> i32 {
    (*program).iterative = iterative != 0;
    0
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
    // One value per pattern entry (or the dense matrix with no pattern): Radau reads both.
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
        view(dirs, ne), view(lat, ne), view(term, ne), &(*program).times, view(grid, npts), &(*program).order, &(*program).pattern, &(*program).mass, (*program).iterative, every != 0,
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
                        view(term as *const i64, ne), &local.times, view(grid as *const f64, npts), &local.order, &local.pattern, &local.mass, local.iterative,
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
