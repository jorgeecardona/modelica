---
title: A Modelica VM runs the model as bytecode in a precompiled Rust engine, so the fast path needs no compiler
status: open
kind: idea
scope: L
owner: jorgeecardona
filed: 2026-11-08
area: sim
verify: test
blocked_by: null
fixed_in: null
---

## What we want

A precompiled Rust engine, shipped in the wheel, that receives the flat model as data
(bytecode) and integrates it. No `rustc`, no C compiler, no per-model build on the user's
machine. When the engine is installed, `simulate()` uses it by default. When it is not
installed, `simulate()` uses the Python interpreter, as it does today.

`rustsim` stays. It is the opt-in path that compiles one model ahead of time
(`docs/rust-backend.md`).

## Measured

On the 80-state VowelA vocal tract model (#4):

| path | wall-clock |
|---|---|
| `simulate()`, Python | 5.0 s |
| `rustsim`, library cached | 0.3 s |
| `rustsim`, first run | about 4 s extra for `rustc` |

## Prior art

- CasADi has an SX virtual machine with this shape. An instruction is
  `{op, dst, src1, src2 | constant}` over one flat work vector, with slots reused after their
  last use ([sx_function.hpp](https://github.com/casadi/casadi/blob/main/casadi/core/sx_function.hpp)).
  A CasADi developer reports that code generation is about 4x faster than the VM
  ([casadi-users](https://groups.google.com/g/casadi-users/c/g_M2-2nxoHA/m/qaE1861MBgAJ)).
- A register VM dispatches about 46% fewer instructions than a stack VM and runs about 1.5x
  faster with switch dispatch (Shi, Casey, Ertl, Gregg, "Virtual Machine Showdown",
  [MURAL](https://mural.maynoothuniversity.ie/10185)).
- CppAD and ADOL-C replay a linear tape with a switch per opcode
  ([CppAD](https://cppad.readthedocs.io/stable-2025/)).
- OpenModelica, Dymola, ModelingToolkit (Julia JIT) and diffsol/DiffSL (Cranelift or LLVM JIT)
  all compile the model. None of them interprets equations
  ([OpenModelica forum](https://openmodelica.org/forum/default-topic/2080-modelica-interpreter.html),
  [diffsl](https://docs.rs/crate/diffsl/0.12.3)).

## Design

### Boundary

Python keeps parsing, flattening, causalization, block ordering (`Emitter.solve`), tearing,
`block_jacobian`, the symbolic tangent walk (`_walk` in `emit.py`), event classification
(`events()`), and the `Code` layout (states, workspace, latches, switching slot).

The engine gets the five functions the Rust unit already has (`_SIGNATURES` in `rustgen.py`):
`rhs`, `observe`, `indicators`, `transition`, `jacobian`. Each one is a program (a tape), not
source text.

The event loop, bisection, latch iteration and chattering counter stay in `md_integrate`
(`rust_radau.py`). They call only `indicators` and `transition`. Moving them to Python would
cross the FFI on every step.

### One runtime, two models

Make `md_integrate` generic over a trait:

```rust
trait Model {
    fn rhs(&mut self, t: f64, y: &[f64], p: &[f64], w: &mut [f64], dy: &mut [f64]);
    fn observe(&mut self, /* ... */);
    fn indicators(&mut self, /* ... */);
    fn transition(&mut self, event: usize, /* ... */);
    fn jacobian(&mut self, /* ... */);
}
```

- `rustgen` emits an `impl Model` with inlined bodies (the `rustsim` path, same speed as today).
- The VM is `impl Model for Program`.

Radau, LU and the event loop then exist once. The runtime becomes a real crate. This replaces the
string concatenation of `CONSTANTS`, `LINALG`, `STEP` and the rest in `rustsim.py`.

### Instruction set

A register tape: a flat `Vec<Instr>` of three-address instructions over one `f64` work vector.
This is not a stack VM (more dispatches) and not a tree (pointer chasing).

| opcode | meaning |
|---|---|
| `Load(dst, src)` | `src` is `Y(i)`, `P(i)`, `W(i)`, `T` or `Const(k)` |
| `Bin(op, dst, a, b)` | the dialect's binary table (`dialect.py`) |
| `Un(op, dst, a)` | the unary table |
| `Call(fn, dst, args)` | the `f64::` builtins and `md_min/max/sign/mod/rem` |
| `Select(dst, c, a, b)` | if-expression: both branches computed, tape stays branch-free |
| `Store(dst, src)` | to `Out(i)` or `W(i)` |
| `Solve(n, A, b, x)` | linear block through the existing `md_solve` |
| `Newton(block)` | runs a residual sub-tape and a Jacobian sub-tape until converged |

- Parameters stay runtime data (`p[]`) and are never folded, so one program serves a whole sweep.
- `transition` is one sub-tape per event, picked by event index. Mode dispatch is one tape per
  mode, as `_dispatch` in `rustgen.py` does it.
- Work-vector slots are reused after a value's last use, as CasADi does.
- A program is serialized as one flat binary buffer. Python objects are never passed across the
  FFI.

`Select` evaluates both branches. A branch that is not taken can produce NaN or a domain error
(for example `if x > 0 then sqrt(x) else 0`). `Select` must discard the unused value without
raising. Check this in the tests.

### Jacobian

Use a symbolic tape built from the existing forward-mode tangents (`_walk`, `tangent`). Do not
use AD inside the VM, because that is a second interpreter design. Do not use finite
differences: for 80 states that is 80 extra `rhs` calls for each Jacobian.

### Lowering

The tape is built from the `Emitter`'s structured output, never by reparsing emitted text. The
regex passes in `rustgen.py` (`_bind`, `_live`) show why reparsing breaks. One option is a
`TAPE` dialect that returns instructions instead of strings.

### Packaging

- Build with pyo3 and maturin as `abi3-py312`: one wheel per platform (linux x86_64 and aarch64,
  macOS, windows x64).
- Build wheels only on release, not on every push. 5 wheels at 3 to 6 shared-runner minutes each
  is about 30 minutes per release.
- `simulate()` tries `import modelica._engine` and falls back to the Python interpreter.

## Expected speed

The tape costs about 3 to 10 ns per operation, against about 0.5 ns inlined. CasADi measures
about 4x between its VM and compiled code. LU, Newton and the event loop stay compiled, and for
stiff models with 50 or more states they take most of the time.

Estimate for VowelA: 0.6 to 1.5 s. That is 3x to 8x faster than Python and 2x to 5x slower than
compiled `rustsim`. Milestone 2 measures this.

## Later option: Cranelift JIT in the engine

Cranelift is a set of pure Rust crates and can live inside the same wheel. The engine would
compile the tape to machine code at load time, in about 1 to 10 ms, with no `rustc` on the
user's machine. That removes the dispatch overhead. Do this only if the interpreter is too slow
after Milestone 2. The tape format is the input in both cases, so the decision can wait.

## Risks

1. A second translation of the `Emitter` output can drift from the others. Mitigation: test it
   differentially against the interpreter, as `tests/test_rustgen.py` does.
2. Semantics of `mod`, `sign`, and NaN in `min`/`max`. Mitigation: reuse the Rust helpers.
3. CI time: the Rust toolchain in the job. Mitigation: cache cargo; build wheels only on release.
4. Dense LU limits every path at hundreds of states. The VM does not change that.

## Milestones

1. `modelica.sim.vm.lower(system) -> Program`, and a Rust `vm::eval`. Test: `rhs`, `observe`,
   `indicators`, `transition` and `jacobian` equal the interpreter at every corpus model's start
   point. Newton blocks may refuse with an `EmitError` at first.
2. `md_integrate` generic over `Model`. Test: the bouncing ball through the VM gives the same
   `event_list` as the interpreter. Benchmark VowelA.
3. Newton blocks, modes and when-transitions. Test: the corpus that `rustsim` passes also passes
   through the VM.
4. Wheels in CI, and `simulate()` uses the engine by default when it is installed.

## Plan to make the engine the default (2026-11-08, Fable review)

1. Fix the emitter bug: a shared subexpression defined inside a Newton loop is read stale after it.
2. `Result.backend` says who integrated. Done in 3960b8e.
3. Measure `steps`, `jacobians` and inner Newton iterations for the engine and `rustsim` on
   vowel_i at 0.05 s and 0.35 s. Benchmark the cargo build, not the `rustc -O` checkout build.
4. Variable-length output: every accepted step when there is no grid, both sides of each event,
   and the start and stop instants. A row count is returned, plus an overflow status; `drive`
   retries with a bigger buffer.
5. Start at `start`: `Loaded.start(t, parameters)`, `t0` in `drive`, one grid function.
6. `sweep` loads the engine once and runs every point on it.
7. Reuse one register frame per program, and factor without copies.
8. Cross-build aarch64 Linux and Windows engines in the manylinux job
   (`MODELICA_ENGINE_LIBRARY` in `hatch_build.py`), plus an arm64 smoke job.
9. macOS arm64 wheel on `saas-macos-medium-m1`, main only.
10. Docs: the `simulate` docstring, the engine page, CHANGELOG.

Deferred: `Code` and `Kernel` on the engine, methods other than Radau, Cranelift, macOS x86_64.

### Step 3: measured on vowel_i (2026-11-09, cargo release build, CPU time)

| stop | path | time | steps | rejected | evaluations | jacobians | factorisations |
|---|---|---|---|---|---|---|---|
| 0.05 s | rustsim | 7.1 s | 12963 | 434 | 93523 | 24 | 991 |
| 0.05 s | engine | 6.7 s | 12963 | 434 | 93523 | 24 | 991 |
| 0.35 s | rustsim | 26.0 s | 103678 | 2843 | 743138 | 141 | 6515 |
| 0.35 s | engine | 53.6 s | 103711 | 2856 | 743447 | 141 | 6530 |

The counters agree to 0.03 %, so both paths integrate the same function. The difference is the
cost per call: about 0.52 ms per step for the engine against 0.25 ms for rustsim on the long
run. The short rustsim run costs more per step (0.55 ms); the cause is not measured. Step 7
(one register frame per program, no copies in `solve`) is the fix to try.

### Step 7: one register file per program (2026-11-09)

A call no longer allocates its register file or the `solve` copies; the program keeps them.
vowel_i over 0.05 s, cargo release build, CPU time: engine 6.7 s before, 4.7 s after, the same
12963 steps. rustsim took 8.3 s in the same session, so the machine was noisy; repeat on a quiet
machine before quoting a ratio.
