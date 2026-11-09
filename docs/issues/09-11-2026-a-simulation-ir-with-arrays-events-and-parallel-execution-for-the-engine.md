---
title: A simulation IR with arrays, events and parallel execution for the engine
status: open
kind: idea
scope: L
owner: jorgeecardona
filed: 2026-11-09
area: sim
verify: test
blocked_by: null
fixed_in: null
---

## What we want

An IR made for simulation, that the precompiled engine runs as data. It must:

- Treat ODEs, DAEs and PDEs as first-class: states, equations over index sets, domains and
  grids, algebraic loops, implicit blocks.
- Treat discrete parts as first-class: zero crossings, `when` clauses, modes, clocks, sampled
  components, event iteration.
- Use every processor the machine has.
- Never compile at simulation time. No `rustc`, no Cranelift, no LLVM, no XLA. The engine is
  built once and shipped in the wheel; the model arrives as data and is interpreted.

This replaces the scalar register tape of
[the VM note](08-11-2026-a-modelica-vm-runs-the-model-as-bytecode-in-a-precompiled-rust-engine.md)
as the main format. The scalar tape stays working while the new IR grows.

## Why the scalar tape is not enough

The tape has one instruction per arithmetic operation, on one number.

- vowel_i is an 80-cell acoustic tube: a 1-D PDE discretized by hand. Its `rhs` is about 1800
  scalar instructions, most of them the same formula once per cell.
- The engine uses 14.6e9 user instructions for a 0.05 s run against 11.2e9 for compiled Rust
  (`perf stat`). Constant folding, preloaded constants and unchecked registers got it there from
  21.1e9. Fused instruction pairs gave nothing (14.66e9 against 14.64e9) and were dropped.
- An interpreter pays its dispatch cost once per instruction. An instruction over a whole array
  pays it once per array. That is how NumPy is fast without compiling, and it is the speed fix
  that keeps the engine free of a compiler.
- Events, modes and latches are lowered to `SELECT`, `STORE_W` and indicator functions. The
  engine cannot see them as events, so it cannot schedule or parallelize around them.

## Draft shape

Two levels.

1. Model level: variables over index sets, equations, domains, events, clocks, modes. This is
   what the Python side produces after flattening and causalization.
2. Execution level: an array tape. One instruction applies to a range of indices (a slice, a
   stencil, a reduction). Scalar code is the length-1 case.

Parallel work comes from three places:

- Ensembles and sweeps: independent runs, one per core.
- Inside one run: array instructions split across threads, Jacobian columns by colouring, a
  parallel or sparse LU for large grids.
- Across components: parts that only meet at events step independently between them.

## Fable review (2026-11-09)

Verdict: add an array-level IR above the scalar tape, not instead of it. The scalar tape stays
the leaf. Sparse or banded LU is the other half of the speed, and does not depend on the IR.

### Prior art to take

- Base Modelica (MCP-0031): arrays survive flattening. Today `FlatModel` scalarizes everything,
  which is why vowel_i is 1800 instructions.
  [ReadMe](https://github.com/HansOlsson/ModelicaSpecification/blob/b31ac8303ac513cc6442221c50c2287b014ec538/RationaleMCP/0031/ReadMe.md)
- MARCO (MLIR `bmodelica`): matching and BLT over array equations with index ranges.
  [slides](https://llvm.org/devmtg/2025-10/slides/student_talks/scuttari.pdf)
- CasADi SX against MX: array nodes pay only when the array is long. Keep scalar ops for short
  and irregular code.
  [FAQ](https://github.com/casadi/casadi/wiki/FAQ:-what-is-the-difference-between-SX-and-MX%3F)
- numexpr: an array VM with no JIT. Bytecode over chunks, chunks spread across cores.
  [intro](https://numexpr.readthedocs.io/en/latest/intro.html)
- FMI 3.0 clocks: periodic, aperiodic and triggered clocks, and clocked partitions.
  [spec](https://fmi-standard.org/docs/3.0/#_clocked_co_simulation)
- ModelingToolkit: an event is a condition and an affect.
  [events](https://docs.sciml.ai/ModelingToolkit/dev/basics/Events/)
- OpenModelica SimCode: Jacobian sparsity pattern and column colouring.
  [guide](https://openmodelica.org/doc/OpenModelicaUsersGuide/1.27/solving.html)
- SUNDIALS: sparse Jacobian in CSC, roots as their own function, KLU.
  [docs](https://sundials.readthedocs.io/en/latest/)

Rejected: an MLIR dependency, hybrid automata as the top level, and jaxpr (no events, no implicit
blocks).

### Levels

1. `ModelIR`: `IndexSet(name, size)`, `Variable` with a shape over index sets, `Equation` written
   once over an index set with stencil reads (`x[i+1]`), `Event`, `Clock`, `ClockedPartition`,
   `StateSet` (today's `Mode`), and an optional `Domain` (grid metadata). `FlatModel` is the case
   where every index set has size 1.
2. `Schedule`: blocks over index ranges (one block for every `i`), the Jacobian `Sparsity` built
   from the stencil offsets, and a `Coloring` of the Jacobian columns.
3. `Tape`: today's opcodes plus array opcodes (`VADD`, `VMUL`, `VCALL1`, `VSHIFT`, `VLOAD_Y`,
   `VSTORE_OUT`, `VSELECT`) over contiguous runs of registers. One dispatch per array, a loop
   inside that the Rust compiler vectorized when the engine was built. Scalar ops stay for index
   sets shorter than about 8; measure the threshold.

PDEs: take the arrays the user discretized. No `grad` or `laplacian` in the IR now; the `Domain`
node and the stencil offsets leave room for a later method-of-lines pass.

### Discrete parts

- `ZeroCrossing(expr, direction, hysteresis)` replaces the `directions` and `latches` arrays
  passed to `md_integrate`.
- `When(condition, body, group)`: the group is the set of events whose bodies write what other
  conditions read; the engine iterates inside a group up to a cap set in the IR.
- A state-set switch is a `When` whose body sets the mode and rebases the states.
- `Clock`: the engine keeps a heap of next clock times, steps to the earliest and runs the
  partition's tape, with no zero-crossing search. `previous()` is a held slot.
- `Workspace` describes the `w` layout, so the engine needs no special switching index.

### Parallelism, by payoff

1. Ensembles and sweeps: one run per core with `std::thread::scope`, one `Scratch` per thread.
   At 80 states this is the only parallelism that pays.
2. Jacobian by colours: a few tangent sweeps instead of one per state; threads only for large n.
3. Array ops split across threads only above about 10k elements per op.
4. Parallel sparse LU: not worth writing. A sequential sparse LU plus a parallel Jacobian.
5. Multirate partitions on threads: deferred; it changes the numerics.

The thread-local `CURRENT` program pointer in `lib.rs` must go first: a parallel run passes the
program through `md_integrate`.

### Solver

- Check that factorisations are reused when the step changes little, as Hairer's Radau does.
- Banded LU first (it covers 1-D PDEs, vowel_i has about 3n non-zeros of n^2), then a general
  sparse LU whose symbolic analysis (ordering, fill) is done in Python and shipped as data.
- Keep Radau. Add Dormand-Prince for non-stiff models and sweeps. BDF only for models above a
  few thousand states.

### Plan

Each step has a test, and every existing model lowers as before when all index sets have size 1.

1. Ensembles on threads: `md_vm_run_many`, no thread-local program. Test: a 16-point sweep equals
   16 sequential runs bit for bit.
2. Banded LU in Radau, bandwidth from `Mode.sparsity`. Test: vowel_i equal to dense within 1e-10.
3. Coloured Jacobian. Test: equal to the dense Jacobian at corpus start points.
4. `IndexSet` and stencil reads survive flattening; `FlatModel.scalarize()` gives today's model.
   Test: the corpus flattens the same after scalarize.
5. Array opcodes in the engine and `vm.py`. Test: vowel_i `rhs` below 100 dispatches, equal to
   the scalar tape within 1e-12.
6. Event and clock nodes in the IR. Test: the event corpus unchanged, and one synchronous `Clock`
   model gives the same samples on both backends.
7. General sparse LU. Test: a 2-D heat grid of 2500 states runs.
8. Array ops split across threads above a threshold. Test: equal to one thread bit for bit.

Deferred: PDE operators, multirate integration, parallel LU, BDF, any JIT.

Traps: a second causalizer for arrays (extend `causalize.py` instead); an array path that skips
the `Emitter`; `x[i+1]` at the last cell must be a lowering error, not a wrap; a versioned flat
buffer format for the FFI.

### Step 1 done: parallel runs (2026-11-09)

`md_vm_run_many` integrates many parameter vectors at once, one thread per core, each thread on
its own copy of the program. `engine.results` (and so `sweep`) uses it for more than one vector.
Each thread sets its own thread-local program pointer, so `md_integrate` did not change.

16 vowel_i runs over 0.02 s, wall clock, 16 logical cores:

| threads | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| time | 27.0 s | 7.2 s | 4.9 s | 4.65 s |

Test: 16 vectors in parallel give bit-identical results to 16 single runs.

### Step 2 done: banded LU in Radau (2026-11-09)

`vm.lower` picks a state order with reverse Cuthill-McKee on `Mode.sparsity`
(`vm.bandwidth_order`) and the engine stores it with `md_vm_order`. Radau builds `mu/h I - J`
in that order. The LU finds the band in the matrix it is given and its loops stop at the band.
Row swaps move only the part right of the pivot column, as LAPACK's `gbtrf` does, so the solve
stays inside the band too.

- vowel_i: 78 states, bandwidth 75 in the natural order, 4 in the new order.
- Only 158 of 2337 steps factor. Most of the gain is in the solves, which run every Newton
  iteration.
- rustsim does not pass an order, so it factors as before.

User instructions for a 0.05 s vowel_i run (`perf stat`, minus a stop=0 run):

| | instructions |
|---|---|
| engine, natural order | 16.1e9 |
| engine, banded order | 7.2e9 |
| rustsim (compiled) | 11.2e9 |

Tests: any state order (the reversed one) gives the natural order's trajectory within 1e-9;
a chain with bandwidth n/2 gets bandwidth 1; an order that is not a permutation is refused.
vowel_i over 0.01 s: same steps and factorisations, largest difference 1.2e-4 on values up to
3.4e6 (4e-11 relative).

### Step 3 not needed: the Jacobian is already sparse (2026-11-09)

`_Tangents` keeps one partial per structural non-zero, so the Jacobian tape is already the
compressed one that colouring would give. A 0.05 s vowel_i run:

| | tape instructions | calls |
|---|---|---|
| `rhs` | 1827 | 93523 |
| `jacobian` | 1984 (222 non-zeros) | 24 |

The Jacobian is about 0.03% of the run. Colouring stays useful only for a Jacobian made by
finite differences, which the engine does not do. The run time is now in `rhs`: step 5 (array
opcodes) is where the next gain is.

### Fable review 2: scalar tape first (2026-11-09)

vowel_i is written as unrolled scalars (`p1..p20`, `u2..u20`), so step 4 does not touch it. Its
`rhs` is not array-shaped; it repeats work:

- 1104 of 1827 instructions read only `time` and constants, including all 244 `sin` calls.
- Radau calls `rhs` about 40 times per step at about 4 distinct times.
- The register file is 17010 values (136 KB) for 1827 instructions, 3 to 4 times the L1 cache.
- 80 `MOVE` and 156 `NEG` can go: `a + (-b)` equals `a - b` bit for bit.

New order, before steps 4 and 5:

1. Time-only hoisting: each function starts with the instructions that read only `time` and
   constants; the engine skips them when `t` equals the last call's `t`. Bit-identical.
2. Register compaction: reuse function-local registers so the file fits in L1. Measure cycles,
   not instructions.
3. Peephole: copy propagation for `MOVE`, `NEG` folded into `ADD` and `SUB`.

Steps 4 and 5 stay, measured on TorusHeat, HeatBar, WaveString and FisherKPP. Rerolling scalar
runs into array ops is rejected: it recovers at the tape what step 4 keeps at the source, and
244 `sin` calls with different constants would still be 244 calls.

### Time-only hoisting done (2026-11-09)

`vm._hoist` moves the instructions that read only `time` and constants to the front of `rhs`,
`jacobian`, `observe` and `indicators`, followed by a new `Op.HOIST`. On vowel_i 1105 of 1828
`rhs` instructions move, all 244 `sin` calls among them.

A cache of one `t` hit almost never: Radau calls `rhs` at its three stage times in turn. The
engine keeps 4 slots per function instead. A slot saves only the moved registers that the rest
of the function reads, and a hit copies them back, so another function that writes the same
registers (the jacobian repeats the rhs's instructions) does no harm.

0.05 s vowel_i, user instructions minus a stop=0 run: 7.46e9 without, 6.26e9 with (16% less).
Bit-identical. The gain is smaller than estimated because `rhs` is now about a quarter of the
run; Newton and the linear algebra are most of the rest. No corpus model has enough time-only
work to be hoisted; a test model with time-driven noise and a state event checks the result.

### Register compaction done (2026-11-09)

`vm._compact` numbers the registers in the order `rhs`, `jacobian`, `indicators`, `observe` and
the rest first use them. Ranges a solve or a Newton block reads move as one. No register is
reused, so results are bit-identical.

vowel_i: 17010 registers before, 5054 after; `rhs` uses registers 0 to 1748 plus constants.
0.05 s run over 3 repeats: L1 data misses 241e6 before, 230e6 after; cycles change by less than
the noise (15.63e9 against 15.52e9, whole process). A small gain; it stays because it is free.

### Peephole done (2026-11-09)

`vm._peephole` reads copies through (`MOVE`) and folds `x + (-y)` into `x - y` and `x - (-y)`
into `x + y`, inside one function; `_prune` drops what nothing reads. vowel_i `rhs`: 1828 to
1748 instructions, no `MOVE` left. Its 156 `NEG` stay: 78 are `DIV` numerators and the rest
feed outputs, and folding `-(a - b)` into `b - a` would change the sign of a zero result.

### Why step 4 still matters: the Python front end scales with scalars (2026-11-09)

On the corpus the engine's run is no longer the cost. TorusHeat (64 states) spends 0.091 s of
0.096 s in `vm.lower` and under 5 ms integrating. Grown as an `n x n` grid over 0.2 s:

| n | states | flatten + causalize | `vm.lower` | `simulate` total |
|---|---|---|---|---|
| 8 | 64 | 0.17 s | 0.10 s | 0.11 s |
| 16 | 256 | 0.65 s | 0.42 s | 0.35 s |
| 24 | 576 | 1.30 s | 1.08 s | 1.39 s |
| 32 | 1024 | 3.61 s | 1.86 s | 1.77 s |

Flattening grows faster than linearly. Every stage handles one scalar per cell, so a 50 x 50
grid (step 7's test) would spend most of its time before the engine starts. Step 4 (index sets
survive flattening) and step 5 (array opcodes) shrink this work by the grid size, not the run.
Step 4 is therefore measured on front-end time, not on engine instructions.

### Fable review 3: the step 4 design (2026-11-09)

Keep a `for` loop as an `ArrayEquation` in `FlatModel` (not a side list), with
`FlatModel.scalarize()` giving today's equations in today's order. Consumers call `scalarize()`
at entry, so nothing breaks while the array path grows.

- Data: `IndexSet(name, start, stop)`, `ArrayEquation(indices, lhs, rhs, span)`, and a
  `Gather(source, table)` of flat element positions per read, one per cell.
- Stencils are not classified. Every index expression is evaluated at every cell and range
  checked: `mod(i - 2, n) + 1` gives the periodic table, `x[i + 1]` at the last cell is an
  error naming the cell.
- Causalize stays one causalizer: structure (matching, Tarjan) per cell, symbols per loop. When
  every cell of a loop is a 1x1 block solved for the same pattern, `_blocks` emits one
  `ArrayBlock`; otherwise the whole loop stays scalar. `scalar_blocks(mode)` is the one fallback.
- Only `Equal` bodies over constant unit-step ranges; everything else unrolls eagerly as today.
  Initial equations unroll eagerly.

Commits: (1) `_affine` stops at a third unknown; (2) the data model and `scalarize()`; (3)
`flatten` emits `ArrayEquation`; (4) `ArrayBlock` in `_blocks`; (5) array opcodes in `vm` and
the engine; (6) `ArrayBlock` in the C and Rust emitters. Expected front end on the 32 x 32
grid: 7.1 s today, about 1.3 s after (5).

Commit 1 done: `_affine` stops once a third unknown has a non-zero coefficient, instead of
differentiating every unknown first. Same result in every case. Causalize on the 32 x 32 grid:
1.74 s to 0.53 s.
