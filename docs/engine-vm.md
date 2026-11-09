# How the engine's virtual machine works

This page describes the virtual machine inside the precompiled engine: the tape format, the
lowering passes, and what the engine does at run time. Read it before you change
`src/modelica/sim/vm.py` or `engine/src/lib.rs`. To use the engine, read
[The precompiled engine](engine.md).

The engine is built once and ships in the wheel. A model arrives as data (a tape) and the engine
interprets it. Nothing is compiled when a simulation runs.

## The pipeline

1. `vm.lower(system)` turns a causalized `System` into a `vm.Program`.
2. `vm.encode(program)` packs it into flat integer and float arrays.
3. `engine.load` passes the arrays to `md_vm_new`, which checks every register and constant
   index and refuses a bad program. `md_vm_order` sets the state order.
4. `md_vm_run` integrates one parameter vector. `md_vm_run_many` integrates many, one thread
   per core, each thread on its own copy of the program.

## The tape

A `Program` holds one `Function` per kernel and mode: `rhs`, `jacobian`, `observe`,
`indicators`, `transition_prefix`, `transition`, plus `parameters`, `initial` and `latch`.

A `Function` is straight-line code, a tuple of `Instr(op, dst, a, b, c)`, plus its Newton
blocks. Operands are register numbers unless the opcode says otherwise.

| Opcodes | What they do |
|---|---|
| `CONST`, `TIME`, `LOAD_Y`, `LOAD_P`, `LOAD_W` | read a constant, `t`, a state, a parameter, a workspace slot |
| `STORE_OUT`, `STORE_W`, `COPY_Y_OUT`, `CLEAR_OUT` | write outputs and the workspace |
| `MOVE`, `ADD` ... `POW`, `NEG`, `NOT`, comparisons, `AND`, `OR` | arithmetic and logic |
| `CALL1`, `CALL2` | a builtin (`c` names it: `sin`, `exp`, `atan2`, ...) |
| `SELECT` | `dst = a != 0 ? b : c` |
| `SOLVE` | a dense linear system over contiguous registers |
| `NEWTON` | run a Newton block until its step is below `1e-12` |
| `HOIST` | marker: everything before it reads only `time` and constants |

The constants live in the last registers. The engine sets them once at load time, so the tape
has no `CONST` instruction for a constant read as a single operand.

## Lowering passes

`vm.lower` runs these passes in order. Each one keeps every result bit for bit.

| Pass | What it does | vowel_i `rhs` |
|---|---|---|
| `_fold` | evaluates constant subexpressions and drops dead instructions (`_prune`) | a third of the tape |
| `_peephole` | reads `MOVE` copies through; `x + (-y)` becomes `x - y` | 1828 to 1748 instructions |
| `_preload` | constants read from their fixed registers | no `CONST` left |
| `_hoist` | moves time-only work before `HOIST` | 1105 instructions moved |
| `_compact` | numbers registers in the order `rhs` first uses them | 17010 to 5054 registers |

`bandwidth_order` picks the state order Radau factors in (reverse Cuthill-McKee on
`Mode.sparsity`). On vowel_i the bandwidth drops from 75 to 4.

The `jacobian` tape is built from `rhs` by forward mode (`_Tangents`). It keeps one partial per
structural non-zero, so it is already sparse.

## At run time

- `Program::run` reads registers without bounds checks. `md_vm_new` checked them all.
- Each function keeps 4 cache slots of the time-only values it last computed, keyed by `t`. Radau
  calls `rhs` at its three stage times in turn, then at the step start. On a hit the engine
  copies the saved values back and starts after `HOIST`.
- Radau builds `mu/h I - J` in the state order. The LU finds the band in the matrix and its
  loops stop there. Row swaps move only the part right of the pivot, as LAPACK's `gbtrf` does.
- The register file and the solve buffers live in a `Scratch` per program copy, so a call
  allocates nothing and threads share nothing.

## Cost on vowel_i

A 0.05 s run, user instructions counted with `perf stat`, minus a run with `stop=0`:

| Version | Instructions |
|---|---|
| first engine | 21.1e9 |
| folding, preloaded constants, unchecked registers | 14.6e9 |
| banded state order | 7.2e9 |
| time-only cache | 6.3e9 |
| compiled Rust backend (`rustsim`), for comparison | 11.2e9 |

16 vowel_i runs over 0.02 s on 16 logical cores take 4.65 s, against 27.0 s on one thread.

## Where the design is

- [The VM note](issues/08-11-2026-a-modelica-vm-runs-the-model-as-bytecode-in-a-precompiled-rust-engine.md): why a VM, the first plan.
- [The simulation IR note](issues/09-11-2026-a-simulation-ir-with-arrays-events-and-parallel-execution-for-the-engine.md): arrays, events, parallel runs, and the measurements behind each step.
