# Lowering

`vm.lower(system)` turns a causalized `System` into a `Program`. This page lists the steps in
order and the rules every step keeps.

## Steps

1. **Write.** `_Lowering` walks each mode's blocks in order, as `Emitter` does for the Python
   path, and writes one tape per kernel. An `ArrayBlock` in `rhs` is written with
   array opcodes ([Array models](arrays.md)).
2. **Jacobian.** `_Tangents` derives `d rhs / d y` from the scalar `rhs` tape by forward mode.
   It keeps one partial per structural non-zero and writes them into a dense row-major output,
   zeroed by one `CLEAR_OUT`.
3. **State order.** `bandwidth_order` picks the order Radau factors in, by reverse
   Cuthill-McKee on `Mode.sparsity`. On WaveString the bandwidth drops from 9 to 2.
4. **Passes.** Each one keeps every result bit for bit.

| Pass | What it does |
|---|---|
| `_fold` | folds constant subexpressions; `_peephole` reads `MOVE` copies through and turns `x + (-y)` into `x - y`; `_prune` drops what nothing reads |
| `_preload` | a constant read as one operand is read from its fixed register, with no `CONST` |
| `_hoist` | in `rhs`, `jacobian`, `observe`, `indicators`: moves the instructions that read only `time` and constants before `HOIST`, when at least 4 of them are arithmetic (loads of `time` and constants do not count) |
| `_compact` | numbers registers in the order `rhs` first uses them, so one call touches few cache lines |

## What the passes save

The `rhs` tape of some examples, with every pass off (`_fold`, `_preload` and `_compact`
replaced by the identity, `hoist=False`, `arrays=False`) and with all of them on, at 0.0.241:

| Model | States | `rhs` instructions | Registers |
|---|---|---|---|
| RCCircuit | 1 | 17 to 14 | 98 to 43 |
| BouncingBall | 2 | 8 to 5 | 47 to 20 |
| RLCCircuit | 2 | 26 to 20 | 131 to 57 |
| DCMotor | 3 | 46 to 31 | 252 to 109 |
| TorusHeat | 64 | 580 to 513, 144 with array opcodes | 2307 to 1319, 1832 with array opcodes |

Among the examples only RoastingMachine gets a `HOIST`: its burner schedule is time-only
arithmetic. The others have fewer than 4 such instructions. It pays most on a model with a
signal source built from many sines, where it skips the source on most `rhs` calls.

## Invariants

Every pass keeps these, and a new pass must too:

- A register never holds a second value, except a Newton block's unknowns. `_compact`
  renumbers; it does not reuse.
- A run that `SOLVE`, a Newton block or an array opcode reads stays contiguous and moves as one.
- A `transition` branch keeps reading the registers its `transition_prefix` wrote.
- `_peephole` and `_hoist` touch only registers written once in the function.
- `_peephole` leaves a function with Newton blocks as it is: a body runs many times, so a copy
  read through could see a later value.
- A function keeps its `tables`; `_compact` renumbers them with the registers.

To check that a step changes nothing, `vm.lower` takes `hoist=False` and `arrays=False`, and
the tests compare trajectories bit for bit.
