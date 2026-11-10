# Changing the VM

This page is for changing the tape format: the encoding both sides agree on, and every place an
opcode lives.

## Encoding

`vm.encode` writes these arrays, which `md_vm_new` reads with fixed row widths (`lib.rs`):

- `code`: 5 integers per instruction (`op, dst, a, b, c`).
- `functions`: 4 per function (first instruction, count, first Newton block, block count).
- `newton`: 7 per block (first unknown, count, body start, body length, matrix, right, step).
- `unknowns`: the Newton blocks' unknowns and every function's `tables`. A `VGATHER` or
  `VSCATTER` names its table by offset in this array; `encode` shifts the offset.
- `constants`: the float pool, loaded into the last registers.
- `views`: per view `base, ndim, *shape, *strides`, every function's `Function.views` in turn.
  A shaped opcode names its view by index in this array; `encode` shifts the index.
  `md_vm_new` refuses a view past the register file or with more than 8 axes, and a shaped
  instruction whose views do not fit its opcode.
- Four scalars beside the arrays: `registers`, `modes`, `events`, `switching` (-1: no mode
  slot). `md_vm_new` also refuses a pool larger than the register file.

`vm.FORMAT` is the tape format number. `md_vm_format` returns the engine's, and
`engine.load` refuses a library whose number differs. Raise both when the encoding changes.

Function index order: `rhs` per mode, `observe`, `indicators`, `transition_prefix`,
`jacobian`, then every `transition` branch, then `parameters`, `initial`, `latch`
(`vm.Encoded` has a method per kernel).

## Adding an opcode

Change all of these in one commit. The engine matches raw numbers, so `vm.Op` and `lib.rs`
agree by hand.

1. `vm.Op`: the number. Add it to `_PURE` when it writes `dst` from register operands only, or
   to `_REMOVABLE` when it writes `dst` from something else (a load). Array opcodes go in
   `_ARRAY`, shaped ones in `_SHAPED` with their read operands in `_VIEW_READS`. Add it to `_FIELDS` with the fields that name a register or a run's first one.
2. `vm.py`: `_operands`, `_apply`, `_reads`, `_writes`, ranges included, and the place that
   emits it (`_Lowering.compute` or `_Lowering.array`, with `_BINARY_OPS` or `_VECTOR_OPS`). For an array opcode,
   `_array_reads`, `_array_writes` and `_array_runs`, so `_compact` keeps its runs together.
3. `_Tangents.step` (or `sstep` for a shaped opcode): its derivative rule. Without it the
   jacobian is wrong, with no error.
4. `_Machine`: the Python evaluator the tests compare the engine with.
5. `_line`, and `_SYMBOLS` for an operator: how `vm.listing` prints it.
6. `lib.rs`: `valid` (`shaped` for a shaped opcode; else `md_vm_new` refuses the program),
   `Program::run`, and `reads` and `writes` (else the time cache saves the wrong registers).
7. A test that runs it in the engine and in `vm.evaluate` and compares the outputs.

## Rules

- The engine is built once. A model never brings code to compile; an unknown opcode, scheme or
  boundary rule is an error.
- A new pass keeps the invariants in [Lowering](lowering.md#invariants).
