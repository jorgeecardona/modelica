# The tape

A tape is the straight-line code of one kernel function. This page defines its parts and shows
how to read one.

## Program and functions

`vm.lower(system)` returns a `vm.Program`. It holds one `vm.Function` per kernel and mode:

| Kernel | What it computes |
|---|---|
| `rhs` | the state derivatives |
| `jacobian` | `d rhs / d y`, row-major, by forward mode |
| `observe` | the observed variables |
| `indicators` | the zero-crossing functions |
| `transition_prefix`, `transition` | what an event changes |
| `parameters`, `initial`, `latch` | parameter vector, start values, relation latches |

A `Function` has three fields:

- `code`: a tuple of `Instr(op, dst, a, b, c)`.
- `newton`: its Newton blocks, each a body of instructions run until the step is small.
- `tables`: register numbers that gather and scatter opcodes read through.

## Registers and constants

All values live in one register file of `Program.registers` doubles. Each register holds one
value per call, and no pass reuses a register for a second value. The exception is a Newton
block's unknowns, which every round updates in place.

The constants are the last `len(Program.constants)` registers. The engine fills them once at
load time, so an instruction reads a constant like any register, and the tape has no load for
it.

## Read a tape

`vm.listing` prints a function. RLCCircuit's `rhs`:

```python
from pathlib import Path

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten, vm

files = ("examples/library/Electrical.mo", "examples/models/RLCCircuit.mo")
system = causalize(flatten([parse(Path(f).read_text()) for f in files], "RLCCircuit"))
program = vm.lower(system)
print(vm.listing(program, program.rhs[0]))
```

```text
   0  r0 = p[0]
   1  r1 = p[1]
   2  r2 = p[2]
   3  r3 = p[3]
   4  r4 = y[0]
   5  r5 = y[1]
   6  r6 = r1 * r4
   7  r7 = r0 - r6
   8  r8 = r0 - r0
   9  r9 = r8 / -1.0
  10  r10 = r5 - r9
  11  r11 = r7 - r10
  12  r12 = -r2
  13  r13 = r11 / r12
  14  r14 = -r13
  15  r15 = -r3
  16  r16 = r4 / r15
  17  r17 = -r16
  18  out[0] = r14
  19  out[1] = r17
```

Twenty instructions: four parameter loads, two state loads, the arithmetic, two outputs. A
constant shows as its value (`-1.0`). No pass turns `r0 - r0` into 0: it is NaN when `r0` is, so the tape keeps the source's
arithmetic.

## Opcodes

Operands are register numbers unless the table says otherwise.

| Opcode | Meaning |
|---|---|
| `CONST` | `dst = constants[a]` (only where a constant cannot be read in place) |
| `TIME` | `dst = t` |
| `LOAD_Y`, `LOAD_P`, `LOAD_W` | `dst = y[a]`, `p[a]`, `w[a]` (state, parameter, workspace) |
| `STORE_OUT`, `STORE_W` | `out[a] = b`, `w[a] = b` |
| `MOVE` | `dst = a` |
| `ADD`, `SUB`, `MUL`, `DIV`, `POW` | `dst = a op b` |
| `LT`, `LE`, `GT`, `GE`, `EQ`, `NE`, `AND`, `OR` | `dst = 1.0` or `0.0` |
| `NEG`, `NOT` | `dst = -a`, `dst = (a == 0)` |
| `CALL1`, `CALL2` | `dst = f(a)` or `f(a, b)`; `c` names `f` (`vm.Builtin`) |
| `SELECT` | `dst = b if a != 0 else c` |
| `SOLVE` | `x = A^-1 b`, dense LU over runs: `dst` is `x`, `a` is `A`, `b` is `b`, `c` is `n` |
| `NEWTON` | run Newton block `a`: body, solve, add the step; stop at `max abs(step) < 1e-12` or after 50 rounds |
| `COPY_Y_OUT`, `CLEAR_OUT` | `out = y`, `out[:a] = 0` |
| `HOIST` | marker: every instruction before it reads only `time` and constants |

The array opcodes work on *runs*: `n` consecutive registers from a first one.

| Opcode | Meaning |
|---|---|
| `VGATHER` | `dst[k] = r[tables[a + k]]`, `k < c` |
| `VSCATTER` | `r[tables[a + k]] = b[k]`, `k < c` |
| `VBROADCAST` | `dst[k] = a`, `k < c` |
| `VADD`, `VSUB`, `VMUL`, `VDIV` | `dst[k] = a[k] op b[k]`, `k < c` |
| `VNEG` | `dst[k] = -a[k]`, `k < c` |
| `VCALL1` | `dst[k] = f(a[k])`, `k < b`, `c` names `f` |
| `VLOAD_Y` | `dst[k] = y[a + k]`, `k < c`; written only by `_fuse`, in `rhs` |
| `VSTORE_OUT` | `out[a + k] = b[k]`, `k < c`; written only by `_fuse`, in `rhs` |

[Array models](arrays.md) shows them in a real tape.
