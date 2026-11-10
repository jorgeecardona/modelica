---
title: Shaped arrays on the tape, so n-dimensional states run without flattening
status: open
kind: idea
scope: L
owner: jorgeecardona
filed: 2026-11-10
area: sim
verify: test
blocked_by: null
fixed_in: null
---

A state written `Real T[nx, ny, nz]` should reach the engine as one array with a shape, and
the tape should act on it by axis. Today the tape has no shape: every array is a flat run of
registers, and the shape is lost at `flatten`.

## Today

- `flatten` writes one scalar equation per cell. `FlatModel.loops` keeps the `for` loop as an
  `ArrayEquation` (index sets, body, shapes) beside the scalars.
- `causalize` records an `ArrayBlock(first, equation)` only when every cell is its own explicit
  1x1 block in cell order. One mode, no alias, no index reduction.
- `vm.lower` writes the block with flat opcodes of length ``c``: `VLOAD_Y`, `VADD`,
  `VMUL`, `VGATHER`, `VSCATTER`, `VSCATTER_ADD`, `VBROADCAST`, `VCALL1`.
- A neighbour read `T[mod(i, n) + 1, j]` is one `VGATHER` through a table with one entry per
  cell. TorusHeat 8 x 8: five tables of 64 entries for one stencil.
- The array path falls back to scalar blocks for an `if`, a comparison, `^`, a two-argument call
  (`max`, `mod`), a loop index used as a value, or a loop whose first or last block is not the
  body at that cell.
- The jacobian is built from the array `rhs`, but 320 of TorusHeat's 414 jacobian instructions
  are scalar `out` stores, and two runs that name one state in one cell fall back to scalar.
- A new operator (`vm.ldu_amul`, `vm.spmv`) is a Python composition of gathers and
  scatter-adds, with tables sized by the number of entries.

The cost grows with the cell count in three places: the gather tables, the lowering time
(`vm.lower` 6 s on 10 000 states), and the jacobian stores.

## Proposal

The engine stays a VM: one precompiled interpreter, the model arrives as data, every loop is
bounded by a shape. Nothing is compiled per model. Reviewed by Fable on 2026-11-10; this
section is the reviewed version.

### Data first

```python
@dataclass(frozen=True, slots=True)
class View:
    base: int                   # register of element (0, 0, ...)
    shape: tuple[int, ...]      # extents, row-major, last index fastest
    strides: tuple[int, ...]    # registers per step on each axis; 0 broadcasts that axis
```

`Function.views: tuple[View, ...]` holds the arrays a function's tape names, as
`Function.tables` holds its gather tables; `encode` pools them. Views index the existing
register file, so Newton blocks, `SOLVE` and the scalar tangents read view elements as
ordinary registers. The iteration shape is the loop domain; each read is a view over it.

- A slice is a view with a moved base and a smaller extent: `T[i - 1]` for `i in 2:n-1`.
- A transpose is a view with swapped strides: a loop `for j, for i` reading `T[i, j]`.
- A broadcast is a view with stride 0: a parameter or `4.0` read in every cell.
- Modelica has no clamp or zero boundary read: edge cells are their own loops, so their reads
  are slices too. Only a periodic read, `mod(i + k - 1, n) + 1`, needs an instruction.

### Opcodes

| Opcode | Meaning |
|---|---|
| `TBINARY v, a, b, op` | elementwise ``op``: add, sub, mul, div, pow, comparisons, and, or, min, max, mod |
| `TUNARY v, a, f` | a builtin of one argument, elementwise |
| `TSELECT v, cond, then, else` | elementwise `if` |
| `TREDUCE v, a, axis, op` | sum, product, min or max along one axis, in row-major order |
| `TINDEX v, axis` | the loop index along ``axis`` as a value (1-based) |
| `ROLL v, a, axis, offset` | ``a`` moved by ``offset`` along ``axis`` with wrap (`np.roll`); ``v`` must not overlap ``a`` |

`VLOAD_Y` and `VSTORE_OUT` move a contiguous view to and from ``y`` and ``out``.
`VGATHER` and `VSCATTER_ADD` stay for irregular reads (meshes, `spmv`). An outer product is
`TBINARY` over stride-0 views; a concatenation writes into slice views of the destination.

### Jacobian

Forward mode over views: the tangents of a cell are keyed by the stencil offset vector, one
view of the state's shape per offset, and tangents with one offset add. Each offset is one
`VSTORE_OUT`. `Program.pattern` lists the entries diagonal by diagonal; the solver keeps
choosing band or CSR (`MdLinear`), and a 3-D model (band ``ny * nz``) goes to BiCGStab.

### Where the shape comes from

Phase 1 keeps the scalar blocks as the truth. `vm.lower` reads `ArrayEquation` and classifies
each subscript by its values over the loop variable: ``i + k`` is a slice, the periodic form is
a `ROLL`, anything else a `VGATHER`. Phase 1 fixes tape size and engine time. It does not fix
front-end time: `flatten` and `causalize` stay per cell. Phase 2 keeps the array through
`causalize` (array-aware matching, BLT, Pantelides), a separate project.

### Traps

- Bit equality: `TREDUCE` sums in row-major order, with the same `binary` and `unary`
  functions in both backends.
- A wrap on an extent of 1 or 2 makes two offsets name one entry: fall back to scalar.
- Every pass that lists registers (`_prune`, `_compact`, `_hoist`, the engine's liveness and
  `valid`) must read a view as its interval. A missed pass overlaps a view silently.
- The engine checks each view once at load, because it runs without bounds checks.
- Old engine wheels must reject a tape with views: the tape format version goes up.

## Plan

1. `View`, `Function.views` and the shaped opcodes in `vm.evaluate`. Test: each against NumPy,
   with slices, transposes and stride 0.
2. The engine encodes, validates and runs views. Test: bitwise equal to `vm.evaluate`; a bad
   view is rejected.
3. Register passes on intervals. Test: `_compact` with overlapping views keeps values.
4. `_Lowering.array` writes views and `ROLL`. Test: TorusHeat same bits, zero gather tables.
5. Remove the fallbacks for `if`, comparisons, `^`, `max`, `mod` and an index as a value.
   Test: one corpus model each, both backends.
6. A non-periodic 2-D heat model (interior loop and edge loops) as slices. Test: same bits.
7. The jacobian by offset. Test: same bits as the scalar jacobian; one store per diagonal;
   extent 2 falls back.
8. A 3-D box heat model. Test: matches the `modelica.sim.fv` tape; the iterative solver runs.
9. Docs: `docs/vm/arrays.md`, `tape.md`, `extending.md`.

Deferred: phase 2, `GEMM`, clamp and zero reads, threads inside one opcode, moving `fv` and
`fem` off the `V*` opcodes.
