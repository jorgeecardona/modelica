---
title: A stencil is emitted as n unrolled scalar blocks, so a method-of-lines model is 230,000 lines of Python at n = 10,000
status: open
area: sim
owner: null
filed: 2026-08-09
scope: L
verify: probe
kind: story
---

## Measured

A 1D heat bar, `Real T[n]` with a three-point stencil written as one `for` equation over the
interior — the shape `docs/ideas/pdes.md:151` recommends and `examples/models/HeatBar.mo`
already carries at `n = 8`. Built at seven sizes on 2026-08-09:

| n | flatten | causalize | emit | exec | blocks | lines | bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.001 s | 0.002 s | 0.004 s | 0.006 s | 10 | 322 | 7 K |
| 100 | 0.010 s | 0.014 s | 0.038 s | 0.052 s | 100 | 2,392 | 69 K |
| 1000 | 0.100 s | 0.161 s | 0.391 s | 0.549 s | 1,000 | 23,092 | 734 K |
| 3000 | 0.301 s | 0.449 s | 1.217 s | 1.622 s | 3,000 | 69,092 | 2.3 M |
| 10000 | 1.076 s | 1.619 s | 4.108 s | 5.563 s | 10,000 | 230,092 | 7.9 M |

One block per cell. The `for` equation is expanded at flattening and never becomes a loop
again: `rhs` at `n = 10` unpacks `T_1_ = y[0]` through `T_9_ = y[8]` as separate statements,
and there is no array left by the time the emitter runs.

Scaling is **linear**, not worse, which is the one good half of this. Where the source goes at
`n = 1000`:

| function | lines |
| --- | --- |
| `observe_jacobian` | 6,013 |
| `jacobian` | 5,009 |
| `observe` | 4,012 |
| `initial` | 3,012 |
| `rhs` | 3,008 |
| `indicators` | 1,008 |
| `transition` | 1,007 |

One `rhs` call at `n = 10000` takes 3.2 ms.

The generated module also allocates `_seed = np.eye(n)` at module level. At `n = 10000` that
is **0.80 GB**, allocated on `exec`, whether or not anything ever calls `jacobian` — measured
peak RSS goes 0.09 to 0.78 GB across the build and the seed accounts for essentially all of
it.

## Inferred

Inferred, not measured: the limit is the *source*, not the arithmetic. 3.2 ms for 10,000
three-point stencil cells is CPython being CPython and would fall by an order of magnitude
under Numba or the C emitter. But 7.9 MB of generated Python is handed to `compile()` on every
build, and `moon run :test-jit` already records what Numba does with long straight-line code.
A model twice this size is not slow, it is impractical.

Also inferred: the fix is not one change. Three separable ones, in increasing size.

1. `_seed` is an identity matrix used one row at a time. `_seed[k]` is a basis vector and
   needs no matrix behind it.
2. A `for` equation whose body has the same shape for every index is *one* block repeated, and
   the emitter could write it as a loop over a slice. That needs the array to survive
   flattening, which today it does not — `Real T[n]` becomes `n` scalars in
   `modelica.ir.flatten`, so this is a change to the flat model and not only to the emitter.
3. The Jacobian's storage and factorisation, which is already filed as
   `the-jacobian-is-emitted-dense`.

## Done

- A method-of-lines model at `n` in the thousands emits source that grows *sub-linearly* in
  `n` for the parts whose structure is uniform. The number to beat is 230,092 lines at
  `n = 10000`; a target is not set here because the shape of the fix decides it.
- `_seed` is gone, or is not an `n x n` allocation. It is the cheapest item on this list and
  it is worth doing alone: 0.80 GB at `n = 10000` for a value that is a basis vector.
- The same corpus tests pass, and `HeatBar` still produces the same numbers. A change that
  makes a stencil compact must not change what it computes.
- The measurement above is re-run and recorded, so the improvement is a number and not a
  claim.

## Deliberately not done

- Nothing about the GPU. `method-of-lines-on-the-device` asked whether the emitter keeps the
  loop, measured that it does not, and said in advance that if it unrolls "the story is about
  the emitter, not about any device". This is that story.
- No `field`, `domain` or `pder`. `docs/ideas/pdes.md` argues against them and this issue
  keeps the hand-written method of lines it recommends.
