---
title: A stencil is emitted as n unrolled scalar blocks, so a method-of-lines model is 230,000 lines of Python at n = 10,000
status: in-progress
area: sim
owner: quiet-tars-187f2
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

## Re-measured 2026-08-13, and the number got worse

The entry says "the measurement above is re-run and recorded, so the improvement is a number
and not a claim". Re-run before touching anything, same model, same machine:

| n | lines 08-09 | lines 08-13 | bytes 08-09 | bytes 08-13 |
| --- | --- | --- | --- | --- |
| 10 | 322 | 458 | 7 K | 12 K |
| 100 | 2,392 | 3,608 | 69 K | 113 K |
| 1000 | 23,092 | 35,108 | 734 K | 1,205 K |

**+52 % at n = 1000**, and `jacobian_sparse` is all of it. Where the source goes now:

| function | lines at n = 1000 |
| --- | --- |
| `observe_jacobian` | 6,011 |
| `jacobian` | 5,008 |
| `jacobian_sparse` | 5,008 |
| `observe` | 4,010 |

`the-jacobian-is-emitted-dense` landed an 88x speedup on this exact model and paid for it in
the currency *this* entry is about. That is not an argument against it — a run that finishes
beats a source that is short — but it moves the target, and the two entries should be read
together.

Emitting got slower too, and by more than the size grew: 0.39 s to 1.83 s at n = 1000, so
about 4.7x. Build (flatten plus causalize) went 0.26 s to 0.50 s, which is the sparsity pass
earning its keep elsewhere.

### What this hands the next person

**`jacobian` and `jacobian_sparse` are the same expressions gathered two ways**, and at
n = 1000 that is 5,008 duplicated lines — 14 % of the whole module. The sparse one fills
`nnz` entries in CSR order; the dense one fills `n` by `n`. One can be had from the other by
scattering, and only one has to be emitted.

That is measured duplication rather than an inference, it is the cheapest item on this
entry's list now, and it does not need narrow tangent rows or a surviving array to collect.
It is not the whole story — a stencil is still `n` unrolled blocks and that is what the
entry is named for — but it is the first thing to take, and unlike the `_seed` item it has
not already been disproven.

### Correcting yesterday's own suggestion, 2026-08-13

The note above says "one can be had from the other by scattering". **That is wrong**, and it
is worth correcting here rather than letting the next person spend a day on it.

Read the two bodies. `differentiate` and `differentiate_sparse` both call `self.tangent(mode)`
-- the whole block walk, which is the 5,000 lines -- and then differ only in their last few
lines: dense writes `out[slot] = row`, sparse writes `out[a:b] = row[_pattern[slot]]`. So the
duplication is real and it is the *walk*, not the output.

But deriving the dense one from the sparse one destroys a property the corpus currently has.
`tests/test_sparsity.py:68` fills the dense Jacobian *independently* and rebuilds it from the
sparse one, and equality is what proves `Mode.sparsity` is exact -- that no non-zero of the
real Jacobian falls outside the pattern, on all 22 models. Scatter the dense one out of the
sparse one and a non-zero outside the pattern is silently dropped, by both, identically. The
test would pass forever and mean nothing.

Going the other way is worse: gathering CSR out of a dense `n` by `n` fill reintroduces the
allocation the sparse path exists to avoid.

**The shape that works** is to emit the walk once and gather twice. The tangent rows go into a
module-level scratch indexed by variable, `jacobian` gathers dense from it and
`jacobian_sparse` gathers CSR from it, and both gathers stay independent so the pattern check
keeps its teeth. The working set does not grow: `differentiate_sparse`'s own docstring says
the rows are already `n` wide and `n` of them are alive whatever the output shape is.

That is a real emitter change rather than a rename, and it is the next thing to take on this
entry. It is still much smaller than narrow rows, which needs `combination` to align two
supports on every add.

### What the shared-walk change actually costs, read off the code

Two details found while reading `emit.py` for it, both of which decide the shape and neither
of which is visible from the docstrings.

**The row index has to be computed once and shared, not derived twice.** `tangent(mode)`
returns `(lines, carried)`, and `carried` is a mapping from variable name to the *local
symbol* holding its row. A scratch indexed by variable needs a name-to-slot order, and the
writer and both gatherers must agree on it. Deriving it by calling `tangent(mode)` again in
each gatherer would agree -- the walk is deterministic -- but it re-runs the whole symbolic
walk three times per mode at emit time, and emit is already the slow half (1.83 s at
n = 1000, against 0.50 s to build). So `Emitter` needs the walk cached per mode, which it
does not have today: `differentiate` and `differentiate_sparse` each call `tangent` fresh.

**`seeds` means there are five walkers, not two.** `tangent(mode, seeds=...)` is also what
`differentiate_parameters`, `differentiate_observed` and `differentiate_observed_parameters`
use -- the whole of the difference is which rows are seeded. So a scratch shared by the two
state-seeded gatherers is the narrow version of this change; the general version is one
scratch per seeding, and `observe_jacobian` at 6,011 lines is the largest single function in
the module and is *not* covered by the narrow one.

Neither is a reason not to do it. They are the reason it is an emitter change with a cache in
it rather than three lines moved, and whoever takes it should know before starting that the
narrow version leaves the biggest function untouched.

## The prerequisite landed, and the arithmetic on the rest is worse than it looked

`1c784ed` caches the forward-mode walk per (mode, seeding). Five methods differentiate and the
seeding is all that separates them -- `differentiate` and `differentiate_sparse` do not differ
even in that -- so the identical walk ran twice for one answer.

    emit at n = 1000     1.83 s  ->  0.30 s     6.1x

Byte-identical output, checked by hashing the whole emitted module for five corpus models.
That buys **emit time, not size**: still 35,108 lines at n = 1000.

### What a shared walk would actually save

Measured at n = 200, splitting each Jacobian into the walk and the lines that write ``out``:

| function | total | output rows | walk |
| --- | --- | --- | --- |
| `jacobian` | 1,007 | 200 | 807 |
| `jacobian_sparse` | 1,007 | 200 | 807 |

So the duplicated part is the 807, not the 1,007 -- the note above called the whole 5,008 at
n = 1000 "duplicated" and that overstates it by a fifth. Sharing the walk saves one copy of
it: about **11 % of the module**, not the 14 % claimed.

And it is not free. The walk's assignments name locals (`_d_T_3_ = ...`), and downstream
expressions read those names. Writing into a scratch means either an extra store per variable
-- which costs exactly the lines the sharing saved, so nothing is gained -- or making the walk
assign into ``rows[k]`` directly, so every downstream read becomes an indexed load in the
hottest function the module has. That is a source-size win paid for in Jacobian evaluation
time, on the path an implicit solver takes several times per step.

**This is a measurement, not a refusal.** 11 % is real and the runtime cost is unmeasured. But
it is a trade rather than a cleanup, and whoever takes it should measure the Jacobian call
before and after rather than assume the indexing is free.

### The bigger target is somewhere else

At n = 1000 the four largest functions are `observe_jacobian` 6,011, `jacobian` 5,008,
`jacobian_sparse` 5,008 and `observe` 4,010 -- 20,037 of 35,108 lines. `observe_jacobian` is
the single largest and it is `d(observed)/dy`, which exists for state estimators; a caller
integrating a bar never touches it. Not emitting what nobody asked for is a larger saving than
sharing a walk, and it costs no runtime at all. That is worth its own entry before this one is
taken further.
