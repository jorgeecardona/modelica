---
title: The Jacobian is emitted dense, though matching already knows where the zeros are
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: 0845bab
kind: story
scope: L
verify: probe
---

## Measured

- `jacobian(t, y, p, w, out)` fills an `n x n` array. The tangents are row vectors seeded from
  `_seed = np.eye(width)` (`src/modelica/sim/emit.py`), so each row is dense storage even when
  most of it is structurally zero.
- The emitter is *already* sparse in one sense, and it is worth being precise about which:
  `emit.combination` leaves a variable out of the sum entirely when its tangent is
  structurally zero, and the docstring says why — "this is the only thing keeping the emitted
  source proportional to the sparsity". So the **source** is sparse and the **storage and the
  linear algebra** are dense.
- The structure needed to do better is already computed. `modelica.ir.causalize` builds
  `work.incidence(equation)` and matches on it; `System.blocks` records which equations must
  be solved together.
- `docs/architecture.md:120` and `docs/ideas/benchmarks.md:63` both claim the sparse pattern as
  something this compiler gets for free: "an analytic *and sparse* Jacobian is free. Its
  pattern falls straight out of the incidence matching that is already built."

## Inferred

At the corpus's size this costs nothing worth measuring — `n` is between 1 and 5, and a dense
5x5 is faster than any sparse structure. The claim is about where it stops being true, and
that is inference: a method-of-lines model with `n` in the thousands, which is what the arrays
work opens up, has a banded Jacobian whose dense storage is `n^2` and whose dense
factorisation is `n^3`. That is the point at which this stops being a cosmetic difference and
becomes the reason a model does not run at all.

So this issue is not "make the corpus faster". It is a prerequisite for the model sizes the
project says it is aiming at, and it should be picked up when those models exist rather than
before.

## Done

- `System` exposes the Jacobian's sparsity pattern as a value — which entries can be non-zero
  — derived from the incidence structure that matching already built, not rediscovered by
  probing or colouring.
- `modelica.sim` can hand that pattern to the solver: SciPy's `Radau` and `BDF` both take a
  `jac_sparsity`, and giving them one is the cheapest possible first use of it.
- The emitted `jacobian` can fill a sparse structure rather than a dense array, for models
  above a stated size, with the dense path kept for small ones because it wins there.
- `verify: probe` — a test proves the pattern is right, and only a measurement on a large
  model proves the change was worth making. Both, and the measurement names its machine.

## Deliberately not done

- No sparse *linear algebra* of our own. SciPy and, later, a C backend have that; the value
  here is the pattern, which nobody else can compute for us.
- No colouring heuristics for finite differences. This project has the exact derivative;
  colouring is what a black-box integrator needs precisely because it does not.

## The model it was waiting for exists, 2026-08-09

This entry says the cost "is inference: a method-of-lines model with `n` in the thousands,
which is what the arrays work opens up ... it should be picked up when those models exist
rather than before". Measured while answering
[`method-of-lines-on-the-device`](03-08-2026-method-of-lines-on-the-device.md):

- A 1D heat bar with a three-point stencil builds at `n = 10000` in 5.6 s. The model exists.
- `jacobian` at that size would fill **0.80 GB** of output for a matrix with three non-zeros
  per row.
- `_seed = np.eye(n)` is a module-level constant of the generated source, so the same 0.80 GB
  is allocated on `exec` *whether or not anyone calls `jacobian`*. Measured peak RSS across
  the build goes 0.09 to 0.78 GB and the seed accounts for essentially all of it.

That last one is worth separating from the rest of this entry, because it is much smaller: the
seed is read one row at a time and `_seed[k]` is a basis vector. Removing the matrix does not
require a sparse Jacobian, a sparse factorisation, or any decision about storage format.

## What landed, 2026-08-10

`Mode.sparsity`, `jacobian_sparse`, and a sparse path in the solver. `tests/test_sparsity.py`,
46 tests. 1282 pass. `verify: probe`; the numbers are below and the machine is named.

- **The pattern is a value on `System`.** `Mode.sparsity` is one dataflow pass over the
  schedule: a variable reaches a state exactly when something it reads does, and the blocks
  are already in an order where everything a block reads was computed earlier. No derivative
  is taken, no expression is evaluated, nothing is probed or coloured. That is the claim
  `docs/architecture.md:120` makes, honoured.

  It is exact on all 22 corpus models -- no non-zero of the real Jacobian falls outside it --
  and tight where it matters: `TwoFalls` 2 of 16, `BouncingBall` 1 of 4, `HeatBar` 22 of 64.
  Conservative in exactly one place, stated at the property: every unknown of a simultaneous
  block gets the union of what the whole block reads.

- **The emitted `jacobian_sparse` fills the non-zeros only**, in CSR order, into a flat array
  of length `nnz` instead of an `n` by `n` one. Checked against the dense one by rebuilding
  it: exactly equal on every corpus model, not to a tolerance, because it is the same
  expressions gathered rather than stored whole.

- **The solver hands SciPy a sparse matrix**, above 16 states and below 25 per cent fill.

- **The measurement**, Intel i9-9880H, CPython 3.12.7, a 1D heat bar to t = 2 at rtol 1e-8:

  | n | nnz | fill | dense | sparse | speedup |
  | --- | --- | --- | --- | --- | --- |
  | 25 | 73 | 11.7 % | 43.0 ms | 20.0 ms | 2.2x |
  | 50 | 148 | 5.9 % | 63.2 ms | 16.6 ms | 3.8x |
  | 100 | 298 | 3.0 % | 3549.8 ms | 181.8 ms | 19.5x |
  | 200 | 598 | 1.5 % | 559.6 ms | 196.3 ms | 2.9x |
  | 400 | 1198 | 0.75 % | 9658.6 ms | 109.3 ms | **88.3x** |

  The row-to-row jumps are step sequences, not noise in the timer: the two paths take
  different steps once the linear algebra differs. The trend is the answer.

  At corpus sizes the two are the same to within noise -- 0.98x to 1.04x over six models -- so
  the threshold is not protecting anything, it is declining to build machinery that does
  nothing. 16 is above the widest model here (10 states) and below where the win starts (25).

### `jac_sparsity` is not the way in, and the entry assumed it was

The Done list said "SciPy's `Radau` and `BDF` both take a `jac_sparsity`, and giving them one
is the cheapest possible first use of it". They do take it, and it would have done nothing:
its own documentation says it "is ignored if `jac` is not `None`", and this project always
has an exact Jacobian.

The real switch is what the callable *returns*. `scipy/integrate/_ivp/radau.py:320` checks
`issparse(J)` and factors with `splu` instead of `lu_factor`. So the pattern had to reach the
solver as a sparse *matrix*, which is why `jacobian_sparse` exists at all rather than the
pattern being passed as an argument.

## Deliberately not done, found while doing it

- **The tangent rows are still `n` wide, and the 0.80 GB is still there.** This entry's
  2026-08-09 note calls `_seed = np.eye(n)` a separable, much smaller half -- "the seed is
  read one row at a time and `_seed[k]` is a basis vector. Removing the matrix does not
  require a sparse Jacobian." That is wrong, and it is worth correcting rather than quietly
  leaving.

  Forward mode here carries **one row vector per variable**, and every state needs its seed
  row, so `n` rows of width `n` are alive at once whatever the seeds are built from. Building
  them lazily moves the allocation; it does not remove it. `_seed[k]` is a *view* into one
  array today, so a lazy version would allocate strictly more.

  The fix is narrow rows -- each variable carrying only the states it reaches, which is
  exactly what `Mode.sparsity` now knows. It is a much larger change than it looks, because
  `combination` sums `partial * row` over rows with *different supports*, and aligning two
  supports on every add is a different generated program. That belongs in its own entry, and
  it now has a pattern to build on.

- **No sparse linear algebra of our own**, as planned. SciPy's SuperLU does the factorisation.
- **No colouring**, as planned. The exact derivative is here; colouring is what a black-box
  integrator needs because it is not.
- **`_SPARSE_FILL = 0.25` was chosen, not measured.** No dense large model exists in this
  corpus to measure it against, and the docstring says so where the constant is.
