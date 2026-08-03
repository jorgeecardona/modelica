---
title: The Jacobian is emitted dense, though matching already knows where the zeros are
status: open
area: sim
owner: null
filed: 2026-08-03
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
