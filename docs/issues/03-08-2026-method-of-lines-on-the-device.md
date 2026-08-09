---
title: The method-of-lines path is the one GPU coupling with a working precedent
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
blocked_by:
  - arrays-and-for-equations-are-refused
kind: idea
scope: L
verify: probe
---

## Measured

- `docs/ideas/vulkan.md:114` names it as the first plausible coupling to `compute-ml`:
  "the method-of-lines path: explicit stepping, fp32 acceptable, and `cml_replay` for the time
  loop. That is the `heat_equation.c` pattern, with the stencil generated from a flat model
  instead of written by hand."
- The precedent is not hypothetical. `compute-ml` ships `examples/heat_equation.c`: a 256x256
  explicit finite-difference stencil, 50,000 steps, entirely on the device, with no host round
  trip per step. It works today.
- It is also the one shape where `compute-ml`'s hardest limit does not bite.
  `docs/ideas/vulkan.md:56` says fp32 is the real blocker for stiff integration; explicit
  stepping of a non-stiff problem at a loose tolerance is exactly what fp32 can carry, and
  the heat equation is that.
- The width is real here in a way it is not elsewhere. `docs/ideas/vulkan.md:36` puts a
  method-of-lines system at `n = 10^4` to `10^6` states, against 2 to 10 for the corpus.

## Inferred

Why this waits on arrays rather than on the GPU work: a method-of-lines model *is* an array
model. `Real T[n]` and a `for` equation over the interior is how it is written, and
`docs/ideas/pdes.md:151` records that as this project's preferred answer to distributed
problems — no new language feature, the discretisation visible in the model where the
modeller can change it. Until that parses, there is no flat model to generate a stencil from,
and the GPU question has no input.

Inferred, not measured: once arrays exist, the interesting question stops being Vulkan and
becomes *whether the emitter recognises the structure at all*. A stencil is a `for` equation
whose incidence pattern is banded. Emitting that as a loop over an array rather than as `n`
unrolled scalar equations is a decision in the emitter, and it is the difference between a
model that compiles in a second and one that emits a hundred thousand lines. That is
CPU-side, it is required either way, and it is probably the real content of this issue.

## What would turn this into a story

Arrays first. Then, in order, and each is a decision that can be taken alone:

1. **Does the emitter keep the loop?** Write a 1D heat bar with `n` in the thousands and look
   at what `emit` produces. If it unrolls, that is the finding and the story is about the
   emitter, not about any device.
2. **Does the time loop leave the host?** The heat-equation precedent keeps it on the device;
   anything with a per-step round trip gives the 100 µs back and more.
3. **Is fp32 honest for this problem?** State the tolerance the answer is claimed at, and
   check it against a CPU run in double precision.

## Deliberately not done

- Nothing about stiff PDEs. An implicit method needs a factorisation, and
  `docs/ideas/vulkan.md:90` records that `compute-ml` has no LU and nothing sparse.
- No `field`, `domain` or `pder`. `docs/ideas/pdes.md` is the standing argument, and this
  issue is deliberately the hand-written method of lines it recommends instead.

## The answer to question 1, 2026-08-09

**The emitter unrolls.** `verify: probe`, and the entry said in advance what to do with this
answer: "If it unrolls, that is the finding and the story is about the emitter, not about any
device." Filed as
[`a-stencil-is-emitted-as-n-unrolled-scalar-blocks`](09-08-2026-a-stencil-is-emitted-as-n-unrolled-scalar-blocks-so-a-method.md).
This entry closes as an idea that produced its story.

A 1D heat bar, `Real T[n]` with a three-point stencil written as one `for` equation over the
interior — exactly the shape `docs/ideas/pdes.md:151` recommends:

| n | flatten | causalize | emit | exec | blocks | lines | bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.001 s | 0.002 s | 0.004 s | 0.006 s | 10 | 322 | 7 K |
| 100 | 0.010 s | 0.014 s | 0.038 s | 0.052 s | 100 | 2,392 | 69 K |
| 1000 | 0.100 s | 0.161 s | 0.391 s | 0.549 s | 1,000 | 23,092 | 734 K |
| 3000 | 0.301 s | 0.449 s | 1.217 s | 1.622 s | 3,000 | 69,092 | 2.3 M |
| 10000 | 1.076 s | 1.619 s | 4.108 s | 5.563 s | 10,000 | 230,092 | 7.9 M |

One block per cell, at every size. The `for` equation is expanded at flattening and never
becomes a loop again — `rhs` at `n = 10` unpacks `T_1_ = y[0]` through `T_9_ = y[8]` as
separate statements, so there is no array left by the time the emitter could keep one.

Two things the entry did not predict, and both matter more than the line count.

**It is linear, not quadratic.** The inference in the entry was that unrolling is "the
difference between a model that compiles in a second and one that emits a hundred thousand
lines". Both halves are true at once: `n = 10000` emits 230,092 lines *and* reaches a kernel
in 5.6 seconds. Nothing blows up. The obstacle is the size of the artifact, not the time to
produce it.

**The seed matrix is the real allocation.** The generated module writes `_seed = np.eye(n)` at
module level. At `n = 10000` that is **0.80 GB**, allocated on `exec`, whether or not anything
calls `jacobian` — measured peak RSS across the build goes 0.09 to 0.78 GB, and the seed is
essentially all of it. It is used one row at a time; `_seed[k]` is a basis vector.

And the arithmetic itself is fine: one `rhs` call at `n = 10000` is 3.2 ms in CPython.

## Questions 2 and 3, not reached

Both are about the device, and both are downstream of an answer that redirects the work.

- **Does the time loop leave the host?** Not asked. `compute-ml` is present as a sibling
  project, so this is reachable; it is reachable *after* there is a compact stencil to send.
- **Is fp32 honest for this problem?** Not asked, for the same reason. Stating a tolerance for
  a kernel that does not exist yet would be stating it for the wrong kernel.

Neither is closed by this entry. They belong to the story above, or to a successor of it, and
they should be asked against source that is a loop rather than 230,000 lines.

## Deliberately not done, found while doing it

- **`the-jacobian-is-emitted-dense` was waiting for exactly this measurement** and now has it.
  That entry says the cost "is inference: a method-of-lines model with `n` in the thousands,
  which is what the arrays work opens up". The model exists now: `n = 10000` means an `n x n`
  output of 0.80 GB, and the note has been added there.
- **The stiffness of this model does not grow with `n`**, so it cannot be used to argue that
  explicit stepping is the right method. `a` in `HeatBar` is already "diffusivity over the
  square of the cell width", so refining the mesh at fixed `a` refines a *different* rod
  rather than the same one more finely. Any claim about implicit against explicit needs `a`
  scaled as `n^2`, and no such claim is made here.
