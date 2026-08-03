---
title: The method-of-lines path is the one GPU coupling with a working precedent
status: open
area: sim
owner: null
filed: 2026-08-03
kind: idea
scope: L
verify: probe
blocked_by:
  - arrays-and-for-equations-are-refused
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
