---
title: 'Spike: can an emitted SPIR-V kernel run an ensemble on compute-ml at all?'
status: open
area: sim
owner: null
filed: 2026-08-03
kind: spike
scope: L
verify: probe
blocked_by:
  - the-ensemble-driver
---

## The question

Can `compute-ml` — the raw-Vulkan compute library in the next directory — run one emitted
kernel over an ensemble of parameter sets, and does it beat the CPU sweep on a real machine?

A spike, not a story. The honest answer may be no, and `docs/ideas/vulkan.md:106` already
records the current position as "not now, do not close the door". This spike is what would
change that position in either direction.

## Measured, from reading compute-ml

`docs/ideas/vulkan.md:54` lists four blockers, hardest first. All four were read out of the
library's source, not assumed:

1. **float32 end to end.** `VkDeviceCreateInfo` passes no `pEnabledFeatures` and no feature
   chain, so `shaderFloat64` is never requested; the dtype enum has no FP64 entry; the host API
   takes `float*`. fp32 gives about seven decimal digits. The reference points in
   `examples/models/*.toml` are generated at `rtol = 1e-12`, and `Robertson` alone spans ten
   orders of magnitude in concentration.
2. **No way to plug in a generated kernel.** The kernel set is closed: `glslc` compiles the
   shaders at build time and embeds them as C arrays. There is no runtime SPIR-V loading, no
   shaderc, no "here is my module and its bindings" entry point. This is the deepest mismatch
   with a project that generates one kernel per model, and the most mechanical to fix.
3. **No linear algebra beyond dense GEMM.** No LU, no triangular solve, nothing sparse. An
   implicit integrator needs a factorisation per Jacobian. Irrelevant for an explicit method.
4. **Submission blocks.** One queue, one fence, about 100 µs of host-visible round trip per
   readback. Fatal with the host in the loop, irrelevant when the time loop is on the device —
   and `cml_replay` already records and replays the dispatches of one step, which is the
   CUDA-graph idea and the thing that makes a device-side loop practical.

Also measured: `compute-ml` ships `examples/heat_equation.c`, a 256x256 explicit
finite-difference stencil run for 50,000 steps entirely on the device. The pattern this project
would need already exists there and works.

## What answers it

Run the ensemble driver's own sweep, unchanged in meaning, on the GPU:

- an explicit, non-stiff corpus model at a loose tolerance — the honest fp32 scope,
- a sweep wide enough to matter (10^3-10^4 parameter sets),
- against the single-threaded CPU sweep on the same machine, both named.

The spike closes when there is a number. Ratios below about 1 on a laptop iGPU are a perfectly
good answer, and `docs/ideas/vulkan.md:73` already predicts one reason for it: consumer parts
run FP64 at 1/32 to 1/64 of FP32, so enabling `shaderFloat64` makes the arithmetic correct
without making it fast.

Blockers 1 and 2 must be settled before any number exists, and both are changes to
`compute-ml` rather than to this repository. Whoever picks this up should decide that first,
and say so here if the answer is that `compute-ml` is not the right host.

## A second vehicle, added 2026-08-03

`jax-is-the-ensemble-vehicle` proposes the same workload on JAX: double precision, `vmap`, and
no device required. [jaxdae](https://arxiv.org/abs/2607.23202) reports batched-gradient wall
time "nearly flat from batch 1 to 1000" through XLA fusion, which is the scaling this spike
would be trying to demonstrate.

That does not close this. It changes what the spike is *for*: the hardware question — can a
generated kernel run on raw Vulkan at all — rather than the only route to a fast ensemble.
Whichever gets a number first should say so in the other.

## Deliberately not done

- Not the stiff solver. Time integration is serial by construction and one trajectory of a
  small system has no width to exploit (`docs/ideas/vulkan.md:35`).
- Not the method-of-lines path, which is the other plausible coupling but waits on arrays and
  `for` equations.
