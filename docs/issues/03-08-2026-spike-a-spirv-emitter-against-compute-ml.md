---
title: "Spike: can an emitted SPIR-V kernel run an ensemble on compute-ml at all?"
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
blocked_by:
  - the-ensemble-driver
kind: spike
scope: L
verify: probe
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

## The answer, 2026-08-10: no, and it is a direction rather than a gap

A spike closes when the question is answered, whichever way. This one is answered without a
GPU number, by the route the entry itself allowed: "Blockers 1 and 2 must be settled before
any number exists, and both are changes to `compute-ml` rather than to this repository.
Whoever picks this up should decide that first, and say so here if the answer is that
`compute-ml` is not the right host."

`compute-ml` is not the right host. `verify: probe`; the four blockers were re-read against
its current source rather than trusted, and two of them have moved.

### The four, a week later

| | as filed, 2026-08-03 | measured 2026-08-10 |
| --- | --- | --- |
| **1. fp32 end to end** | "passes no `pEnabledFeatures` and no feature chain, so `shaderFloat64` is never requested" | **moved, and still real.** `src/backend/vulkan/vk_context.c:129` now sets `pEnabledFeatures` and requests `shaderFloat64` when the device has it; `src/shaders/dot_f64.comp` exists with a Kahan f32 fallback. But `CmlDtype` is still `CML_FLOAT32`, `CML_FLOAT16` (planned), `CML_Q8_0`, `CML_Q4_0`, `CML_Q2_0`, `CML_Q1_0` — no fp64 — and every host entry point takes `const float *`. f64 lives *inside* a reduction shader; it never crosses the boundary. |
| **2. no runtime kernel loading** | closed kernel set, `glslc` at build time | **unchanged.** No public header takes SPIR-V. `vkCreateShaderModule` is internal to `src/backend/vulkan/vk_pipeline.c:20`. |
| **3. no linear algebra beyond GEMM** | no LU, nothing sparse | **unchanged for a factorisation.** `936de72` adds conjugate gradient, which is iterative and matrix-free — a different thing from the LU an implicit integrator wants, and irrelevant to the explicit method this spike scoped anyway. |
| **4. submission blocks** | ~100 µs round trip, `cml_replay` mitigates | **unchanged and correct as filed.** `cml_replay` is at `include/compute_ml/context.h:31`. |

### What makes it a direction

The dtype ladder is fp32, fp16, Q8, Q4, Q2, Q1. `compute-ml` is an ML inference library and it
is getting *more* quantised, which is the right direction for what it is for. This corpus's
reference points are generated at `rtol = 1e-12`, and `Robertson` alone spans ten orders of
magnitude in concentration.

So the ask is not "finish two features". It is: add an fp64 storage dtype the library has no
other use for, and add a runtime kernel-loading entry point it also has no other use for, in
someone else's repository, for one consumer. That is a fork wearing a contribution's clothes.

### Two things that were not knowable when this was filed

**Vulkan is reachable here, and CUDA is not.** Both GPUs enumerate and all three devices
report `shaderFloat64 = true`:

    GPU0  Intel UHD 630 (CFL GT2)   Mesa 26.1.5      shaderFloat64 = true
    GPU1  Quadro T2000              NVIDIA 550.163   shaderFloat64 = true
    GPU2  llvmpipe (LLVM 21.1.8)                     shaderFloat64 = true

Measured while answering `jax-is-the-ensemble-vehicle`, `cuInit(0)` returns 999 on the same
machine. So the hardware question *could* have been asked through Vulkan, and the obstacle is
the library rather than access to a device. Worth recording, because "we could not get to a
GPU" would have been a much weaker reason to close this.

**The workload has a number.** `jax-is-the-ensemble-vehicle` closed the same day at 383x over
the single-threaded sweep at n = 1000, per-point cost flat from 100 to 1000, in float64, on
CPU with no device at all, agreeing with the interpreted tier to 1.22e-08. The entry said
"whichever gets a number first should say so in the other". That one did.

## Deliberately not done, found while doing it

- **No SPIR-V emitter was written**, and none should be until something wants one. The
  emitter itself is the easy half — `modelica.sim.dialect` already showed that a second
  target is a table of spellings plus the statements a language needs — and it would have
  been written against a host that cannot store its numbers.
- **This does not say Vulkan is the wrong vehicle**, only that this library is the wrong host
  for this workload. A future ensemble backend on raw Vulkan, or on a compute library built
  for fp64, is a different entry and this one does not prejudge it.
- **`compute-ml`'s own direction is not criticised.** Quantisation is correct for inference.
  The mismatch is that a Modelica corpus and a 1-bit weight format want opposite things from
  the same buffer.
