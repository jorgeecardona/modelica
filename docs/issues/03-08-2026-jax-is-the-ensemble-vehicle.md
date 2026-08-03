---
title: JAX gives the ensemble what Vulkan was being asked for, in double precision, today
status: open
area: sim
owner: null
filed: 2026-08-03
kind: idea
scope: M
verify: probe
blocked_by:
  - the-ensemble-driver
---

## Measured

`docs/ideas/vulkan.md` picks the ensemble — one compiled model over a matrix of parameter
sets — as the first workload with real width, and then records four things that block
`compute-ml`: it is float32 end to end, there is no way to load a generated kernel at
runtime, there is no linear algebra beyond dense GEMM, and submission blocks.

[jaxdae](https://arxiv.org/abs/2607.23202), July 2026, has the same workload working, and it
is worth reading because of how much of this project's own design it shares:

- **Pantelides with dummy derivatives**, chosen the same way — "by bipartite matching on the
  equation sparsity pattern, which equations to differentiate";
- adaptive **BDF, Radau and Rosenbrock**, with state events and a jump map at the event time;
- acausal modelling through typed effort-flow connectors and a `connect()`, assembled from a
  Python API — which is what `modelica.build` is;
- batched-gradient wall time that "stays nearly flat from batch 1 to 1000" through XLA
  fusion, where an implementation without native vectorisation is "linear in batch".

Also measured, and it is the point: JAX is double precision when asked, it runs on CPU with
no device at all, and `vmap` is the whole of the ensemble API.

## Inferred

Comparing the two vehicles for the *same* workload, from what each is documented to do
rather than from having run either here:

| | compute-ml / SPIR-V | JAX |
| --- | --- | --- |
| double precision | not requested; fp64 is 1/32 to 1/64 on consumer parts | yes |
| loading a generated kernel | no runtime entry point; needs shaderc | `jit` of emitted source |
| linear algebra | dense GEMM only, no LU | full |
| batching | one thread per trajectory, hand-written | `vmap` |
| runs with no GPU | no | yes |
| what it costs us | a SPIR-V emitter and changes to another library | an emitter to JAX-flavoured Python |

The inference: **the SPIR-V spike is asking a hardware question, and the ensemble is a
software one.** A JAX backend answers the ensemble question on any machine, in the precision
the corpus's reference points demand, and it happens to run on a GPU if one is there.

What JAX does not settle is the *stiff single trajectory*, which is where `docs/ideas/
vulkan.md:35` already says no width exists. Nothing here changes that.

## What would turn this into a story

The ensemble driver first — it is the API either vehicle sits behind, and this issue waits on
it deliberately. Then a measurement, not an opinion:

- emit a corpus model as JAX-traceable Python from the same `FlatModel`,
- sweep 1, 10, 100 and 1000 parameter sets under `vmap`, in float64, on CPU and on whatever
  device is present,
- against the single-threaded sweep, on a named machine,
- and check the answers against the interpreted tier, which is the standing rule for every
  backend here.

If the curve is flat, that is the ensemble solved and the GPU question becomes a footnote. If
it is not, this closes with the numbers and the SPIR-V spike keeps its place.

## Deliberately not done

- No suggestion to depend on JAX. It would be a backend behind an extra, like `sundials` and
  `verify`, and the core stays free of dependencies.
- No adoption of jaxdae itself. It does not read Modelica, which is the entire front end, and
  its solvers are a second implementation of what `modelica.sim` already has. What is being
  borrowed is the evidence, and possibly the emitter target.
- The SPIR-V spike is not closed by this. Two vehicles for one workload is a comparison worth
  keeping open until one of them has a number.
