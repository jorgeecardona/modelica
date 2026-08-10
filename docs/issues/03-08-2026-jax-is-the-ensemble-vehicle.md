---
title: JAX gives the ensemble what Vulkan was being asked for, in double precision, today
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
blocked_by:
  - the-ensemble-driver
kind: idea
scope: M
verify: probe
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

## The measurement, 2026-08-10

**The curve is flat.** Per this entry's own words, "that is the ensemble solved and the GPU
question becomes a footnote". `verify: probe`; everything below was run.

Intel i9-9880H, JAX 0.11.0, `jax_enable_x64`, CPU. `RLCCircuit`, two states, 41 output points
to t = 20, sweeping `r.R` over the batch. Both sides at rtol = atol = 1e-8, and both sides
running an **explicit Runge-Kutta** -- `jax.experimental.ode.odeint` against
`modelica.sim.sweep` with `method="RK45"` -- because the default `Radau` would have compared
two solvers and called the difference vectorisation.

| n | jax `vmap` | per point | `sweep` | per point | ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1 ms | 0.08 ms | 12.8 ms | 12.79 ms | 165x |
| 10 | 0.6 ms | 0.06 ms | 94.1 ms | 9.41 ms | 146x |
| 100 | 1.8 ms | 0.02 ms | 756.2 ms | 7.56 ms | 417x |
| 1000 | 20.8 ms | 0.02 ms | 7957.2 ms | 7.96 ms | 383x |

Per-point cost under `vmap` falls from 0.08 ms to **0.02 ms and stays there** from 100 to
1000 -- the plateau jaxdae reported, reproduced here. The single-threaded sweep is flat at
about 8 ms per point, which is what "linear in batch" looks like.

Against the *default* Radau sweep the ratio is 3151x at n = 1000. That number is real and it
is not the one to quote: most of the extra is scipy's implicit method against XLA's explicit
one, not vectorisation.

**Agreement with the interpreted tier: 1.22e-08**, over every state at every output point of
every batch, at a solver tolerance of 1e-8. That is the standing rule for a backend here and
it holds.

### What the probe emits, and what it refuses

`jax_source` walks `System.blocks` and writes one functional line per block, reusing
`Emitter.expr` for the arithmetic -- the decisions stay in `modelica.sim.emit`, which is the
same split the C backend uses. What it cannot write, measured over the corpus:

    13/22 models emitted:
      DCMotor, HeatBar, MassSpringDamper, NewtonCooling, PIControl, ProbeLC,
      ProbeRC, RCCircuit, RLCCircuit, Robertson, TwoBodyHeat, TwoFalls, VanDerPol

    9 refused:
      AtwoodPendulum    simultaneous block
      BouncingBall      1 events
      BumpStop          1 events
      DrainingTank      1 events
      Pendulum          simultaneous block, 2 events, two state sets
      PhaseChange       simultaneous block, 6 events
      RoastingMachine   9 events
      Setback           1 events
      TwoInertias       simultaneous block, 2 events, two state sets

That list is the honest boundary of this result. The width question is answered for the shape
the probe covers; **whether a JAX backend can carry events and simultaneous blocks is a
different question**, and it is a question about `lax.cond`, `lax.while_loop` and a jump map
rather than about hardware. jaxdae answers yes to it, which is evidence and not a measurement
made here.

## The comparison this settles

The entry set up two vehicles for one workload and asked for a number. The number says the
software vehicle wins the ensemble question on a machine with no working GPU driver at all:
383x over the loop at n = 1000, in float64, with the answers holding to the solver's own
tolerance.

`docs/ideas/vulkan.md`'s four blockers against `compute-ml` are all still true and none of
them was tested here -- they did not have to be, because the workload they were blocking is
now done elsewhere.

## Deliberately not done, found while doing it

- **No JAX dependency, and none proposed.** The probe ran in a throwaway environment
  (`uv venv` plus `jax[cpu]`). If this became a backend it would sit behind an extra, like
  `verify` and `sundials`.
- **No GPU number.** The machine has a Quadro T2000 -- Turing, where fp64 runs at 1/32 rate,
  which is exactly the objection `docs/ideas/vulkan.md:56` raises -- and the stock jaxlib is
  CPU-only. A CUDA jaxlib install was started and had not finished when this was written; it
  is a download of about a gigabyte for a card whose double-precision throughput is the thing
  under suspicion. **The CPU number is the one that decides this entry**, because it is
  already 383x and it needs no device at all, which was the argument.
- **`odeint` has no event handling**, so the probe could not have integrated a model with
  events even if it could emit one. A real backend would need `diffrax`, which does, and
  which is a second dependency.
- **The SPIR-V spike is not closed by this**, as this entry required. What has changed is
  what it is for: it is now a hardware question about `compute-ml`, and no longer the only
  route to a wide ensemble.
