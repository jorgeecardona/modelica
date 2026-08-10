---
title: "Stages 2 and 3 of the benchmark: HIRES, E5, Ring Modulator, and ScalableTestSuite"
status: open
area: sim
owner: null
filed: 2026-08-10
scope: L
verify: probe
kind: story
---

## Measured

`scripts/bench.py` exists and `moon run :bench` runs it. Stage 1 of
`docs/ideas/benchmarks.md:84` is measured and published in
[`work-precision-harness`](03-08-2026-work-precision-harness.md): `Robertson` and `VanDerPol`,
three methods, an error axis, on a named machine.

Stages 2 and 3 are not, and neither is blocked on the harness. What each needs, measured:

- **Stage 2 -- `HIRES`, `E5`, `Ring Modulator`.** None is in `examples/models/`. `HIRES` is 8
  states, `E5` is 4 with a very wide range, `Ring Modulator` is 15 and is the one where the
  linear solve starts to dominate. Each has to be written as a `.mo`, written again as its
  Python twin, and given a `.toml` with reference points -- the corpus rule -- before it can
  be timed. Adding a problem to `PROBLEMS` in the harness is one line; getting the model right
  is not.
- **Stage 3 -- `ScalableTestSuite`.** An external Modelica library, not vendored here. It is
  the only stage where compile time is the thing under test rather than the integrator, and
  the only one with a public baseline: OpenModelica publishes per-branch results against it.

## Inferred

Inferred from stage 1 rather than from having run these: the reference is the hard part, not
the timing. On `VanDerPol` two solvers at rtol = atol = 1e-12 agree only to 1.6e-8 relative,
so a self-computed reference supports an error axis to 1e-6 and no deeper. `E5` spans a much
wider dynamic range than either stage-1 problem, so the same limit will bite sooner and harder.

Which makes the Bari published solutions a prerequisite for stage 2 in a way they were only an
improvement for stage 1. That is inference from one problem's behaviour, not a measurement of
`E5`.

## Done

- The three stage-2 problems are corpus models, each written twice with a `.toml`, and each
  appears in the harness's `PROBLEMS`.
- Their reference solutions come from the Bari test set, not from this harness. Where a
  published value is used, the `.toml` says which one.
- The work-precision tables are published with the four things
  `docs/ideas/benchmarks.md:94` demands: the problem, the tolerance sweep, the reference, and
  the machine.
- Stage 3 is a separate decision recorded either way: either `ScalableTestSuite` is vendored
  or fetched and the compile-time numbers are published against OpenModelica's, or it is
  written down here why not.

## Deliberately not done

- Not a CI gate, for the reason `work-precision-harness` already gives.
- No new harness. `scripts/bench.py` takes a problem, a horizon and a tolerance list; these
  are entries in a table.
