---
title: "Stages 2 and 3 of the benchmark: HIRES, E5, Ring Modulator, and ScalableTestSuite"
status: in-progress
area: sim
owner: quiet-tars-187f2
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

## HIRES landed 2026-08-13 — one of the three, and a new kind of reference point

`examples/models/HIRES.{mo,py,toml}`, and `"HIRES": (321.8122, (1e-4, 1e-6, 1e-8))` in
`scripts/bench.py`'s `PROBLEMS`.

Eight states, the equations exactly as the Bari report prints them, ``y0 = (1, 0, 0, 0, 0, 0,
0, 0.0057)``, ``0 <= t <= 321.8122``. The horizon is not derived from anything and the report
says so: "The end point of the integration interval, 321.8122, was chosen arbitrarily".

### The reference is somebody else's

This entry said "the reference is the hard part, not the timing", and named the Bari published
solutions as a prerequisite. Taken literally rather than approximated: the `.toml` point is
Table II.1.1 of report `hires.pdf` from *Test Set for Initial Value Problem Solvers*, release
2.4 (Mazzia and Magherini, University of Bari and INdAM, February 2008), computed there by
RADAU5 on a Cray C90 with ``uround = 1.01e-19``.

Agreement, measured:

| tolerance | worst relative difference from the published values |
| --- | --- |
| 1e-8 | 8.6e-8 |
| 1e-10 | 7.5e-10 |
| 1e-12 | **5.1e-11** |

`y7 + y8 = 0.0057` to machine precision throughout, which is conservation of the enzyme and
follows from `der(y7) + der(y8) = 0` by inspection.

**`tests/test_examples.py::SOURCES` gained a fourth kind, `published`.** Its own docstring
already admitted the gap -- ``analytic`` and ``reference`` are "both of them things this
project could in principle have got wrong in the same way twice" -- and every `source =
"reference"` in the corpus until now meant an independent integration *by this project*. A
published number cannot be wrong in the same way twice because nothing about it came from
here. It is required to carry a `citation` naming the work **and the table**, since a test set
prints run characteristics beside its solutions and only one of those is the answer.

That requirement is a small change with a wide blast radius, and it is the part of this work
that outlives HIRES: the corpus can now hold somebody else's arithmetic and say whose.

### The harness runs it

    HIRES, to t = 321.812, 8 states
      method       rtol    seconds   rel error   steps
      Radau       1e-04    0.0390s    4.04e-03      81
      Radau       1e-06    0.0452s    1.59e-04     131
      Radau       1e-08    0.0990s    1.57e-06     290
      BDF         1e-04    0.0222s    2.04e-01     130
      BDF         1e-06    0.0409s    1.68e-03     214
      BDF         1e-08    0.0764s    1.68e-05     358
      LSODA       1e-04    0.0057s    7.09e-02     160
      LSODA       1e-06    0.0116s    1.18e-03     309
      LSODA       1e-08    0.0227s    6.76e-05     575

Which is the comparison this problem was chosen for: LSODA is four to seven times faster than
Radau here and is not on Robertson, because HIRES is *mildly* stiff -- about three decades of
eigenvalue spread against Robertson's nine -- so a method that switches has somewhere to
switch to. Not published as a table yet; that is the next paragraph.

### What is left on this entry

- **`E5` and `Ring Modulator`.** Nothing done. `E5` is the one this entry predicts will be
  hardest, and the prediction stands: the published reference matters more there than here
  precisely because its dynamic range is wider.
- **The work-precision tables.** The numbers above were run on this machine and pasted here.
  `docs/ideas/benchmarks.md:94` wants the problem, the sweep, the reference and the machine
  published together, and pasting into a register entry is not that.
- **Stage 3, `ScalableTestSuite`.** Untouched, and still needing a decision recorded either
  way.
