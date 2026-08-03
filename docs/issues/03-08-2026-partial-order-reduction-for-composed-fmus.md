---
title: Composed FMUs with rollback is the workload that would justify partial-order reduction
status: open
area: verify
owner: null
filed: 2026-08-03
kind: idea
scope: L
verify: probe
blocked_by:
  - ssp-read-the-archive-and-run-the-master
---

## Measured

- `docs/ideas/verification.md:290` records the open item, and it is the only one left in that
  page: "Composed FMUs with rollback are the workload that would justify the partial-order
  reduction that musil's README lists as missing. Nothing here has been big enough to need it
  yet."
- What exists: `compose(...)` and `channel_actions` already cover several components with an
  unreliable link over every interleaving (`docs/ideas/verification.md:80`), and
  `canonicalize=` is available as symmetry reduction over interchangeable components, validated
  by `symmetry_reduction_sound` (`docs/ideas/verification.md:264`).
- Also measured, and it is why this is an idea rather than a story: **state explosion has not
  happened here.** `docs/ideas/verification.md:262` calls it "a fourth limit, but a soft one,
  because these models are small". `tests/test_ssp_master.py` runs today.

## Inferred

The trigger is inferred from the arithmetic of interleavings, not observed: a co-simulation
master over *n* components, each of which may step, discard, save or restore, has a state space
that grows factorially in the orderings even when the per-component state is tiny. Two
components are checkable today. Five with rollback is where `check` plausibly stops returning.

The order of remedies is also inference, and it matters because they are not equally cheap.
`canonicalize=` (symmetry reduction) is already implemented and should be tried **first** — many
SSP components are interchangeable, which is exactly its precondition. Partial-order reduction
is the second lever and it lives in musil, not here.

## What would turn this into a story

A measurement, and it needs the SSP master to exist first:

- compose *n* real components with rollback, *n* = 2, 3, 4, 5, and record the state count and
  the wall time of `check`,
- then the same with `canonicalize=` over the interchangeable ones,
- and say at which *n* it stops being usable.

If symmetry reduction carries it to a realistic system, this closes as "not needed, here is the
curve" — a good outcome, and the honest one to record. If it does not, the numbers are the
argument for partial-order reduction in musil, and this becomes a story **in that repository**
rather than in this one.

## Deliberately not done

Nothing is being asked of musil today. `docs/ideas/verification.md:294` states the standing
constraint on the relationship: neither project depends on the other at runtime, musil is an
extra here, and a verification tool is not a reason to break the core's freedom from
dependencies.
