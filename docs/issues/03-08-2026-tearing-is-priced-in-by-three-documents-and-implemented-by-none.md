---
title: Tearing is priced in by three documents and implemented by none
status: in-progress
area: ir
owner: cute-replicant-6e76d
filed: 2026-08-03
kind: story
scope: L
verify: test
---

## Measured

- `grep -rniE "tearing|\btear\b" src/` returns nothing. There is no tearing pass.
- `docs/architecture.md:125` lists "make the system smaller before the solver sees it —
  tearing and BLT turn a simultaneous block of *n* equations into a much smaller torn system"
  as one of the three things this compiler can do that a precompiled library cannot.
  `docs/ideas/benchmarks.md:66` repeats the claim, and `docs/ideas/backends.md:46` lists
  tearing among the shared work that happens once above every backend.
- The corpus already has blocks that would pay: `AtwoodPendulum` produces a 4x4 implicit
  block, the largest in the corpus (recorded in `logs/2026-08-02.md`), and `Pendulum`'s mode
  dispatcher carries a 3x3 linear solve (`tests/test_jit.py:44`).

So three documents state a capability the code does not have. That is the finding.

## Inferred

The cost of the missing pass is a Newton iteration over *n* unknowns where a torn system
would iterate over one or two, with the rest recovered by forward substitution. On the
corpus's current block sizes (4x4) the absolute saving is small. It grows with the model, and
the `ScalableTestSuite` staging in `docs/ideas/benchmarks.md` is where it would show. This
is inference, not measurement — nothing here has been timed with and without tearing.

## Done

- `modelica.ir.causalize` returns, for each simultaneous block, a set of torn variables and
  the residual equations over them, and the remaining variables as an ordered assignment
  sequence.
- `modelica.sim.emit` writes the torn form: assignments, then a Newton iteration over the
  torn unknowns only.
- `AtwoodPendulum`'s 4x4 block solves with at most 2 torn unknowns, and its reference points
  in `examples/models/AtwoodPendulum.toml` hold unchanged to the tolerance already recorded
  there. Same trajectory, smaller system — that equality is the test.
- The interpreter (`modelica.ir.evaluate`) and the emitted kernel agree on the torn form, in
  the differential style `tests/test_jit.py` already uses.

## Deliberately not done

No heuristic tuning. Tearing quality is an NP-hard choice and the literature's usual answer
is a greedy heuristic; this issue asks for a *correct* torn form with any admissible choice,
not the best one. Picking a better heuristic is a separate, measurable follow-up, and it
needs the harness from the work-precision issue to be worth arguing about.