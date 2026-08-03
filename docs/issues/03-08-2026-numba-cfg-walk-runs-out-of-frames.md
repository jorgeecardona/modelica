---
title: Numba's recursive CFG walk runs out of frames on RoastingMachine, at about 975 of 1000
status: open
area: sim
owner: null
filed: 2026-08-03
kind: bug
scope: XS
verify: probe
---

## Measured

- Compiling `RoastingMachine` with `jit=True` raised `RecursionError: maximum recursion depth
  exceeded` inside Numba's own control-flow-graph walk (`logs/2026-08-01.md`, session
  `rusty-replicant-32c43`).
- The cause was measured rather than guessed: the model's gas schedule is a chain of nine
  `elseif` branches, and that chain costs about **975 frames** of Numba's recursive CFG walk
  against CPython's default limit of **1000** (`logs/2026-07-30.md`, where the first diagnosis —
  "cannot be compiled" — was corrected to "marginal").
- So whether it compiles depends on how deep the caller's stack already is. It is not a hard
  failure. It is a failure that depends on who is calling.
- `RoastingMachine` is not in `tests/test_jit.py`'s `COVERED` tuple (`Robertson`, `Pendulum`,
  `BouncingBall`, `DrainingTank`), so nothing in the suite compiles the branch chain and CI
  cannot see this.

## Inferred

The 975 figure is one measurement on one machine with one Numba version; the *margin* is what
matters, not the number. Any model with a long `elseif` chain — a lookup table, a gain
schedule, a piecewise characteristic curve — is in the same position, and a schedule with
eleven branches instead of nine would fail outright. This is inferred from the shape of the
walk, not measured on an eleven-branch model.

## Done

- `modelica.sim.kernel` raises the recursion limit around the Numba compile and restores it
  afterwards, in a `try`/`finally`, with the reason written at the call site and this issue
  named.
- `RoastingMachine` joins `COVERED` in `tests/test_jit.py`, so `moon run :test-jit` compiles the
  nine-branch chain and compares it against the interpreter like the other four.
- `verify: probe` because a green test in the normal environment proves nothing here — Numba is
  in no dependency group and `tests/test_jit.py:32` skips itself without it. What closes this is
  `moon run :test-jit` passing with `RoastingMachine` in it.

## Deliberately not done

**The branch chain stays.** Flattening the gas schedule into arithmetic was considered and
declined on 2026-07-30: the chain is a recorded gas trace from a real roast, and flattening it
would trade the example's one real property — that it is measured, not invented — for an opt-in
path no test exercises. The model is not the problem; the compiler's recursion budget is.
