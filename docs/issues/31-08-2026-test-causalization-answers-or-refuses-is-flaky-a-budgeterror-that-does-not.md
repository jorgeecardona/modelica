---
title: "test_causalization_answers_or_refuses is flaky: a BudgetError that does not reproduce"
status: open
area: sim
owner: null
filed: 2026-08-31
scope: S
verify: test
kind: bug
---

## Measured

- 2026-08-31, full suite on a loaded machine (load ~8, swap in use):
  `tests/test_causalize_properties.py::test_causalization_answers_or_refuses` failed with
  `hypothesis.errors.FlakyFailure` — "Failed on the first call but did not on a
  subsequent one". The underlying raise is the suite's own `BudgetError`
  (tests/test_causalize_properties.py:183) surfacing from `is_constant` under
  `_reduce_index` → `_differentiate` → `work.constants()` (src/modelica/ir/causalize.py:157).
- The generated reproducer (verbatim from hypothesis):
  `model M` with `Real x0..x3(start = 1, fixed = true)` and equations
  `x0 + x0 = x0; der(x0) + x0 = der(x1); x0 = x0 + x0 + der(x1);` — a singular system
  headed into index reduction.
- Hypothesis also warned it spent >5 minutes shrinking and gave up. The isolated file run
  took 343 s.
- The same test passed when the file was rerun. `.hypothesis/` keeps the failing example,
  so a later run starts from it (the gitignore comment records that design).

## Inferred

A branch budget is deterministic for a fixed input, so "fails once, passes on retry" means
the budget's accounting is not a function of the input alone — shared or cumulative state
across hypothesis calls (the counter not reset between examples, or reset depending on
which example ran before). That is in those words an inference; nobody has traced the
counter yet.

## Done

Either of these observations closes it:
- The budget counter is shown to reset per example, and the flake is reproduced or ruled
  out with the counter fixed — the reproducer model above run twice in one process gives
  the same branch count both times.
- The test's budget mechanism is redesigned so a `BudgetError` is deterministic for a
  given generated source, and 20 consecutive full-file runs show no FlakyFailure.
