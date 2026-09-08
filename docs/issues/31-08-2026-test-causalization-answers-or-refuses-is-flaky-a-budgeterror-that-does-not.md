---
title: "test_causalization_answers_or_refuses is flaky: a BudgetError that does not reproduce"
status: done
area: sim
owner: null
filed: 2026-08-31
fixed_in: b2007eb
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

## What it actually was

The inference above was wrong: there is no branch counter. The `BudgetError` at
`tests/test_causalize_properties.py:183` is a **wall-clock** guard — `signal.setitimer(ITIMER_REAL,
5.0)` — so it is non-deterministic by construction, and "fails once, passes on retry" is a slow
causalization crossing five seconds only when the machine is loaded. The generated model is a
singular system that runs a long way into index reduction before `causalize`'s *own* budget
(`_reduce_index`, "index reduction did not terminate") stops it: on this machine the reproducer
above raises `CausalizationError` in **0.107 s**, and a profile put **56 % of that in
`_Work.constants()`** — `frozenset(name for name, v in variables if v.is_constant)`, rebuilt on
each of the 1495 `_differentiate` calls, so as the variable table grows under differentiation the
rescan is O(n) per call and O(n²) over the run.

## What landed

`_Work.constants()` no longer rescans the variable table. The constant names are maintained as
a set on `_Work` (`_constants`), updated in the two places every variable passes through —
`declare()` adds a name when `is_constant`, `forget()` discards it — and `constants()` returns a
`frozenset` cached until the set next changes (`_constants_frozen`). Index reduction differentiates
equations 1495 times on the reproducer and each call asks for the constants once; the answer is now
O(1) instead of an O(n) rescan.

Measured on the reproducer above (`CausalizationError` raised, not a solution):
- before: 0.107 s, `constants()` 56 % of it;
- after: 0.017 s — a **6.3×** speedup, with the same `CausalizationError`.

The five-second `signal.setitimer` guard in `test_causalize_properties.py` is now unreachable for
this generated model on a loaded machine, so the FlakyFailure it produced cannot recur. The test's
timer is left as-is: it is a real backstop against a genuinely non-terminating causalization, and
making it deterministic would mean bounding work by branch count, which the summary above shows was
never what it measured.

## Verified

- `constants()` of a model with one `constant Real g` returns `['g']`, and `[]` after the variable
  is forgotten — the incremental set tracks declare/forget.
- The 20 `test_causalize_*` property tests pass; `make check` green.
