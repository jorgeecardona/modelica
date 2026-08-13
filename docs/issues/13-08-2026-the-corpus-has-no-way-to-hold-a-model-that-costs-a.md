---
title: The corpus has no way to hold a model that costs a minute to simulate, and Ring Modulator costs 74 seconds
status: open
area: ci
owner: null
filed: 2026-08-13
kind: story
scope: M
verify: probe
---

## Measured

Every model in `examples/models/` is simulated by **seven** test call sites, each of which
integrates the whole horizon:

| where | runs |
| --- | --- |
| `tests/test_examples.py::test_simulates_as_specified` | 1 |
| `tests/test_examples.py` (Jacobian against a difference quotient, twice) | 2 |
| `tests/test_fold.py::test_the_two_modes_integrate_to_the_same_trajectory` | 2 |
| `tests/test_verify.py::test_event_iteration_settles_at_every_event` | 1 |
| `tests/test_radau.py::test_the_fused_loop_reaches_the_reference_points` | 1 |
| `tests/test_solution_set.py::test_the_original_equations_hold_along_the_trajectory` | 1 |

That is fine at the corpus's current scale: the whole suite is about 8 minutes and no single
model is a measurable part of it. It stops being fine at the first expensive model.

**Ring Modulator is that model.** Written out and measured on 2026-08-13 (the `.mo` is in
`10-08-2026-stages-2-and-3-of-the-benchmark-hires-e5-ring-modulator-and`):

    15 states, index 1, 23 blocks
    rtol = atol = 1e-6, t in [0, 1e-3]
    73.8 s, 119,723 steps
    worst relative difference from the published reference: 6.35e-5, on y14

It is not slow because anything here is wrong. The Bari run characteristics for the same
problem show RADAU5 taking 102,515 steps at rtol = 1e-7, so ~120,000 steps is what the problem
costs: a 10 kHz carrier, four diodes with an `exp` nonlinearity, and 1 ms of simulated time.

Seven runs at 74 seconds is **about nine minutes**, which would roughly double the suite for
one model.

## Inferred

Inferred, not measured: this will happen again, and not only for benchmark problems. A method-
of-lines bar at `n = 1000`, an FMU composed of several components, or a full day of a thermal
model are all things this project intends to hold, and none of them is a second's work.

Also inferred: the fix is not "make it faster". 120,000 steps is the problem's own number, and
the corpus's job is to hold models that are *interesting*, not models that are *cheap*.

## Done

- A corpus model can say it is expensive, and the seven runners above honour that -- by
  running it once rather than seven times, by running it on a shorter horizon than its
  reference point needs, or by being marked and deselected by default. Which of the three is
  the decision this issue exists to make.
- Whatever is chosen, **`test_simulates_as_specified` still runs**. A model whose reference
  point is never checked is a file, not a test, and the point of adding a published reference
  was to have something check it.
- The suite's total time before and after is recorded, on a named machine, the way
  `work-precision-harness` records its numbers.
- Ring Modulator lands as a corpus model, which is what closes the stage-2 half of
  `stages-2-and-3-of-the-benchmark`.

## Deliberately not done

- No `pytest -m slow` deselected in CI by default. That is one of the three options above and
  it is the one that quietly stops testing things; if it is chosen it should be chosen on
  purpose and written down, not arrived at.
- Nothing about making the integration faster. `jit=True` would help and is not the point: the
  suite runs interpreted on purpose, and a corpus that only holds models fast enough to run
  seven times is a corpus shaped by its harness.
