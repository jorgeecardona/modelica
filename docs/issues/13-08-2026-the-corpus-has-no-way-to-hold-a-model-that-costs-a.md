---
title: The corpus has no way to hold a model that costs a minute to simulate, and Ring Modulator costs 74 seconds
status: in-progress
area: ci
owner: quiet-tars-187f2
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

## Two of the seven were pure waste, 2026-08-13

Before deciding how the corpus should hold an expensive model, two of the seven call sites
turned out not to need what they were paying for. Neither change weakens anything.

### `test_verify` ran a duplicate on every model with no events

Read `check_run`: it returns ``[check_iteration(kernel)]`` and then loops over
``result.events``. With no events the loop body never runs, so the trajectory is computed and
thrown away and what remains is exactly what `test_event_iteration_settles` already asserts two
functions above. Sixteen of 25 corpus models are event-free.

    tests/test_verify.py     144.6 s  ->  97.1 s      (-33 %)
    AtwoodPendulum alone      31.7 s  ->  skipped

### `test_fold` integrated twice at 1e-10, for a claim that needs no accuracy at all

Its assertion is that the folded and unfolded builds agree **bit for bit**. Two builds that
take the same steps agree or do not agree whatever those steps are worth, so the tolerance was
buying nothing. Floored at 1e-6, the same rule the difference-quotient tests already use:

    tests/test_fold.py       182.4 s  ->  24.1 s      (-87 %)
    VanDerPol                107.1 s  ->  12.6 s
    AtwoodPendulum            51.9 s  ->   5.3 s

All 53 tests still pass, and the claim is unchanged.

### The same floor is *not* safe for `test_solution_set`, which is a finding

Tried, measured, reverted. Four models fail at the floor -- `PhaseChange` 1.7,
`RoastingMachine` 0.67, `BumpStop` 1.0e-7, `Setback` -- and every one on an equation containing
a *latched* relation, at an event instant. The schedule freezes such a relation at the last
event and the check re-decides it from the sampled values, so a sample the other side of an
event compares two different models. Filed as
[`the-solution-set-check-re-decides-latched-relations`](13-08-2026-the-solution-set-check-re-decides-latched-relations-instead-of-reading-the.md).

## What the remaining five cost, and what each actually needs

With those two done, this is the table the decision should be made from:

| call site | needs the whole horizon | needs the spec's tolerance |
| --- | --- | --- |
| `test_simulates_as_specified` | **yes** -- the reference point is at the end | **yes** |
| `test_radau` (fused loop) | **yes** -- same reason | **yes** |
| `test_solution_set` | no -- 20 samples | yes, until the latch issue closes |
| difference quotient x2 | no -- 12 plausible states | no, already floored |
| `test_fold` x2 | no -- bit for bit | no, now floored |
| `test_verify` | only if the model has events | yes |

**Two call sites genuinely need the expensive run; five do not.** For Ring Modulator that is
about 150 s rather than 520 s, which is a 30 % increase on the suite rather than a doubling --
a number somebody can accept or refuse, which is what this entry was filed to make possible.

The remaining decision is unchanged in shape and much smaller in stakes: whether the three
"no" rows should share one cheap trajectory, and whether a spec may name a shorter horizon for
them than its reference point needs.
