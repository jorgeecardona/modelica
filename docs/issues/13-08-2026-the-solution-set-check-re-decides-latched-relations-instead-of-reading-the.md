---
title: The solution-set check re-decides latched relations instead of reading the latch, so it depends on where the samples fall
status: open
area: verify
owner: null
filed: 2026-08-13
kind: bug
scope: S
verify: test
---

## Measured

`tests/test_solution_set.py` evaluates each original flat equation against the trajectory
using `modelica.ir.evaluate.evaluate`, which computes a relation like ``time < 66.7`` from the
values in front of it. The schedule does not: `modelica.sim.emit` replaces every *watched*
relation by a read of the value latched at the last event, which is the whole point of a latch.

So the two disagree wherever a sampled column and the event that last flipped the latch fall
on opposite sides of the relation.

Measured on 2026-08-13 by loosening this file's integration tolerance to a 1e-6 floor (the
rule `tests/test_fold.py` uses). Four of 25 models fail, and **all four fail on an equation
containing a latched relation**:

| model | worst relative residual | the equation |
| --- | --- | --- |
| `PhaseChange` | **1.735** | ``Q = (if time < t_out then Q_furnace else 0) - ...`` at t = 520 |
| `RoastingMachine` | **0.674** | the nine-way ``burner`` schedule, at t = 93 |
| `BumpStop` | 1.0e-7 | ``stop.f = if stop.s_rel > stop.gap then ...`` at t = 8.474 |
| `Setback` | -- | (same shape) |

t = 93 and t = 520 are event instants of those models. `RoastingMachine`'s residual of 0.67 is
not a numerical wobble: it is the difference between two arms of the schedule, 85 against 27.

At the specs' own tolerances all 25 pass, which is why this was not visible when the file was
written the same day. **That is luck, not correctness** -- the sample columns happen to miss
the crossings. Any change to `SAMPLES`, to a spec's tolerance, or to the solver's step
selection can move them onto one.

## Inferred

Inferred, not measured: the fix is to evaluate the original equations the way the schedule
does, which means the environment must answer a watched relation from `w` rather than from the
values. `Code.events` carries each `Trigger.condition` and its latch slot, and
`modelica.verify._Discrete.workspace` already builds exactly that mapping for the model
checker -- so the pieces exist and are not wired together here.

Also inferred: this is a defect in the *test*, not in the engine. A latched relation not being
re-decided between events is correct and deliberate, and `docs/ideas/verification.md` argues
for it at length. The test is the thing making a claim it cannot support.

## Done

- `tests/test_solution_set.py` reads latched relations from the workspace rather than
  re-deciding them, so a sample landing next to an event compares the same model to itself.
- Its integration tolerance can then be floored at 1e-6 like the other trajectory-only tests,
  and the four models above pass at that floor. That is the observable that closes this: the
  floor is the test, not a separate one.
- The docstring paragraph in that file that now reads "an equation containing a latched
  relation is a different case" says instead that it is not.

## Deliberately not done

- No change to how the engine latches. It is right.
- Not fixed by sampling away from events. Choosing columns that avoid the crossings would make
  the test pass and would remove the only points where the question is interesting.
