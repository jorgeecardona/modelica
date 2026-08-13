---
title: The standstill counter cannot fire, because a chattering run advances by the re-arm distance and that is a thousand times the standstill threshold
status: done
area: sim
owner: quiet-tars-187f2
filed: 2026-08-13
fixed_in: fcef55c
kind: bug
scope: S
verify: test
---

## Measured

`modelica.sim.solve` watches for a run that stops advancing and raises "the model is
chattering" past `_CHATTER` consecutive zero-length segments. On the standard chattering
model it never raises. It never even reaches 1.

```modelica
model Chattering
  Real x(start = 1, fixed = true);
equation
  der(x) = if x > 0 then -1 else 1;
end Chattering;
```

A relay with no hysteresis: crossing the threshold reverses the flow that crossed it, so
`x = 0` cannot be settled on. `simulate(..., stop=5.0)`, instrumented at `_Run._fired` and
stopped by hand after 45 seconds:

| | |
| --- | --- |
| events | 24,273 in 45 s (539 / s) |
| time reached | **1.00004854300402** of 5.0 |
| segment length | median 2e-9, min 1e-9, max 2e-9 |
| `_STANDSTILL * max(1, \|t\|)` at t = 1 | 1e-12 |
| segments below that threshold | **0 of 24,272** |
| `chatter` counter high-water mark | 0, against `_CHATTER = 100` |

The run stalls exactly where `x` reaches zero and then crawls forward a nanosecond at a time,
forever, in silence.

1e-9 is not an arbitrary number. It is `_REARM` (`src/modelica/sim/solve.py:55`), the distance
the engine itself holds a just-fired crossing away from zero so that SciPy does not re-report
it. So a chattering run advances at exactly the pace the engine sets for it, and
`src/modelica/sim/solve.py:292`

```python
if at - t >= _STANDSTILL * max(1.0, abs(at)):
    chatter, stalled_from = 0, len(events)
```

reads that pace as progress and resets the counter.

Measured on 2026-08-13. `tests/test_chattering.py` carries the model and an expected failure
stating the two constants' relation directly.

## Inferred

Inferred, not measured: `_STANDSTILL` and `_REARM` were chosen independently and never read
against each other. `_STANDSTILL` is documented as a *relative* segment length and `_REARM` as
an *absolute* hold-off, so nothing in either docstring puts them on the same axis -- and on
the same axis they are three orders of magnitude apart, the wrong way round.

Also inferred: this is not a tuning problem, and raising `_STANDSTILL` to 1e-9 would be the
wrong fix. The threshold is relative, so at `t = 1e6` a relative 1e-9 is an absolute 1e-3, and
a genuine event 1e-4 later would be called a standstill. The two quantities have different
units and the comparison needs to say so -- something nearer

    at - t <= max(_STANDSTILL * max(1.0, abs(at)), 2 * _REARM)

with the second term justified rather than picked, since the measured segments were 1e-9
*and* 2e-9 and the reason for the doubling is not established here.

## Done

- The relay above raises `SimulationError` naming `x > 0`, in seconds rather than never.
  `Plan._chattering` already writes that message and already tallies `Event.cause`; nothing
  is reaching it.
- `tests/test_chattering.py::test_a_segment_no_longer_than_the_rearm_distance_counts_as_standstill`
  passes, and its `xfail` marker is deleted rather than flipped.
- The relation between the two constants is stated where they are defined. Whatever the fix,
  the next person changing `_REARM` has to see that `_STANDSTILL` depends on it.
- No corpus model changes. Every one of them advances by far more than a nanosecond per
  event, so a threshold raised to the re-arm distance cannot reach them -- and that is a
  claim to check, not to assume.

## Deliberately not done

- No change to `_CHATTER`. 100 consecutive stalled segments is a fine number; the counter
  never gets to 1, so the number it stops at was never the problem.
- Nothing about *solving* chattering -- no sliding-mode regularisation, no automatic
  hysteresis. The engine's job here is to say the run cannot progress and stop, which is what
  the counter was written to do.
- No opinion on what the model checker could say. That stays
  [`chattering-across-instants`](03-08-2026-chattering-across-instants.md), which measured
  that `check_iteration` reports `ok` at every point of this model -- correctly, since every
  instant does settle.

## Fixed 2026-08-13

The test asks about the **span** of `_CHATTER` consecutive events rather than the length of
each one, and `_STANDSTILL` goes 1e-12 to 1e-4.

```python
if at - stalled_at >= _STANDSTILL * max(1.0, abs(at)):
    chatter, stalled_from, stalled_at = 0, len(events), at
else:
    chatter += 1
```

A span needs no knowledge of the indicator's slope, which is what the absolute floor this
entry sketched (`2 * _REARM`) would have needed. `_REARM` is a distance in *indicator* units
and becomes a distance in time only after dividing by ``d indicator / dt``, which is the
model's business and not the loop's -- the measured 1e-9 and 2e-9 segments are that quotient
with a slope of exactly 1, and a relay with a gain of 1000 would have produced 1e-12 instead.
So the doubling this entry asked to justify turned out not to need justifying: it is the wrong
quantity to build a rule on.

### The number, with both sides measured

In the units the loop computes -- span of `_CHATTER` consecutive events over `max(1, |t|)`:

| | |
| --- | --- |
| tightest 100-event span, corpus | **7.9e-1** (`Pendulum`; `Thermostat` 9.9e-1) |
| the relay | **2.0e-7** |

Six and a half orders of magnitude of daylight. `1e-4` sits near the middle of it on a log
scale: 500x clear of the relay, 7,900x clear of the tightest real model. No corpus model has
more than 33 events in its entire run, so none of them needs 100 inside any window at all.
`tests/test_chattering.py::test_the_standstill_span_clears_the_corpus_by_orders_of_magnitude`
asserts the margin, so a future model that narrows it says so there rather than in a run that
stops for a reason nobody expected.

### After

    Chattering: 102 events at t = 1.000000201 without advancing -- the model is chattering.
    What kept firing: 'x > 0 changed' x102

0.19 s, against 45 s and still going. The second model in the same file, a relay against a
setpoint, raises in 0.26 s. `Plan._chattering` was already written and already tallied
`Event.cause`; nothing had ever reached it.

1468 pass, typecheck 0, and no corpus model changed -- the margin above is why, and the suite
is the check.

### Not done

The `FIGHTING` model in the same test file still raises SciPy's `ValueError`. That is
[`a-scipy-valueerror-escapes-simulate`](13-08-2026-a-scipy-valueerror-escapes-simulate-when-two-crossings-sit-on-zero-at.md)
and it fails before any of this is reached.
