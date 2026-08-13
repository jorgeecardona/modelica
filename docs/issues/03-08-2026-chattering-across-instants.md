---
title: Chattering across instants is watched by a counter, and the checker cannot see it
status: done
area: verify
owner: quiet-tars-187f2
filed: 2026-08-03
fixed_in: 5d92c1c
kind: idea
scope: M
verify: probe
---

## Measured

- `modelica.verify` is exact *within* one instant. `docs/ideas/verification.md:47` gives the
  reason: at one point of simulated time `y` does not move, so the discrete layer is a finite
  machine over the latch bits and enumerating it approximates nothing.
- `docs/ideas/verification.md:245` states the boundary in the same breath, and it is the first
  of the three hard limits: "**One instant at a time.** Freezing `y` is what buys exactness,
  and it is also the boundary. Each event acts at most once, so this says nothing about
  chattering *across* instants. That stays a runtime question, watched by the standstill
  counter in `modelica.sim.solve`. `check_run` narrows the gap... Narrowing is not closing."
- The runtime side is real and is all there is: `src/modelica/sim/solve.py:219-233` counts
  events that arrive within `_STANDSTILL` of each other and raises "the model is chattering"
  past `_CHATTER` of them. That is a backstop, not a diagnosis — it fires after the fact and
  names no relation.

## Inferred

What chattering across instants actually is, and why the instant-local checker cannot reach
it: a relation whose crossing the transition moves *back* across, so the model advances time
by an epsilon and immediately crosses again. Each instant is individually settled and
confluent — the checker says so, correctly — and the sequence still does not progress. It is a
liveness property of the *run*, not of the instant.

Inferred consequence, and it is why this is an idea rather than a story: making the checker
see it means giving it a notion of time advancing, which means either an abstraction of `y`
(the thing `docs/ideas/verification.md:26` records as tried, unnecessary, and dropped) or a
different instrument entirely. Reaching for the abstraction again without a concrete failing
model would be repeating a decision that was already made on evidence.

There is also a cheaper thing that is not the checker at all: when the counter fires, the
engine *knows* which indicators kept firing. Saying "`level > 0` fired 100 times at
t = 3.0000001" instead of "the model is chattering" is a diagnosis rather than an alarm, and
it needs no new theory.

## What would turn this into a story

Either of these, and the first is much cheaper:

1. **A better message.** The standstill counter reports which triggers fired and how often,
   from `code.events` — which already carries the source text of each relation. That is a
   small, self-contained change and it is worth doing whether or not anything else here is.
2. **A model that chatters.** No corpus model does. Write one deliberately — a relay with no
   hysteresis is the standard example — and see what the engine says today. Only with that in
   hand is it worth asking what the checker could have said, and whether `check_run` on that
   trajectory finds anything the counter did not.

## Deliberately not done

- No predicate abstraction of the continuous state. It was planned once, turned out to be
  unnecessary when freezing the instant made the model exact, and is recorded as such in
  `docs/ideas/verification.md:283`. Bringing it back needs a failing model as evidence, not a
  hypothetical.
- No change to `_STANDSTILL` or `_CHATTER`. Tuning a backstop is not the same as diagnosing
  what it catches.

## Answered 2026-08-13 — both items done, and the premise was wrong

This entry asked for two things and said the first was much cheaper.

**1. A better message. Already there.** `Plan._chattering` tallies `Event.cause` and names
what kept firing; `src/modelica/sim/solve.py:303` is the method and its docstring is the
argument this entry made. It landed before this entry was reread.

**2. A model that chatters.** Written, in `tests/test_chattering.py`, and it is the standard
one -- a relay with no hysteresis, `der(x) = if x > 0 then -1 else 1`, which cannot settle on
`x = 0` because crossing the threshold reverses the flow that crossed it.

### What the checker says: `ok`, at every point, and it is right

`check_iteration` at `x = +1`, `x = 0` and `x = -1`, all three: *event iteration settles, and
settles in one place, from all 2 configurations*. Exactly what this entry inferred -- each
instant is individually confluent and the sequence still does not progress. Nothing about the
instant-local checker needs to change, and nothing here is evidence for bringing back the
predicate abstraction `docs/ideas/verification.md:283` dropped.

### What the engine says: nothing, forever

This is the part the entry got wrong. It says "The runtime side is real and is all there is".
Measured, the runtime side does not fire:

| | |
| --- | --- |
| events | 24,273 in 45 s |
| time reached | 1.00004854 of a 5-second horizon |
| segment length | median 2e-9, min 1e-9, max 2e-9 |
| standstill threshold at t = 1 | 1e-12 |
| segments short enough to count | **0 of 24,272** |
| `chatter` counter | 0, against `_CHATTER = 100` |

A chattering run advances at exactly `_REARM`, the distance the engine holds a just-fired
crossing off zero -- and that is a thousand times the standstill threshold, so every segment
resets the counter. The backstop this entry treats as the one real defence has never been
reachable by the model it exists for.

Filed as [`the-standstill-counter-cannot-fire`](13-08-2026-the-standstill-counter-cannot-fire-because-a-chattering-run-advances-by-the.md),
with the reason a straight `_STANDSTILL = 1e-9` is the wrong fix (the threshold is relative
and `_REARM` is absolute).

### And a second thing, found on the way

A two-relation variant fails in 0.01 s with `ValueError: f(a) and f(b) must have different
signs` -- SciPy's `brentq`, out of `simulate` unchanged, with no model name and no time. Filed
as [`a-scipy-valueerror-escapes-simulate`](13-08-2026-a-scipy-valueerror-escapes-simulate-when-two-crossings-sit-on-zero-at.md).
Unrelated to chattering: any model whose indicator is exactly zero at `t = 0` is a candidate.

### So this idea is closed

It asked what would turn it into a story, and named two things. Both are done, and what they
produced is two bugs against the runtime rather than any change to the checker. That is the
answer: **the gap this entry describes is real and is not where the work is.** The instant is
exact, the sequence is a liveness property nothing here can reach, and the thing that was
supposed to catch it at runtime was broken the whole time.

A spike that returns "no" has succeeded, and so has an idea that turns into two bugs somewhere
else.
