---
title: Chattering across instants is watched by a counter, and the checker cannot see it
status: in-progress
area: verify
owner: quiet-tars-187f2
filed: 2026-08-03
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
