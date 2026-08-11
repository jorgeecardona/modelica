---
title: The checker enumerates latches, and a discrete variable is now the other half of the state
status: open
area: verify
owner: null
filed: 2026-08-11
kind: story
scope: M
verify: test
---

## Measured

`modelica.verify` builds the model checker's state out of `Trigger.latch` alone. That was
complete until 2026-08-11: a latch was the only discrete thing an emitted kernel carried,
and `docs/ideas/verification.md:44` says so — "the latch slots of `w`, plus `pre`-valued
variables when they land".

They have landed. `3d68521` gives a `when` body the ability to assign a discrete variable
that persists, so `examples/models/Thermostat` carries two of them (`heating`, `cycles`) in
workspace slots the checker does not look at.

The measurement that entry recorded, over every corpus model with events, is what makes the
gap concrete — `check_iteration` starting from every latch configuration at a fixed state:

| model | latches | starting configurations | settled configurations |
| --- | --- | --- | --- |
| `RoastingMachine` | 9 | 512 | 1 |
| `BouncingBall`, `DrainingTank`, `Setback` | 1 | 1–2 | 1 |

One settled configuration from 512 starts, every time. That is the correct answer *for a
latch*: a latch is the cached value of a relation the model can evaluate, so the event
iteration repairs whatever you set it to. It is the wrong answer for a memory, and the
checker currently cannot tell the difference because it never enumerates one.

## Inferred

Inferred, not measured: this is the point at which the liveness check stops being a
formality, and `docs/ideas/verification.md:118` predicted exactly that — "An event acts at
most once per instant, and that is what makes the iteration terminate *today*. When discrete
variables and `pre()` land, that guarantee goes away. A condition can then be driven back
and forth within one instant."

Whether it *does* go away in `Thermostat` is not known. Its two `when` clauses assign the
same variable from different thresholds, which is the shape that could in principle drive a
condition back and forth in one instant — and the model settles in practice. The checker is
what would turn "settles in practice" into "cannot fail to settle", which is the whole
argument of that page.

Also inferred: the state grows. A discrete variable is not a bit, so the checker cannot
enumerate its value the way it enumerates `2^k` latch configurations. What it can enumerate
is the *reachable* set from a given start, which is what `check` already does — the change
is to the state's definition, not to the search.

## Done

- `modelica.verify`'s `Instant` carries the remembered slots beside the latches, and
  `transition` writing one is a move the checker sees.
- `check_iteration` reports **more than one settled configuration** for a model with a
  memoried flag, and a test asserts it. The 512-starts-1-settled measurement above becomes
  the *contrast* that explains what a latch is, rather than the rule.
- `Thermostat` is checked: the iteration settles at every instant a real run stops at
  (`check_run`), or a counterexample says which sequence does not.

## Deliberately not done

- Nothing about chattering *across* instants. That is
  [`chattering-across-instants`](03-08-2026-chattering-across-instants.md) and freezing the
  instant is still what buys exactness here.
- No attempt to enumerate a discrete `Real` over its range. A counter has no finite range and
  the reachable set from a start is the only tractable question.
