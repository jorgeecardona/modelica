---
title: The checker enumerates latches, and a discrete variable is now the other half of the state
status: done
area: verify
owner: quiet-tars-187f2
filed: 2026-08-11
fixed_in: 3f44912
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

## Done 2026-08-13

`Instant` carries `memory`, the discrete-time variables in `Code.remembered` order, and
`Code` carries `remembered` so nothing outside `modelica.sim.emit` has to re-derive it from
the `when` bodies.

### What it was, measured

`Thermostat` at `T = 18`, below `T_low`, so the heat-on branch is enabled. Before:

    latches=- fired=[]      state (18.0,)
    latches=- fired=[0]     state (18.0,)

Two states that differ only in `fired`, which the outcomes drop -- so the settled state
compared **equal to the one it started from**, and the report said "settles in one place"
about a model that had set `heating` and incremented `cycles`. After:

    latches=- memory=0,0 fired=[]     state (18.0,)
    latches=- memory=1,1 fired=[0]    state (18.0,)

### The "more than one settled configuration" the entry asked for

It is not in the corpus, and that is the honest finding rather than a gap in the work.
`Thermostat`'s two thresholds are `T <= 19` and `T >= 21`, which cannot both hold at one
instant, so nothing races. `RoastingMachine` has nine latches and no memory at all.

The shape that *does* diverge is two `when` bodies enabled together where one reads what the
other writes, and `tests/test_verify.py::CONTESTED` is the smallest legal one -- each variable
assigned in exactly one `when`, so it is not a single-assignment violation:

      when x <= 2   then a = pre(a) + 1;  end when;
      when x <= 2.5 then b = a;           end when;

Below `x = 2` both hold. Measured on that model, at `y = 1.0`:

| | report |
| --- | --- |
| before | `event iteration settles, and settles in one place, from all 1 configurations` |
| after | `settles somewhere different depending on the order events are applied in` |
| | `memory=1,0` \| `memory=1,1`, identical latches |

Red then green, and it is the `docs/ideas/verification.md:118` prediction arriving: "A
condition can then be driven back and forth within one instant."

It lives in the test file and not in `examples/models`. The corpus is models worth
simulating; this is order-dependent by construction, which is not a property an example
should quietly carry.

### `check_run` reads the memory off the trajectory

Starting the counter from `initial` at an instant halfway through a run checks a state the
run never reached -- `Thermostat` at t = 40 has come on twice. `check_run` now takes the
remembered variables out of `result.values` at the event's column, the same way it already
took the continuous state, and `check_iteration`/`iteration` grew a `memory=` keyword for it.

### Deliberately not done

- The memory is a starting *point*, not a set to enumerate. A latch is a bit and there are
  `2**k`; `cycles` is a counter with no finite range. What is enumerated is the reachable set
  from the start, which is what `check` already does -- as the entry itself said.
- Nothing about chattering across instants. Still
  [`chattering-across-instants`](03-08-2026-chattering-across-instants.md).

1454 pass, typecheck 0.
