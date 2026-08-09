---
title: Detecting charge and drop from the bean probe — the models are fine, the decision layer is wrong
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
blocked_by:
  - no-output-jacobian
fixed_in: dd4d769
kind: idea
scope: L
verify: probe
---

## Measured

The problem is real and the numbers are from a real session
(`apps/roaster-bridge/captures/field/2026-07-30-sincerely-5batch/raw-samples.jsonl`, 4387
samples, five batches on a 15 kg Loring):

- The bridge's live detector placed **1 of 5 charges correctly** on 30 Jul. The operator
  hand-fixed the rest. It is not a thresholding bug: a charge and a drop both move the bean
  probe fast and far, and no rule on the signal alone separates them.
- A model does separate them, because the two have different *futures*. After a charge the
  probe is buried in coffee and must turn and climb the bean curve; after a drop it hangs in air
  and relaxes toward the loop. So: two filters, two hypotheses, and the one that keeps
  predicting the next sample wins. `LOADED` is the `RoastingMachine` model (4 states) under an
  EKF; `EMPTY` is the probe alone relaxing toward measured `inlet`, 1 state.
- Prototype (`scratchpad/detect.py`, not in the repository), scored against `global-pass.json`
  over the whole session:
  - naive flip-flop: all 10 events found (5 charges, 5 drops), lag -20 to +30 s, **12 false
    positives**;
  - nearly every false positive was a drop/charge *pair* 200-240 s into a roast — where the
    anti-chatter refractory expired and the loaded filter's tracking error peaks;
  - tightening `Q` on `T_bean` alone: no help;
  - raising the refractory to 700 s using the real roast duration: **worse**;
  - requiring the physical plunge signature (probe falls >= 40 K in 60 s; the real batches fall
    107-152 K): false positives **12 to 2**.
  - best so far: `Q_bean 0.2, dwell 12, threshold 12` -> 10 firings, 8/10 matched, 2 false
    positives, worst lag 68 s.

## Inferred

The conclusion drawn there, and it is inference from three symptoms rather than a measurement:
**the two models are fine and the decision layer is wrong** — a hard two-state flip-flop.

1. One bad transition desynchronises everything after it (a `charge` called during preheat put
   the machine out of phase for the whole session).
2. The refractory is doing work a *duration model* should do. Roasts ran 834-863 s, which is a
   strong prior, and a single scalar gate uses almost none of it.
3. There is no way to say "probably loaded". A boolean forces commitment at the worst moment.

The standard answer is an **IMM with a semi-Markov duration model**: both hypotheses alive with
probabilities, mixed through a transition matrix, with the duration distribution carrying
"fourteen minutes" properly. Then no single sample commits anything, and out-of-phase is not a
reachable state.

## Where this belongs, which is the point of filing it here

**Not in this repository.** The IMM is about 200 lines and it belongs in the crm bridge. What
this repository owes the problem is the three things the bridge cannot produce for itself, and
each is a separate entry or already exists:

- **F** — `kernel.jacobian`, exact and source-transformed. Exists.
- **H** — `observe_jacobian`. Does not exist; this issue waits on it.
- **the discrete half as an object** — `code.events`, `Trigger.latch`, `indicators`,
  `transition`, and `modelica.verify` to enumerate them. Exists, and is unexplored for this
  purpose: the engine already models "door open" and "charged" as latched discrete state, which
  is the same object the estimator is trying to infer.

## What would turn this into a story here

An answer to one question: can the estimator's discrete hypothesis be *the model's own latch
configuration*, enumerated by `modelica.verify`, instead of a hand-written two-state machine in
the consumer? If yes, that is a real feature of this toolchain and it gets its own entry. If no,
this closes as "the pieces are exported, the filter lives downstream", which is a legitimate
outcome and should be written down rather than left open.

## Deliberately not done

No filter, no IMM, no detector in `modelica.sim`. A Modelica compiler that ships a Kalman filter
has stopped being a Modelica compiler.

## The answer, 2026-08-09

**No.** The estimator's discrete hypothesis cannot be the model's latch configuration, and the
reason is not a missing feature — it is that a latch is the wrong kind of object. This closes
the way the entry said it would if the answer came out no: *the pieces are exported, the filter
lives downstream*.

`verify: probe`. Everything below was run, not reasoned.

### A latch is a function of the state, so it carries no belief

Starting `modelica.verify.check_iteration` from **every** latch configuration, at a fixed
state, over every corpus model that has events:

| model | latches | starting configurations | settled configurations |
| --- | --- | --- | --- |
| `RoastingMachine` | 9 | 512 | 1 |
| `Pendulum` | 2 | 1 | 1 |
| `TwoInertias` | 2 | 1 | 1 |
| `BouncingBall` | 1 | 1 | 1 |
| `BumpStop` | 1 | 2 | 1 |
| `DrainingTank` | 1 | 2 | 1 |
| `Setback` | 1 | 2 | 1 |

Five hundred and twelve starting configurations, one place to land. Set the latches to
anything and the event iteration repairs them, because a latch is the cached value of a
relation the model can evaluate for itself.

What it is a function *of* is visible by moving the argument. Same state, different time:

    t =   0.0 s   512 -> 1   111111110
    t =  80.0 s   512 -> 1   011111110
    t = 300.0 s   512 -> 1   001111110
    t = 600.0 s   512 -> 1   000111110
    t = 700.0 s   512 -> 1   000001110
    t = 800.0 s   512 -> 1   000000000

That is the burner schedule, read back out of the latch word.

An IMM hypothesis has to be the opposite thing. The premise of this entry is that "a charge and
a drop both move the bean probe fast and far, and no rule on the signal alone separates them" —
so `LOADED`/`EMPTY` is *precisely* the information the state does not carry. A latch is
precisely the information the state does carry. They are complements, and no amount of
plumbing turns one into the other.

There is a second, independent reason. `LOADED` and `EMPTY` have different state vectors — four
states against one. A latch does not change the state vector. The object that does is `Mode`,
and modes come out of index reduction, not out of a modeller declaring two regimes.

### And the construct that would be needed does not exist

Filed separately as
[`a-when-clause-can-only-reinit-a-state-so-no-discrete-variable`](09-08-2026-a-when-clause-can-only-reinit-a-state-so-no-discrete-variable.md).
Three legal Modelica models, asked for a kernel:

    refused  Boolean, plain equation:        DerivativeError: a '>' expression has no derivative
    refused  discrete Real, in a when:       EmitError: no translation for Assign(...) inside a when-clause
    refused  counter with pre(), in a when:  EmitError: no translation for Assign(...) inside a when-clause

`modelica.sim.emit.Emitter.statement` matches `reinit` of a state and nothing else. The middle
two *causalize* — they fail at emission. So this engine has no discrete variable that remembers
at all, and "the engine already models 'door open' and 'charged' as latched discrete state" —
written in the Measured section above, on 3 Aug — **is wrong**. It models the burner schedule
as latched relations. That is a different thing and it was worth finding out.

### What is exported, which is what this repository owed the problem

All three, and the one that was blocking now exists. `RoastingMachine` at t = 300 s:

| piece | what it is | measured |
| --- | --- | --- |
| **F** | `kernel.jacobian` | 4x4, exact, source-transformed |
| **H** | `kernel.observe_jacobian` | 30x4, exact; relative error 1.3e-10 against a central difference quotient, and the worst entry is the *quotient's* — `d C_bean / d m_w` is exactly `cp_water = 4186` and the quotient says 4186.000001 |
| **the discrete half** | `code.events` (9), `Trigger.latch`, `indicators`, `transition`, `modelica.verify` | 512 configurations enumerated in seconds |

The rows the bridge reads, for free:

    H['probe'] = [0.0, 0.0, 1.0, 0.0]
    H['lag']   = [0.0, 1.0, -1.0, 0.0]

The measurement is exactly the third state, so the `LOADED` filter's H is a selector and not a
computation. `lag` — how far the coffee runs ahead of its own probe, the quantity the operator
actually cares about — is the difference of two states and is not measured at all, which is the
case for having the model in the loop rather than a rule on the signal.

## Deliberately not done, found while doing it

- **The IMM is still downstream and should be.** Nothing here changed that; what changed is
  that it is now a decision rather than a gap, and the last piece it was waiting on shipped.
- **If the memoried-discrete issue lands, this question is worth asking again** — and the
  answer would still be no for `LOADED`/`EMPTY` specifically, because of the state-dimension
  argument, which no discrete flag fixes. What it would make possible is a model that carries
  its own regime, which is a different and probably better thing to build.
- **`observed` is 30 variables wide on a 4-state model**, so H is 30x4 and 26 of its rows are
  parameters and derivatives the estimator has no use for. Selecting rows is the caller's
  business today. Worth noticing before somebody multiplies the whole thing every step.
