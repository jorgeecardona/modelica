---
title: Detecting charge and drop from the bean probe — the models are fine, the decision layer is wrong
status: open
area: sim
owner: null
filed: 2026-08-03
kind: idea
scope: L
verify: probe
blocked_by:
  - no-output-jacobian
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
