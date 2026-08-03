---
title: There is no output Jacobian, so every estimator finite-differences H by hand
status: in-progress
area: sim
owner: quiet-tars-187f2
filed: 2026-08-03
kind: story
scope: S
verify: test
---

## Measured

- `modelica.sim.emit` writes five functions: `rhs`, `observe`, `jacobian`, `indicators`,
  `transition` (`src/modelica/sim/emit.py:48`). `jacobian` is `rhs` differentiated by source
  transformation, forward mode, one block at a time (`src/modelica/sim/emit.py:55`) — so
  **F = d(rhs)/dy is exact and already emitted**.
- There is no equivalent for `observe`. `grep -n "observe" src/modelica/sim/emit.py` shows it
  emitted as a value function only. **H = d(observe)/dy does not exist.**
- Measured consequence, from `logs/2026-07-30.md`: building the observability matrix
  `[H; HF; HF^2; HF^3]` for `RoastingMachine` required central differences for H, while F came
  straight from `kernel.jacobian`. The same session's EKF work
  (`scratchpad/ekf.py`, not committed) did the same thing again.
- The log records the fix in one line: "The engine has no output Jacobian (`d observe / dy`); H
  had to be finite-differenced. Small extension of the existing source transformation."

## Inferred

Inferred, not measured: finite-differencing H injects truncation error into exactly the place
that hurts — the Kalman gain — and it costs *n* extra `observe` calls per update. Neither cost
has been quantified here. What *is* measured is that two independent pieces of work needed H
and both had to invent it, which is the argument for putting it where F already is.

The general point is worth stating because it will come up again: an observed variable is an
ordinary expression over states and parameters, so its derivative is available by the same
forward-mode transformation that produced `jacobian`. Nothing new has to be invented; a second
caller of an existing pass has to be written.

## Done

- `modelica.sim.emit` emits `observe_jacobian(t, y, p, w, out)` — d(observed)/d(states), by the
  same source transformation as `jacobian`, with the same workspace conventions.
- `Kernel` exposes it beside `jacobian`, and `build(jit=True)` compiles it like the others.
- Tested the way the emitter's other derivatives are: against central differences on every
  corpus model, at the tolerance `tests/test_examples.py:384` already uses for the exact-versus
  -differenced comparison of `jacobian`.
- `tests/test_jit.py` covers it for the compiled path.

## What landed, and the two things worth knowing

The forward-mode walk in `emit.differentiate` was split into `emit.tangent`, which propagates
tangents through the blocks and returns what it reached; `jacobian` reads it off at the state
derivatives and `observe_jacobian` reads it off at the variables. One transformation, two
readings, so there is no second implementation to keep in step. 28 difference-quotient checks
pass across the corpus (14 models x 2 derivatives), and `moon run :test-jit` compares it
compiled against interpreted on five models.

1. **The aliases had to be differentiated here.** `jacobian` never touches them — causalization
   eliminates an alias before the solver sees it — but `observe` writes every one, and an alias
   is exactly what a model is usually measured *through*. So `observe_jacobian` differentiates
   `system.aliases` after the blocks, and `test_the_output_jacobian_agrees_with_a_difference_quotient`
   is the only check in the suite that covers those rows at all.
2. **Named `observe_jacobian`, not `sensitivity`.** *Sensitivity* has an owner in this field:
   the derivative with respect to a **parameter**, which identification and uncertainty
   propagation ask for. That is this same transformation with a different seed, it will want
   the name, and spending it here would have cost more than the shorter word is worth.

## Deliberately not done

- No estimator, no filter, no Kalman anything in `modelica.sim`. This issue delivers the
  derivative and nothing above it. The notebook that uses it is a separate entry, and the
  argument for keeping filtering *out* of the library is in the roast-event-detection entry.
- No sensitivity of the observed variables to *parameters* (d(observe)/dp). That is what
  parameter identification wants, it is the same transformation again, and it should be its
  own issue once something needs it.
