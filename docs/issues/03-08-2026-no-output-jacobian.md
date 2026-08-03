---
title: There is no output Jacobian, so every estimator finite-differences H by hand
status: open
area: sim
owner: null
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

## Deliberately not done

- No estimator, no filter, no Kalman anything in `modelica.sim`. This issue delivers the
  derivative and nothing above it. The notebook that uses it is a separate entry, and the
  argument for keeping filtering *out* of the library is in the roast-event-detection entry.
- No sensitivity of the observed variables to *parameters* (d(observe)/dp). That is what
  parameter identification wants, it is the same transformation again, and it should be its
  own issue once something needs it.
