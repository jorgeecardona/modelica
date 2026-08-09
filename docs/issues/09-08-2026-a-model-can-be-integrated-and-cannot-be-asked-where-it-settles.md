---
title: A model can be integrated and cannot be asked where it settles
status: open
area: sim
owner: null
filed: 2026-08-09
kind: story
scope: M
verify: test
---

## Measured

- `modelica.sim.solve` exposes exactly one entry point, `simulate` (`src/modelica/sim/solve.py:100`).
  Everything else in the module is the run loop and its parts — `_Run`, `_segment`, `_settle`,
  `_sample`, `_result`. There is no call that answers "what does this model settle to".
- The pieces a steady solve needs are already emitted and already exercised. `rhs(t, y, p, w, out)`
  *is* the residual, and `jacobian` is d(rhs)/dy **exactly**, by source transformation, not by
  difference quotient (`src/modelica/sim/emit.py:55`). A Newton iteration on `rhs(...) = 0` calls
  both and needs nothing that does not exist.
- `examples/*.toml` states what a model must do as `[[simulate.point]]` — a time, values, an
  `rtol` and a `source` — and `[[simulate.invariant]]` with `expr`/`min`/`max`
  (`examples/models/NewtonCooling.toml`). There is no section for an equilibrium.
- `initial-equations-are-carried-and-never-read` is the neighbouring open issue: an
  `initial equation` is parsed, flattened, carried on `System`, and read by nothing.
- **Public evidence for why this is a second entry point rather than a nicety**, from a JuliaHub
  post about Dyad
  (<https://www.linkedin.com/posts/hvac-share-7491182510975107074-Kp5Q/>): one acausal model of a
  three-zone commercial building serving both steady-state HVAC sizing and 24-hour dynamic
  analysis, with the argument stated as *"the same equations, parameters, and thermal network
  remain consistent across workflows — reducing model drift, manual updates, and engineering
  risk."* Sizing and transient are the two things an engineer does with one thermal model. This
  toolchain offers one of them.

## Inferred

- Inferred, not measured: steady state and initialization are the same nonlinear solve with
  different unknowns held fixed, so one Newton driver serves both and
  `initial-equations-are-carried-and-never-read` becomes a caller rather than a second
  implementation. Nobody has tried to share them here, so this is an expectation, not a finding.
- Inferred: the hard part is not Newton, it is *which* residual. `der(x) = 0` is unambiguous for
  a plain ODE. With a latched relation in `w` an equilibrium is per-mode, and a model with more
  than one admissible state set has one equilibrium per set. The corpus contains both kinds
  (`DrainingTank`, `BouncingBall`).

## Done

- `modelica.sim` gains an entry point beside `simulate` that solves `rhs(t, y, p, w) = 0` by
  Newton on the emitted `jacobian`, returns the state vector and the observed values there, and
  reports honestly when it does not converge or when the matrix is singular.
- The latches are set the way `simulate` sets them at the initial point, and the answer says
  which mode it is an equilibrium **of**.
- `examples/*.toml` gains a `[steady]` section carrying the equilibrium values with a `source`,
  the same way every reference point already carries one.
- At least two corpus models carry one with an analytic source: `NewtonCooling` settles at
  `T_amb`; `TwoBodyHeat` settles at the heat-capacity-weighted mean of its two initial
  temperatures. Both are one line of algebra, so the test checks the solver instead of trusting
  it.
- The test that ties the two analyses together: integrating far enough under constant inputs
  arrives at the `[steady]` answer within tolerance. That is the "no model drift" claim made
  checkable rather than asserted.

## Deliberately not done

- No continuation and no homotopy, and therefore not the `homotopy()` operator. Newton from the
  `start` attributes is the first pass; a model that needs a path to find its equilibrium is a
  separate issue with its own evidence.
- **No optimisation-based sizing.** "Sizing" in the post's sense picks a parameter to hit a
  target, which is a solve over *parameters*. That is the calibration machinery, not this.
- No claim of stability. `VanDerPol` has a limit cycle around an unstable fixed point; the entry
  point reports what Newton found and does not claim anything settles there.
