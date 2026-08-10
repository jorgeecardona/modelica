---
title: A model can be integrated and cannot be asked where it settles
status: done
area: sim
owner: cute-replicant-6e76d
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

## What landed

`modelica.sim.steady`, `Steady`, `SteadyError`. `tests/test_steady.py`, 14 tests. 1305 pass.

    >>> steady(system)
    RLCCircuit settles at l.i = 0, c.v = 1 (residual 0.00e+00, 1 steps)

Newton on the emitted `rhs` and `jacobian`, and nothing new is emitted for it -- `rhs` already
*is* the residual and `jacobian` is d(rhs)/dy exactly. The answer carries every observed
variable at that point, the residual it reached, and which state set it is an equilibrium
**of**.

**Four corpus models carry a `[steady]` section with `source = "analytic"`**, where the entry
asked for two:

| model | settles at | why, in one line |
| --- | --- | --- |
| `NewtonCooling` | `T = 25` | `der(T) = -k (T - T_amb)` vanishes only at `T_amb` |
| `RCCircuit` | `c.v = 1` | no current at rest, so the capacitor holds the source |
| `RLCCircuit` | `l.i = 0, c.v = 1` | the same, with one more state; independent of R, L and C |
| `PIControl` | `plant.y = 1, controller.x = 0.5` | integral control has no steady-state error *by construction* -- the integrator stops only where the error is zero |

**And the two analyses agree.** Integrating each far enough under constant inputs arrives at
the `[steady]` answer to 1e-6. That is the "no model drift" claim made checkable rather than
asserted: one set of equations, two workflows, one answer.

### `TwoBodyHeat` corrects the entry

This entry expected it to "settle at the heat-capacity-weighted mean of its two initial
temperatures". That is true of the **trajectory** and it is not a root of the residual. Two
masses joined by a conductor stop moving whenever their temperatures are *equal* -- at any
common value -- so `rhs = 0` has a one-parameter family of solutions and its Jacobian is
singular everywhere on it. Picking the right member needs the conserved energy as an extra
equation, which a steady solve on `rhs` alone does not have.

So it is **refused**, with that in the message, rather than returning whichever member the
arithmetic drifted to. The inference in this entry that "steady state and initialization are
the same nonlinear solve with different unknowns held fixed" is the same observation from the
other side: what is missing in both cases is the extra equations.

`VanDerPol` behaves as the entry predicted: Newton lands on the origin, the unstable fixed
point its limit cycle goes round, and nothing here claims the model settles there.
`DrainingTank` is refused too -- its square root goes imaginary under a Newton step and the
residual is NaN.

## Deliberately not done, found while doing it

- **No continuation, no homotopy, no line search**, as the entry required. Every model that
  converges here does so in one step, because each is linear in its states; a model that needs
  a path is a different entry point.
- **No stability claim**, as required. `Steady` reports the residual and the mode and says
  nothing about what a trajectory does near the point.
- **`values=` overrides parameters** the way `modelica.sim.sweep` does, so an equilibrium can
  be asked for at another operating point without editing the model. That is the half of
  "sizing" that is a solve over *states*; the half that solves over *parameters* is the
  calibration machinery and is still not here.
- **The `[steady]` section is read by `tests/test_steady.py` and not by `test_examples.py`.**
  It could move; it is where the analysis it checks lives.
