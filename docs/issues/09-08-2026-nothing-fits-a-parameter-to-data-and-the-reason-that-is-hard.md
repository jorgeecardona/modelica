---
title: Nothing fits a parameter to data, and the reason that is hard is that parameters trade off
status: in-progress
area: sim
owner: quiet-tars-187f2
filed: 2026-08-09
blocked_by:
  - the-parameter-derivative-has-a-name-reserved-for-it-in-the-emitter
kind: idea
scope: L
verify: probe
---

## Measured

- No entry point in `modelica.sim` fits anything. `simulate` (`src/modelica/sim/solve.py:100`)
  runs a model forward with the parameters it was given, and that is the whole surface.
- **Estimation has been done here twice, both times by hand and outside the package.**
  `logs/2026-07-30.md` records building the observability matrix `[H; HF; HF^2; HF^3]` for
  `RoastingMachine`, with `F` from `kernel.jacobian` and `H` central-differenced; the same
  session's EKF lived in `scratchpad/ekf.py` and was never committed. `an-estimation-notebook` is
  the open issue for the second of those. `4686cd1` fixed the `H` half by emitting
  `observe_jacobian`.
- **The public statement of the difficulty**, from a post by Anton Evdokimov on welding
  heat-source calibration
  (<https://www.linkedin.com/posts/dr-anton-evdokimov_heat-source-calibration-looks-simple-but-share-7485439140239691776-hL40/>):
  a heat source model has a thermal efficiency (how much energy enters the workpiece) and spatial
  parameters (how concentrated it is). *"Temperatures close to the weld line are influenced by
  both the thermal efficiency and the heat source shape"* — so many parameter pairs produce
  nearly identical fusion boundaries while giving different thermal cycles everywhere else. His
  procedure is two staged fits ordered by sensitivity: efficiency first, from thermocouples far
  from the weld where the shape barely matters; then the radius against the fusion cross-section,
  with efficiency held.

## Inferred

- Inferred, not measured: **a least-squares fit over both parameters at once reproduces exactly
  the failure he describes.** It converges to *a* pair that matches the data it was shown, is
  wrong everywhere else, and reports a small residual while being wrong. A fit without an
  identifiability diagnostic is a machine for producing confident wrong parameters, and that is
  the argument for building the diagnostic first.
- Inferred: the diagnostic is the conditioning of the sensitivity matrix over a chosen set of
  measurements — the same construction as the observability matrix already built by hand for
  `RoastingMachine`, with parameters in place of states. Two parameters that move every
  measurement together give a near-singular matrix, and **that is computable from the model
  alone, before any data exists.** The same matrix answers his "use the far thermocouple first"
  advice: it names the measurement whose sensitivity to one parameter is large and to the other
  small.
- Inferred: this needs the *trajectory* sensitivity dy(t)/dp, not the partials alone, because y
  depends on p. That means integrating the forward-sensitivity equations alongside the state,
  which is a solver change sitting on top of the emitter change its blocker delivers.

## Done

Three pieces, in this order, because the diagnostic is useful without the fit and the fit is
dangerous without the diagnostic.

- **Trajectory sensitivity:** dy/dp integrated alongside y, from the partials
  `the-parameter-derivative-has-a-name-reserved-for-it-in-the-emitter` emits.
- **An identifiability report** over a named parameter set and a named measurement set: the
  sensitivity matrix, its conditioning, and which parameters it cannot separate. Model only, no
  data required.
- **A fit** that takes measurements and a parameter subset, holds the rest, and reports the
  residual *together with* the conditioning it fit at. Asked to fit parameters the report says
  are inseparable, it says so rather than returning a number.
- `verify: probe`, deliberately. A green test on synthetic data proves the optimiser converges,
  which is not the claim. The claim is that the procedure recovers known parameters from
  real-shaped data, and the way to check it here is `RoastingMachine` against a Cropster roast
  log — the same sensing that `green-moisture-as-a-measurement` and
  `detecting-charge-and-drop-from-the-bean-probe` already reach for.

## Deliberately not done

- No Bayesian posterior, no MCMC, no confidence intervals. A conditioning number and a residual
  answer the question the post asks. Uncertainty quantification is a separate direction and would
  need its own argument in `docs/ideas/`.
- No global optimiser. Local and gradient-based, from the parameters the model already declares —
  the sensitivities exist precisely so that a gradient method is available.
- **No automatic sensor placement.** The report says which measurements separate which
  parameters; deciding where a thermocouple actually goes depends on things the model does not
  know, and presenting a ranking as a recommendation would overstate it.
- No state estimation. An EKF tracks states with parameters known; this fits parameters with
  states unknown. They share `observe_jacobian` and nothing else, and merging them would produce
  a joint estimator nobody asked for.

## The first of the three pieces landed 2026-08-13

`modelica.sim.sensitivity`. ``S' = (df/dy) S + (df/dp)`` integrated alongside ``y``, with both
partials taken from the emitted `jacobian` and `jacobian_parameters` -- exact, by source
transformation, so nothing in it differences anything. `Sensitivity` carries ``dy/dp`` and,
through the chain rule over `observe_jacobian` and `observe_jacobian_parameters`,
``d(observed)/dp`` -- which is the one a caller wants, because a measurement is almost never a
state.

### Checked against the thing it replaces

Central differences of two runs, ``h = 1e-5`` relative, both at ``rtol = 1e-9``, every observed
variable at eight points:

| | worst relative disagreement |
| --- | --- |
| `NewtonCooling`, `d/dk` | 1.8e-7 |
| `RCCircuit`, `d/dr.R` | 8.1e-10 |
| `RCCircuit`, `d/dc.C` | 8.9e-7 |
| `MassSpringDamper`, `d/dspring.c` | 7.0e-8 |

About where a quotient of two 1e-9 trajectories should land, which is the point: the
disagreement belongs to the quotient. `tests/test_sensitivity_forward.py`, 7 tests.

### Two refusals, both measured into existence

**A model with events is refused.** This was going to be an approximation until
`RoastingMachine` was tried: integrated through the augmented system with no event handling,
its *own trajectory* -- before any sensitivity is read -- comes out **8.8e-2** from
`simulate`'s, because the latches are set once and never refreshed. And ``S`` needs a jump
condition at a crossing, since a `when` that reinitialises a state moves ``dy/dp`` by an
amount depending on the crossing time, which itself depends on ``p``. Eight corpus models have
events; returning a number that is quietly wrong for all of them is not an option.

**A kernel built without `sensitivity=` is refused**, rather than returning the zeros its
empty `jacobian_parameters` would produce. Zeros are a plausible answer -- a parameter nothing
depends on has exactly that sensitivity -- so the failure has to be loud.

### And one thing left open on purpose

``S(0) = dy0/dp`` defaults to zero and is the caller's to supply. A start value written in
terms of a parameter has it non-zero, and **no emitted function computes it**: `initial` maps
``p`` to ``y0`` and its derivative is written nowhere. `sensitivity` does not guess, because a
wrong ``S(0)`` is wrong for the whole run and reads like a modelling error rather than an
unset argument.

### What is left on this entry

The other two pieces, unchanged: the identifiability report over a parameter set and a
measurement set, and the fit that refuses to fit what the report says is inseparable. Both now
have their input.

Also new, and both blocking a *complete* answer rather than the next piece:

- **`dy0/dp` is not emitted.** Until it is, a model whose starts depend on the fitted
  parameters needs its `S(0)` written by hand.
- **Events.** An identifiability report on `RoastingMachine` -- which is the model this entry
  names for its probe -- cannot be built until forward sensitivity crosses an event.

1554 pass, typecheck 0.

## The second piece landed the same day: the identifiability report

`modelica.sim.identifiability`. The scaled sensitivity matrix over a parameter set and a
measurement set, its singular values, and the combination the measurements see least. **No
data required**, which is this entry's own argument for building it before the fit.

### The failure this entry describes, on a corpus model, with the exact answer

`RCCircuit`, fitting `r.R` and `c.C` from the capacitor voltage:

    RCCircuit: 2 parameters from 1 measurements, condition 1.95e+14
      invisible combination: -0.707*r.R +0.707*c.C

``c.v`` decays as ``exp(-t/(R*C))``, so it sees the *product* and nothing else. Double ``R``
and halve ``C`` and every sample is unchanged. The weakest direction comes out at exactly 45
degrees -- ``sqrt(1/2)`` to three places -- which is the analytic null of ``log R + log C``
rather than an approximation of it, because the columns are scaled to ``d/d log p``.

That is the welding failure from the LinkedIn post, in four lines and without a workpiece: a
least-squares fit over both would converge to whichever pair the optimiser wandered into,
report a small residual, and be wrong about the current.

### And the advice, arrived at from the model

His procedure is two staged fits ordered by sensitivity -- the far thermocouple first, where
the shape barely matters. The same shape here. ``r.i`` is ``C * d(c.v)/dt``, so it carries
``C`` on its own:

| measured | condition |
| --- | --- |
| `c.v` | **1.95e14** |
| `c.v` and `r.i` | **3.32** |

Nothing was measured to find that out. It is a statement about the model and the sensor set,
which is what this entry inferred it would be.

### The negative case, so the report is not a machine for finding something

`NewtonCooling`'s ``k`` and ``T_amb`` from ``T`` read **2.92**; `MassSpringDamper`'s stiffness
and damping from the position read **5.06**. Fourteen orders from the RC reading, which is the
gap `Identifiability.inseparable`'s default threshold of 1e-6 sits in the middle of on a log
scale.

### One thing a caller has to know

**A large condition number is bounded by the integration tolerance.** An invisible direction's
singular value is round-off, so what comes out measures how accurately the sensitivity was
integrated: the RC reading is 1.7e10 at the default `rtol = 1e-6` and 1.9e14 at 1e-9. The
*conclusion* is identical at both and the separable numbers do not move at all -- but the big
number is not a property of the model and the docstring says so.

`tests/test_sensitivity_forward.py`, 12 tests.

### What is left

The third piece: a fit that takes measurements and a parameter subset, holds the rest, and
reports the residual *together with* the conditioning it fit at -- refusing, rather than
returning a number, when this report says the subset is inseparable. Its input now exists.

Unchanged from the note above, and both still blocking a complete answer: `dy0/dp` is not
emitted, and forward sensitivity refuses a model with events -- which `RoastingMachine`, the
model this entry names for its probe, has nine of.
