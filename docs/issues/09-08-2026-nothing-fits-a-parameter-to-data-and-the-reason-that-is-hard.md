---
title: Nothing fits a parameter to data, and the reason that is hard is that parameters trade off
status: open
area: sim
owner: null
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
