---
title: The EKF results live in a log and a deleted scratch file, not in the documentation
status: open
area: docs
owner: null
filed: 2026-08-03
kind: story
scope: M
verify: inspect
blocked_by:
  - no-output-jacobian
---

## Measured

The site has seven executed notebooks (`docs/notebooks/`) and none of them is about estimation.
The work exists and it is not published. From `logs/2026-07-30.md`, built in
`scratchpad/ekf.py` — a scratch file, not in the repository:

- Twin experiments, so there is a truth to score against: simulate `RoastingMachine`, sample at
  the bridge's 1.5 s, add the probe's real noise, then ask the filter to recover what it was not
  told. F came from `kernel.jacobian`, exact and source-transformed.
- On the **real** batch-1 curve, one-step prediction of the probe:

      open loop (model alone)   1.674 K
      EKF, Q_mach 4             0.397 K
      EKF, Q_mach 40            0.371 K
      EKF, Q_mach 200           0.345 K

- Convergence: start the filter 60 K off in `T_mach` and it converges in about 180 s. RMSE after
  settling 2.126 K, against 2.121 K from a correct start — statistically identical.
- **A correction that is the most useful thing in the whole exercise.** An earlier session had
  reported `T_mach` "formally observable and practically not", from the instantaneous
  observability matrix `[H; HF; HF²; HF³]`: rank 4 everywhere, condition 1e6-1e7, weak direction
  almost purely `T_mach`. That was the wrong metric. It asks what can be recovered from a short
  burst; a filter running 576 samples across a varying gas schedule accumulates far more.
- **A recommendation killed by its own experiment.** The same session had recommended adding
  `inlet` as a second measurement. Measured:

      probe only                     T_mach rmse 2.121 K
      + machine sensor, sigma 5 K               2.496 K
      + machine sensor, sigma 20 K              4.866 K

  A second sensor made it worse — the filter already beats that sensor's own noise, so it
  injects noise rather than information. The planned inlet/return restructuring was dropped.

## Inferred

Why this belongs on the site rather than in a log: it is the clearest demonstration of what
`docs/architecture.md:3` claims — every stage is a value you can hold. The filter needs F and H
and the engine has them exactly, because the model is symbolic. That is an argument no page
currently makes, and the numbers above make it in one screen.

Inferred, and worth flagging to whoever writes it: `RoastingMachine` is a good subject because
the data is real, and a bad one because the notebook cannot ship the 4387-sample capture. A
twin experiment on a corpus model is self-contained; the real-data numbers can be quoted.

## Done

- `docs/notebooks/estimating.ipynb`, executed by the docs build like the other seven, with a nav
  entry: what a state estimator needs, where F and H come from in this toolchain, a twin
  experiment on a corpus model, and the two corrections above stated as corrections.
- Uses `observe_jacobian` for H rather than central differences — which is why this waits on
  that issue.
- No filter enters `modelica.sim`. The notebook is the deliverable; the library keeps its
  derivatives and stops there.

## Deliberately not done

No smoother, no UKF, no particle filter. One EKF, on a model whose numbers are already
measured.
