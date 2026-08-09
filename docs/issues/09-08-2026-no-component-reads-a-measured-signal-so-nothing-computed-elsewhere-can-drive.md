---
title: No component reads a measured signal, so nothing computed elsewhere can drive a model
status: open
area: sim
owner: null
filed: 2026-08-09
kind: story
scope: S
verify: test
---

## Measured

- `examples/library/Blocks.mo` holds six blocks: `Constant`, `Gain`, `Feedback`, `PI`,
  `FirstOrder`, `Limiter` (`examples/library/Blocks.mo:16-56`). None of them carries data.
- Every source in the corpus is a constant or an expression in `time` written into the model
  text. The largest is `RoastingMachine`'s burner schedule — a chain of conditionals over `time`
  inside the model itself, and the corpus's first use of `time` in an equation
  (`logs/2026-08-01.md`). So a schedule is expressible, and only by editing the model.
- **Nothing takes a series from outside.** A measured trajectory, or a field a different tool
  computed, cannot drive anything here.
- Arrays landed on 2026-08-04 — the language half in `18e0f5a`, the flattening half in `5cb8957`
  — so a parameter matrix of samples is expressible now, where before this it was not.
- **Public evidence for the shape of the need**, from a post by Benjamin Turner, P.E.
  (<https://www.linkedin.com/posts/benjamin-turner-p-e-75599214_asme-asme-fea-share-7490842131209596928-IqJv/>):
  a coupled workflow where temperatures computed in one tool are imported into a second to drive
  a thermo-mechanical fatigue analysis. The whole value of the coupling is that the second model
  is driven by the first one's output instead of by an assumed load.
- The Modelica Standard Library's name for this component is
  `Modelica.Blocks.Sources.CombiTimeTable`.

## Inferred

- Inferred: **a table is a source of time events, not only of values.** A linear interpolant has
  a corner at every sample, and an integrator that steps across one is integrating a different
  function on either side of it — the same failure mode `emit`'s latching already exists to
  prevent (`src/modelica/sim/emit.py:40`). The honest version announces the sample instants as
  time events rather than letting the step controller discover them as stiffness.
- Inferred: file reading is the part that does not belong inside the language. A parameter matrix
  is expressible today; a path to a `.csv` is an external function and a promise about the
  filesystem, and it would break the property that a model here needs no other program.

## Done

- `examples/library/Blocks.mo` gains a `TimeTable`: a parameter matrix of `[t, y]` rows, linear
  interpolation between rows, held flat outside the range, and the sample instants announced as
  time events.
- Its twin in `examples/library/Blocks.py`, so the two front ends still agree the way every other
  component does — `library-twins-are-only-compared-as-text` is the standing caveat on how that
  agreement is currently checked.
- A corpus model that uses it, with a `.toml` whose reference points come from evaluating the
  interpolant independently of the simulation.
- The test that catches the event handling being wrong: the value *at* a sample instant is exact
  and does not depend on which side the solver approached it from.

## Deliberately not done

- No file reading — no `.csv`, no `.mat`, no path of any kind. The data is a parameter, and a
  parameter is set by whoever builds the model. `write-results-other-tools-can-read` is the issue
  about file formats and it points the other way, outward.
- No spline and no `smoothness` argument. Linear first. A C¹ interpolant changes what the events
  mean and deserves its own evidence.
- No two-dimensional table, no extrapolation modes beyond holding flat.
