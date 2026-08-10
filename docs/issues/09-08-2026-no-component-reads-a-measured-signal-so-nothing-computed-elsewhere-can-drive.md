---
title: No component reads a measured signal, so nothing computed elsewhere can drive a model
status: blocked
area: sim
owner: null
filed: 2026-08-09
blocked_by:
  - a-parameter-array-cannot-be-given-its-values-parameter-real-t-4
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

## Blocked 2026-08-10: the TimeTable cannot be written at all

Picked up as a scope-S entry and put down without claiming it. The Done list asks for "a
parameter matrix of `[t, y]` rows, linear interpolation between rows". Two separate things
stop that, both measured rather than reasoned about, and neither is in this repository's
`sim` layer.

**An array cannot be given its values.**

    parameter Real ts[4] = {0.0, 1.0, 2.0, 3.0};
    EmitError: array equations are not supported yet

`HeatBar` declares `Real T[n]` and works because every array in it takes a *scalar* start that
broadcasts. A table needs a different value per element and there is no way to say so. Filed
as [`a-parameter-array-cannot-be-given-its-values`](10-08-2026-a-parameter-array-cannot-be-given-its-values-parameter-real-t-4.md),
which this now waits on.

**And a lookup cannot be searched for.**

    for i in 1:3 loop
      if x < ts[i + 1] then ...
    FlattenError: function 'lookup' uses a 'for' loop, which cannot be turned into an
    expression; statements here may assign and branch, and nothing else

Functions are inlined into an expression, and a loop with a data-dependent exit is not one. So
even with array bindings, the interpolation would have to be written as an unrolled chain for
one fixed table size -- a `TimeTable4` -- which is not the component this entry asks for.

The two are separable and the first is much the smaller: an array binding is `n` scalar
equations after flattening, which is what `for` equations already produce. The second is a
question about what inlining means, and may be answered by not inlining.

Worth naming what this costs today: `RoastingMachine`'s burner schedule is the shape a
modeller is forced into instead -- nine literal thresholds typed into the model by hand, for
one machine, because there is no way to hand a component a table.
