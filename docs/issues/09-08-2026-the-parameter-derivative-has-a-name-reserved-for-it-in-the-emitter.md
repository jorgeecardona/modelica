---
title: The parameter derivative has a name reserved for it in the emitter and no code behind it
status: open
area: sim
owner: null
filed: 2026-08-09
kind: story
scope: S
verify: test
---

## Measured

- `modelica.sim.emit` writes `rhs`, `observe`, `jacobian`, `indicators`, `transition`, and since
  `4686cd1` also `observe_jacobian` (`src/modelica/sim/emit.py:48`). Every one of them already
  takes the parameters as an argument: the signature is `observe_jacobian(t, y, p, w, out)`.
- **The emitter states this gap in its own docstring**, where it explains why `observe_jacobian`
  did not take the shorter name (`src/modelica/sim/emit.py:869`):

  > Emitted as ``observe_jacobian`` and not as ``sensitivity``. The word *sensitivity* has an
  > owner in this field: it is the derivative with respect to a **parameter**, which is what
  > identification and uncertainty propagation ask for. That is the same transformation again
  > with a different seed, it will want the name, and taking it here for a different derivative
  > would cost more than the shorter word is worth.

- The transformation it refers to is already split for reuse. `emit.differentiate` became
  `emit.tangent`, which propagates tangents through the blocks and returns what it reached;
  `jacobian` reads it off at the state derivatives, `observe_jacobian` reads it off at the
  variables (`4686cd1`). One walk, two readings.
- **Public evidence that this derivative is what practice asks for**, from a post by Anton
  Evdokimov on welding heat-source calibration
  (<https://www.linkedin.com/posts/dr-anton-evdokimov_heat-source-calibration-looks-simple-but-share-7485439140239691776-hL40/>):
  a heat source has a thermal efficiency and a spatial radius, both of which move the
  temperature near the weld, so many pairs fit the same data. His procedure is ordered by which
  measurement is sensitive to which parameter — efficiency from thermocouples far from the weld
  line, where the shape barely matters, then the radius against the fusion boundary. That
  ordering is read off d(observed)/d(parameter) and off nothing else.

## Inferred

Inferred, not measured: the seed is the only difference. `tangent` currently seeds a unit vector
in the states; seeding it in a parameter propagates through the same blocks, and the same two
readings apply — at the state derivatives it is d(rhs)/dp, at the variables it is
d(observed)/dp. Neither has been written, so neither cost is known.

An ordering constraint worth stating before either lands: `parameters-as-constants-or-as-arguments`
proposes burning parameters in as literals, and a literal has no derivative. The two are not in
conflict — the constant-folded kernel is the fast path and the sensitivity kernel is a second
build — but whichever lands second has to say so.

## Done

- `emit` emits the two parameter partials, d(rhs)/dp and d(observed)/dp, by the same forward-mode
  walk as `jacobian` and `observe_jacobian`, with the same workspace conventions.
- The caller names which parameters to differentiate against; it is not all of them. A corpus
  model declares dozens of parameters and an identification asks about three.
- `Kernel` exposes them beside the other derivatives, and `build(jit=True)` compiles them.
- Tested the way the other two derivatives are: against central differences on every corpus
  model, at the tolerance `tests/test_examples.py` already uses for the exact-versus-differenced
  comparison of `jacobian`.

## Deliberately not done

- **No trajectory sensitivity.** dy(t)/dp is *not* the partial — y itself depends on p, so the
  useful quantity for a fit comes from integrating the forward-sensitivity equations alongside
  the state. That is a solver change on top of this one, and it belongs to whatever fits
  parameters. This issue delivers the partials that change needs.
- No fit, no optimiser, no uncertainty propagation.
- No d(rhs)/dp for the *discrete* state. A latched relation is frozen between events by
  construction; what a parameter does to the instant an event fires is a different question.
