---
title: A phase-change kinetics example — the reachable half of the welding animation
status: open
area: sim
owner: null
filed: 2026-08-03
kind: idea
scope: M
verify: test
---

## Measured

- On 2026-08-02 Jorge pointed at two LinkedIn animations and said "I want to be able to generate
  images like those ones". The two were answered differently
  (`logs/2026-08-02.md`):
  - **The MathWorks one is reachable** — a mechanism animation is a trajectory drawn as a shape
    in space. That was built: `examples/animate.py` renders any corpus model from an `[animate]`
    block in its own `.toml`, and `AtwoodPendulum` was written for it.
  - **The Evdokimov one is not, as posted.** Its image is a *field* over a mesh under a moving
    heat source — FEM, and out of scope for this toolchain.
- What was recorded as in scope, and not built: "the kinetics (JMAK, Koistinen-Marburger) on a
  prescribed thermal cycle: a lumped ODE with events, same shape as RoastingMachine's gas
  schedule. Not built; he did not pick it."
- `examples/models/RoastingMachine.mo` is the precedent for the shape: a prescribed schedule in
  `time`, time events, and a state that responds to it. It is the corpus's first model with
  `time` in an equation (`logs/2026-08-01.md`).

## Inferred

Why it is worth a model, inferred from what the corpus does not yet contain: every example here
is mechanics, electrics, thermal or fluid. Phase-transformation kinetics is a different
*physics* over the same mathematics — Johnson-Mehl-Avrami-Kolmogorov for diffusive
transformation, Koistinen-Marburger for the martensitic one, both driven by a prescribed
temperature history, both producing fractions that must sum to one. That last constraint is an
invariant of exactly the kind `examples/*.toml` already checks, and it comes from the physics
rather than from the modeller.

The inferred payoff is a Modelica argument, not a metallurgy one: the same toolchain that reads
a DC motor reads a steel transformation, and both end at the same `FlatModel`.

## Done

- A corpus model: a prescribed thermal cycle (heat, hold, quench), JMAK for the diffusive part,
  Koistinen-Marburger for the martensitic part, phase fractions as states.
- Written twice as every example is, with a `.toml` whose invariant is that the fractions sum to
  one over the whole run, and whose reference points come from an independent integration.
- The transformation-start events are real state events, not sampled — that is the part that
  makes it a simulation problem rather than a plotting exercise.
- An `[animate]` block if the result reads better as a picture; it is not required.

## Deliberately not done

- No field, no mesh, no moving heat source. That is FEM, it is out of scope, and
  `docs/ideas/pdes.md` is the standing argument for keeping it out.
- Nobody asked for this. It was offered, not picked, and it is recorded here so that "we thought
  of it and did not do it" is visible rather than forgotten — which is what this folder is for.
