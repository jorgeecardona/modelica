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
    heat source — FEM, and out of scope for this toolchain. The post is
    <https://www.linkedin.com/posts/dr-anton-evdokimov_the-temperature-field-tells-you-where-the-share-7487531703264051200-e_f7/>:
    austenite forming under a moving source, then ferrite, pearlite, bainite or martensite on
    cooling, ending at a hardness distribution in X60 steel. It names the three kinetic models
    by name — JMAK, Kirkaldy-Venugopalan, Koistinen-Marburger.
- Also stated in that post, and separable from the mesh: **each phase has its own thermal and
  mechanical properties**, and the transformations are what set residual stress, hardness and
  distortion. A commenter on it (Kaveh Alipour) adds that the austenitisation temperature is not
  a constant ~900 °C but depends on composition, `Ac1` and `Ac3`.
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

Inferred, and the part that decides whether this is one model or two: **if the phase-dependent
properties feed back, the problem stops being a post-processing exercise.** A specific heat and
a conductivity that are functions of the fractions put the fractions back into the thermal
equation they were driven by, so temperature and transformation are one coupled nonlinear
system rather than a cycle computed first and a kinetics integrated after it. The corpus has
nothing of that shape: `RoastingMachine` has a heat capacity that falls as the beans dry
(`C_bean`), which is the same idea with one fraction, and this would be the same idea with
four that must sum to one. Which of the two versions to write is the open question — the
uncoupled one is a smaller model and a weaker claim.

## Done

- A corpus model: a prescribed thermal cycle (heat, hold, quench), JMAK for the diffusive part,
  Koistinen-Marburger for the martensitic part, phase fractions as states.
- Written twice as every example is, with a `.toml` whose invariant is that the fractions sum to
  one over the whole run, and whose reference points come from an independent integration.
- The transformation-start events are real state events, not sampled — that is the part that
  makes it a simulation problem rather than a plotting exercise.
- An `[animate]` block if the result reads better as a picture; it is not required.
- A decision recorded either way on the coupling above: if the properties feed back, the model
  says so and the algebraic loop it creates is part of what the example teaches; if they do not,
  the `.toml`'s `teaches` field does not claim a coupling that is not there.
- Hardness as an observed variable — a weighted sum over the fractions — because that is the
  quantity the post ends on and it costs one alias.

## Deliberately not done

- No field, no mesh, no moving heat source. That is FEM, it is out of scope, and
  `docs/ideas/pdes.md` is the standing argument for keeping it out.
- Nobody asked for this. It was offered, not picked, and it is recorded here so that "we thought
  of it and did not do it" is visible rather than forgotten — which is what this folder is for.
