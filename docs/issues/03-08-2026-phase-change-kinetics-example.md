---
title: A phase-change kinetics example — the reachable half of the welding animation
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: cbe2cb3
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

## What landed

`examples/models/PhaseChange.{mo,py,toml}`. 1216 tests pass, `mkdocs build --strict` clean.

A 1 kg bar goes into a furnace, comes out at 520 s, cools in air, is quenched in oil at
700 s. Four phases, seven states, and a hardness at the end.

| bullet | what landed |
| --- | --- |
| prescribed cycle, JMAK, Koistinen-Marburger, fractions as states | yes; `T`, four fractions and two reaction clocks are the seven states |
| written twice, with the sum-to-one invariant and reference points from an independent integration | yes; the twin renders the `.mo` byte for byte, and the four reference points come from scipy at rtol = atol = 1e-12 |
| transformation starts are real state events, not sampled | yes; 14 events, all but the two furnace times are crossings of `Ac1`, `Ps` and `Ms` |
| `[animate]` if it reads better as a picture | no; a bar with no geometry has nothing to draw |
| a decision recorded on the coupling | **coupled**, and it is the point of the model. See below |
| hardness as an observed variable | yes, a weighted sum: 520 HV at the end |

### The coupling: it is real, and it is a 2x2

The entry asked which version to write and said the uncoupled one was "a smaller model and a
weaker claim". Coupled, and by a route the entry did not name.

The feedback is not only the specific heat. Koistinen-Marburger is **not a rate law** -- it
says `x_m = 1 - exp(-alpha (Ms - T))`, a fraction fixed by how far below `Ms` the bar has
got, with no time in it. Differentiated against the austenite still available it becomes
`dx_m/dt = -alpha x_aust dT/dt`, so **the martensite rate reads the temperature's own
derivative**. The energy balance reads the martensite rate through the latent heat. The two
are one simultaneous block, which causalization finds:

    blocks: ... (1, ('r_pear',)), (1, ('der_T', 'r_mart')), (1, ('der_x_aust',)) ...

It is linear in both unknowns, so it solves without iteration. The corpus had no block that
came from the *physics* rather than from a loop of components, and that is what this example
is for.

### What the model does

Reaches 759 degC, austenitises fully during the hold, forms 38 % pearlite cooling in air,
quenches to 60 % martensite, and keeps **2.4 % retained austenite** -- a real feature of an
incomplete quench, not a numerical residue. `total` holds to **1.3e-15** over the whole run.

Cross-checked against Koistinen-Marburger's own closed form rather than only against the
integrator: austenite at the `Ms` crossing is 0.6261, so the law predicts
`0.6261 * (1 - exp(-0.011 * 296)) = 0.6020` and the model gives 0.5987. The 0.55 % gap is
named in the `.toml` rather than smoothed over -- the closed form assumes nothing else
competes below `Ms`, and here the pearlite window closes exactly *at* `Ms`.

## Deliberately not done, found while doing it

- **The latent heat has a ceiling, and it is physics rather than a fudge.** Solving the block
  by hand gives `der(T) = -UA (T - T_amb) / (m (cp - L_mart alpha x_aust))`. The denominator
  changes sign when `L_mart * alpha` exceeds `cp`, and then cooling the bar heats it. With
  `alpha = 0.011` the ceiling is 44.5 kJ/kg; real medium-carbon values run 20 to 80, so the
  ceiling is reachable. Above it the transformation releases heat faster than the quench
  removes it -- recalescence, which is a real phenomenon that this *formulation* cannot
  represent: it needs KM as a function of the lowest temperature reached, which is another
  state. `L_mart = 21000` is inside the physical range with a factor of two of margin, and
  there is an invariant on `r_mart` holding the trajectory to it.
- **The pearlite reaction is gated on its window and the first version was not.** Left
  ungated it kept converting austenite to pearlite after the quench, because `tau_pear`
  stops but the rate does not depend on it stopping. Below `Ms` that austenite is *retained*,
  not pearlite. It changed the answer from 28 % martensite to 60 %.
- **`max` has no derivative rule**, so the martensite rate cannot be written
  `max(0, -alpha x dT)`. The guard is the `T < Ms` branch instead, and the latent-heat
  ceiling is what makes that guard sufficient. A `max`/`min` rule in `modelica.ir.derive` is
  a real gap and is not filed here because nothing else in the corpus wants one yet.
- **`modelica.build` could not say `and`.** Python cannot overload `and`, so the Python front
  end had no spelling for a condition on a temperature *window* -- and the twin has to render
  the same text as the `.mo`, so nesting two `if_`s was not an option. `and_`, `or_` and
  `not_` are new, and the model is what found them missing.
- **No `Ac1`/`Ac3` composition dependence**, which the commenter on the original post asked
  for. It is a second temperature and a second reaction and it does not change the shape of
  what this example teaches.
- **No animation**, as above. A hardness bar chart over time would be a plot, not a picture
  of the model.
