---
title: Nothing checks that index reduction preserves the solution set, because that needs numbers and the generator makes structure
status: open
area: verify
owner: null
filed: 2026-08-13
kind: story
scope: M
verify: test
---

## Measured

`03-08-2026-property-based-testing-for-causalization` listed three properties and
`tests/test_causalize_properties.py` delivers two of them. The third is not there:

> **Differentiating preserves the solution set**: a trajectory admissible before reduction is
> admissible after it, checked by residual evaluation at sampled points.

Everything in that file is structural. `grep -c "assert" tests/test_causalize_properties.py`
counts assertions about incidence: which block reads which name, whether the matching is a
bijection, whether a block is square. No number is evaluated anywhere in it, and the module
docstring says so as a design statement rather than as a gap.

What that leaves unchecked is the pass with the most room to be wrong. Index reduction
*rewrites the equations*: `_differentiate` (`src/modelica/ir/causalize.py:431`) appends
`d/dt` of an equation, and `_reduce_index` repeats until every equation can take a top-order
variable. A differentiated constraint has the same structure as the original whatever the
arithmetic did to it, so every property in that file passes on a `d/dt` that is wrong.

The corpus does check this, end to end, on `examples/models/Pendulum.mo` and four others --
they are index-reduced and their `.toml` records numbers that a run has to reproduce. That is
five models, all of them written by somebody who was thinking about index reduction.

## Inferred

Why it was not done with the rest, inferred from writing the file rather than from an attempt
that failed: the property needs a *consistent point* to evaluate at, and a generated model
with random coefficients rarely has one.

The check is: take the flat equations before causalization, take the schedule after it,
evaluate the schedule at some state, and assert every original residual is near zero. For an
index-1 model that is a solve away. For an index-2 model the original constraint holds only
where the state is consistent -- which is the whole reason the constraint had to be
differentiated -- so the point has to be *found* first, by solving the original algebraic
equations, on a random system whose Jacobian is singular as often as not.

So this is not "the same file plus twenty lines". It needs a generator that makes index-2
models on purpose with a known consistent point, which is a second piece of work
(`solvable` builds triangles, and a triangle is index one by construction).

## Done

- A property that evaluates: for a generated model with a known consistent point, the
  schedule's answer satisfies the residuals of the **original** flat equations, not the
  reduced ones. Sampled at several points, not one.
- The generator that makes those models: a constraint over positions with an algebraic flow,
  the shape `tests/test_causalize_properties.py::test_a_constraint_between_two_states_is_reduced_not_refused`
  writes by hand three times. The point is consistent by construction, so nothing has to
  solve for one.
- A number for how far the residual is allowed to be from zero, and the reason for it. Not a
  guessed `1e-9`: index reduction on a model with a coefficient of 5 and a state of 4 has a
  scale, and the tolerance has to be stated relative to it.

## Deliberately not done

- No comparison against another tool. That is
  `reachability-and-agreement-against-a-reference-tool` and it answers a different question --
  whether this reads the same Modelica, not whether its own reduction is sound.
- Nothing about the two bugs the structural properties already found. Those are
  `causalization-never-returns-on-a-three-equation-linear-ode` and
  `a-triangular-model-is-refused-as-structurally-singular`, and both would want fixing before
  a numerical property is added on top -- a generator that hangs cannot sample anything.
