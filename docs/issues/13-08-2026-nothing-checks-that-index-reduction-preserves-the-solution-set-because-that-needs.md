---
title: Nothing checks that index reduction preserves the solution set, because that needs numbers and the generator makes structure
status: done
area: verify
owner: quiet-tars-187f2
filed: 2026-08-13
fixed_in: 2e1e0df
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

## Done 2026-08-13

`tests/test_solution_set.py`. Take the flat equations *before* causalization, take the
trajectory the causalized model produced, evaluate one against the other at twenty points of
each run. 24 tests, 50 s.

`modelica.ir.evaluate.evaluate` already existed and reads `der(x)` as a lookup of the literal
key ``der(x)``, so the whole harness is a name map: `Result` carries a row per observed
variable, and `Mode.derivative` supplies the second spelling for a derivative alias
elimination renamed.

### The corpus, measured

Every one of the 23 models, at twenty points each. Worst relative residual of any original
equation anywhere: **6.2e-15** (`AtwoodPendulum`). `DrainingTank` 3.1e-13, everything else at
or below 1e-15. `Pendulum` is index 3 and its ``x^2 + y^2 = L^2`` holds to **5.6e-17**.

`TOLERANCE = 1e-11`, which is thirty times the worst real reading and five orders below the
solvers' own ``rtol``. That gap is the argument, not slack: these residuals are not limited by
the integration tolerance, because an algebraic equation is *solved* at each point rather than
integrated. A residual anywhere near ``rtol`` means an equation stopped being solved.

### The generator this entry said was needed

Two positions moved by one shared flow and tied by a constraint --
``der(x) = c1*u + f1``, ``der(y) = c2*u + f2``, ``a*x + b*y = k`` -- with ``k`` computed from
the drawn starts so the consistent point is built in rather than solved for. This entry
anticipated that correctly and it is why the generator is 30 lines instead of a study in
singular matrices.

Two things it found that the entry did not anticipate:

- Only ``x`` may be `fixed`. A constraint between two states leaves one of them to the
  constraint, and pinning both is an over-determined initialisation rather than an index-2
  model.
- **Not both coefficients may be 1.** ``x + y = k`` is an *alias*, the shape `_pair`
  substitutes away, so it never reaches index reduction and the model comes out index 1.
  Hypothesis found that on its second draw.

``u`` is checked against its closed form ``-(a*f1 + b*f2)/(a*c1 + b*c2)`` as well as the
residual, because a reduction that satisfied the constraint by solving for the wrong variable
would keep the residual at zero and still be wrong.

### Does it have teeth? Five mutations of `d/dt`, measured

Real edits to `modelica.ir.derive._binary`, one rule at a time, not monkeypatches:

| mutation | `Constrained` | `Pendulum` | `TwoInertias` |
| --- | --- | --- | --- |
| product rule drops a term | caught | caught | caught |
| power rule forgets the exponent | passed | **0.39** | passed |
| sum rule drops the right term | passed | caught | **0.71** |
| difference becomes a sum | **1.2** | passed | **2.0** |
| unary minus loses its sign | **1.3** | passed | **1.4** |

**Five of five caught, none by every model.** A number is a residual; "caught" without one is
a model that stopped being causalizable or divided by zero, which is also an answer. No single
model is enough, which is the argument for running this over the whole corpus rather than over
the two index-reduced examples.

And the honest half: **on these five mutations the generated family catches nothing the corpus
does not.** Its value is being generated -- hundreds of coefficient combinations rather than
three fixed models -- and holding the index-2 shape on purpose rather than by accident of two
mechanical examples. That is not evidence of extra coverage and the file says so.

### Deliberately not done

- No mutation testing in CI. The table above was produced by editing `derive.py` and reverting
  it; automating that is a separate tool and a separate argument about run time.
- The tolerance is not derived from the solver's. It is deliberately far below it, for the
  reason stated above, and a model that legitimately needed a looser one would be evidence
  that something is being integrated that ought to be solved.

1472 pass before this, plus 24 here, typecheck 0.
