---
title: A triangular model is refused as structurally singular once alias elimination renames a variable into a derivative
status: in-progress
area: ir
owner: quiet-tars-187f2
filed: 2026-08-13
kind: bug
scope: M
verify: test
---

## Measured

This model is refused:

```modelica
model M
  Real x0(start = 1, fixed = true);
  Real a0;
  Real a1;
equation
  der(x0) = a0;
  a0 + (x0 + 1) + x0 = a1;
  a0 = x0 + x0;
end M;
```

    M: nothing determines a1 -- the model is structurally singular

It is not singular. Read the equations in the other order and it is a triangle:
`a0 = 2*x0`, then `a1 = a0 + 2*x0 + 1`, then `der(x0) = a0`. Three equations, three unknowns,
one obvious schedule.

Found by `tests/test_causalize_properties.py`'s `solvable` generator, which builds models
that are triangular by construction and then shuffles them. **9 of 120 draws were refused**
when it first ran on 2026-08-13; the three lines above are the shrink.

The trigger is narrow, measured by varying one thing at a time:

| model | |
| --- | --- |
| the one above | refused |
| `a0 + 1 = a1` (the shift is a constant, not `2*x0`) | fine |
| `a0 = x0` (the alias is unit, not scaled) | fine |
| `der(x0) = a1` instead of `der(x0) = a0` | fine |

## Inferred

The mechanism, read off `_eliminate`, `_priority` and `_Matching` and confirmed as far as the
first elimination by tracing `_eliminate` -- the elimination below is measured, the matching
step after it is inferred:

- `der(x0) = a0` is an affine relation between two unknowns, so `_pair` takes it, and
  `_rank` prefers to lose `der_x0` (`_keep` subtracts one for a name in `work.integral`). So
  `der_x0 := a0` and `derivative[x0] = a0`. **`a0` is now the derivative of `x0`** -- the
  documented move, the one that keeps `mass.v` rather than inventing `der_mass_s`.
- Traced: that is the *only* elimination. The remaining equations are
  `a0 + (x0 + 1) + x0 = a1` and `a0 = x0 + x0`, with unknowns `a0`, `a1` and `x0`.
- `_priority` (`src/modelica/ir/causalize.py:508`) sorts by differentiation level, higher
  first -- "higher derivatives go before lower ones, which is what keeps states at the bottom
  of each differentiation chain". `a0` is now level 1, `a1` is level 0, so the first equation
  reaches for `a0` and takes it.
- The second equation has only `a0` and `x0`, `a0` is taken, and `x0` is *free*, so `augment`
  takes it without displacing anything -- `free` is tried before `rest`
  (`src/modelica/ir/causalize.py:578`). Both equations are matched. `a1` is the leftover.
- `_states` (`src/modelica/ir/causalize.py:786`) then refuses, correctly: `a1` is unmatched
  and has no derivative, so nothing would ever give it a value.

Both matchings are **maximum**. Only one of them leaves a state as the leftover, and
`_Matching`'s own docstring says that is the whole game: "Any maximum matching is structurally
valid, but they are not equally good: the unknowns left over become the states, so the choice
among them *is* state selection."

So this is not a matching that failed. It is a matching that succeeded at the wrong thing,
and the augmenting search stopped as soon as every equation had *something* rather than
continuing until the leftovers were state-shaped.

## Done

- The model above causalizes, with `x0` as the state, and a test holds it.
  `tests/test_causalize_properties.py::test_a_model_with_a_causal_order_is_never_refused`
  carries it as `xfail(strict=True)` today; closing this means deleting the marker.
- The `solvable` property run stops needing its `assume` past `CausalizationError`, or the
  remaining refusals are explained rather than skipped.
- The rule that decides it is written down where `_priority` and `_Matching.augment` are, not
  in a commit message: preferring a *free* unknown over displacing is a heuristic, and this is
  the case where it costs the answer.

## Deliberately not done

- No change to `_keep` or to which variable alias elimination loses. Renaming `a0` into
  `der(x0)` is right and is what makes results readable; the failure is downstream of it.
- Nothing about the hang the same generator found, which is a different defect in the same
  file: `causalization-never-returns-on-a-three-equation-linear-ode`.

## Fixed 2026-08-13

The diagnosis above was right about the mechanism and wrong about where to fix it.

The first attempt did what this entry's Done list suggested -- made `augment` prefer a free
unknown that nothing integrates, so the equation for `a0 = 2*x0` would displace rather than
consume `x0`. That fixed the model above and 60 of 60 refusals in one sample. Then the
property run found another:

```modelica
  a0 = 2 * x0;
  a2 = x0 + a1;
  a1 = 2 * a0;
  der(x0) = x0;
  der(x1) = x0;
```

still refused, because the retreat happens **one level down**: `a2`'s equation displaces
`a0`'s, and *that* one takes `x0` on its way out. A local preference cannot see it. Measured
over 4,000 generated triangular models, the local fix took the refusal rate from 10.0 % to
0.82 % -- twelve times better and still wrong.

### What it is instead

Two passes over the same augmenting search, in `_Matching.run`. A variable that has a
derivative in the system can be integrated, so leaving it free costs nothing -- it becomes a
state. A variable with no derivative must be computed or it is determined by nothing. So the
first pass hides **every** state candidate, and only the equations that failed get a second
pass with them allowed.

`_Matching` already had `excluded` for exactly this shape ("unknowns no equation may claim --
the states, when a selection is being *tested*"), so the change adds to it and puts it back.

### Measured

| | refused, 4,000 generated triangular models |
| --- | --- |
| before | 400 (10.0 %) |
| the local fix | 33 (0.82 %) |
| two passes | **0** |

The same 4,000 draws, derandomized over eight seeds, so the three rows are comparable.

**The corpus is byte-identical**: every one of the 23 models causalizes to the same index, the
same modes, the same state selection per mode, the same block shapes and torn counts, and the
same alias set. Dumped as JSON before and after and diffed.

The second pass is load-bearing rather than a formality, and the corpus says exactly where:
**2 of 23 models need it -- `Pendulum` and `TwoInertias`, six equations each** -- and those
are the two that are index-reduced. That is the shape to expect, since a dummy derivative is
precisely an equation that has to solve for something integrable.

`tests/test_causalize_properties.py` stops assuming past refusals: a `solvable` draw that is
refused is now a test failure, which is what it should always have been.

1465 pass, typecheck 0.
