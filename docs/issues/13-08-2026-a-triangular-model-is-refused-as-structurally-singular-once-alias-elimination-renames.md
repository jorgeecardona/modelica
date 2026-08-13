---
title: A triangular model is refused as structurally singular once alias elimination renames a variable into a derivative
status: open
area: ir
owner: null
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
