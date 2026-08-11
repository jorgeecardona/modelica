---
title: A Boolean set by a relation is differentiated instead of latched
status: open
area: ir
owner: null
filed: 2026-08-11
kind: bug
scope: M
verify: test
---

## Measured

```modelica
model M
  Real x(start = 1, fixed = true);
  Boolean hot;
equation
  der(x) = -1;
  hot = x > 0.5;
end M;
```

fails with `DerivativeError: a '>' expression has no derivative`.

The equation `hot = x > 0.5` reaches causalization as an ordinary equation with one unknown,
so `solve_for` differentiates the residual to find the coefficient of `hot` — and a relation
has no derivative. The failure is correct for what the code asked; the question is wrong.

This was split out of `a-when-clause-can-only-reinit-a-state-so-no-discrete-variable`
(landed 2026-08-11, `3d68521`), whose measurement table lists it as the first of three
failures a modeller meets. That entry fixed the other two — a `when` body may now assign a
discrete variable and `pre()` reads the pre-event value — and deliberately did not touch
this one, because it is a change in a different pass.

## Inferred

Inferred, not measured: a `Boolean` whose equation is a bare relation is the *same object*
this engine already has machinery for. `modelica.sim.emit._watched` collects every relation
that appears inside an equation and gives each a latch slot in `w`; `indicators` watches it
for a crossing and `transition` flips it. A named Boolean is that, with a name.

So the fix is likely to be recognising the shape before causalization asks for a derivative:
an equation `b = <relation>` where `b` is Boolean becomes a watched relation whose latch slot
*is* `b`'s slot, and `b` stops being an unknown for the matching to solve for. What makes
this a real change rather than a rename is that the variable has to leave the equation set —
matching counts equations against unknowns, so removing one of each keeps the model balanced.

Not inferred, and worth stating because it bounds the work: a Boolean assigned inside a
`when` already works, because that path goes through `_remembered` and never reaches the
differentiator. This entry is only about the *equation* form.

## Done

- `Boolean hot; hot = x > 0.5;` builds, simulates, and `hot` reports 0 or 1 in the result.
- It is latched, not recomputed: the model gains an event where the relation crosses, and
  `hot` holds its value between events like every other watched relation.
- The flat model stays balanced, and a test asserts the equation and the unknown both left.
- `a > b and c < d` — a Boolean set by a *logical combination* of relations — either works
  or is refused by name. Silence there would be the same defect one level up.

## Deliberately not done

- No `Boolean` state, no `pre()` of a relation-valued variable. A latch is repaired by the
  event iteration and is not memory; the memory case is the `when` form, which already
  works.
- Nothing about `Integer` variables set by equations. Same shape, different question, and no
  caller is asking.
