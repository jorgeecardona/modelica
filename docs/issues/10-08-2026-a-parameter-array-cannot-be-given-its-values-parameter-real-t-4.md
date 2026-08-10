---
title: "A parameter array cannot be given its values: `parameter Real t[4] = {0, 1, 2, 3}` is refused"
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-10
scope: M
verify: test
kind: story
---

## Measured

`examples/models/HeatBar.mo` declares `Real T[n]` and it works, so arrays exist. What does not
work is giving an array its *values*. Measured 2026-08-10:

```
model T
  parameter Real ts[4] = {0.0, 1.0, 2.0, 3.0};
  ...
```

    EmitError: array equations are not supported yet

The binding on an array-valued component becomes an array equation -- `ts = {0.0, 1.0, ...}` --
and `modelica.sim.emit` refuses those. `HeatBar` gets away with it because every one of its
arrays is given a *scalar* start ``Real T[n]`` with a scalar ``start``, which broadcasts;
the moment a component needs different values per element there is no way to say so.

A second wall stands next to it, and a table lookup meets both. A function body cannot loop:

```
    for i in 1:3 loop
      if x < ts[i + 1] then ...
```

    FlattenError: function 'lookup' uses a 'for' loop, which cannot be turned into an
    expression; statements here may assign and branch, and nothing else

Functions are inlined into an expression here (`modelica.ir.flatten._inline`), and a loop with
a data-dependent exit is not one.

## Inferred

Inferred from where these two bite rather than from a survey: between them they rule out every
component whose behaviour is *tabulated*. `no-component-reads-a-measured-signal` asks for a
`TimeTable` with "a parameter matrix of `[t, y]` rows", and neither the matrix, nor a pair of
vectors, nor a search over either can be written today. `RoastingMachine`'s burner schedule is
the shape a modeller is forced into instead: nine literal thresholds typed out by hand, in the
model, for one machine.

Also inferred: the two are separable and the first is much the smaller. An array binding is
`n` scalar equations after flattening, which is exactly what `for` equations already produce;
the loop-in-a-function is a question about what inlining means and may be answered by not
inlining.

## Done

- `parameter Real t[4] = {0.0, 1.0, 2.0, 3.0}` flattens to four scalar parameters with those
  four values, and the same for a non-parameter array.
- A binding whose length does not match the declaration is refused and says both numbers.
- Written twice, as every corpus feature is, and the Python front end can say it.
- A test that the values arrive *in order*: an array binding that reversed itself would pass
  every structural check and every length check.

## Deliberately not done

- No matrices. `[a, b; c, d]` is refused by the parser with its own message
  (`src/modelica/lang/parse.py`), and a table of `[t, y]` rows is expressible as two vectors.
- Nothing about loops in function bodies. That is the other wall and it is its own entry.

## What landed

`modelica.ir.flatten._element_of`, and `modelica.build.array`.
`tests/test_array_binding.py`, 6 tests. 1311 pass.

    parameter Real ts[4] = {0.0, 1.0, 2.5, 3.0};

flattens to ``ts[1] = 0.0``, ``ts[2] = 1.0``, ``ts[3] = 2.5``, ``ts[4] = 3.0`` -- four
parameters with four values, in the order they were written.

The change is where the array component is already expanded. `_array` makes one scalar
component per index and every one of them was carrying the *whole* literal as its binding,
which is what became "array equations are not supported yet". Each now takes its own element.
Nested literals index successively, so a two-dimensional declaration works the same way.

- **Order is the load-bearing test.** A binding that reversed itself, or gave every element
  the first value, has the same number of parameters, the same shapes and the same equations;
  only the numbers say otherwise.
- **A short binding is refused and says both numbers**, because silently short reads as a zero.
- **A scalar binding still broadcasts.** `Real T[n](start = T_0)` gives every cell the same
  start and every array in the corpus relies on it -- indexing into a scalar is the obvious way
  to make this feature work and `HeatBar` stop. It has its own test, holding `HeatBar` to the
  same 54.21324087 it has always given.
- **Both front ends say it.** `modelica.build.array(0.0, 1.0, 2.5)` renders `{0.0, 1.0, 2.5}`.
  A Python list cannot stand in: `_lift` turns a value into an expression and a list is not
  one, so the literal needed a name.

## Deliberately not done, found while doing it

- **This does not unblock `no-component-reads-a-measured-signal` on its own.** That entry has
  two walls and this was the smaller. The other stands: a function body cannot loop, so a
  table *lookup* still cannot be searched for, and an interpolation would have to be an
  unrolled chain for one fixed table size. Its `blocked_by` needs re-reading rather than
  clearing.
- **No matrices.** `[a, b; c, d]` is still refused by the parser with its own message. A
  nested literal `{{1, 2}, {3, 4}}` against a two-dimensional declaration does work, which is
  the same information said the way this IR already holds it.
- **An array binding whose value is an expression per slot works** -- `{k, 2 * k, k + 1}` --
  because each element is bound to whatever was written there and evaluated in dependency
  order like any other parameter. There is a test.
