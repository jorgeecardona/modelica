---
title: "A parameter array cannot be given its values: `parameter Real t[4] = {0, 1, 2, 3}` is refused"
status: open
area: sim
owner: null
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
