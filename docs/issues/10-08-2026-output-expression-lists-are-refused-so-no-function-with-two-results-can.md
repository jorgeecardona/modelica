---
title: Output expression lists are refused, so no function with two results can be called
status: open
area: lang
owner: null
filed: 2026-08-10
scope: M
verify: test
kind: story
---

## Measured

`src/modelica/lang/parse.py:293` raises `output expression lists are not supported yet` for
an equation whose left side is a parenthesised list. Measured on 2026-08-10:

```
model T
  Real a; Real b; Real x(start = 0, fixed = true);
equation
  der(x) = 1;
  (a, b) = f(x, 1.0);
end T;
```

    ModelicaSyntaxError: output expression lists are not supported yet (<modelica>, line 6)

This is standard Modelica -- specification §8.3.1, the equation form `(out1, out2, ...) =
function_call`. It is the *only* way to call a function that returns more than one value, so
today a function in this toolchain can return exactly one thing.

Also refused by the same rule, and this is what found it: `spatialDistribution`, whose
signature in §3.7.4.2 is `(out0, out1) = spatialDistribution(...)`. Every built-in operator
with two results needs this before it can be written at all.

## Inferred

Inferred from reading the parser rather than from building it: the shape is bounded. The
left side becomes a list of `Ref`s, the equation becomes one node holding several targets and
one call, and flattening prefixes each target the way it already prefixes one. Causalization
then has an equation that determines *n* unknowns at once, which is the part that is not
free -- `modelica.ir.causalize` matches one equation to one unknown, and a call with two
results is one equation matched to two.

The likely answer, also inferred: split it at flattening into *n* equations, each assigning
one target from an indexed result of one shared call, so that everything downstream keeps
seeing one equation per unknown. That is what turns the feature from a causalization change
into a parser and flattening change, and it is worth checking before committing to it.

## Done

- `(a, b) = f(x)` parses, flattens and reaches a `System` with `a` and `b` each determined.
- A function that returns two values is declarable and callable, with a test that the two
  results are not silently the same value.
- The `.mo` and the Python front end both say it, since every corpus feature is written twice.
- `unparse(parse(text)) == text` holds for the new form, which is the round-trip every other
  node in `modelica.lang` is held to.

## Deliberately not done

- Nothing about `spatialDistribution` itself. That is
  `03-08-2026-spatialdistribution.md`, which waits on this and on two other things it names.
