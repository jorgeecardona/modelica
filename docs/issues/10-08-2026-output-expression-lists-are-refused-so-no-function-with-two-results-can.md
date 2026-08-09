---
title: Output expression lists are refused, so no function with two results can be called
status: done
area: lang
owner: cute-replicant-6e76d
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

## What landed

`ir.CallEquation`, parsed, unparsed, flattened and expressible from both front ends.
`tests/test_output_list.py`, 11 tests. 1227 pass, docs strict clean.

Each Done bullet:

- **`(a, b) = f(x)` parses, flattens and reaches a `System`** with both targets determined.
- **A two-result function is declarable and callable**, and there is a test that the two
  results are not the same value — the failure that passes every structural check and shows
  up only in the numbers.
- **Both front ends say it.** `modelica.build.results(a, b) == call("f", x)` renders
  `(a, b) = f(x);`. `==` builds an equation everywhere else in that module and it builds one
  here too; the difference is which kind.
- **`parse(unparse(x)) == parse(x)`** for all three forms, including the skipped result.

### The design, which is the part worth reading

**Not a tuple.** There is no tuple *value* in this IR, and adding one would reach every
expression walker in the project to serve a form that only ever appears at the head of an
equation. `ir.CallEquation` is its own node with `targets` and one `call`.

**Flattening turns one of them into one ordinary `ir.Equal` per wanted output**, so
causalization, the emitter, the C backend, the folded mode and `modelica.verify` go on seeing
one equation determining one unknown, and none of them learns that a call can return a pair.
The whole feature stops at `modelica.ir.flatten`.

That works because functions here are *inlined*: `_inline` already runs the body symbolically
and reads the expression assigned to the single output. `_inline_all` is the same run read
out per output. **One run, not one per output** — two results of one call are two views of
one computation, and a body with a branch run twice could take a different arm each time and
return a pair that never existed. There is a test on the emitted text for exactly that.

## Deliberately not done, found while doing it

- **The parser rewinds, and it has to.** An equation may open with a bracket for two reasons,
  an output list or a parenthesised expression, and one token of lookahead does not separate
  them: `(a + b) = c` is a legal equation. `_output_list` reads the list and puts the cursor
  back when what follows the closing bracket is not `=`. That case has its own test, because
  it is the one that makes the rewind necessary rather than tidy.
- **`(, b) = f(x)` works** — a skipped result, which the specification allows and which costs
  one `None` in the targets tuple.
- **A subscripted target is not supported.** `(a[i], b) = f(i)` inside a `for` would need the
  loop substitution to reach into a target, and the parser reads a target as a plain
  reference, so it cannot arise. Stated at the substitution rather than left for someone to
  discover.
- **Nothing checks that a function's outputs are used in declaration order** beyond position,
  because Modelica does not name them at the call site. `(a, b) = split(x)` binds `a` to the
  first declared output whatever it is called.
- **`spatialDistribution` is still blocked.** This was its first prerequisite and it has two
  more, both named in `03-08-2026-spatialdistribution.md`: the workspace cannot hold a
  growable buffer, and the kernel has no accepted-step hook.
