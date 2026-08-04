---
title: Arrays and `for` equations are refused, and the distributed-model workaround needs them
status: done
area: lang
owner: quiet-tars-187f2
filed: 2026-08-03
fixed_in: 92e77e4
kind: story
scope: L
verify: test
---

## Measured

- `src/modelica/lang/parse.py:85`, `_UNSUPPORTED["for"]`: "'for' loops and array equations are
  not supported yet". `_refuse_unsupported` raises on the keyword at five call sites, so the
  refusal is by name and not a misparse.
- `tests/test_lang.py:306` pins that refusal. Deleting the feature quietly is therefore
  impossible; deleting the *refusal* is part of closing this.
- No declaration in `modelica.ir` carries a dimension. `FlatVariable` is scalar, and
  `modelica.ir.flatten` expands components but has no array expansion.
- Every model in `examples/models/` is scalar. Nothing in the corpus exercises a subscript.

## Inferred

This is the largest single conformance gap, and it is load-bearing for more than conformance.
`docs/ideas/pdes.md` settles the project's position on distributed models: option 1, the
preferred one, is "a method of lines written by hand, with arrays and `for` equations — it is
plain Modelica, it works today, and it is what industry actually does". That path is
unreachable in this toolchain until arrays exist. The same is true of the
`ScalableTestSuite` staging in `docs/ideas/benchmarks.md`, whose models take a size parameter
and are written with array equations.

The inference is about *importance*, not about difficulty: nothing has been prototyped, so the
scope of L is a judgement from the number of stages that must change (lexer subscripts, AST
dimensions, flattening expansion, the incidence graph, the emitter's indexing), not a
measurement.

## Done

- `Real T[n];` and `for i in 2:n-1 loop ... end for` parse into IR values that carry their
  dimensions and their iteration range.
- Flattening expands them into `n` scalar variables and `n-2` scalar equations, so everything
  below `modelica.ir.flatten` is unchanged and still sees scalars. This is the cheap design
  and it should be tried first; it is also what makes the size parameter a *structural*
  parameter rather than a runtime one, which is a limit worth stating out loud when it lands.
- A new corpus model — a 1D heat bar under the method of lines, `n` in the tens — parses,
  flattens, simulates, and agrees with an independently integrated reference in its `.toml`.
- The `for` entry is gone from `_UNSUPPORTED` and its case is gone from `tests/test_lang.py`.

## Prior art, added 2026-08-03

[pymoca](https://github.com/pymoca/pymoca) — a Modelica compiler in Python, BSD licensed, used
by Deltares under RTC-Tools — implements all of this, and its test corpus names the cases:
`Array3D`, `ArrayExpand`, `ArrayExpressions`, `MatrixExpressions`, `ForLoop`, `NestedForLoop`,
`DelayForLoop`, `IndexedConnect`, `ArrayConnectParam`. It is an ANTLR front end with a
separate `tree` module that flattens, so the flattening reads apart from the grammar.

Two of those are worth studying before designing this. `ArrayConnectParam` and
`IndexedConnect` are connections *into* an array of components, which is where scalarising at
flatten time either works or stops working, and it is not in the sketch above.

This is also the construct that most limits what models this toolchain can be given: a
worked, permissively-licensed implementation makes it cheaper than the `L` suggests. See
`docs/ideas/landscape.md`.

## Deliberately not done

- No `field`, `domain` or `pder`. `docs/ideas/pdes.md` records why: every attempt at native
  PDE support since 2000 failed, and the position of this project is that a construct whose
  meaning depends on a numerical method the tool picks silently does not belong in the core.
- No array-valued *expressions* (matrix multiplication, `sum`, slicing as a value). Scalarising
  at flatten time is a smaller and separately testable claim. Array expressions can follow.
