---
title: `algorithm` sections are refused, and with them `while`, `break` and `return`
status: done
area: lang
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: 8fe9bbe
kind: story
scope: L
verify: test
---

## Measured

`src/modelica/lang/parse.py:85`, `_UNSUPPORTED`, refuses five keywords that are one feature:

    algorithm   "algorithm sections are not supported yet"
    while       "'while' loops are not supported yet"
    break       "'break' is not supported yet"
    return      "'return' is not supported yet"
    external    "external functions are not supported yet"

`tests/test_lang.py:306` asserts each refusal by name.

`modelica.ir` has equations and no statements: `grep -n "class .*Statement" src/modelica/ir/`
returns nothing. There is no assignment node, no `if` statement, no loop node.

## Inferred

An algorithm section is not an equation section with a different spelling. It is *causal*:
statements execute in order, a variable may be assigned more than once, and the section as a
whole behaves like a function from its inputs to its outputs. That has three consequences the
rest of the pipeline currently assumes away.

1. Matching (`modelica.ir.causalize`) matches one equation to one unknown. An algorithm section
   matches as a *group*: all of its outputs at once, against the section as one node.
2. Symbolic differentiation (`modelica.ir.derive`) walks expressions. It cannot differentiate a
   statement sequence, so a model whose state derivative comes out of an algorithm section has
   no analytic Jacobian, and the engine would have to fall back to finite differences — the
   exact thing `docs/architecture.md:120` says this compiler does not have to do.
3. `modelica.sim.emit` writes straight-line arithmetic. Statements need control flow in the
   emitted source, which is new but mechanical.

Point 2 is the reason this is not "just parsing". It is inference, not measurement: no model
here has yet needed it.

## Done

- `modelica.ir` has statement value types: assignment, `if`/`elseif`/`else`, `for`, `while`,
  `break`, `return`, and a `function` class with `input`/`output` declarations.
- `modelica.lang.parse` reads algorithm sections and functions; `modelica.lang.unparse` writes
  them back, and `parse(unparse(x)) == x` holds over a new corpus entry that uses one.
- `modelica.ir.causalize` treats a section as one node matching all of its outputs, and the
  incidence structure records that grouping rather than hiding it.
- A corpus model calls a user-written `function` in an equation, simulates, and agrees with
  its reference points.
- The five entries are gone from `_UNSUPPORTED` and from `tests/test_lang.py`.

## Deliberately not done

- `external` stays refused. An external function is a call into a shared library, which is an
  FFI question and not a language question, and it belongs with the FMI work rather than here.
  Keeping it in `_UNSUPPORTED` when the other four leave is deliberate, and this note is why.
- No attempt to differentiate through a statement sequence. A model whose derivative depends
  on an algorithm section should say so and fall back, loudly, not silently.

## What landed, and the Done bullet that was refused instead

Four commits: `dbf0599` the language, `2193a9c` the function inliner, `f5f78b2` the corpus
model, `8fe9bbe` sections in a model. 721 tests pass.

**The third Done bullet was not built, and this is the reason.** It asked `modelica.ir.causalize`
to treat a section as one node matching all of its outputs, with the grouping recorded in the
incidence structure. What was built instead is *symbolic execution at flatten time*: a section
becomes one equation per variable it assigns, and a call to a `function` becomes the expression
its body computes. Straight-line assignment is substitution; an `if` is a merge, each arm run
against its own copy and the disagreements becoming an `ir.IfExpr` over the branch conditions.

The argument for the change is this issue's own Inferred section, point 2. A section carried
down to causalization as a node is a node `modelica.ir.derive` cannot differentiate, so any
model whose state derivative passed through one would lose its analytic Jacobian and fall back
to difference quotients — which `docs/architecture.md:120` says this compiler does not do, and
which the last "deliberately not done" above forbids doing silently. Converting at flatten time
means the node never exists: causalization, the BLT sort, the differentiator and the emitter all
meet ordinary expressions, and `emit` needed no control flow at all.

The price is the subset, and it is stated where it is enforced rather than here. `for`, `while`,
`break` and `return` cannot become expressions and are refused by name, in the words the modeller
wrote. A grouped node is what would be needed to accept them, so if a model in this corpus ever
needs a loop in a section, that bullet comes back as its own issue with a real instance behind
it — which it does not have today.

One refusal was added that the issue did not anticipate: `y := y + 1`, where the same section
assigns `y`, reads a value that does not exist. A continuous variable has no previous value.
Emitted literally it is `y = y + 1`, which has no solution and which the solver would be blamed
for.

`external` stayed refused as planned, so four entries left `_UNSUPPORTED`, not five. The Done
bullet says five and the note above it says why one stays; the note is the correct half.
