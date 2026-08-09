---
title: IR nodes carry no source span, so no error below the parser can point at the equation
status: in-progress
area: ir
owner: cute-replicant-6e76d
filed: 2026-08-03
kind: bug
scope: M
verify: test
---

## Measured

- `grep -rn "span\|Span" src/modelica/ir/nodes.py` returns **nothing**. No IR value carries a
  position.
- The positions exist, and they stop at the parser. `src/modelica/lang/lexer.py:67` gives every
  token a `line` and a `column`, and `lexer.error` builds a syntax error that shows the offending
  line with a caret. `src/modelica/lang/lexer.py:7` states the intent: "Every token carries its
  line and column, so an error raised **three stages later** can still…" — and three stages later
  there is nothing to carry it in.
- Two documents claim otherwise:
  - `docs/architecture.md:146`: "AST nodes are frozen dataclasses that carry source spans."
    True of tokens; not true of `modelica.ir` values, which are what the parser returns.
  - `docs/ideas/backends.md:22`: the interpreter "reports an error in *Modelica* terms:
    'division by zero in equation 12 of BouncingBall: ball.v / ball.m'. It can, because it still
    holds the IR node **and the source span**."
- What the interpreter actually says, from `src/modelica/ir/evaluate.py:93`:
  `nothing in the environment answers 'x'`. No equation, no model, no line. The same at
  lines 79, 85, 102 and 115.

So the second-tier error messages, which `docs/ideas/backends.md` gives as the main reason for
keeping an interpreter at all, are not better than the compiled path's in the way that page
claims. They are better in *kind* (a named failure rather than a NaN) and not in *location*.

## Inferred

Where this hurts is inferred rather than observed, because the corpus is small and every model
fits on a screen: on a model of any size, "nothing answers 'Rw'" without a line number is a
search. That exact message appeared once already, in a real bug — a modification on an inner
component bound `R` to a name that did not exist, and code generation died on "nothing computes
'Rw'" (`CHANGELOG.md`, unreleased). It was found by reading, not by being told where.

Design inference: the span has to be optional and it has to not break structural equality, or it
will break two things this project relies on. `parse(unparse(x)) == x` compares IR trees, and
`tests/test_examples.py:239` compares flat models from two front ends — and one of those front
ends is Python, where there is no source text at all. So a span must be excluded from `__eq__`
and from the content hash that `docs/ideas/backends.md:70` makes the compiled-kernel cache key.
A `field(compare=False)` on a frozen dataclass does exactly that; whoever picks this up should
check the hash path deliberately rather than assume it.

## Done

- Expression, equation and declaration nodes carry an optional span. `modelica.lang.parse` fills
  it; `modelica.build` leaves it `None`, and that asymmetry is fine and stated.
- The span does not participate in equality or hashing, and the two equivalence tests above pass
  unchanged — which is the observation that proves it.
- `EvaluationError` and the flattener's "nothing computes X" name the model, the equation and the
  line when a span is available, and degrade to today's message when it is not.
- One test asserts the *good* message on a model with a deliberate error, so the improvement
  cannot silently rot back.

## Deliberately not done

- No span on flattened variables' *names*. A flat variable is a path, not a location, and
  several equations may produce it.
- The two documents above get corrected in the same change. Leaving a page claiming a
  capability that was just implemented differently is how the next reader gets misled.
