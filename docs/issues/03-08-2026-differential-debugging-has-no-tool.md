---
title: Differential debugging is the argument for two tiers, and there is no tool that does it
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
blocked_by:
  - ir-nodes-carry-no-source-span
kind: story
scope: M
verify: test
---

## Measured

- `docs/ideas/backends.md:27` sells it as a capability: "It gives **differential debugging**.
  When a compiled run disagrees with an interpreted run, step both and find the first equation
  where they diverge. You cannot even ask that question of a generated-C pipeline, because there
  is nothing to compare against."
- `grep -rniE "diverge|differential debug|step both" src/` finds one hit, and it is a comment:
  `src/modelica/sim/kernel.py:162`, arguing that `error_model="numpy"` must be set or
  "differential debugging is worth nothing if the fast path can answer a question the slow path
  answers differently". So the *precondition* has been maintained carefully and the tool it
  exists for was never written.
- What exists instead is a bulk comparison: `tests/test_jit.py` runs compiled against
  interpreted on four models and asserts they agree. That answers "do they agree" and says
  nothing about **where** they stop agreeing, which is the whole feature.

## Inferred

The gap is inferred to be small in code and large in value. The pieces are all present: the
interpreter (`modelica.ir.evaluate`), the emitted kernel, the block structure in
`System.blocks`, and a workspace layout shared by both tiers
(`src/modelica/sim/emit.py:48`). What is missing is a driver that runs both over the same
inputs *block by block* and stops at the first mismatch, plus a way to name what it stopped at
— which is why this waits on IR nodes carrying a span. Without one, the answer is "block 7",
and the reader still has to find block 7 in the model they wrote.

Also inferred, and it is the reason this is worth more than a debugging convenience: this is the
tool that would settle a disagreement between the fused-step JIT and the current path, between
the C emitter and the interpreter, and between an exported FMU and the model it came from. Every
backend issue in this register produces a second implementation of the same arithmetic, and each
one needs this to be diagnosable when it disagrees.

## Done

- `modelica.sim` gains a comparison driver: given a model and a starting state, evaluate every
  block in both tiers, and return the first block whose values differ by more than a stated
  tolerance — with the equation, the variables it computes, and both tiers' numbers.
- It works for any pair of tiers, not only interpreter-versus-Numba, because the C emitter and
  the FMU exporter will both need it.
- A test that plants a deliberate divergence (an emitted function patched to return a wrong
  value for one block) and asserts the driver names that block and no other. A debugging tool
  whose failure mode is untested is a debugging tool that will be distrusted at the exact moment
  it matters.
- The notebook or docs page that shows the workflow, since the capability is currently promised
  in `docs/ideas/backends.md` and demonstrated nowhere.

## Deliberately not done

No stepping through *time*. This compares one evaluation at one state, which is where a
divergence starts. Following a trajectory until the two runs separate is a different and larger
tool, and it should not be attempted before the single-instant one is trusted.

## What landed

`modelica.sim.compare`, exported as `modelica.sim.diverges`. `tests/test_compare.py`, nine
tests. `docs/diverging.md`, in the nav, and `docs/ideas/backends.md` now points at it instead
of only promising it. 1054 tests pass; `mkdocs build --strict` is clean.

Each Done bullet:

- **The driver.** `diverges(system, left, right, *, t, y, p, w, absolute, relative)` returns
  a `Divergence` or `None`. It carries the block index, the variables that differ, both
  tiers' numbers, the block's equations unparsed back to Modelica, and the span.
- **Any pair of tiers.** The two arguments are *observers* — anything with the emitted
  `observe` signature — not two kernels. A `Kernel` passes itself; a C unit passes a ctypes
  wrapper; an FMU will pass whatever wraps an FMU. There is a test that passes a plain
  closure, because the interpreter is not privileged and the pairs that will matter (C
  against interpreter, fused-step against per-block, FMU against its model) do not all
  contain it.
- **The planted-divergence test.** Three models, planted one block at a time, asserting the
  driver names *that* block and no other. The second half is the hard one and is not taken on
  trust: `_spread` counts how many blocks the wrong value actually reached, and the
  propagation test fails if the answer is one — otherwise it would be testing an easy case
  dressed as a hard one.
- **The page.** `docs/diverging.md`, with the workflow, a real report, and the two "why"
  sections: why the first block and not all of them, and why one state and not a run.

### How it reads the two tiers, which was the design question

Neither tier has an instrumentation hook, and one that grew one would no longer be the tier
that ships. So nothing is instrumented. Both are asked for `observe`, which writes **every**
variable the model has, and the block schedule attributes each variable to the block that
computes it.

That works because of a fact worth measuring rather than assuming: over the corpus, every
name in every `Block.solved`, in every mode, is in `Code.observed` — 0 missing. The
block-level picture is already inside a value both tiers agree to produce.

### The blocker was closed and the span still did not arrive

This waited on `ir-nodes-carry-no-source-span`, which landed. Measured at the start of this
work: **0 of 86** block equations in the corpus carried a span. `ir.Equal` has the field and
the parser fills it; every rewrite between the parser and the block dropped it.

Five one-line changes, each of them a place where an equation is rewritten but is still the
same equation to a reader:

| where | what it does |
| --- | --- |
| `flatten.py:511` | the `for`-equation expansion |
| `flatten.py:1339` | the prefixing walk that flattens a component's equations |
| `causalize.py:_named` | naming derivatives |
| `causalize.py:_apply_to` | alias elimination |
| `causalize.py` torn residuals | rewriting a torn block as a residual |

After them: **77 of 86**. The nine without one are equations nobody wrote — connection
equations and equations index reduction differentiated — and they should not have a line.

## Deliberately not done, found while doing it

- **No stepping through time**, as the entry planned. `diverges` takes the state; finding the
  instant is the caller's.
- **A connection equation could carry the span of its `connect(...)`** and does not. That is
  most of the remaining nine, and it is a change in `flatten`'s connection expansion rather
  than a one-liner, because one `connect` becomes several equations.
- **An alias divergence has no block**, and is reported with `position` of `None` rather than
  passed over. Aliases are recovered after the schedule runs, so two tiers can agree on every
  block and still disagree on a variable a user reads. There is a test.
- **The comparison is over `observe`, so it is as fine-grained as the block schedule and no
  finer.** A block that solves five variables simultaneously is named as a block; which of
  the five equations inside it is wrong is not answered. For a 1x1 block, which is most of
  them, the two are the same thing.
- **Scaling a value plants nothing where the value is zero**, which at a start point is most
  of them. `DCMotor` starts at rest, so the first version of the test multiplied `0.0` by
  `1.001` and passed by finding no divergence at all. The planting is additive now, and the
  reason is in the helper's docstring.
