---
title: Differential debugging is the argument for two tiers, and there is no tool that does it
status: open
area: sim
owner: null
filed: 2026-08-03
kind: story
scope: M
verify: test
blocked_by:
  - ir-nodes-carry-no-source-span
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
