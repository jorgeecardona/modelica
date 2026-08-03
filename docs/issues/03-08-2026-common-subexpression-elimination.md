---
title: The IR nodes were made hashable for CSE, and no stage does CSE
status: open
area: ir
owner: null
filed: 2026-08-03
kind: story
scope: M
verify: test
---

## Measured

- `src/modelica/ir/nodes.py:18` says the nodes are frozen and slotted so that "stages can use
  them as dictionary keys for common-subexpression elimination and for caching". The property
  is real: expression nodes are hashable and structurally equal.
- No stage uses it that way. `grep -rniE "subexpression|\bcse\b" src/` finds only that
  docstring line.
- `docs/ideas/backends.md:46` lists common-subexpression elimination among the work that
  "happens once, above all of them" — the shared symbolic stage every backend inherits.
- Where it would pay is measurable today: `modelica.sim.emit` differentiates `rhs` by source
  transformation, forward mode (`src/modelica/sim/emit.py:55`). Forward-mode differentiation
  duplicates subexpressions by construction — every product rule re-emits both factors — so the
  Jacobian source is where the repetition lives.

## Inferred

The size of the win is not measured. Reading the emitted Jacobian for a corpus model would
settle it in minutes, and whoever picks this up should do that *first* and record the ratio
here, because it decides whether this is worth doing at all. Numba and a C compiler both do
their own CSE on the source they receive, so the benefit may be mostly in **compile time and
readability** rather than in run time — and readability matters more here than it usually
would, because `docs/ideas/backends.md:8` makes debuggability the whole argument for owning the
compiler.

Stating that plainly: this issue may close as "measured, not worth it in the emitted source",
and that is a legitimate outcome.

## Done

- A CSE pass over the expression IR that returns a value like every other stage: an ordered
  list of named subexpressions plus the rewritten expressions, not a mutated tree.
- `modelica.sim.emit` writes the named subexpressions as assignments before the block that uses
  them, so the generated source is smaller *and* more readable, not just smaller.
- The differential test that already exists is the correctness bar: interpreter and emitted
  kernel agree, before and after, on every corpus model.
- The ratio recorded, on at least one corpus model: source size and node count with and without
  the pass, for `rhs` and for `jacobian` separately.

## Deliberately not done

- Not a general optimiser. No constant propagation beyond what `modelica.ir.derive.simplify`
  already does, no algebraic rewriting, no reassociation — reassociation changes floating-point
  results, and this pipeline compares against reference points at 1e-10 and tighter.
