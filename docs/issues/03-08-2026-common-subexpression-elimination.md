---
title: The IR nodes were made hashable for CSE, and no stage does CSE
status: done
area: ir
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: bf2e996
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

## The measurement, which this issue asked for first

Taken on the **emitted Python**, not on the IR. That matters: the Jacobian is built by
forward-mode differentiation *inside* the emitter and does not exist in the IR at all, so
measuring the IR would have missed the case the Measured section points at. Trees against the
graphs they collapse to, summed over the corpus:

| | before | after | source | repetition left |
| --- | --- | --- | --- | --- |
| `rhs` | 593 nodes | 484 | 18.4% smaller | 22.3% → 4.8% |
| `jacobian` | 1608 nodes | 1155 | 28.2% smaller | 32.8% → 6.4% |

The guess in Measured was right: the Jacobian repeats itself half again as much as the
right-hand side it came from. Worst single case before the pass was `BumpStop`'s Jacobian at
47%, then `AtwoodPendulum`'s at 41%.

So it was worth doing, and the answer to the Inferred section's open question is: **the win
is in source size, not measured in run time**, and no run-time claim is made here. What is
left — 4.8% and 6.4% — is repetition the pass declines on purpose: inside conditional arms,
and where a name's first use is inside a Newton loop.

## What landed

`45a4158` the measurement and the pass, `bf2e996` the emitter. 865 tests pass.

**Two silent bugs, both found by the corpus rather than by reasoning.** They are the finding
worth carrying out of this issue, because both are properties of *where* a name may go and
neither is obvious from the pass in isolation.

1. **Hoisting to the top of a function is wrong.** What a function emits is a sequence: block
   two reads what block one assigned, and a Newton block reads an unknown that exists only
   inside its loop. A name lifted above its inputs still runs — the Python local holds
   whatever the previous step left in it — so it computes a number, and the wrong one. It
   moved `BumpStop`'s conserved energy by 2.3e-8 and broke `AtwoodPendulum`'s Jacobian
   against its difference quotient. Each name is now defined immediately before the first
   line that reads it.
2. **Naming before latching is wrong.** `_latched` matches a watched relation as a whole
   node, so naming a subexpression inside one leaves a relation that no longer matches, and
   the branch goes back to being re-decided on every evaluation — the failure `emit`'s own
   docstring describes as the tank draining to minus one machine epsilon and oscillating
   forever. `expr` latches first and names second.

Both were caught because every corpus model checks an exact Jacobian against a difference
quotient, and several check a conserved quantity. That is what those tests are for, and this
is the first change that needed them.

## Deliberately not done, found while doing it

- **No cross-block hoisting**, per bug 1. A subexpression shared between two blocks is named
  at its first use in the earlier one, which is correct but leaves the second use to a name
  that is already in hand — so nothing is lost. What *is* left on the table is a repeat whose
  first use is inside a Newton loop: it is recomputed per iteration, exactly as before.
- **No run-time measurement.** Numba and a C compiler do their own CSE on the source they
  receive, so a speed claim here would need a work-precision harness that does not exist yet
  (`work-precision-harness`). The claim made is about the source.
