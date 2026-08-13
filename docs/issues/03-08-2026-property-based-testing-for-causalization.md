---
title: Causalization is tested by examples only, and its properties are stateable
status: in-progress
area: verify
owner: quiet-tars-187f2
filed: 2026-08-03
kind: story
scope: M
verify: test
---

## Measured

- `modelica.verify` model-checks the discrete half exhaustively — event iteration, dynamic state
  selection, the FMI mode machine, the SSP master (`docs/ideas/verification.md`). None of it
  touches causalization.
- `docs/ideas/verification.md:255` says why, and says it as a limit rather than an omission:
  "It is the wrong tool for causalization. Matching, BLT sorting and Pantelides are graph
  algorithms over unbounded inputs, and not finite automata. Their properties — the matching is
  perfect, index reduction terminates, differentiating preserves the solution set — **want
  property-based testing over generated models**."
- What holds `modelica.ir.causalize` today is `tests/test_causalize.py` plus the corpus: fifteen
  models, each with hand-written expectations in its `.toml`. Every one of them was written by a
  person who was thinking about the same cases the implementation was thinking about.
- No model generator exists: `grep -rn "hypothesis" tests/ pyproject.toml` returns nothing.

## Inferred

Where the bugs are, inferred from the shape of the algorithms rather than from a failure: the
corpus is fifteen models that a person chose, so it is dense where intuition is dense and empty
where it is not. Matching and Pantelides are exactly the kind of code whose failures live in
structures nobody would think to draw — a model that is structurally singular in two
overlapping subsets, an equation that mentions a derivative and its own state, a block that
becomes trivial after index reduction.

This project has already been bitten once by the corresponding gap on the discrete side:
`docs/ideas/verification.md:113` records that "a run visits the configurations its own
trajectory reaches, and the bug is in the other one". The same sentence holds with "a corpus
covers the structures its author thought of".

## Done

- A generator for structurally valid flat models: random variables, random equations over them,
  a controlled proportion of derivatives, and a size parameter. It generates the *structure*;
  the numbers can be arbitrary because every property below is structural.
- Three properties, run over generated models:
  1. **The matching is perfect** where the model is balanced, and where it is not, the
     unmatched variables and equations are reported rather than a wrong answer being returned.
  2. **Index reduction terminates**, and the index it reports does not decrease when an
     equation is differentiated.
  3. **Differentiating preserves the solution set**: a trajectory admissible before reduction
     is admissible after it, checked by residual evaluation at sampled points.
- A generated counterexample, when one is found, is shrunk to a minimal model and added to the
  corpus as a `.mo`/`.py`/`.toml` triple — so the register of what broke stays in the same place
  as everything else.

## Deliberately not done

- No model checking here. That would repeat the mistake `docs/ideas/verification.md:255` warns
  against: these are graph algorithms over unbounded inputs, and enumeration is the wrong
  instrument for them.
- No claim about *numerical* correctness of the selected states. Whether a chosen dummy
  derivative stays regular is a statement about numbers, and it is already checked from the
  other side by `check_selection`.

## Done 2026-08-13

`tests/test_causalize_properties.py`, `hypothesis` in the dev group. Two generators and five
properties, 8 passing and 2 expected failures, 29 s.

**It found three things on the day it was written, and two of them are bugs.**

### 1. Causalization never returns on a three-equation linear ODE

    der(x0) = x2;
    der(x1) = 2 * x0;
    der(x2) = x0;

Over 90 s and given up on. `der(x0) = x2` makes `x2` the derivative of `x0`, `der(x2) = x0`
makes `x0` the derivative of `x2`, and `_Work.level` walks that map with a `while`. Filed as
[`causalization-never-returns-on-a-three-equation-linear-ode`](13-08-2026-causalization-never-returns-on-a-three-equation-linear-ode-because-alias-elimination.md).
Neighbours: drop the scale on the middle line, 2.0 ms; break the cycle, 1.6 ms.

### 2. A triangular model refused as structurally singular

    der(x0) = a0;
    a0 + (x0 + 1) + x0 = a1;
    a0 = x0 + x0;

`M: nothing determines a1 -- the model is structurally singular`, on a model that reads
`a0 = 2*x0`, `a1 = a0 + 2*x0 + 1`, `der(x0) = a0`. **9 of 120 `solvable` draws refused.**
Filed as [`a-triangular-model-is-refused-as-structurally-singular`](13-08-2026-a-triangular-model-is-refused-as-structurally-singular-once-alias-elimination-renames.md).

### 3. A property that was wrong, which is also what this is for

`test_every_state_has_a_derivative_something_computes` first asserted that a state's
derivative is produced by a *block*, and failed on `der(x0) = x0; der(x1) = x0` -- where
`derivative[x1] = x0` is correct and `x0` is a state, so it is known rather than computed. The
engine was right and the property was wrong. Stating it correctly is the value: a rule nobody
had written down now is, in the file that enforces it.

### What the properties are

| | |
| --- | --- |
| a schedule only reads what it has already computed | the BLT property, per block, in order |
| the matching is perfect and no variable is solved twice | square blocks, a bijection, nothing stranded |
| every state has a derivative something computes | or that is already known |
| a model with a causal order needs no index reduction | `index == 1` on a triangle |
| causalization answers or refuses | `System` or `CausalizationError`, on unstructured input |

`solvable` builds a triangle and then shuffles the equations and flips their sides, so the
order has to be rediscovered rather than read off. `arbitrary` promises nothing -- 102 of 120
draws are refused -- and carries only the last property.

### Deliberately not done

- **No residual property.** "Differentiating preserves the solution set" is in this entry's
  Done list and is not here. It needs a numerical harness (build a kernel, evaluate the
  *original* flat equations at the schedule's answer) and random coefficients make that a
  study in singular matrices and NaNs rather than in index reduction. Filed separately.
- **No generator for index-2 models.** `solvable` is triangular by construction, so
  Pantelides is exercised by three hand-written models at the end of the file and not by the
  generator. A generator that makes structurally singular models *on purpose* is its own
  piece of work.
- **No shrunk counterexample added to `examples/models`.** This entry asked for that; the two
  found are three lines each, they live in the test file beside the property that found them,
  and neither is a model worth simulating.

The budget is worth knowing about: a generated draw gets 5 s of wall clock through
`signal.setitimer`, because finding #1 means a draw can hang the suite and hypothesis's
`deadline` measures a call that returned.

1462 pass, 2 xfail, typecheck 0.
