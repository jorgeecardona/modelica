---
title: Causalization is tested by examples only, and its properties are stateable
status: open
area: verify
owner: null
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
