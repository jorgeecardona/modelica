---
title: Tearing is priced in by three documents and implemented by none
status: done
area: ir
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: cdbbdcf
kind: story
scope: L
verify: test
---

## Measured

- `grep -rniE "tearing|\btear\b" src/` returns nothing. There is no tearing pass.
- `docs/architecture.md:125` lists "make the system smaller before the solver sees it —
  tearing and BLT turn a simultaneous block of *n* equations into a much smaller torn system"
  as one of the three things this compiler can do that a precompiled library cannot.
  `docs/ideas/benchmarks.md:66` repeats the claim, and `docs/ideas/backends.md:46` lists
  tearing among the shared work that happens once above every backend.
- The corpus already has blocks that would pay: `AtwoodPendulum` produces a 4x4 implicit
  block, the largest in the corpus (recorded in `logs/2026-08-02.md`), and `Pendulum`'s mode
  dispatcher carries a 3x3 linear solve (`tests/test_jit.py:44`).

So three documents state a capability the code does not have. That is the finding.

## Inferred

The cost of the missing pass is a Newton iteration over *n* unknowns where a torn system
would iterate over one or two, with the rest recovered by forward substitution. On the
corpus's current block sizes (4x4) the absolute saving is small. It grows with the model, and
the `ScalableTestSuite` staging in `docs/ideas/benchmarks.md` is where it would show. This
is inference, not measurement — nothing here has been timed with and without tearing.

## Done

- `modelica.ir.causalize` returns, for each simultaneous block, a set of torn variables and
  the residual equations over them, and the remaining variables as an ordered assignment
  sequence.
- `modelica.sim.emit` writes the torn form: assignments, then a Newton iteration over the
  torn unknowns only.
- `AtwoodPendulum`'s 4x4 block solves with at most 2 torn unknowns, and its reference points
  in `examples/models/AtwoodPendulum.toml` hold unchanged to the tolerance already recorded
  there. Same trajectory, smaller system — that equality is the test.
- The interpreter (`modelica.ir.evaluate`) and the emitted kernel agree on the torn form, in
  the differential style `tests/test_jit.py` already uses.

## Deliberately not done

No heuristic tuning. Tearing quality is an NP-hard choice and the literature's usual answer
is a greedy heuristic; this issue asks for a *correct* torn form with any admissible choice,
not the best one. Picking a better heuristic is a separate, measurable follow-up, and it
needs the harness from the work-precision issue to be worth arguing about.

## What landed

`cdbbdcf`. 905 tests pass. Every corpus block that was simultaneous:

| model | block | iterates on | recovered |
| --- | --- | --- | --- |
| `AtwoodPendulum` | 4x4 | 2 | `der_vr`, `der_om` |
| `Pendulum` | 3x3 | 1 (`F`) | `der_vx`, `der_vy` |
| `TwoInertias` | 3x3 | 1 | 2 |

The Done bullet named `AtwoodPendulum` at "at most 2 torn unknowns"; it is exactly 2, and
its reference points hold at the tolerance recorded before tearing existed. That equality —
same trajectory, smaller system — is the whole test, and it comes free from the corpus.

**The design decision is that the assignments are folded back into the residuals.** What
`_tear` returns is an ordinary `Block` over the torn unknowns, so `emit` solves it with the
`linear` and `newton` it already had and the only new code is recovering the substituted
variables afterwards, plus their tangents, which are the chain rule an explicit assignment
already gets. Nothing downstream had to learn what tearing is.

## Deliberately not done, found while doing it

- **It removed a compiled test's coverage, and that is restored in the same commit.**
  `Pendulum` used to be the only model in `tests/test_jit.py::COVERED` emitting
  `np.linalg.solve` with a 2-D right-hand side. Tearing reduced its block to one unknown, so
  it stopped emitting the matrix path — and nothing else in that list ever did.
  `AtwoodPendulum` tears to two and still emits it, so it joined the list. Worth recording as
  a *shape*: a pass that makes generated code simpler can delete the only instance of a
  construct a test was there to compile.
- **No timing.** The issue's Inferred section says the saving is small at 4x4 and grows with
  the model, and that remains unmeasured — `work-precision-harness` is what would settle it.
  What is measured here is the size of the system, not the time it takes.
- No heuristic tuning, as planned. The greedy choice is `sorted(pending)` broken by how many
  unused equations want each unknown, which is admissible and arbitrary among ties.