---
title: `spatialDistribution()` — the one PDE the Association standardized, and we do not have it
status: blocked
area: sim
owner: null
filed: 2026-08-03
kind: story
blocked_by:
  - output-expression-lists-are-refused-so-no-function-with-two-results-can
scope: L
verify: test
---

## Measured

- `grep -rn "spatialDistribution" src/` returns nothing. The operator is not in the lexer's
  builtins, not in the expression IR, and not in `modelica.sim`.
- It is standard Modelica, since 3.2 (March 2010) — `docs/ideas/pdes.md:26` dates it, and
  §3 of the specification defines it.
- `docs/ideas/pdes.md:155` states this project's position on it in as many words: "Add
  `spatialDistribution()` as a builtin operator in the expression IR, with its
  transport-buffer semantics in `modelica.sim`. This is standard Modelica, so it belongs on the
  roadmap on its own merits. **It is a conformance item, and not a PDE feature.**"

## Inferred

Why it is worth doing while the general PDE question stays closed: it is the only distributed
construct with a known-good algorithm and clear industrial demand — pipelines, gas networks,
district heating, conveyor belts, transmission lines (`docs/ideas/pdes.md:133`). It is
implemented as a *transport buffer inside the tool*, not as a general discretization, which is
exactly why it survived when every general proposal died.

The implementation shape is inferred from the specification rather than from having built it:
the operator carries state between steps (the buffer of in-flight values), so it is neither a
pure expression nor an event. That makes it the first construct in this pipeline whose value
depends on the *history* of the run, and the interesting design question is where that buffer
lives — most likely the workspace `w`, beside the latches, since `docs/ideas/verification.md:44`
already treats `w`'s tail as the model's discrete state.

## Done

- `spatialDistribution(in0, in1, x, positiveVelocity, initialPoints, initialValues)` parses and
  reaches the IR as a node, not as a generic function call.
- Its transport-buffer semantics run in `modelica.sim`, with the buffer in the workspace so that
  the discrete half stays inspectable and `modelica.verify` can still enumerate it.
- A corpus model uses it — a transport delay along a pipe at varying velocity is the canonical
  one — with reference points computed independently in its `.toml`.
- The edge cases the specification calls out are covered by tests: velocity reversal, and a
  query point outside the initial support.

## Deliberately not done

- No `field`, no `domain`, no `pder`, and no general PDE support. `docs/ideas/pdes.md` is the
  argument and it is long: every attempt since 2000 failed, the specification ticket was closed
  as expired in March 2016, and a construct whose meaning depends on a numerical method the
  tool picks silently cannot be checked at any stage of this pipeline.

## Re-scoped 2026-08-10, from M to L, and blocked

Claimed, measured, and handed back rather than half-built. Three things stand between this
entry and its Done list, and none of them was in the estimate. Each was measured, not read.

**1. The operator cannot be written down.** Its signature is
`(out0, out1) = spatialDistribution(...)` — an output expression list, which
`src/modelica/lang/parse.py:293` refuses outright:

    ModelicaSyntaxError: output expression lists are not supported yet

Filed as
[`output-expression-lists-are-refused`](10-08-2026-output-expression-lists-are-refused-so-no-function-with-two-results-can.md),
which this now waits on. It is a language feature in its own right — no function with two
results is callable today — and it is a prerequisite rather than part of this.

**2. The workspace cannot hold the buffer.** `Code.workspace` is an `int`: `w` is a flat
float array whose length is fixed when the model is emitted. A transport buffer holds one
entry per sample still in flight, and how many that is depends on the velocity and the step
sizes of the run. The Done bullet asking for the buffer "in the workspace so that the
discrete half stays inspectable" is asking for a growable value in a fixed-length array of
doubles. Either the buffer is capped and the cap is part of the model, or it lives outside
`w` and `modelica.verify` cannot enumerate it. That is a design decision this entry made in
advance without knowing `w` was fixed.

**3. There is no accepted-step hook, and the count is the argument.** A transport buffer may
only be appended when a step is *accepted*; appending inside `rhs` records the solver's trial
evaluations, including the ones it throws away. Measured on `RCCircuit`:

    225 rhs calls for 32 accepted output samples

Seven evaluations per sample. `modelica.sim.kernel.Kernel` has nine entry points —
`parameters`, `initial`, `latch`, `rhs`, `jacobian`, `observe`, `observe_jacobian`,
`indicators`, `transition` — and not one of them means "a step was accepted". This operator
would be the first construct whose value depends on the history of the run, and the kernel
contract has no place for one.

`scipy.integrate.solve_ivp` with `dense_output=True` does report the accepted step times
after a segment, so a hook is constructible. What it does not solve is the short-delay case:
when the transport delay is shorter than the segment the solver is currently integrating,
`rhs` needs history the buffer does not have yet, and that is the classic delay-differential
problem — iterate the segment, or refuse delays under one step. Either is a decision about
the *solver*, not about the operator.

## What a next attempt should do first

Not write the operator. Decide the two questions above and record the answers:

- Does the buffer live in `w` with a cap the model declares, or outside `w` with
  `modelica.verify` told it cannot see it?
- Does the solver iterate a segment for a short delay, or does the operator refuse one?

Both are answerable in prose before any code, and both are why this is L.
