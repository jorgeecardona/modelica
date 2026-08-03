---
title: '`spatialDistribution()` — the one PDE the Association standardized, and we do not have it'
status: open
area: sim
owner: null
filed: 2026-08-03
kind: story
scope: M
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
