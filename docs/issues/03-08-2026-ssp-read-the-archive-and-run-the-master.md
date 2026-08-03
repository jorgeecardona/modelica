---
title: SSP 2.0 — the master's protocol is specified and checked, and nothing reads an .ssp
status: open
area: ssp
owner: null
filed: 2026-08-03
kind: story
scope: L
verify: test
blocked_by:
  - fmi-import
---

## Measured

- `src/modelica/ssp/__init__.py`: "Reading `.ssp` archives, resolving the connection graph and
  doing the actual stepping are not implemented yet."
- What exists is `modelica.ssp.master` — the protocol as pure transitions: what phase the
  master is in, what moves are legal, what each move does. It was written *before* the master
  so that the master would have something to be checked against
  (`docs/ideas/verification.md:207`), and `tests/test_ssp_master.py` checks it.
- That check already found something a test with cooperative FMUs never would:
  **`canGetAndSetFMUState` is optional in FMI, and a master that assumes it is broken.** Four
  moves from the start — save a, begin, step b, a discards — no move puts `b` back
  (`docs/ideas/verification.md:226`). The counterexample is in the docs with its trace.
- Stage 8 of the build order (`docs/architecture.md:266`). `docs/architecture.md:238` lists the
  four parts of SSP 2.0 that matter to the data model: FMI 3.0 components, the
  `structuralParameter` connector kind, clocked connectors, and arrays in connectors and
  connections.

## Inferred

Blocked by FMI import, and that is a real ordering rather than a preference: an `.ssd` names
components which are FMUs, so a master with nothing to step is untestable. Inferred, but only
just — one could stub the components. A stubbed co-simulation master would be checked against
the same abstraction it was derived from, which is precisely the failure mode `check_open`
exists to avoid.

The rollback finding above predicts where the implementation will hurt: the master must handle a
component that legally declines to save its state, and the checked protocol already says what it
must do about it. That is knowledge available before the first line is written.

## Done

- A `.ssp` archive (a zip) parses: `SystemStructure.ssd` for components, connectors, connections
  and nested systems; `.ssv` parameter sets; `.ssb` bindings — all into frozen dataclasses.
- The composition graph resolves against the FMUs it references, using `modelica.fmi`'s importer.
- The master steps them, and its legal moves are `modelica.ssp.master`'s — the module that was
  checked, not a second implementation of the same idea. That is the whole point of having
  written it first, and a reviewer should be able to see each move in the code.
- A system of at least two components simulates, including one that declines
  `canGetAndSetFMUState`, and the run is correct rather than merely not crashing.
- Arrays in connectors wait on the arrays issue in `lang`; say so in the code where they are
  refused.

## Deliberately not done

- No clocked connectors and no DCP in the first pass. Name them as unsupported rather than
  half-reading them, the way `modelica.lang.parse` names its refusals.
