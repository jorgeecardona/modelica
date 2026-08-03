---
title: FMI 3.0 export — a System as an FMU, and the two routes disagree in the docs
status: open
area: fmi
owner: null
filed: 2026-08-03
kind: story
scope: L
verify: test
---

## Measured

- Nothing is implemented (`src/modelica/fmi/__init__.py`). Stage 7 of the build order
  (`docs/architecture.md:265`).
- The guard is already written and already checked: `Lifecycle.call` refuses an illegal call,
  which is meant to become `fmi3Error` instead of a corrupted instance
  (`docs/ideas/verification.md:186`, reader 1 of the mode table).
- **Two documents describe different deliverables**, and this is the finding worth recording:
  - `docs/architecture.md:236`: "a `System` plus a generated `modelDescription.xml`, with the
    FMU's C entry points backed by the Python evaluator".
  - `docs/ideas/backends.md:42`: C is "one translation unit — FMU export, where a shared
    library is the deliverable".
- `docs/ideas/verification.md:178` records a constraint that binds either route: Event Mode
  calls `fmi3UpdateDiscreteStates` repeatedly until the discrete states settle, so **the FMI
  API itself mandates the event iteration** already implemented in `modelica.sim.solve`. The
  exported FMU must expose that loop, not hide it.

## Inferred

The two routes are not equally good and the difference should be decided in the open:

- **Python-backed.** Fast to build, and the FMU only loads where CPython and this package are
  installed. That is not an FMU in the sense anybody else means it, and shipping one would
  invite a bug report from a tool that cannot load it.
- **C translation unit.** Self-contained, loadable by Dymola, OpenModelica and FMPy, and needs
  the C emitter first.

Inference: the second is the only one that satisfies the promise on the README's front page
("speaks the two exchange standards"), and the first is at best a development shortcut. If it
is built, it should be labelled as such in the docs, and this disagreement in the two pages
should be resolved rather than left for a reader to find.

## Done

- `modelica.fmi` writes an FMU from a `System`: a valid `modelDescription.xml` (variables,
  causality, variability, model structure, unit definitions) plus a binary, zipped to the FMI
  3.0 layout.
- The runtime guard is `modelica.fmi.modes.Lifecycle`, not a second copy of the rules.
- Event Mode iterates `fmi3UpdateDiscreteStates` to a fixed point, and that is the same
  iteration `modelica.sim.solve` runs — one implementation, reached from two sides.
- Round trip: export a corpus model, import it with this package's own importer **and** with
  an independent one (FMPy), simulate both, and match the model's own reference points in its
  `.toml` to the tolerance already recorded there. Two importers, because agreeing with
  ourselves proves nothing.
- The docs disagreement above is resolved in whichever page turns out to be wrong.

## Deliberately not done

- Not blocked on the C emitter in the register, deliberately: a Python-backed prototype is a
  legitimate way to learn the XML and the layout. But it does not close this issue, and the
  FMPy leg of the test is what enforces that.
