---
title: FMI 3.0 import — the mode table exists, the FMU reader does not
status: open
area: fmi
owner: null
filed: 2026-08-03
kind: story
scope: L
verify: test
---

## Measured

- `src/modelica/fmi/__init__.py` says it plainly: "Nothing is implemented here yet except the
  part that has to come first: the **calling sequence**, in `modelica.fmi.modes`."
- What does exist is not small. `modelica.fmi.modes` holds the FMI 3.0 state machine — every
  mode, and which `fmi3*` call is legal in each — verified against the specification sources
  and checked by `modelica.verify.check_modes` (`tests/test_fmi_modes.py`). Two real
  corrections came out of checking it, recorded in `docs/ideas/verification.md:195`: Event Mode
  was wrongly placed in the Scheduled Execution table, and "an instance can always be freed"
  is a reachability property rather than a liveness one.
- `docs/architecture.md:229` describes the intended reader: `modelDescription.xml` into
  dataclasses (variables, causality, variability, model structure, unit definitions), and the
  shared library through `ctypes`.
- Stage 6 of the build order (`docs/architecture.md:263`): "read and simulate an existing FMU.
  This checks the variable and model structure data model against real files."

## Inferred

The value of doing import before export is inferred, and the reason is the one the build order
gives: an importer is checked by *other people's files*. An exporter checked only by our own
importer proves that two of our own beliefs agree with each other. Every real FMU in the wild
is a test case this repository did not write.

`Lifecycle.call` already refuses an illegal call (`docs/ideas/verification.md:186`), so the
driver written for import gets its legality guard for free and cannot emit a sequence the table
forbids. That is the payoff of having written the table first, and it should be used rather
than reimplemented.

## Done

- `modelDescription.xml` parses into `modelica.fmi` dataclasses, frozen like everything else, so
  an FMU's interface is a value you can print and compare.
- The shared library loads through `ctypes` and the driver's call sequence goes through
  `Lifecycle`, so an illegal order is refused here rather than inside somebody's C.
- Both interface types run: Model Exchange (the FMU gives derivatives, `modelica.sim` integrates
  through the same `Problem` and `Solver`) and Co-Simulation (the FMU integrates itself between
  communication points).
- At least one third-party FMU from the FMI cross-check repository simulates end to end and
  matches its published reference trajectory. That FMU, or a script that fetches it, is in the
  repository, because a test that depends on a file nobody can find is not a test.

## Prior art, added 2026-08-03

Two things exist that this should use rather than rebuild.

- **[fmi-cross-check](https://github.com/modelica/fmi-cross-check)** is the corpus: FMUs from
  many vendors, organised by FMI version, interface type and platform, each with reference
  results. It is the acceptance test named above, and the files are somebody else's, which is
  the point.
- **[FMPy](https://github.com/CATIA-Systems/FMPy)** does this job well and is BSD licensed:
  pure Python and `ctypes`, FMI 1.0, 2.0 and 3.0, both interface types, and it validates
  `modelDescription.xml` against the XML schema rather than trusting it. It also ships the
  `cross_check` module that drives the repository above. Schema validation before parsing is
  the cheap habit to copy — an FMU that misdescribes its own interface should be refused by
  name, the way `modelica.lang.parse` refuses a construct.

Worth weighing against `fmi-export` when picking one up: reading FMUs is well served in Python
already, and writing them is the half that needs a compiler. See `docs/ideas/landscape.md`.

## Deliberately not done

- No export. It is stage 7 and its own entry.
- No source-code FMUs, no `fmi3` binary variants beyond the platform CI runs on. Say which
  platform in the test, and skip elsewhere.
