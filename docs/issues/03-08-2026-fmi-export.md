---
title: FMI 3.0 export — a System as an FMU, and the two routes disagree in the docs
status: in-progress
area: fmi
owner: quiet-tars-187f2
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

## `modelDescription.xml` landed 2026-08-13 — the half that needs no compiler

`modelica.fmi.describe(system)` writes the description an importer reads *before* it runs a
step: the variable table, causality and variability, units, start values, and the model
structure. No zip, no binary, no `fmi3` entry points, and this settles nothing about the two
routes -- the entry's argument that a Python-backed FMU "is not an FMU in the sense anybody
else means it" is untouched by writing XML.

`tests/test_fmi_description.py`, 52 tests, corpus-wide rather than against a fixture.

### The three mappings, and why each is the way it is

**Value references are `Code.observed` order.** A variable is a string here and a 32-bit
integer across the boundary, and every later call refers to it by that number. Taking the
order the emitted `observe` already fills means a caller holding one array and one description
needs no second mapping, and a test asserts the two agree for all 25 models.

**`parameter` becomes `fixed`, not `tunable`.** A tunable parameter may be changed in
Initialization Mode *and* between communication points; Modelica's `parameter` is constant
through a run by definition. Calling it tunable would promise an importer a change the
equations do not support.

**A state points at its derivative, which is the opposite direction from `Mode.derivative`.**
FMI declares a continuous state by naming its derivative's value reference. It is the one
mapping here that is easy to write backwards, so the test checks it against the schedule and
not against itself -- and for an aliased model the answer is the modeller's own name rather
than `der_x`.

### What the corpus found

**A name the model observes but never declared.** `VanDerPol`'s `der_x` is one: `der(x)` is
`y`, so alias elimination removed the variable while `observe` still fills a slot for it and
the model structure still points at it. Skipping it left a table with a hole exactly where an
importer is told to look. Seven models had one; the writer now synthesises the entry.

### Still to do on this entry

Everything else: the binary, the zip, the `fmi3` entry points, the `Lifecycle` guard wired to
them, and the round trip through FMPy. And the docs disagreement between
`docs/architecture.md:236` and `docs/ideas/backends.md:42` is **not** resolved -- writing the
description does not decide whether the deliverable is Python-backed or a C translation unit,
and that decision should be made in the open rather than by whichever file gets written next.

### Unit definitions, and the schema order that hides a mistake

`UnitDefinitions` is the Done list's fourth item and it landed with the description. Two things
in it are worth having written down.

**The order is fixed and invisible.** FMI 3.0's schema puts `UnitDefinitions` *before*
`ModelVariables`, and a document with the right content in the wrong order reads perfectly
fine to a person while a validating importer rejects it. The first version had it after. A
test asserts the order rather than the content.

**The table of decompositions is deliberately short.** The corpus uses 37 distinct unit
strings; `_BASE` knows 21. The rest -- ``W/K``, ``N.m.s/rad``, ``1/(K.s)`` -- get a `<Unit>`
with a name and no `<BaseUnit>`, which is legal and says exactly what is true: this is the
string the model used and nothing here claims to convert it. Parsing Modelica's unit grammar
to derive exponents is a real piece of work; guessing is worse than declining.

``degC`` is why the table exists rather than being skipped entirely. It is the only unit in the
corpus that is not a product of powers -- converting it to kelvin means adding 273.15, the
``offset`` attribute is the only place that can be said, and getting it wrong is a
twenty-degree error in somebody else's plot.
