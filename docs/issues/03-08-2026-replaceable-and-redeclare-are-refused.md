---
title: `replaceable` and `redeclare` are refused, and the standard library is built on them
status: in-progress
area: lang
owner: cute-replicant-6e76d
filed: 2026-08-03
kind: story
scope: L
verify: test
---

## Measured

- `src/modelica/lang/parse.py:85`: `replaceable` → "'replaceable' elements are not supported
  yet", `redeclare` → "'redeclare' is not supported yet". Both asserted at
  `tests/test_lang.py:306`.
- `modelica.ir.flatten` resolves `extends` and applies modifications, but a modification can
  only change a *value* today. `ir.Modification` carries `value` and `entries`
  (`src/modelica/lang/parse.py:_merge`); it has no way to say "and this component is now a
  different class".
- The example libraries in `examples/library/` (`Electrical`, `Rotational`, `Machines`,
  `Blocks`) were written without any replaceable element, so nothing in the corpus is blocked
  today.

## Inferred

This is the gate on reading the Modelica Standard Library, and therefore on reading almost any
real industrial model. `Modelica.Fluid` parameterises every component by its medium package;
`Modelica.Electrical` and `Modelica.Mechanics` use replaceable connectors and replaceable
sub-components throughout. A toolchain that refuses `redeclare` can read the corpus written for
it and very little that was written for anybody else. That is a statement about the MSL's
construction, not a measurement taken here — no MSL file has been run through this parser.

The design consequence to watch: redeclaration happens *before* flattening produces equations,
so it belongs in the instantiation step in `modelica.ir.flatten`, not in a post-pass. Getting
that wrong means two implementations of name resolution, which `docs/architecture.md:217`
already names as the mistake the diagram layer was written to avoid.

## Done

- `replaceable` on a component or a class parses, carrying its constraining clause
  (`replaceable package Medium = PartialMedium constrainedby PartialMedium`).
- `redeclare` inside a modification parses and reaches flattening as a value, not as text.
- `modelica.ir.flatten` applies it: the redeclared component instantiates from the new class,
  and the constraining clause is checked — a redeclaration that does not satisfy it is refused
  with the two class names in the message.
- A new corpus entry: one model, two redeclarations of the same component (say a resistor and
  a capacitor into the same slot), each flattening to a different `FlatModel` and each
  simulating against its own reference points. Two results from one model text is the
  observation that closes this.
- Both entries are gone from `_UNSUPPORTED` and from `tests/test_lang.py`.

## Prior art, added 2026-08-03

[pymoca](https://github.com/pymoca/pymoca) implements this and tests four separate cases:
`RedeclareComponents`, `RedeclareClass`, `RedeclareInExtends`, `RedeclareTypeCompatibility`.
The fourth is the one this issue is most likely to get wrong — a redeclaration that does not
satisfy its constraining clause — and their test is worth reading before ours is written. They
also carry `ExpandableConnectorBus`, which is the next thing an MSL reader needs and is not in
this issue. BSD licensed. See `docs/ideas/landscape.md`.

## Deliberately not done

- No `inner`/`outer` here, though the MSL uses them together with replaceable media. That is
  its own issue and its own scoping rule.
- No claim about reading the MSL. Redeclaration is *necessary* for that and is nowhere near
  sufficient — arrays, algorithm sections and enumerations all stand in the way too. Anybody
  closing this should resist writing "the MSL now parses" in the changelog.
