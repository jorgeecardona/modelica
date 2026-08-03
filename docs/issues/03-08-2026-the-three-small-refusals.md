---
title: '`import`, `enumeration` and `inner`/`outer` — the three small refusals'
status: open
area: lang
owner: null
filed: 2026-08-03
kind: story
scope: M
verify: test
---

## Measured

Three entries in `src/modelica/lang/parse.py:85`, grouped here because each is small on its
own and none is worth a separate register entry:

    import        "'import' is not supported yet; refer to classes by their full name"
    enumeration   "enumerations are not supported yet"
    inner/outer   "'inner'/'outer' elements are not supported yet"

All three are pinned at `tests/test_lang.py:306`.

Measured about each:

- **`import`** is pure name resolution. `modelica.ir.nodes:338` already walks a dotted path
  through nested classes (`candidates = self.classes`, then `found.classes`), which is the
  machinery an import alias would feed. Nothing below the parser would change.
- **`enumeration`** needs a value type. `modelica.ir` has `Real`, `Integer`, `Boolean` and
  `String`; an enumeration is a fourth kind with a fixed set of literals and an ordering.
  `modelica.sim.emit` puts everything in a float workspace `w`, so an enumeration would have
  to emit as its ordinal — which is what the specification says it is.
- **`inner`/`outer`** is a scoping rule: an `outer` element binds to the nearest enclosing
  `inner` of the same name. `modelica.ir.flatten` has `Scope` and `Found` (public, per
  `docs/architecture.md:217`), so there is a place for it to live.

## Inferred

`inner`/`outer` is the one with a hidden cost. Its industrial use is the `World`/`system`
object — one gravity vector, one ambient pressure, reached by every component without being
wired. That means a component's equations can depend on a declaration that is nowhere in its
own text, which is exactly the sort of thing that makes a diagram lie. Whoever implements it
should check `modelica.diagram.schematic` afterwards, or the picture will show a model the
flattener does not simulate — the failure `docs/architecture.md:217` was written to prevent.

Ordering, inferred from that: `import` (smallest, no semantics), then `enumeration` (needs a
type), then `inner`/`outer` (needs a scoping rule and a diagram check).

## Done

- `import A.B.C;`, `import D = A.B.C;` and `import A.B.*;` resolve during flattening, and a
  corpus model refers to a library class by an alias.
- `type Color = enumeration(red, green, blue);` parses, flattens with its ordinal encoding, and
  a model that switches on one simulates.
- An `outer` component binds to the nearest enclosing `inner`, an unmatched `outer` is refused
  by name with its own message, and `modelica.diagram.schematic` draws the model that flattens.
- All three entries are gone from `_UNSUPPORTED` and from `tests/test_lang.py`.

## Deliberately not done

These three are grouped for the register, not for the branch. Each closes on its own test and
they can land in three commits; the grouping only says none of them is large enough to be
worth its own entry here.
