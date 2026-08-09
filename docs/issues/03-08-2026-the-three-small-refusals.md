---
title: `import`, `enumeration` and `inner`/`outer` — the three small refusals
status: done
area: lang
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: 3976eb4
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

## What landed

Three commits, in the order this issue proposed: `f52f711` `import`, `91b2751`
`enumeration`, `3976eb4` `inner`/`outer`. 845 tests pass. Every Done bullet is met, and
`_UNSUPPORTED` is down to two entries — `external` and `operator` — from twelve when the
register was filled.

Three corpus entries, one per feature: `RLCCircuit` gained `import E = Electrical;`
(it names five classes from one package, so an alias earns its place); `Setback` is new, a
room whose setpoint an enumeration selects; `TwoFalls` is new, two bodies sharing one gravity.
All three carry analytic reference points.

**The `inner`/`outer` diagram check found the thing it was written to find.** The Inferred
section above predicted it, and it was real: `modelica.diagram._members` walked every
member, so a body declaring `outer Env env` drew an `env` box *inside itself* — an instance
the flat model has no variable for, and two bodies would have shown two environments where
the model has one. `_members` now skips an `outer`, and `tests/test_scoping.py` holds it
there.

Two findings worth carrying forward, neither predicted here:

- **The Python front end had no `==`.** `Expression.__eq__` builds an equation, so a
  *comparison* — `if schedule == Schedule.day` — had no operator left to spell it.
  `modelica.build.equal` is the mirror of the `not_equal` that already existed for the same
  reason, and `Setback` is the first model that needed it.
- **A name written in Python is not always the name written in the `.mo`.** An alias, an
  enumeration literal and a redeclared class are all named from *outside* their package by
  the builder, and the `.mo` may write them shorter. `_shorten` applies a class's own imports
  when building it, which is what keeps the twins equal; the enumeration and redeclare cases
  carry a documented caveat instead, because no model in the corpus needs them shortened.

## Deliberately not done, found while doing it

- **`inner outer` on one declaration** is refused in the parser by name. It declares an inner
  *and* refers to an outer, which is a third rule rather than the two together, and half-
  reading it would give a model a shared object that is quietly not shared.
- **An enumeration literal has no description string in the IR.** Modelica lets each literal
  carry one; the parser reads them and drops them, because the printer has nowhere to put
  them and the round trip is the contract.
- No `String` or `Boolean` switch on an enumeration beyond `==` and `<>`. The ordering
  operators (`<`, `>`) are defined on enumerations by the specification and nothing here
  refuses them, but nothing tests them either.
