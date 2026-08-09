---
title: Redeclaring a class is parsed and refused, and it is what a medium package is
status: done
area: ir
owner: cute-replicant-6e76d
filed: 2026-08-09
fixed_in: b6a76fd
kind: story
scope: L
verify: test
---

## Measured

- `replaceable package Medium = Water constrainedby Water;` parses and reaches the IR as a
  nested `ir.Class` with `replaceable=True` and `constraining="Water"` (`5a033e4`).
  `rig(redeclare package Medium = P.Water)` parses too, and the entry carries the replacement
  as an `ir.Class` on `ir.Modification.redeclare`.
- **Flattening refuses it by name.** `src/modelica/ir/flatten.py`, in `instantiate`: an entry
  carrying a redeclaration whose key matches no *component* is an error, and a redeclared class
  always matches no component. The message is "redeclaring a class is not supported yet".
  `tests/test_redeclare.py::test_redeclaring_a_class_is_refused_by_name` holds it.
- Redeclaring a *component* is done and is a different mechanism: `_redeclared` swaps the
  component's `type_name` and resolves it in the scope that wrote the redeclaration.
- `modelica.ir.flatten.Scope` resolves a name against the enclosing lexical chain and then the
  program (`Scope._find_head`). There is nothing per-instance in it.

## Inferred

Inferred, not measured: the two are not the same job, and that is why one landed without the
other. A component redeclaration changes *one declaration*, which is a value the instantiator
already holds. A class redeclaration changes what a **name resolves to inside a whole
instance** — every `Medium.p` in the body of that instance, at every depth. The same class
instantiated twice with different media would then need `Medium` to resolve two ways at once,
and a `Scope` that answers from the program and the lexical chain cannot say that.

So the shape this needs, inferred: a per-instance type environment threaded through
`instantiate`, consulted by `Scope.resolve` before the lexical chain. That is the same
"one implementation of name resolution" constraint `docs/architecture.md:217` names, so it
belongs in `Scope` and not beside it.

Why it matters, and this is a claim about the MSL rather than a measurement here:
`Modelica.Fluid` parameterises every component by a replaceable medium package, so this is the
gate on that library specifically, in the way `replaceable-and-redeclare-are-refused` was the
gate on the rest. pymoca tests it as `RedeclareClass`.

## Done

- `modelica.ir.flatten` applies a class redeclaration: inside the instance that received it,
  the replaceable class's name resolves to the replacement, at every depth of that instance
  and in no other.
- The constraining clause is checked the way a component's is, with both class names in the
  message when it fails.
- Two instances of the same class with different redeclared packages flatten to different
  models in one program, which is the observation that shows the environment is per-instance
  rather than global.
- A corpus entry, or a test-only library if the corpus has no use for a medium: this closes on
  either, and the corpus is preferable.
- The refusal in `instantiate` and its test are gone.

## Deliberately not done

- No `inner`/`outer`. The MSL uses them together with replaceable media and they are their own
  issue.
- No claim about reading `Modelica.Fluid`. This is necessary for it and far from sufficient.

## What landed

`b6a76fd`. 913 tests pass.

The inferred shape was right: a per-instance type environment on `Scope`. `Scope.bindings` is
a pair list of the names an instance has had replaced, consulted **before** the lexical chain
— a redeclaration replaces a declaration, so it has to beat the class that declared it — and
carried down by `inside()`, because the binding belongs to the instance and not to the class.
`_rebind` applies them before `effective` runs, which matters: the members an `extends` brings
in resolve their type names through it too.

The observation that closes it:

    a.element.R      <- `Rig` with its declared default, a resistor
    b.element.C      <- the same `Rig`, with a capacitor redeclared into it

Two instances of one class, one flat model, different classes inside. That is what a lexical
scope cannot say and what this was for.

Two scopes again, and the same way to get it wrong that the component version had. The
replacement was written by whoever instantiated, so it resolves one level out; the
`constrainedby` was written inside the class that owns the slot, so it resolves there. The
first attempt got the second one backwards and failed on `Water` not being in scope where
`P.Water` is.

## Deliberately not done, found while doing it

- **A package constant is still out of reach, and it is the real gate on `Modelica.Fluid`.**
  `Media.Water.rho` in an equation does not resolve — with or without a redeclaration, so it
  is not a gap in this feature. A dotted reference to a constant declared inside a package is
  a lookup the flattener does not do at all. A medium package is mostly constants, so
  redeclaring one is not yet *useful* even though it now works; the tests parameterise a
  component's type instead, and `tests/test_redeclare_class.py` records the limit where
  somebody will meet it. That lookup is the next issue for anyone heading toward the fluid
  libraries.
- No corpus entry. The Done list allowed a test-only library, and the corpus has no model
  that wants a swappable class today — `Electrical.Probe` already covers a swappable
  *component*, which is the shape a reader of the corpus will meet first.
