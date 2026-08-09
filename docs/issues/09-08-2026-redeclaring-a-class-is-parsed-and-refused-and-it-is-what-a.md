---
title: Redeclaring a class is parsed and refused, and it is what a medium package is
status: in-progress
area: ir
owner: cute-replicant-6e76d
filed: 2026-08-09
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
