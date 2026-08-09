---
title: A `redeclare` written as a class element is refused, and that is how a base class is specialised
status: in-progress
area: lang
owner: cute-replicant-6e76d
filed: 2026-08-09
kind: story
scope: M
verify: test
---

## Measured

- `src/modelica/lang/parse.py`, in `_long_definition`: a `redeclare` at the head of a class
  element raises "a 'redeclare' element in a class body is not supported yet; redeclare in a
  modification instead". `tests/test_redeclare.py::test_a_redeclare_element_in_a_class_body_is_refused_by_name`
  holds it.
- The other spelling works. `rig(redeclare Capacitor element)` — a redeclaration *inside a
  modification* — parses, reaches the IR on `ir.Modification.redeclare`, and is applied by
  `modelica.ir.flatten` (`f28baf2`).
- `ir.Class` has no field a class-body redeclaration could be stored in. `components`,
  `extends` and `classes` all hold declarations of new things; nothing holds "and this
  inherited thing is now different".

## Inferred

Inferred: the two spellings differ in *when* they can be applied, which is why one was easy.
A modification is applied at instantiation, where the enclosing class is already merged. A
class-body redeclaration modifies something an `extends` brought in, so it cannot be resolved
until `effective()` has folded the base chain — and `effective()` runs after the parser has
already had to decide what the element was.

The likely shape, inferred and not tried: the parser stores it as a modification on the class
rather than as an element, and `effective()` applies it to the member the base contributed,
in the same place it already applies an `extends` clause's own modification entries
(`flatten.effective`, the `picked_up` line). If that is right this is smaller than it looks,
which is why the scope here is M and not L.

Why it matters: this is how the MSL specialises a partial base. `Modelica.Electrical.Analog.
Basic` and the fluid libraries both write `redeclare` as an element under an `extends`, so a
reader that only takes the modification spelling reads half the idiom. pymoca tests it as
`RedeclareInExtends`.

## Done

- `redeclare <declaration>;` in a class body parses and reaches the IR without being confused
  for a new component.
- `modelica.ir.flatten` applies it to the member the `extends` chain contributed, and refuses
  by name when no inherited member carries that name — the mirror of the check `instantiate`
  already makes for an unmatched redeclaration in a modification.
- The constraining clause is honoured the same way, through the same `_check_constraint`.
- A test that a class specialised this way and the same class specialised by a modification
  flatten to the same model. Two spellings of one meaning is the claim, and it is checkable.
- The refusal in `_long_definition` and its test are gone.

## Deliberately not done

- No `redeclare replaceable`, which re-opens the slot for a further redeclaration down the
  chain. It parses today and would need a rule about which constraint then applies.
