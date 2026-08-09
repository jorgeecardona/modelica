---
title: A model with no Placement is drawn by a force layout, and a force layout knows no conventions
status: in-progress
area: diagram
owner: cute-replicant-6e76d
filed: 2026-08-09
kind: bug
scope: L
verify: inspect
---

## Measured

Re-measured on 2026-08-09 after `7780e61`, which invalidated the first version of this entry.
That version described one wire crossing one resistor in `RLCCircuit`; `RLCCircuit` now
carries `Placement` annotations that put it on a loop, so that geometry is gone and the
finding underneath it is bigger.

- **Ten of the sixteen corpus models carry no `Placement` at all**:
  `grep -c Placement examples/models/*.mo` reports zero for `RCCircuit`, `DCMotor`,
  `MassSpringDamper`, `BouncingBall`, `DrainingTank`, `AtwoodPendulum`, `BumpStop`,
  `ProbeLC`, `RoastingMachine` and `Setback`.
- For those, `modelica.diagram.layout.place` calls `_force` — a spring layout, 400
  iterations (`layout.py:STEPS`). `_from_model` is only reached when *every* component has a
  placement (`_placed`, all-or-nothing).
- What `_force` produces for `RCCircuit`, a four-component series loop, rendered today:
  **the ground is at the top of the picture** with a stalk about a third of the image tall;
  the source is in the middle of the canvas; the capacitor is on the left and the resistor
  along the bottom; and the loop reads as a spiral rather than a rectangle.

## Inferred

A force layout optimises a quantity nobody drawing a schematic has ever optimised. It has no
notion of ground being down, of a source being on the left, of a loop being a rectangle, or
of current flowing clockwise from the positive terminal. Those are conventions, and a reader
of a circuit diagram uses them to understand the picture before reading a single label — so a
picture that ignores them is not merely untidy, it is slower to read than the source text it
replaces.

Inferred, and the reason this is scope L rather than a tweak: the fix is not a better force
layout, it is a *different kind of layout*. Nearly every electrical and mechanical model in
this corpus is a single series loop with stubs hanging off it, and for that shape the
placement is not a search at all — walk the cycle and put the components on a rectangle. The
hard part is deciding what to do with a model that is not one loop, and the honest answer is
to fall back to what exists rather than to guess.

## Done

- A model with no `Placement` whose connection graph is a single cycle is drawn on a
  rectangle: the source on the left edge and vertical with its positive pin at the top, the
  series components along the top rail in cycle order, the last one vertical on the right,
  the return along the bottom.
- A degree-one component hanging off the loop — a ground — is placed *below* the node it
  attaches to, never above it.
- A component on a vertical side is rotated, which `7780e61` made possible: `Box.turned`
  turns the icon and its pins together.
- A model whose graph is not a single cycle still gets the force layout, and that fallback is
  stated where it happens rather than discovered.
- `RCCircuit`, `MassSpringDamper` and `DCMotor` are the three to look at, because they are a
  loop, a loop in another domain, and a model with a branch. `verify: inspect` — the docs
  site is where they are looked at.

## Deliberately not done

- No general orthogonal router with obstacle avoidance, and no attempt to lay out a model
  that is genuinely a mesh. One loop covers the corpus; a router is a much larger thing to
  own and to test, and `docs/architecture.md:222` already constrains what the output may say.
- No change to `_from_model`. A model that carries placements is drawn where the modeller put
  things, which is what every Modelica tool does and is why two tools agree.
