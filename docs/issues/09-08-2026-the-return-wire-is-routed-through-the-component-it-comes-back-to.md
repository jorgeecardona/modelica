---
title: A model with no Placement is drawn by a force layout, and a force layout knows no conventions
status: done
area: diagram
owner: cute-replicant-6e76d
filed: 2026-08-09
fixed_in: b54ce31
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

## What landed

`b54ce31`, on top of `7780e61` which made rotation possible. 913 tests pass.

`modelica.diagram.layout._loop` places a single-cycle model on a rectangle. Which models
take it, measured over the corpus:

| layout | models |
| --- | --- |
| loop | `RCCircuit`, `ProbeRC`, `ProbeLC`, `PIControl` |
| placed | `RLCCircuit` |
| force | `BumpStop`, `DCMotor`, `MassSpringDamper`, `TwoBodyHeat`, `TwoFalls`, `TwoInertias` |

`verify: inspect`, and the artifact is the built page: `site/notebooks/drawing/index.html`
renders `RLCCircuit` as a rectangle — source vertical on the left with `+` at the top, `r`
and `l` along the top rail, `c` vertical on the right, ground below the junction, values on
every part. Looked at, not inferred from element counts.

Three things had to be right and each was wrong once first, which is the useful part:

1. **A junction is a node, not a shortcut.** Walking *through* one makes every pair of pins
   meeting at a dot adjacent, so a ground on a three-way node comes out with degree two and
   looks like part of the loop — and then no model is ever a simple cycle.
2. **Which part takes the left edge** is decided by the model text: one of the two parts
   meeting the grounded node, and of those the one declared first. Not a guess about which
   is a source.
3. **Which way round the loop is walked.** Current leaves a source at the terminal the
   ground is *not* on. The other direction puts the loop's last part on the top rail and
   crosses every wire.

## Deliberately not done, found while doing it

- **`MassSpringDamper` does not take the loop layout**, and its picture is fine anyway. Its
  spring and damper are parallel between two nodes, so after the stubs are removed the cycle
  has only two *components* in it and the guard rejects it. Two parallel parts between two
  nodes is a real shape and a second convention — spring above, damper below, wall left, mass
  right — and it is not this one.
- **`DCMotor` is still poor**, and it is the honest remaining gap. It branches and it nests,
  so it is neither a loop nor one level, and the force layout does what a force layout does.
  Whoever takes that needs a layout for hierarchy, not for loops.
- No orthogonal router with obstacle avoidance, as planned.
