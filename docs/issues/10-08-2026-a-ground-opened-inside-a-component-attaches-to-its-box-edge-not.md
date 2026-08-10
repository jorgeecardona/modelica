---
title: A ground opened inside a component attaches to its box edge, not to its pin
status: open
area: diagram
owner: null
filed: 2026-08-10
scope: S
verify: inspect
kind: bug
---

## Measured

`examples/library/Electrical.mo:58` places `Ground`'s only connector at the top of its icon:

    Pin p annotation(Placement(transformation(extent = {{-4.0, 86.0}, {4.0, 94.0}})));

and the icon draws a stem from `{0, 90}` down to `{0, 20}` with three bars under it. So the
one place a wire may meet a ground is the top of that stem.

`RLCCircuit` draws it correctly -- the wire comes down from the junction into the top of the
stem. `DCMotor` rendered at `depth=2` does not: the wire arrives horizontally from the left
and stops beside the symbol at about the height of the bars, below the stem entirely. Read
off the rendered SVG on 2026-08-10, after the routing work in `3f26811` and its successor.

## Inferred

Inferred from the two cases differing rather than from reading the resolution: `render._pin`
returns `None` for this edge, so `_corner` picks a box side instead, and the side facing the
junction is the left one. Why it returns `None` when the same component resolves in
`RLCCircuit` is the thing to find -- the candidates are the port name on the edge not matching
the icon's connector, and the junction node the wire actually attaches to being synthetic.

Worth stating because it decides the size: if it is a name mismatch this is small, and if the
pin is genuinely unavailable at depth 2 then `_corner` needs to know that a component with
exactly one connector has only one place a wire may meet it, whatever the geometry suggests.

## Done

- A ground drawn at any depth attaches at the top of its stem, and every other component with
  a placed connector attaches at that connector.
- A test in the shape the others in `tests/test_diagram_layout.py` have: for every edge whose
  endpoint declares a `Placement` for that connector, the drawn attachment point is that
  connector's position and not a box side.

## Deliberately not done

- No change to the fallback for a component with no placed connectors. A box side is the right
  answer there and it is what most of the corpus uses.
