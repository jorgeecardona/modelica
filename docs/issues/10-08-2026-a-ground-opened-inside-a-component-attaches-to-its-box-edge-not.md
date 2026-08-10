---
title: A wire leaves a pin and runs straight back through the component it just left
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

## Re-diagnosed 2026-08-10, and the first diagnosis was wrong

`_pin` is not the problem. It resolves for every ground at every depth -- checked directly:

    RLCCircuit@1  ground  port='p'  icon=True  ports=[('p', (0.0, 90.0))]  pin=(180.4, 248.6)
    DCMotor@1     ground  port='p'  icon=True  ports=[('p', (0.0, 90.0))]  pin=(331.3, 269.7)
    DCMotor@2     ground  port='p'  icon=True  ports=[('p', (0.0, 90.0))]  pin=(596.0, 343.2)

The wire starts exactly there. What it does next is the fault. The ground's pin is at the
*top* of its stem, and at depth 2 the thing it connects to is below and to the left:

    depth 1   M 331.3 269.7  L 331.3 174.8  L 320.4 174.8      up and away -- correct
    depth 2   M 596.0 343.2  L 596.0 390.0  L 451.6 390.0      down, through the whole symbol

So the wire leaves the top of the stem, runs back down through the bars, and exits below the
box. From outside it reads as a wire that stops beside the ground rather than joining it.

**Three wires do this**, measured over the corpus at both depths, counting only a segment that
crosses more than a third of the way into the box:

    DCMotor@2: motor.p -> motor.winding    back through motor.winding
    DCMotor@2: motor.winding -> motor.coil back through motor.winding
    DCMotor@2: ground -> ·7                back through ground

**And no test could have caught them.** `test_no_wire_runs_through_a_component_it_does_not_touch`
skips the edge's own two endpoint boxes, by name, because a wire obviously has to reach them.
That exclusion is right at the attachment and wrong everywhere else: a wire may touch its own
box at the pin and nowhere else.

## Done

- A wire leaves its attachment point outward and does not re-enter the box it left. Where the
  far end is on the opposite side, it goes *around* -- which is the route
  `render._under` already writes for a feedback edge, applied to a second case.
- The clearance test stops excluding the endpoint boxes wholesale. It excludes a small disc
  around each attachment point instead, so the pin is allowed and the rest of the box is not.
- The three above go to zero and the measurement is recorded.

## Deliberately not done

- Nothing about `_pin`, which was the first diagnosis and is not the fault. The original text
  is left above rather than rewritten, because a wrong diagnosis that reads plausibly is worth
  keeping visible next to the measurement that overturned it.
