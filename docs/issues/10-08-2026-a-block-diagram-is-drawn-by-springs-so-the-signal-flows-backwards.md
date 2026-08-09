---
title: A block diagram is drawn by springs, so the signal flows backwards and the captions float
status: open
area: diagram
owner: null
filed: 2026-08-10
scope: L
verify: inspect
kind: story
---

## Measured

Read off the rendered images on 2026-08-10, after `b896fbf` fixed the ground, the diverged
canvas and the doubled port label. What that commit did not touch:

**`PIControl` reads backwards.** It is a causal block diagram — `setpoint`, `error`,
`controller`, `plant`, every connector an input or an output — and it is placed by
`modelica.diagram.layout._force`, which has no notion of signal flow. In the rendered image
the signal goes bottom to top, then right to left, then up and right again. `setpoint` sits
about 250 px below `error` with nothing in between. A reader of a control diagram expects
left to right with the feedback path returning underneath, and that is a convention exactly
like "ground goes down" — `layout._loop` already applies the electrical one.

**A port label lands on a component.** After the doubled `flange` was dropped, the survivor in
`DCMotor` sits across the motor's left edge, half on the box and half off it. `_port_label`
places a name 13 px along the line and 6 px to the side of it, and never asks what is there.

**Wires take staircases.** `DCMotor`'s `n` pin leaves the motor rightward, turns up, and turns
right again to reach the junction, over a span where a straight line and one corner would do.
`render._route` produces the jog; `render._straighten` does not remove it.

**A caption is 96 px from what it names.** `layout.CAPTION = 96.0`. On `RLCCircuit` the word
`source` floats in the middle of the loop, well away from the circle, and `c` sits far to the
right of the capacitor plates. Compared against a drawing from an EDA tool side by side, this
is the most visible remaining difference: everything else is roughly in the right place.

**Ink over canvas is 10 to 19 per cent** for every model that is not the edgeless grid.
`tests/test_diagram_layout.py` holds a floor of 8 per cent, which catches divergence and
nothing else; the loop rectangles are simply larger than their contents need.

## Inferred

Inferred, not measured: the four are one shape of problem seen four times. Placement asks
"where is there room" and never "what does this mean" — which is what
`03-08-2026-the-return-wire-is-routed-through-the-component-it-comes-back-to` concluded for
circuits and answered with `_loop`. The block-diagram case is the same argument in another
domain and has an equally standard answer: rank the nodes by signal flow, put each rank in a
column, and route the edges that go backwards underneath.

The other three are local and probably independent of it.

## Done

- A schematic whose connectors are all causal is placed left to right by signal flow, one
  column per rank, with feedback edges routed below the row rather than through it.
  `PIControl` is the model to look at: `setpoint`, `error`, `controller`, `plant` in that
  order across the page.
- A model that is neither one loop nor causal still gets the force layout, and that fallback
  is stated where it happens.
- A port label does not overlap any component box. There is a test, in the shape the ones in
  `tests/test_diagram_layout.py` already have.
- A caption sits beside the symbol it names, close enough that no reader has to guess which
  one it belongs to. The number replacing `CAPTION = 96.0` is measured against the widest
  caption in the corpus rather than chosen.
- The ink fraction floor in `tests/test_diagram_layout.py` rises, and the new value is
  whatever the improved layouts actually reach.

## Deliberately not done

- No orthogonal router with obstacle avoidance. The staircase above is worth removing where
  `_straighten` can already see it; a real router is a much larger thing to own and to test.
- Nothing about hierarchy. `DCMotor` nests and branches, and a layout for hierarchy is its
  own problem — recorded in the entry named above and still open.
