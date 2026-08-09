---
title: A block diagram is drawn by springs, so the signal flows backwards and the captions float
status: done
area: diagram
owner: cute-replicant-6e76d
filed: 2026-08-10
scope: L
verify: inspect
fixed_in: e2d7386
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

## What landed

1186 tests pass, `mkdocs build --strict` is clean, and `tests/test_diagram_layout.py` is 87
tests. `verify: inspect` — the artifacts are the rendered images, looked at.

Each Done bullet:

- **`modelica.diagram.layout._signal`.** A schematic whose every connector is an input or an
  output is placed left to right by signal flow: `_flow` points each connection from its
  output end to its input end, `_forward` breaks the cycles with a depth-first walk,
  `_ranked` gives the longest path, and each column is ordered by the median position of
  what feeds it. `PIControl` now reads **setpoint, error, controller, plant**, and the
  feedback leaves the plant's bottom, runs left under the row, and comes up into `error`
  labelled `u2`. 473x410 at 11.9 % ink before; 548x198 at 21.3 % now.
- **The fallback is stated.** `_signal` returns False for a physical network or a model with
  no connectors, and `place` falls through to `_loop` and then to `_force`, in that order,
  with the reason at each.
- **A port name is not written over a component.** `_port_label` steps the word out along the
  wire — 13, 18, 24, 31, 40 px — until it clears **both** boxes, and draws nothing when
  nowhere on that wire is clear. `test_a_port_name_is_not_written_over_a_component` holds it
  over the corpus.
- **A caption sits beside its symbol.** `_reach` measures the icon's ink instead of using
  `FRAME`. On `RLCCircuit` the word `source` was 82 px from the circle and is now 8 px past
  its edge; `c` likewise. Side by side with the EDA drawing this entry was measured against,
  it was the last visible difference.
- **The ink floor is 9.5 %**, up from 8. The corpus minimum is `MassSpringDamper` at 10.9 %.

### The two things that were wrong on the first attempt

**Measuring the icon along its own axis changed nothing.** A component's leads run out to the
frame edge on the icon's x-axis, so the widest ink is always 100 and `_reach` returned
`FRAME` for every part. A turned component's caption goes *across* its axis, and across is
where the symbol actually ends — the source's circle is radius 40 in a 200-wide square.

**The feedback left the plant's left edge.** `_corner` aims a wire at the other box, which
for a backward edge is the side facing backwards, so the line came out of the plant and
immediately crossed it. A wire that goes under leaves and arrives at the *bottom*, and the
corner is aimed at the dip rather than at the far box.

## Deliberately not done, found while doing it

- **`Layout.below` is a new field and the renderer reads it.** A feedback dip needs canvas
  the bounds do not know about, since bounds are computed from boxes. It is zero for every
  layout but the signal one, and `_under` refuses to dip into room that was not reserved.
- **Port labels have their own CSS class now** (`port`, not `small`). They shared a class
  with the values written *inside* a box, so the first version of the label test flagged
  `k = 1` for being drawn on the box it belongs to. A test that cannot tell two things apart
  is a test of neither.
- **`DCMotor` is still the weak one**, and it is not this entry's. It nests and it branches,
  so it is neither one loop nor causal and it falls to the springs. The staircase in its `n`
  wire is also still there: `_route` knows the two boxes at the ends and nothing in between,
  and a real router is a much larger thing to own. Both belong to
  `03-08-2026-the-return-wire-is-routed-through-the-component-it-comes-back-to`, which
  already records the hierarchy gap.
- **No test asserts the *order* of a signal layout.** The geometric rules catch a diverged or
  illegible picture; that `PIControl` reads setpoint-first is checked by looking at it, which
  is what `verify: inspect` asks for. A test naming the expected column order would pin one
  model's answer rather than the convention.
