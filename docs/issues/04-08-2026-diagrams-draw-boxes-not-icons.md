---
title: Diagrams draw boxes, and the icons that would make them read like circuits now survive parsing
status: open
area: diagram
owner: null
filed: 2026-08-04
kind: story
scope: L
verify: inspect
---

## Measured

- `modelica.diagram` draws a box per component and a line per connection. The positions come
  from `modelica.diagram.layout`, a force layout started from a circle
  (`docs/architecture.md:226` says why: a random start would make the docs site diff on every
  build). Nothing in the model influences the picture except its *topology*.
- **The data a real diagram needs now survives the parser**, as of `0bafb30`. Three fields
  hold it, and all three are `ir.Modification` values that were previously counted over and
  discarded:
  - `Class.annotation` — `Icon(graphics = {...})` and `Diagram(...)`: what a component
    *looks like*;
  - `Component.annotation` — `Placement(transformation(extent = {...}, rotation = ...))`:
    where an instance sits and how big it is;
  - `Connect.annotation` — `Line(points = {{x, y}, ...}, color = {0, 0, 255})`: the route a
    wire takes, including its corners.
  `ir.Array` and `Call.named` were added for exactly this: `{{-100, -100}, {100, 100}}` and
  `Rectangle(extent = ..., lineColor = ...)` are what the whole graphics language is made of.
- **No file in this repository carries one.** `grep -rln "annotation" examples/` matches only
  `examples/animate.py`, which is Python. Every library and model here was written by hand
  without graphics.

## Measured again, from the rendered DCMotor

Read off `docs/notebooks/drawing.ipynb`'s own output and confirmed in
`src/modelica/diagram/render.py`. Icons are not the only thing wrong, and the other two are
ours to fix whether or not any icon ever exists:

- **Wires are straight lines from centre to centre**, clipped at each box's border by
  `_corner` (`render.py:64`). Nothing avoids anything: the motor-to-ground wire runs across
  the picture and past whatever is in its way. A real schematic is *orthogonal* — segments
  meet at right angles and turn around components.
- **Port labels collide.** `_port_label` (`render.py:119`) puts the connector name 13 px along
  the wire and 6 px to the side, with no awareness of any other label. When two connectors
  leave one component in nearly the same direction — the motor's `p` and `n` both heading to
  the supply — the two names land on top of each other, which is the `p p` overlap in the
  picture, and `flange` sits half on top of the friction box.
- **A connection set of three or more gets a centroid node** (`render.py:143`,
  `node.kind == "junction"`), drawn as a bare dot in open space with wires radiating from it.
  It is *true* — three connectors do share one node — but with nothing else at that point it
  reads as a stray mark rather than as a junction.

The third item is worth keeping and drawing better rather than removing: a junction is real
electrical structure. The first two are defects.

One observation about the reference pictures that decides part of the design: **they have no
port labels at all.** A resistor's pins are visible because the icon draws them, so the wire
simply arrives at a place that already means something. Our labels exist to make up for the
boxes. Where an icon supplies the pin, the label should go.

## Inferred

The thing worth understanding before starting: **OpenModelica and Dymola do not generate
those pictures.** A resistor looks like a resistor because somebody drew it, once, in the
`annotation(Icon(...))` of `Modelica.Electrical.Analog.Basic.Resistor`, and every diagram
that uses one renders that drawing. The picture is *content*, and the tool is a renderer.

So this is two pieces of work that fail separately:

1. **A renderer for the graphics language.** `Rectangle`, `Line`, `Polygon`, `Ellipse`,
   `Text`, `Bitmap`; `extent`, `origin`, `rotation`, `lineColor`, `fillColor`,
   `fillPattern`, `lineThickness`; the icon coordinate system and how `Placement` maps it
   into the diagram. This is mechanical, it is well specified, and it is entirely inside
   `modelica.diagram`.
2. **Icons to draw.** Without them item 1 changes nothing a reader can see, because this
   corpus has none. Two ways to get them, and they are not exclusive: draw icons for
   `examples/library/*.mo` by hand, which is a few dozen lines of annotation per component
   and makes `RLCCircuit` render as a circuit; or read a library that already has them,
   which means the MSL and therefore `redeclare`, `import`, arrays and enumerations first.

Inferred and worth stating because it decides the order: **item 1 with hand-drawn icons for
our own library is the cheap half and shows the whole result.** A resistor, a capacitor, an
inductor, a ground and a source is five icons, and it turns the corpus's circuit models into
pictures that look like the screenshots. The MSL route is gated behind four `lang` issues.

The current layout does not go away. A model with no `Placement` still has to be drawn, and
the force layout is what draws it — so the two coexist, with the annotation taken when it is
there. That fallback is also what keeps the docs site building while the icons are written.

## Done

- `modelica.diagram` reads `Icon`, `Placement` and the connection `Line` from the IR, and
  renders them: a component is drawn as its own icon at its own place, and a wire follows the
  points the model gives rather than a computed route.
- A model with no graphics draws exactly as it does today. That is a test: every existing
  diagram test still passes unchanged, because nothing in the corpus has annotations yet.
- `examples/library/Electrical.mo` gains icons for its components, and its Python twin the
  same annotations, so the two front ends still agree. `RLCCircuit` then renders as a circuit
  diagram rather than as five boxes.
- The SVG still names no colour of its own where the model does not: `docs/architecture.md:222`
  requires that one image read correctly on a light page and a dark one, so a component with
  no `lineColor` keeps using `currentColor`. Where the model *does* give a colour, that colour
  is used — a blue signal wire is blue in both themes.
- The output is still deterministic. A rebuild of the docs site produces no diff.
- `verify: inspect` — the artifact is the evidence. A test can assert the SVG parses, is
  stable and names the right components; whether it *looks* like a circuit is a thing a person
  checks by looking, and the docs site is where they look.

## Deliberately not done

- No editor, no round trip. Reading and rendering annotations is not the same as writing
  them back, and a diagram editor is a different project.
- No `Bitmap` with external image references in the first pass. An icon that reaches for a
  file outside the model breaks the promise that a picture needs no other program.
- No attempt to *infer* an icon for a component that has none. Guessing what a component
  looks like from its equations is how a picture starts lying about the model.
