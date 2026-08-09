"""Where each box goes.

The renderers need coordinates, and nothing in the standard library gives them. So this
module computes them, in one of two ways:

* A **schematic** has no direction. A resistor does not come before a capacitor; they are
  simply joined. So it gets a force layout: every box pushes every other box away, every
  connection pulls its two boxes together, and the arrangement settles.
* A **block graph** and an **incidence graph** both have a direction. One block runs
  before another, and an equation is solved for a variable. So they get a layered layout:
  rows down the page, in the order the solver uses.

Two properties matter more than beauty. The result must be the same every time, because
these images go into a documentation site that a test rebuilds. And the code must need no
other package, because the core of this project has no dependencies. The force layout
therefore starts from a circle instead of from random positions, and it uses a fixed
number of steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from math import cos, radians, sin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelica.diagram.graph import Graph, Node

__all__ = ["Box", "Layout", "place"]

CHAR = 7.0
"""Width of one character at the font size the renderer uses."""

LINE = 15.0
"""Height of one line of text."""

PAD = 10.0
"""Space between the text and the edge of its box."""

MARGIN = 24.0
"""Space between the drawing and the edge of the image."""

JUNCTION = 9.0
"""Diameter of the dot where more than two connectors meet."""

PAIR = 2
"""Boxes needed before there is anything to lay out."""


@dataclass
class Box:
    """One node placed on the canvas. ``x`` and ``y`` are the centre."""

    node: Node
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    lines: tuple[str, ...] = ()
    """The text to draw, first line is the label."""
    rotation: float = 0.0
    """Degrees counter-clockwise, carried from the node so the icon and its pins agree."""
    caption: str = "right"
    """Which side a turned component's name goes, so it never lands inside the loop."""

    def turned(self, x: float, y: float) -> tuple[float, float]:
        """A point of the icon's own frame, rotated by this box's rotation.

        One place, used by both the pin positions and the drawing, because a pin that turns
        with the symbol and a symbol that turns without its pins are the same bug seen twice.
        """
        if not self.rotation:
            return x, y
        angle = radians(self.rotation)
        cosine, sine = cos(angle), sin(angle)
        return x * cosine - y * sine, x * sine + y * cosine

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def top(self) -> float:
        return self.y - self.height / 2


@dataclass
class Layout:
    """Every box, and how big the image has to be to hold them."""

    boxes: dict[str, Box] = field(default_factory=dict[str, Box])
    width: float = 0.0
    height: float = 0.0

    def __getitem__(self, node_id: str) -> Box:
        return self.boxes[node_id]

    def bounds(self, ids: list[str]) -> tuple[float, float, float, float]:
        """The rectangle that holds all of ``ids``, as left, top, width, height."""
        boxes = [self.boxes[i] for i in ids if i in self.boxes]
        left = min(box.left for box in boxes)
        top = min(box.top for box in boxes)
        right = max(box.left + box.width for box in boxes)
        bottom = max(box.top + box.height for box in boxes)
        return left, top, right - left, bottom - top


# --------------------------------------------------------------------------------------
# Box sizes
# --------------------------------------------------------------------------------------


def _short(type_name: str) -> str:
    """``Electrical.Resistor`` shown as ``Resistor``. The package is in the caption."""
    return type_name.rpartition(".")[2]


def _lines(node: Node) -> tuple[str, ...]:
    if node.kind == "junction":
        return ()
    lines = [node.label] if node.label else []
    if node.type_name and node.kind in ("component", "port"):
        lines.append(_short(node.type_name))
    elif node.type_name and node.kind == "block":
        lines.append(node.type_name)
    lines.extend(node.detail[:3])
    return tuple(lines)


ICON = 76.0
"""How wide a component that draws itself is, in the picture.

Square, because an icon's own frame is square and squeezing it would misdraw every
proportion the author chose. 76 keeps a 4-unit stroke at about 1.5 px, which is a line
rather than the grey smear a hairline becomes.
"""


def _sized(node: Node) -> Box:
    lines = _lines(node)
    if node.kind == "junction":
        return Box(node=node, width=JUNCTION, height=JUNCTION, lines=())
    if node.icon:
        # The name is part of the drawing -- an icon says where it wants it -- so the box
        # carries no text lines of its own.
        return Box(node=node, width=ICON, height=ICON, lines=(), rotation=node.rotation)
    longest = max((len(line) for line in lines), default=1)
    width = max(longest * CHAR + 2 * PAD, 44.0)
    height = max(len(lines), 1) * LINE + 2 * PAD
    return Box(node=node, width=width, height=height, lines=lines)


# --------------------------------------------------------------------------------------
# The force layout, for schematics
# --------------------------------------------------------------------------------------


def _placed(boxes: dict[str, Box]) -> bool:
    """Whether the model arranged itself. All or nothing: a half-placed model is a mess."""
    drawn = [b for b in boxes.values() if b.node.kind == "component"]
    return bool(drawn) and all(b.node.at is not None for b in drawn)


def _from_model(graph: Graph, boxes: dict[str, Box], spread: float = 1.6) -> None:
    """Put every box where the model said, in the diagram's own -100..100 coordinates.

    This is what a Modelica tool does and it is why two tools draw the same model the same
    way. ``spread`` converts those coordinates to pixels; it is the one number here that is
    taste rather than data.
    """
    for box in boxes.values():
        if box.node.at is not None:
            box.x, box.y = box.node.at[0] * spread, -box.node.at[1] * spread
    _junctions(graph, boxes)


def _junctions(graph: Graph, boxes: dict[str, Box]) -> None:
    """A junction has no place of its own, so it goes to the middle of what it joins.

    The pins, not the components. A node joining three pins that all face the same way
    belongs beside them; put it at the centre of the *boxes* instead and every wire has to
    wrap around its own component to reach it.
    """
    for box in boxes.values():
        if box.node.kind != "junction":
            continue
        reached = [
            _pin_at(boxes[far], port)
            for edge in graph.edges
            for far, port in (
                (edge.target, edge.target_port)
                if edge.source == box.node.id
                else (edge.source, edge.source_port)
                if edge.target == box.node.id
                else (None, ""),
            )
            if far is not None and far in boxes
        ]
        if reached:
            # The median, not the mean. A set of three is usually two wires meeting at one
            # terminal, and the median lands *on* that terminal while the mean floats off it
            # and makes both wires stair-step to reach the dot.
            box.x = _median([x for x, _ in reached])
            box.y = _median([y for _, y in reached])


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _pin_at(box: Box, port: str) -> tuple[float, float]:
    """Where a connector sits on a box, or the box's middle when the model does not say."""
    found = next((p for p in box.node.ports if p.name == port and p.at), None)
    if found is None or found.at is None:
        return box.x, box.y
    scale = box.width / 200.0
    x, y = box.turned(*found.at)
    return box.x + x * scale, box.y - y * scale


STEPS = 400
"""Iterations of the force layout. Enough to settle every model in the corpus."""


def _force(graph: Graph, boxes: dict[str, Box]) -> None:
    """Push every box apart, pull connected boxes together, and cool down."""
    ids = list(boxes)
    count = len(ids)
    if count < PAIR:
        for box in boxes.values():
            box.x = box.y = 0.0
        return

    # A circle, not a random scatter: the same model must draw the same picture every
    # time, or every rebuild of the documentation site produces a diff.
    span = 120.0 * math.sqrt(count)
    for index, node_id in enumerate(ids):
        angle = 2 * math.pi * index / count
        boxes[node_id].x = span * math.cos(angle)
        boxes[node_id].y = span * math.sin(angle)

    # Siblings of the same parent pull together, so an opened component stays in one
    # place instead of spreading its insides across the picture.
    pulls = [(e.source, e.target, 1.0) for e in graph.edges if e.source in boxes]
    pulls += [
        (a.id, b.id, 0.35)
        for a in graph.nodes
        for b in graph.nodes
        if a.parent and a.parent == b.parent and a.id < b.id
    ]

    ideal = span / math.sqrt(count)
    temperature = span / 4
    for _ in range(STEPS):
        shift = {node_id: [0.0, 0.0] for node_id in ids}
        for i, first in enumerate(ids):
            for second in ids[i + 1 :]:
                dx = boxes[first].x - boxes[second].x
                dy = boxes[first].y - boxes[second].y
                distance = max(math.hypot(dx, dy), 0.01)
                push = ideal * ideal / distance
                shift[first][0] += dx / distance * push
                shift[first][1] += dy / distance * push
                shift[second][0] -= dx / distance * push
                shift[second][1] -= dy / distance * push
        for source, target, weight in pulls:
            if source not in boxes or target not in boxes:
                continue
            dx = boxes[source].x - boxes[target].x
            dy = boxes[source].y - boxes[target].y
            distance = max(math.hypot(dx, dy), 0.01)
            pull = distance * distance / ideal * weight
            shift[source][0] -= dx / distance * pull
            shift[source][1] -= dy / distance * pull
            shift[target][0] += dx / distance * pull
            shift[target][1] += dy / distance * pull
        for node_id in ids:
            dx, dy = shift[node_id]
            distance = max(math.hypot(dx, dy), 0.01)
            step = min(distance, temperature)
            boxes[node_id].x += dx / distance * step
            boxes[node_id].y += dy / distance * step
        temperature *= 0.985

    _separate(boxes)


def _separate(boxes: dict[str, Box]) -> None:
    """Push overlapping boxes apart. The force layout works on points, not rectangles."""
    ids = list(boxes)
    gap = 16.0
    for _ in range(60):
        moved = False
        for i, first in enumerate(ids):
            for second in ids[i + 1 :]:
                one, two = boxes[first], boxes[second]
                overlap_x = (one.width + two.width) / 2 + gap - abs(one.x - two.x)
                overlap_y = (one.height + two.height) / 2 + gap - abs(one.y - two.y)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                moved = True
                if overlap_x < overlap_y:
                    push = math.copysign(overlap_x / 2, one.x - two.x or 1.0)
                    one.x += push
                    two.x -= push
                else:
                    push = math.copysign(overlap_y / 2, one.y - two.y or 1.0)
                    one.y += push
                    two.y -= push
        if not moved:
            return


# --------------------------------------------------------------------------------------
# The layered layout, for graphs that have a direction
# --------------------------------------------------------------------------------------


def _layers(graph: Graph, boxes: dict[str, Box]) -> list[list[str]]:
    """Put each node one row below every node it depends on."""
    if graph.kind == "incidence":
        rows: dict[int, list[str]] = {0: [], 1: []}
        for node_id, box in boxes.items():
            rows[0 if box.node.kind == "equation" else 1].append(node_id)
        return [rows[0], rows[1]]

    depth = dict.fromkeys(boxes, 0)
    for _ in range(len(boxes)):
        changed = False
        for edge in graph.edges:
            if edge.source in depth and edge.target in depth:
                wanted = depth[edge.source] + 1
                if depth[edge.target] < wanted:
                    depth[edge.target] = wanted
                    changed = True
        if not changed:
            break
    rows = {}
    for node_id, level in depth.items():
        rows.setdefault(level, []).append(node_id)
    return [rows[level] for level in sorted(rows)]


def _order(graph: Graph, rows: list[list[str]]) -> None:
    """Sort each row by the average position of what it joins in the row above."""
    joins: dict[str, list[str]] = {}
    for edge in graph.edges:
        joins.setdefault(edge.target, []).append(edge.source)
        joins.setdefault(edge.source, []).append(edge.target)
    for index in range(1, len(rows)):
        above = {node_id: position for position, node_id in enumerate(rows[index - 1])}

        def key(node_id: str, above: dict[str, int] = above) -> tuple[float, str]:
            positions = [above[other] for other in joins.get(node_id, []) if other in above]
            return (sum(positions) / len(positions) if positions else 1e9, node_id)

        rows[index].sort(key=key)


def _layered(graph: Graph, boxes: dict[str, Box]) -> None:
    rows = _layers(graph, boxes)
    _order(graph, rows)
    vertical = 34.0
    horizontal = 22.0
    if graph.kind == "incidence":
        # Two columns, equations on the left, so an edge crosses the gap and the
        # bipartite shape is the first thing a reader sees. The gap is wide, because
        # every line lives in it.
        x = 0.0
        for row in rows:
            width = max((boxes[node_id].width for node_id in row), default=0.0)
            y = 0.0
            for node_id in row:
                boxes[node_id].x = x + boxes[node_id].width / 2
                boxes[node_id].y = y + boxes[node_id].height / 2
                y += boxes[node_id].height + 8.0
            x += width + 170.0
        return
    y = 0.0
    for row in rows:
        height = max(boxes[node_id].height for node_id in row)
        width = sum(boxes[node_id].width for node_id in row) + horizontal * (len(row) - 1)
        x = -width / 2
        for node_id in row:
            boxes[node_id].x = x + boxes[node_id].width / 2
            boxes[node_id].y = y + height / 2
            x += boxes[node_id].width + horizontal
        y += height + vertical


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------
# The loop layout: what a person draws when they draw a circuit
# --------------------------------------------------------------------------------------

CAPTION = 96.0
"""Room to leave beside a component whose name is written outside its box."""

RAIL = 150.0
"""Horizontal spacing between components on a rail. Wide enough for a two-line caption."""

RISE = 200.0
"""Distance between the top rail and the bottom one."""

STUB = 90.0
"""How far below its node a ground hangs. Far enough that its symbol clears the rail."""


def _neighbours(graph: Graph, boxes: dict[str, Box]) -> dict[str, dict[str, str]]:
    """Which nodes touch which, and through which of the near one's own pins.

    A junction stays a node here, and that is the correction worth stating: walking *through*
    one makes every pair of pins that meet at a dot adjacent, so a ground on a three-way node
    comes out with degree two and looks like part of the loop. Kept as a node it is what it
    is -- a corner the loop passes through, with the ground hanging off it.
    """
    touching: dict[str, dict[str, str]] = {i: {} for i in boxes}
    for edge in graph.edges:
        if edge.source not in boxes or edge.target not in boxes:
            continue
        touching[edge.source][edge.target] = edge.source_port
        touching[edge.target][edge.source] = edge.target_port
    return touching


def _cycle(touching: dict[str, dict[str, str]]) -> tuple[list[str], dict[str, str]] | None:
    """The single loop through every node, and the stubs hanging off it.

    ``None`` when the model is not one loop with stubs, which is the whole admission this
    layout makes: it knows one shape, it recognises that shape, and anything else goes back
    to the force layout rather than being bent into a rectangle it is not.
    """
    stubs = {name: next(iter(near)) for name, near in touching.items() if len(near) == 1}
    loop = {
        name: {other for other in near if other not in stubs}
        for name, near in touching.items()
        if name not in stubs
    }
    if len(loop) < PAIR + 1 or any(len(near) != PAIR for near in loop.values()):
        return None
    start = min(loop)
    order = [start]
    previous, current = "", start
    while True:
        step = next((n for n in sorted(loop[current]) if n != previous), None)
        if step is None or step == start:
            break
        order.append(step)
        previous, current = current, step
    return (order, stubs) if len(order) == len(loop) else None


def _turn(box: Box, wanted: str, *, up: bool) -> None:
    """Stand a component on end, with ``wanted`` at the top when ``up``.

    Which way it turns is decided by where the pin already is, not by a table of component
    types. A rotation of -90 sends the icon frame's ``x = -90`` edge upward and +90 sends it
    down, so the pin that has to be at the top picks the sign.
    """
    found = next((p for p in box.node.ports if p.name == wanted and p.at), None)
    if found is None or found.at is None:
        box.rotation = -90.0
        return
    on_the_left = found.at[0] < 0
    box.rotation = -90.0 if on_the_left is up else 90.0


def _loop(graph: Graph, boxes: dict[str, Box]) -> bool:
    """Place a series loop the way a schematic is drawn, or say it cannot.

    The conventions, and they are the whole algorithm: the source stands on the left edge
    with its positive pin at the top; the series parts run along the top rail in the order
    the loop visits them; the last one stands on the right edge; the return is the bottom
    rail; and anything hanging off the loop -- a ground -- goes *below* what it hangs from.
    """
    touching = _neighbours(graph, boxes)
    walked = _cycle(touching)
    if walked is None:
        return False
    order, stubs = walked
    # The junctions came along to keep the loop honest; they are corners, not components,
    # and `_junctions` puts each one where its own wires meet once the parts are placed.
    parts = [name for name in order if boxes[name].node.kind != "junction"]
    if len(parts) < PAIR + 1:
        return False
    order_written = {name: index for index, name in enumerate(boxes)}
    # Which part takes the left edge. A ground is taken at the source's own terminal, so the
    # left edge belongs to one of the two parts meeting the grounded node -- and of those, to
    # whichever the modeller declared first. That is not a guess about which is a source: it
    # is the order the model is written in, and a model is written source-first for the same
    # reason it is drawn that way.
    grounded = set(stubs.values())
    candidates = [name for name in parts if set(touching[name]) & grounded] or parts
    anchor = min(candidates, key=lambda name: order_written[name])
    cut = order.index(anchor)
    order = order[cut:] + order[:cut]
    # And which way round it. Current leaves a source at the terminal the ground is *not*
    # on, so the walk steps away from the grounded node first; going the other way puts the
    # loop's last part on the top rail and crosses every wire.
    if len(order) > 1 and order[1] in grounded:
        order = [order[0], *reversed(order[1:])]
    parts = [name for name in order if boxes[name].node.kind != "junction"]

    first, *rest = parts
    span = max(len(rest) - 1, 1)
    width = span * RAIL
    boxes[first].x, boxes[first].y = 0.0, RISE / 2
    boxes[first].caption = "left"
    _turn(boxes[first], _towards(touching, first, rest[0]), up=True)
    along = rest[:-1] if len(rest) > 1 else rest
    for index, name in enumerate(along):
        boxes[name].x = width * (index + 0.5) / max(len(along), 1) if len(rest) > 1 else width / 2
        boxes[name].y = 0.0
    if len(rest) > 1:
        last = rest[-1]
        boxes[last].x, boxes[last].y = width, RISE / 2
        _turn(boxes[last], _towards(touching, last, rest[-2]), up=True)
    for name, hangs in stubs.items():
        boxes[name].x = width / 2 if boxes[hangs].node.kind == "junction" else boxes[hangs].x
        boxes[name].y = RISE + STUB
    _junctions(graph, boxes)
    return True


def _towards(touching: dict[str, dict[str, str]], near: str, far: str) -> str:
    """The pin ``near`` uses to reach ``far``, following a junction between them if there is one."""
    direct = touching[near].get(far)
    if direct is not None:
        return direct
    for other, port in touching[near].items():
        if far in touching.get(other, {}):
            return port
    return ""



def place(graph: Graph) -> Layout:
    """Give every node a position and report the size of the image that holds them."""
    # A node that contains other nodes is drawn as the frame around them, so it gets no
    # box of its own.
    parents = {node.parent for node in graph.nodes if node.parent}
    boxes = {node.id: _sized(node) for node in graph.nodes if node.id not in parents}
    if not boxes:
        return Layout(width=2 * MARGIN, height=2 * MARGIN)

    if graph.kind == "schematic":
        if _placed(boxes):
            _from_model(graph, boxes)
        elif not _loop(graph, boxes):
            # Not one loop with stubs, so there is no convention to apply and the springs
            # are still the best available answer. Said here rather than left to be found.
            _force(graph, boxes)
    else:
        _layered(graph, boxes)

    frame = 14.0 if parents else 0.0
    # A caption written beside a turned component sits outside its box, and a box is all the
    # bounds know about. Without this the name is cropped by the viewBox -- on whichever side
    # it was written, so both get the room.
    turned = [box for box in boxes.values() if box.rotation]
    before = CAPTION if any(box.caption == "left" for box in turned) else 0.0
    after = CAPTION if any(box.caption != "left" for box in turned) else 0.0
    left = min(box.left for box in boxes.values()) - frame - before
    top = min(box.top for box in boxes.values()) - frame
    for box in boxes.values():
        box.x += MARGIN - left
        box.y += MARGIN - top
    width = max(box.left + box.width for box in boxes.values()) + MARGIN + frame + after
    height = max(box.top + box.height for box in boxes.values()) + MARGIN + frame
    return Layout(boxes=boxes, width=width, height=height)
