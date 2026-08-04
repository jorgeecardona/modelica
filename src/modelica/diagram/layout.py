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
        return Box(node=node, width=ICON, height=ICON, lines=())
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


def _from_model(boxes: dict[str, Box], spread: float = 1.6) -> None:
    """Put every box where the model said, in the diagram's own -100..100 coordinates.

    This is what a Modelica tool does and it is why two tools draw the same model the same
    way. ``spread`` converts those coordinates to pixels; it is the one number here that is
    taste rather than data.
    """
    for box in boxes.values():
        if box.node.at is not None:
            box.x, box.y = box.node.at[0] * spread, -box.node.at[1] * spread


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


def place(graph: Graph) -> Layout:
    """Give every node a position and report the size of the image that holds them."""
    # A node that contains other nodes is drawn as the frame around them, so it gets no
    # box of its own.
    parents = {node.parent for node in graph.nodes if node.parent}
    boxes = {node.id: _sized(node) for node in graph.nodes if node.id not in parents}
    if not boxes:
        return Layout(width=2 * MARGIN, height=2 * MARGIN)

    if graph.kind == "schematic":
        _from_model(boxes) if _placed(boxes) else _force(graph, boxes)
    else:
        _layered(graph, boxes)

    frame = 14.0 if parents else 0.0
    left = min(box.left for box in boxes.values()) - frame
    top = min(box.top for box in boxes.values()) - frame
    for box in boxes.values():
        box.x += MARGIN - left
        box.y += MARGIN - top
    width = max(box.left + box.width for box in boxes.values()) + MARGIN + frame
    height = max(box.top + box.height for box in boxes.values()) + MARGIN + frame
    return Layout(boxes=boxes, width=width, height=height)
