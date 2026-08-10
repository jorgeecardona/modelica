"""Three ways to draw a :class:`~modelica.diagram.graph.Graph`.

Each one is for a different reader:

* :func:`svg` writes an image and needs no other program. A notebook shows it, a browser
  shows it, and a Markdown file can include it. This is the default.
* :func:`dot` writes Graphviz source. Graphviz has a better layout than this package
  does, so use it when you have Graphviz and the model is large.
* :func:`mermaid` writes Mermaid source. A documentation site draws it in the browser,
  which keeps the text of the page and the picture in the same file.

The SVG uses ``currentColor`` for every line and every character, and it fills the boxes
with the same colour at low opacity. Nothing here names a colour. So the image reads
correctly on a white page and on a dark page, and it needs no second copy and no script.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Final

from modelica.diagram.icon import FRAME, Disc, Label, Stroke
from modelica.diagram.layout import CHAR, LINE, PAD, place

_STRAIGHT: Final = 1e-9
"""Cross product below which three points count as one line. Coordinates are pixels."""

_BOTH_LABELS: Final = 60.0
"""How long a line has to be before two port names fit along it without touching."""

if TYPE_CHECKING:
    from modelica.diagram.graph import Edge, Graph
    from modelica.diagram.layout import Box, Layout

__all__ = ["dot", "mermaid", "svg"]

HEADER = 40.0
"""Height of the title band above the drawing."""

PORT_LABEL = 6
"""Characters a connector name can have before the label is dropped as clutter."""

CAPTION = 110
"""Characters of the caption to keep. A longer one would set the width of the image."""


# --------------------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------------------

_STYLE = """
.mdg { font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.mdg text { fill: currentColor; }
.mdg .box { fill: currentColor; fill-opacity: .06; stroke: currentColor;
            stroke-opacity: .55; stroke-width: 1; }
.mdg .port { stroke-dasharray: 4 3; }
.mdg .state .box, .mdg .solve { fill-opacity: .16; }
.mdg .frame { fill: none; stroke: currentColor; stroke-opacity: .3;
              stroke-dasharray: 5 4; }
.mdg .link { stroke: currentColor; stroke-opacity: .65; stroke-width: 1.4; fill: none; }
.mdg .weak { stroke-opacity: .3; stroke-width: 1; }
.mdg .dot { fill: currentColor; }
.mdg .mark { stroke: currentColor; stroke-linecap: round; stroke-linejoin: round; }
.mdg .name { font-weight: 600; }
.mdg .type, .mdg .small, .mdg .port { font-size: 10.5px; opacity: .62; }
.mdg .title { font-size: 14px; font-weight: 600; }
"""


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _corner(box: Box, towards_x: float, towards_y: float) -> tuple[float, float]:
    """Where a wire attaches to a box: the **middle of the side** that faces the other one.

    Not the point where a ray from centre to centre happens to leave the rectangle, which is
    what this did before. That point slides along the edge as the other box moves, so two
    components sitting at the same height attached at two *different* heights and the wire
    between them had to jog -- two corners and a step of a few pixels, for nothing.

    A side has one middle and it does not move. Two boxes facing each other across a gap
    attach at the same height, and `_route` then has a straight line to draw.
    """
    dx, dy = towards_x - box.x, towards_y - box.y
    if dx == 0 and dy == 0:
        return box.x, box.y
    if abs(dx) >= abs(dy):
        return (box.left + box.width if dx > 0 else box.left), box.y
    return box.x, (box.top + box.height if dy > 0 else box.top)


def _edge_svg(edge: Edge, layout: Layout, kind: str) -> list[str]:
    """One line. A layered graph keeps its lines in the gap between the rows.

    In a schematic the boxes are scattered, so a line goes from centre to centre and
    stops at the edge of each box. In an incidence or a block graph the boxes are in
    columns or rows, and a line from centre to centre would cross the boxes between them.
    So those lines leave the facing side and curve through the gap, where nothing else is.
    """
    if edge.source not in layout.boxes or edge.target not in layout.boxes:
        return []
    source = layout[edge.source]
    target = layout[edge.target]
    weak = " weak" if edge.kind == "incidence" else ""
    marker = ' marker-end="url(#mdg-arrow)"' if edge.directed else ""

    if kind == "incidence":
        x1, y1 = source.left + source.width, source.y
        x2, y2 = target.left, target.y
        bend = max((x2 - x1) / 2, 12.0)
        path = f"M {x1:.1f} {y1:.1f} C {x1 + bend:.1f} {y1:.1f} {x2 - bend:.1f} {y2:.1f} {x2:.1f} {y2:.1f}"
        return [f'<path class="link{weak}" d="{path}"{marker} />']
    if kind == "blocks":
        x1, y1 = source.x, source.top + source.height
        x2, y2 = target.x, target.top
        bend = max((y2 - y1) / 2, 10.0)
        path = f"M {x1:.1f} {y1:.1f} C {x1:.1f} {y1 + bend:.1f} {x2:.1f} {y2 - bend:.1f} {x2:.1f} {y2:.1f}"
        return [f'<path class="link{weak}" d="{path}"{marker} />']

    start = _pin(source, edge.source_port)
    end = _pin(target, edge.target_port)
    under = _under(source, target, layout)
    # A wire going under leaves and arrives at the *bottom*. Aiming it at the other box
    # instead -- which is what a forward wire does -- puts it on the side facing backwards,
    # so `PIControl`'s feedback came out of the plant's left edge and crossed the block it
    # was leaving.
    aim = (source.x, under) if under is not None else (target.x, target.y)
    back = (target.x, under) if under is not None else (source.x, source.y)
    x1, y1 = start or _corner(source, *aim)
    x2, y2 = end or _corner(target, *back)
    path = _route(
        (x1, y1),
        (x2, y2),
        source if start else None,
        target if end else None,
        under=under,
        obstacles=tuple(layout.boxes.values()),
    )
    parts = [f'<path class="link{weak}" d="{path}" fill="none"{marker} />']
    # A drawn pin needs no name beside it: the icon already says where the wire attaches,
    # and the label was only ever standing in for a pin nobody could see.
    if start is None:
        parts.extend(_port_label(edge.source_port, x1, y1, source, target))
    if end is None and _room(edge, (x1, y1), (x2, y2), near=start is None):
        parts.extend(_port_label(edge.target_port, x2, y2, target, source))
    return parts


def _room(edge: Edge, here: tuple[float, float], there: tuple[float, float], *, near: bool) -> bool:
    """Whether the far end's name can be written without landing on the near end's.

    Both names sit 13 px in from their own end, so a line shorter than about 60 px prints
    them on top of each other. `DCMotor` did exactly that: ``flange`` over ``flange``,
    twice, between the damper and the motor.

    Two ends of one connection often carry the same word, and then one is enough however
    long the line is -- the reader learns nothing from the second.
    """
    if not near:
        return True
    if edge.source_port == edge.target_port:
        return False
    return ((there[0] - here[0]) ** 2 + (there[1] - here[1]) ** 2) ** 0.5 > _BOTH_LABELS


def _pin(box: Box, port: str) -> tuple[float, float] | None:
    """Where a named connector sits on ``box``, when its own model says so."""
    if not box.node.icon:
        return None
    found = next((p for p in box.node.ports if p.name == port and p.at), None)
    if found is None or found.at is None:
        return None
    scale = box.width / (2 * FRAME)
    # Turned with the symbol. A pin that stays put while its icon rotates is a wire that
    # arrives where the pin used to be, which is the one way rotation can be worse than none.
    x, y = box.turned(*found.at)
    return box.x + x * scale, box.y - y * scale


def _stub(point: tuple[float, float], box: Box | None) -> tuple[tuple[float, float], bool]:
    """A short segment straight out of a pin, along whichever axis it faces.

    A wire that leaves a resistor sideways and a wire that leaves it downward mean the same
    thing to the equations and different things to a reader. Leaving along the pin's own
    axis is what makes the drawing agree with the symbol.
    """
    if box is None:
        return point, True
    x, y = point
    dx, dy = x - box.x, y - box.y
    reach = box.width * 0.16
    if abs(dx) >= abs(dy):
        return (x + reach if dx > 0 else x - reach, y), True
    return (x, y + reach if dy > 0 else y - reach), False


def _under(source: Box, target: Box, layout: Layout) -> float | None:
    """The depth a backward wire runs at, or ``None`` when the wire goes forwards.

    A feedback line leaves a block on the right and arrives at one on its left, so any route
    between them crosses whatever is in between -- in `PIControl` that is the controller,
    with the wire straight through it. Taking it under the whole row is what a control
    diagram does, and the room for it was kept by `layout.place`.

    Only where that room exists: `Layout.below` is zero for every layout but the signal one,
    and a dip into a canvas that did not reserve it would be clipped.
    """
    if not layout.below or target.left + target.width >= source.left:
        return None
    return max(box.top + box.height for box in layout.boxes.values()) + layout.below / 2


def _route(  # noqa: PLR0913 - two points, two boxes, and the two ways round an obstacle
    start: tuple[float, float],
    end: tuple[float, float],
    source: Box | None,
    target: Box | None,
    *,
    under: float | None = None,
    obstacles: tuple[Box, ...] = (),
) -> str:
    """An orthogonal wire: out of each pin along its own axis, then a Z between the stubs.

    Two corners rather than one. A single corner has to pick an axis, and when it picks the
    one a stub already points along, the wire doubles back over itself -- which is what a
    reader sees as a line running past a component and stopping in mid-air.

    ``under`` overrides all of that with the one route a backward wire can take: down out of
    the source, along below everything, and up into the target.

    ``obstacles`` are the boxes the middle segment must miss. The two at the ends are always
    among them; the rest is why a wire in `DCMotor` no longer runs through the converter it
    has nothing to do with.
    """
    a, _ = _stub(start, source)
    b, _ = _stub(end, target)
    if under is not None:
        return _path(_straighten([start, (start[0], under), (end[0], under), end]))
    a, a_horizontal = _stub(start, source)
    b, b_horizontal = _stub(end, target)
    if a_horizontal and b_horizontal:
        middle = _clear((a[0] + b[0]) / 2, a[0], b[0], obstacles, vertical=True)
        points = [start, a, (middle, a[1]), (middle, b[1]), b, end]
    elif not a_horizontal and not b_horizontal:
        middle = _clear((a[1] + b[1]) / 2, a[1], b[1], obstacles, vertical=False)
        points = [start, a, (a[0], middle), (b[0], middle), b, end]
    elif a_horizontal:
        points = [start, a, (b[0], a[1]), b, end]
    else:
        points = [start, a, (a[0], b[1]), b, end]
    return _path(_straighten(points))


def _clear(
    middle: float,
    first: float,
    second: float,
    obstacles: tuple[Box, ...],
    *,
    vertical: bool,
) -> float:
    """Put the Z's middle line where it crosses no component.

    Two pins facing the same way put the midpoint *behind* one of them, so the wire runs back
    across the symbol it just left before turning. That was the original case and the fix was
    to step past whichever of the two boxes it landed in.

    Every box, now, and not only the two at the ends. A wire between two components can run
    straight through a third that has nothing to do with it -- `DCMotor` drew its winding to
    its coil through the middle of the converter -- and a router that knows only its own
    endpoints cannot see that.

    Candidates are the two sides of every box in the way, plus the midpoint itself; the one
    nearest the midpoint that clears *all* of them wins. If none does, the midpoint stands:
    a wire through a box is bad and a wire teleported across the picture to avoid one is
    worse, and there is no third option without a real router.
    """

    def spans(box: Box) -> tuple[float, float]:
        low = (box.left if vertical else box.top) - 6
        return low, low + (box.width if vertical else box.height) + 12

    def blocked(where: float) -> bool:
        return any(low < where < high for low, high in map(spans, obstacles))

    if not blocked(middle):
        return middle
    edges = [edge for box in obstacles for edge in spans(box)]
    # Between the two ends, or the line stops being a route from one to the other.
    inside = [e for e in edges if min(first, second) - 12 <= e <= max(first, second) + 12]
    open_ones = [e for e in inside if not blocked(e)]
    return min(open_ones, key=lambda e: abs(e - middle)) if open_ones else middle


def _straighten(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop repeats, reversals, and any point the wire does not actually turn at.

    A pin's stub points *out* of its component, which is right when the wire leaves that
    way and wrong when the wire arrives from that side: the path then runs past the pin and
    returns along itself, which a reader sees as a line ending in mid-air. Removing the
    reversal is better than never stubbing, because the stub is what keeps a wire from
    emerging out of the middle of a symbol.

    Collinear points go too, and that is the rule "a straight line unless a corner is
    necessary" made true of the *output* rather than only of the intent. The stub and the Z's
    midline are constructed whatever the geometry, so a wire between two boxes that face each
    other was emitted as five points on one line -- geometrically straight, and carrying four
    joints for a reader and a diff to wonder about.
    """
    kept: list[tuple[float, float]] = []
    for point in points:
        if kept and point == kept[-1]:
            continue
        if len(kept) >= 2:  # noqa: PLR2004 - a reversal needs two points behind it
            (ax, ay), (bx, by) = kept[-2], kept[-1]
            first = (bx - ax, by - ay)
            second = (point[0] - bx, point[1] - by)
            if first[0] * second[0] + first[1] * second[1] < 0:
                kept.pop()
                if kept and point == kept[-1]:
                    continue
            elif abs(first[0] * second[1] - first[1] * second[0]) < _STRAIGHT:
                kept.pop()  # straight on: the middle point is not a corner
        kept.append(point)
    return kept


def _path(points: list[tuple[float, float]]) -> str:
    return " ".join(
        f"{'M' if index == 0 else 'L'} {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points)
    )


def _port_label(name: str, x: float, y: float, near: Box, far: Box) -> list[str]:
    """The connector name, written just outside the box the line leaves.

    It sits beside the line and not on it. A character with a line through it is worse
    than no character.
    """
    if not name or len(name) > PORT_LABEL:
        return []
    dx, dy = far.x - near.x, far.y - near.y
    length = max((dx * dx + dy * dy) ** 0.5, 0.01)
    # Stepped out along the wire until the word clears the box it belongs to. One fixed
    # offset was enough while every wire left sideways; a feedback line leaves downward, and
    # 13 px down from a pin on the bottom edge is still inside the block -- `PIControl` put
    # the plant's `y` on top of its own frame.
    at_x = at_y = 0.0
    clear = False
    for reach in (13.0, 18.0, 24.0, 31.0, 40.0):
        at_x = x + dx / length * reach - dy / length * 6
        at_y = y + dy / length * reach + dx / length * 6 + 3.5
        # Both boxes, not just the one it names: stepping out of its own component walks it
        # toward the other, and a word on the wrong box is no more readable than a word on
        # the right one.
        clear = not _touching(at_x, at_y, name, near) and not _touching(at_x, at_y, name, far)
        if clear:
            break
    if not clear:
        # Nowhere along this wire is clear, which means the two boxes are close enough that
        # the wire barely exists. The pin is visible; the word would only be a smudge.
        return []
    return [
        (
            f'<text class="port" x="{at_x:.1f}" y="{at_y:.1f}" '
            f'text-anchor="middle">{_escape(name)}</text>'
        )
    ]


def _touching(x: float, y: float, name: str, box: Box) -> bool:
    """Whether a name centred at ``x, y`` would be drawn over ``box``.

    The word is measured, not guessed at: `text-anchor` is ``middle`` and the small face is
    about four fifths of the body character, which is the same arithmetic
    `tests/test_diagram_layout.py` uses to decide two labels have collided.
    """
    half = len(name) * CHAR * 0.8 / 2
    return (
        x - half < box.left + box.width
        and x + half > box.left
        and y - LINE / 2 < box.top + box.height
        and y + LINE / 2 > box.top
    )


def _reach(box: Box) -> float:
    """How far the icon's *ink* extends sideways, in icon coordinates.

    Not `FRAME`, which is the 200 by 200 square every Modelica icon is drawn in and has
    nothing to do with how big the drawing inside it is. A voltage source is a circle of
    radius 40 in that square, so a caption placed at the frame's edge floats 60 units of
    white away from the symbol it names -- which on `RLCCircuit` put the word ``source``
    in the middle of the loop and ``c`` well to the right of the capacitor.

    Measured **across** the icon and not along it. A component's leads run along its own
    x-axis out to the frame edge, so the widest ink is always 100 and measuring that changed
    nothing -- the first version of this did exactly that. The caption of a turned component
    goes beside it, which is across its axis, and across is where the symbol actually ends.

    Falls back to the frame for an icon that draws nothing, where there is no ink to measure.
    """
    far = 0.0
    for shape in box.node.icon.shapes:
        if isinstance(shape, Stroke):
            far = max(far, *(abs(y) for _, y in shape.points))
        elif isinstance(shape, Disc):
            far = max(far, abs(shape.y) + shape.ry)
    return far or FRAME


def _caption(box: Box, name: str, label: Label, scale: float) -> list[str]:
    """A component's name, and the settings written at the instance under it.

    ``R = 1`` belongs on the picture for the same reason the name does: a schematic without
    values is a topology, and the question a reader brings to one is usually numeric.
    """
    shown = list(box.node.detail[:2])
    step = label.size * scale * 1.15
    if box.rotation:
        # Beside a turned component, reading down: the space above it is a lead. Which side
        # is the layout's call -- a name written into the middle of a loop is worse than no
        # name -- and the default is the right, where a single component has room.
        outward = -1.0 if box.caption == "left" else 1.0
        left = box.x + outward * (_reach(box) * scale + 8)
        anchor = "end" if box.caption == "left" else "start"
        first = box.y - label.size * scale * 0.6
    else:
        # Above an upright one, and the *stack* grows upward so the last line clears the
        # symbol. Putting the name where the icon asked and the values below it writes them
        # straight across the drawing, which is what the first attempt did.
        left, anchor = box.x + label.x * scale, "middle"
        first = box.y - label.y * scale + label.size * scale * 0.35 - step * len(shown)
    head = f'<text class="name" x="{left:.1f}" y="{first:.1f}" text-anchor="{anchor}">'
    written = [f"{head}{_escape(name)}</text>"]
    written += [
        f'<text class="small" x="{left:.1f}" y="{first + step * (index + 1):.1f}" '
        f'text-anchor="{anchor}">{_escape(detail)}</text>'
        for index, detail in enumerate(shown)
    ]
    return written


def _icon_svg(box: Box) -> list[str]:
    """Draw a component's own icon, mapped from its -100..100 frame into the box.

    The y axis flips: Modelica draws with y up and SVG with y down. Nothing here names a
    colour -- every stroke is ``currentColor`` -- so one file reads on a light page and a
    dark one, which an icon that hard-coded ``{0,0,0}`` could not.
    """
    scale = box.width / (2 * FRAME)

    def at(x: float, y: float) -> tuple[float, float]:
        turned_x, turned_y = box.turned(x, y)
        return box.x + turned_x * scale, box.y - turned_y * scale

    parts = [f'<g class="{box.node.kind} icon">']
    for drawn in box.node.icon.shapes:
        stroke = (
            f' stroke-width="{max(drawn.thickness * scale, 0.6):.2f}"'
            if not isinstance(drawn, Label)
            else ""
        )
        fill = ' fill="currentColor"' if getattr(drawn, "filled", False) else ' fill="none"'
        match drawn:
            case Stroke(points=points, closed=closed):
                path = " ".join(
                    f"{'M' if index == 0 else 'L'} {x:.1f} {y:.1f}"
                    for index, (x, y) in enumerate(at(px, py) for px, py in points)
                )
                parts.append(
                    f'<path class="mark" d="{path}{" Z" if closed else ""}"{stroke}{fill} />'
                )
            case Disc(x=x, y=y, rx=rx, ry=ry):
                cx, cy = at(x, y)
                parts.append(
                    f'<ellipse class="mark" cx="{cx:.1f}" cy="{cy:.1f}" '
                    f'rx="{rx * scale:.1f}" ry="{ry * scale:.1f}"{stroke}{fill} />'
                )
            case Label(text=text):
                # A name is read, not drawn: it never turns with the symbol. A rotated
                # component with its label on its side is how a schematic stops being
                # readable at exactly the moment it gets tidy. Where the icon wanted it --
                # usually above -- is where a lead now runs, so a turned component takes its
                # name beside it instead, which is where a schematic puts it anyway.
                shown = text.replace("%name", box.node.label)
                parts.extend(_caption(box, shown, drawn, scale))
    if box.node.comment:
        parts.append(f"<title>{_escape(box.node.comment)}</title>")
    parts.append("</g>")
    return parts


def _box_svg(box: Box) -> list[str]:
    node = box.node
    if node.kind == "junction":
        return [f'<circle class="dot" cx="{box.x:.1f}" cy="{box.y:.1f}" r="{box.width / 2:.1f}" />']
    if node.icon:
        return _icon_svg(box)
    dashed = " port" if node.kind == "port" else ""
    heavy = " solve" if node.kind == "block" and node.type_name != "assignment" else ""
    parts = [
        f'<g class="{node.kind}">',
        (
            f'<rect class="box{dashed}{heavy}" x="{box.left:.1f}" y="{box.top:.1f}" '
            f'width="{box.width:.1f}" height="{box.height:.1f}" rx="5" />'
        ),
    ]
    y = box.top + PAD + LINE - 4
    for index, text in enumerate(box.lines):
        css = "name" if index == 0 else "type"
        parts.append(
            f'<text class="{css}" x="{box.x:.1f}" y="{y:.1f}" '
            f'text-anchor="middle">{_escape(text)}</text>'
        )
        y += LINE
    if node.comment:
        parts.append(f"<title>{_escape(node.comment)}</title>")
    parts.append("</g>")
    return parts


def _frames_svg(graph: Graph, layout: Layout) -> list[str]:
    """A dashed rectangle around every opened component, with its name on it."""
    parts: list[str] = []
    for parent in sorted({node.parent for node in graph.nodes if node.parent}):
        inside = [node.id for node in graph.nodes if node.parent == parent]
        placed = [i for i in inside if i in layout.boxes]
        if not placed:
            continue
        left, top, width, height = layout.bounds(placed)
        parts.append(
            f'<rect class="frame" x="{left - 12:.1f}" y="{top - 20:.1f}" '
            f'width="{width + 24:.1f}" height="{height + 32:.1f}" rx="7" />'
        )
        label = graph[parent].label if parent in graph else parent
        parts.append(
            f'<text class="small" x="{left - 6:.1f}" y="{top - 8:.1f}">{_escape(label)}</text>'
        )
    return parts


def svg(graph: Graph) -> str:
    """Draw the graph as one standalone SVG element."""
    layout = place(graph)
    caption_text = (
        graph.caption if len(graph.caption) <= CAPTION else graph.caption[: CAPTION - 3] + "..."
    )
    # The title band must not be cut off, so it has a say in how wide the image is.
    width = max(layout.width, len(graph.name) * 9 + 40, len(caption_text) * 6.4 + 20, 220.0)
    height = layout.height + HEADER
    body: list[str] = []
    body.extend(_frames_svg(graph, layout))
    for edge in graph.edges:
        body.extend(_edge_svg(edge, layout, graph.kind))
    for box in layout.boxes.values():
        body.extend(_box_svg(box))
    drawing = "\n".join(body)
    caption = (
        f'<text class="small" x="0" y="30">{_escape(caption_text)}</text>' if caption_text else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="mdg" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" height="{height:.0f}" '
        f'role="img" aria-label="{_escape(graph.kind)} of {_escape(graph.name)}">'
        f"<style>{_STYLE}</style>"
        f'<defs><marker id="mdg-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" fill-opacity=".65" />'
        f"</marker></defs>"
        f'<g transform="translate(6 4)">'
        f'<text class="title" x="0" y="14">{_escape(graph.name)}</text>{caption}</g>'
        f'<g transform="translate(0 {HEADER:.0f})">{drawing}</g>'
        f"</svg>"
    )


# --------------------------------------------------------------------------------------
# Graphviz
# --------------------------------------------------------------------------------------


def _quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _dot_label(graph: Graph, node_id: str) -> str:
    node = graph[node_id]
    if node.kind == "junction":
        return ""
    lines = [node.label, *([node.type_name] if node.type_name else []), *node.detail[:3]]
    return "\\n".join(_quote(line) for line in lines if line)


def dot(graph: Graph) -> str:
    """Write the graph as Graphviz source. Render it with ``dot -Tsvg``."""
    engine = "neato" if graph.kind == "schematic" else "dot"
    rank = "LR" if graph.kind == "incidence" else "TB"
    out = [
        f'digraph "{_quote(graph.name)}" {{',
        f"  // layout: {engine}",
        f'  graph [rankdir={rank}, fontname="monospace", label="{_quote(graph.caption)}",',
        "         labelloc=b, fontsize=10, splines=true];",
        '  node [shape=box, style=rounded, fontname="monospace", fontsize=10];',
        '  edge [fontname="monospace", fontsize=9];',
    ]
    parents = {node.parent for node in graph.nodes if node.parent}
    for parent in ["", *sorted(parents)]:
        children = [
            node for node in graph.nodes if node.parent == parent and node.id not in parents
        ]
        if parent:
            out.append(f'  subgraph "cluster_{_quote(parent)}" {{')
            out.append(f'    label="{_quote(graph[parent].label)}"; style=dashed;')
        for node in children:
            shape = {"junction": "point", "port": "box", "equation": "box"}.get(node.kind, "box")
            style = "dashed,rounded" if node.kind == "port" else "rounded"
            style = "filled,rounded" if node.kind in ("state", "block") else style
            out.append(
                f'    "{_quote(node.id)}" [label="{_dot_label(graph, node.id)}", '
                f'shape={shape}, style="{style}"];'
            )
        if parent:
            out.append("  }")
    for edge in graph.edges:
        attributes: list[str] = []
        if not edge.directed:
            attributes.append("dir=none")
        if edge.source_port:
            attributes.append(f'taillabel="{_quote(edge.source_port)}"')
        if edge.target_port:
            attributes.append(f'headlabel="{_quote(edge.target_port)}"')
        if edge.kind == "incidence":
            attributes.append("color=gray60")
        written = f" [{', '.join(attributes)}]" if attributes else ""
        out.append(f'  "{_quote(edge.source)}" -> "{_quote(edge.target)}"{written};')
    out.append("}")
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# Mermaid
# --------------------------------------------------------------------------------------


def _mermaid_id(node_id: str) -> str:
    """Mermaid identifiers take letters, digits and underscores, and nothing else."""
    return "n" + "".join(c if c.isalnum() else "_" for c in node_id)


def mermaid(graph: Graph) -> str:
    """Write the graph as Mermaid source, without the surrounding code fence."""
    direction = "LR" if graph.kind in ("schematic", "incidence") else "TB"
    out = [f"graph {direction}"]
    parents = {node.parent for node in graph.nodes if node.parent}
    for parent in ["", *sorted(parents)]:
        children = [
            node for node in graph.nodes if node.parent == parent and node.id not in parents
        ]
        indent = "    " if parent else "  "
        if parent:
            out.append(f'  subgraph {_mermaid_id(parent)}["{graph[parent].label}"]')
        for node in children:
            text = "<br/>".join([node.label, *([node.type_name] if node.type_name else [])])
            shape = {
                "junction": ("((", "))"),
                "port": ("([", "])"),
                "block": ("[[", "]]"),
                "state": ("([", "])"),
                "equation": ("{{", "}}"),
            }.get(node.kind, ("[", "]"))
            body = "" if node.kind == "junction" else f'"{text}"'
            out.append(f"{indent}{_mermaid_id(node.id)}{shape[0]}{body}{shape[1]}")
        if parent:
            out.append("  end")
    for edge in graph.edges:
        arrow = "-->" if edge.directed else "---"
        ports = " ".join(p for p in (edge.source_port, edge.target_port) if p)
        label = f"|{ports}|" if ports else ""
        out.append(f"  {_mermaid_id(edge.source)} {arrow}{label} {_mermaid_id(edge.target)}")
    return "\n".join(out)
