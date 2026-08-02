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
from typing import TYPE_CHECKING

from modelica.diagram.layout import LINE, PAD, place

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
.mdg .name { font-weight: 600; }
.mdg .type, .mdg .small { font-size: 10.5px; opacity: .62; }
.mdg .title { font-size: 14px; font-weight: 600; }
"""


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _corner(box: Box, towards_x: float, towards_y: float) -> tuple[float, float]:
    """Where the line from the centre of ``box`` leaves it, aimed at a point."""
    dx = towards_x - box.x
    dy = towards_y - box.y
    if dx == 0 and dy == 0:
        return box.x, box.y
    half_w = box.width / 2 + 2
    half_h = box.height / 2 + 2
    scale_x = half_w / abs(dx) if dx else float("inf")
    scale_y = half_h / abs(dy) if dy else float("inf")
    scale = min(scale_x, scale_y)
    return box.x + dx * scale, box.y + dy * scale


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

    x1, y1 = _corner(source, target.x, target.y)
    x2, y2 = _corner(target, source.x, source.y)
    parts = [
        (
            f'<line class="link{weak}" x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}"{marker} />'
        )
    ]
    parts.extend(_port_label(edge.source_port, x1, y1, source, target))
    parts.extend(_port_label(edge.target_port, x2, y2, target, source))
    return parts


def _port_label(name: str, x: float, y: float, near: Box, far: Box) -> list[str]:
    """The connector name, written just outside the box the line leaves.

    It sits beside the line and not on it. A character with a line through it is worse
    than no character.
    """
    if not name or len(name) > PORT_LABEL:
        return []
    dx, dy = far.x - near.x, far.y - near.y
    length = max((dx * dx + dy * dy) ** 0.5, 0.01)
    at_x = x + dx / length * 13 - dy / length * 6
    at_y = y + dy / length * 13 + dx / length * 6 + 3.5
    return [
        (
            f'<text class="small" x="{at_x:.1f}" y="{at_y:.1f}" '
            f'text-anchor="middle">{_escape(name)}</text>'
        )
    ]


def _box_svg(box: Box) -> list[str]:
    node = box.node
    if node.kind == "junction":
        return [f'<circle class="dot" cx="{box.x:.1f}" cy="{box.y:.1f}" r="{box.width / 2:.1f}" />']
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
