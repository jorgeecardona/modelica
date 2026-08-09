"""A component's ``Icon`` annotation, read into shapes a renderer can draw.

A Modelica icon is not a picture file. It is a list of graphical primitives written in the
model's own text, in a square frame that runs from -100 to 100 with **y pointing up**, and
it is the reason a resistor drawn by any tool looks like a resistor: the drawing travels
with the model. :mod:`modelica.lang.parse` keeps annotations, so the primitives arrive here
as ordinary IR values -- an ``ir.Array`` of ``ir.Call``, each call carrying its arguments by
name.

This module turns those into four shapes and nothing else. The Modelica graphics language
has more (fill patterns, line patterns, arrows, bitmaps); what is here is the subset the
corpus draws with, and anything unrecognised is skipped rather than approximated. An icon
that renders *almost* right is worse than a component that falls back to a labelled box,
because the first one is a drawing of a model nobody wrote.

Colour is deliberately not read. :mod:`modelica.diagram.render` draws every stroke in
``currentColor`` so that one SVG is legible on a light page and a dark one, and an icon that
insisted on ``{0,0,0}`` would be invisible on half of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelica.ir import nodes as ir
from modelica.ir.evaluate import EvaluationError, evaluate

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Disc", "Frame", "Label", "Shape", "Stroke", "placement", "shapes"]

FRAME = 100.0
"""Half the width of an icon's own coordinate system: it runs -100 to 100 in both axes."""


@dataclass(frozen=True, slots=True)
class Stroke:
    """A polyline or a polygon: ``Line``, ``Polygon``, and ``Rectangle`` as four corners.

    A rectangle becomes a closed stroke rather than its own shape because nothing downstream
    needs to know the difference, and a corner radius the renderer cannot honour is better
    dropped here than half-drawn there.
    """

    points: tuple[tuple[float, float], ...]
    thickness: float = 4.0
    closed: bool = False
    filled: bool = False


@dataclass(frozen=True, slots=True)
class Disc:
    """An ``Ellipse``, given by the two corners of the box that bounds it."""

    x: float
    y: float
    rx: float
    ry: float
    thickness: float = 4.0
    filled: bool = False


@dataclass(frozen=True, slots=True)
class Label:
    """A ``Text``. ``%name`` is the instance's own name, filled in when it is drawn."""

    x: float
    y: float
    text: str
    size: float = 22.0


type Shape = Stroke | Disc | Label


@dataclass(frozen=True, slots=True)
class Frame:
    """Everything an icon draws, and whether it is worth drawing at all."""

    shapes: tuple[Shape, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.shapes)


def shapes(annotation: ir.Modification) -> Frame:
    """The drawable shapes of ``annotation(Icon(graphics = {...}))``, or an empty frame."""
    icon = annotation.entries.get("Icon") if annotation else None
    graphics = icon.entries.get("graphics") if icon else None
    drawn = graphics.value if graphics else None
    if not isinstance(drawn, ir.Array):
        return Frame()
    return Frame(tuple(_shapes(drawn.elements)))


def _shapes(elements: tuple[ir.Expr, ...]) -> Iterator[Shape]:
    for element in elements:
        if not isinstance(element, ir.Call):
            continue
        arguments = dict(element.named)
        shape = _one(element.name, arguments)
        if shape is not None:
            yield shape


def _one(  # noqa: PLR0911 -- one case per primitive
    name: str, arguments: dict[str, ir.Expr]
) -> Shape | None:
    thickness = (
        _number(arguments.get("lineThickness")) or _number(arguments.get("thickness")) or 4.0
    )
    filled = _fill(arguments)
    match name:
        case "Line":
            points = _points(arguments.get("points"))
            return Stroke(points, thickness) if len(points) > 1 else None
        case "Polygon":
            points = _points(arguments.get("points"))
            return Stroke(points, thickness, closed=True, filled=filled) if points else None
        case "Rectangle":
            extent = _extent(arguments.get("extent"))
            if extent is None:
                return None
            (x0, y0), (x1, y1) = extent
            corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            return Stroke(corners, thickness, closed=True, filled=filled)
        case "Ellipse":
            extent = _extent(arguments.get("extent"))
            if extent is None:
                return None
            (x0, y0), (x1, y1) = extent
            return Disc(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                rx=abs(x1 - x0) / 2,
                ry=abs(y1 - y0) / 2,
                thickness=thickness,
                filled=filled,
            )
        case "Text":
            extent = _extent(arguments.get("extent"))
            text = arguments.get("textString")
            if extent is None or not isinstance(text, ir.StringLiteral):
                return None
            (x0, y0), (x1, y1) = extent
            return Label(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=text.value,
                size=_number(arguments.get("fontSize")) or 22.0,
            )
        case _:
            return None


def _fill(arguments: dict[str, ir.Expr]) -> bool:
    """Whether the shape is filled, which is the one thing `fillPattern` is read for."""
    pattern = arguments.get("fillPattern")
    return isinstance(pattern, ir.Ref) and pattern.parts[-1] != "None"


def _number(node: ir.Expr | None) -> float | None:
    """A literal, or ``None`` when the icon says something this cannot know.

    An icon may compute from a parameter (`DynamicSelect`), and one that does is skipped
    rather than guessed at.
    """
    if node is None:
        return None
    try:
        value = evaluate(node, {})
    except EvaluationError:
        return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _points(node: ir.Expr | None) -> tuple[tuple[float, float], ...]:
    if not isinstance(node, ir.Array):
        return ()
    found: list[tuple[float, float]] = []
    for element in node.elements:
        pair = _pair(element)
        if pair is None:
            return ()
        found.append(pair)
    return tuple(found)


def _pair(node: ir.Expr) -> tuple[float, float] | None:
    if not isinstance(node, ir.Array) or len(node.elements) != 2:  # noqa: PLR2004 - a point
        return None
    x, y = (_number(part) for part in node.elements)
    return None if x is None or y is None else (x, y)


def _extent(node: ir.Expr | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    points = _points(node)
    return (points[0], points[1]) if len(points) == 2 else None  # noqa: PLR2004 - two corners


def rotation(annotation: ir.Modification) -> float:
    """``Placement(transformation(rotation = 90))`` -- how far the instance is turned.

    Degrees, counter-clockwise, in the diagram's own frame. Zero when nothing was written,
    which is most components: a resistor drawn lying down is a resistor lying down, and only
    the ones on the vertical sides of a loop ask to be stood up.
    """
    place = annotation.entries.get("Placement") if annotation else None
    transformation = place.entries.get("transformation") if place else None
    entry = transformation.entries.get("rotation") if transformation else None
    value = entry.value if entry is not None else None
    match value:
        case ir.RealLiteral(value=found) | ir.IntLiteral(value=found):
            return float(found)
        case ir.Unary(op="-", operand=ir.RealLiteral(value=found) | ir.IntLiteral(value=found)):
            return -float(found)
        case _:
            return 0.0


def placement(annotation: ir.Modification) -> tuple[float, float] | None:
    """Where a connector sits, from ``annotation(Placement(transformation(extent = ...)))``.

    The centre of the extent, in the enclosing icon's own -100..100 frame. This is how
    Modelica says where a pin is, and it is why a wire can arrive *at the pin* rather than
    at the middle of whatever box the tool happened to draw. Without it a diagram has to
    guess, and every tool guesses differently.
    """
    if not annotation:
        return None
    place = annotation.entries.get("Placement")
    transformation = place.entries.get("transformation") if place else None
    extent = (
        _extent(transformation.entries["extent"].value)
        if transformation and "extent" in transformation.entries
        else None
    )
    if extent is None:
        return None
    (x0, y0), (x1, y1) = extent
    return ((x0 + x1) / 2, (y0 + y1) / 2)
