"""Every schematic in the corpus, against the conventions a reader of a diagram uses.

A picture cannot be tested by comparing pixels. A reference image goes stale on the first
font change and tells you *that* something moved, never *what is wrong*. What can be tested
is the thing a reader actually complains about, and every complaint so far has been
geometric: the ground is not at the bottom, two labels sit on top of each other, the boxes
are lost in an ocean of white.

So each test here is one convention, written as a predicate over `modelica.diagram.layout`.
When one fails it names the model and the rule, which is what a reference image cannot do.

The numbers in the bounds are not taste. Each was measured over the corpus first and set
just outside what the worst honest layout produces, so a regression trips it and a redesign
that is merely different does not.
"""

from __future__ import annotations

import itertools
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelica import diagram
from modelica.diagram.layout import CHAR, LINE, Box, Layout, place
from modelica.diagram.render import svg

if TYPE_CHECKING:
    from modelica.diagram.graph import Graph

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MODELS = sorted((EXAMPLES / "models").glob("*.mo"))


def _layout(model: Path) -> tuple[Graph, Layout]:
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    files = [str(model), *(str(EXAMPLES / "library" / f"{n}.mo") for n in spec["requires"])]
    graph = diagram.of(*files)
    return graph, place(graph)


#: Models whose `.mo` declares no components at all -- a flat pile of equations, so there is
#: no schematic to have an opinion about. Listed rather than skipped by a size test, because
#: a model that *lost* its components should fail here rather than quietly pass.
FLAT = {
    "AtwoodPendulum",
    "BouncingBall",
    "DrainingTank",
    "HeatBar",
    "NewtonCooling",
    "Pendulum",
    "RoastingMachine",
    "Robertson",
    "Setback",
    "VanDerPol",
}

DRAWN = [m for m in MODELS if m.stem not in FLAT]


def _overlap(a: Box, b: Box, gap: float = 0.0) -> bool:
    return (
        a.left < b.left + b.width + gap
        and b.left < a.left + a.width + gap
        and a.top < b.top + b.height + gap
        and b.top < a.top + a.height + gap
    )


@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_a_model_with_components_draws_them_and_one_without_draws_nothing(model: Path) -> None:
    """The `FLAT` list is checked in both directions, so it cannot rot."""
    _, layout = _layout(model)
    if model.stem in FLAT:
        assert not layout.boxes, f"{model.stem} is listed as flat and drew boxes"
    else:
        assert layout.boxes, f"{model.stem} drew nothing"


@pytest.mark.parametrize("model", DRAWN, ids=[m.stem for m in DRAWN])
def test_no_two_boxes_overlap(model: Path) -> None:
    """Two components in the same place is the one fault no reader can work around."""
    _, layout = _layout(model)
    clashes = [
        (i, j)
        for (i, a), (j, b) in itertools.combinations(layout.boxes.items(), 2)
        if _overlap(a, b)
    ]
    assert clashes == []


@pytest.mark.parametrize("model", DRAWN, ids=[m.stem for m in DRAWN])
def test_the_ground_is_below_everything_it_connects_to(model: Path) -> None:
    """The first convention anybody names, and the one that was broken.

    `DCMotor` drew its ground at the top of the picture, stem upward, because a force layout
    optimises a quantity that has never heard of gravity. Measured here rather than looked
    at: the ground's centre is below the centre of every node it joins.
    """
    graph, layout = _layout(model)
    grounds = [
        node.id
        for node in graph.nodes
        if node.id in layout.boxes and "ground" in (node.type_name + node.label).lower()
    ]
    for ground in grounds:
        others = [layout[o] for o in graph.neighbours(ground) if o in layout.boxes]
        assert others, f"{ground} connects to nothing"
        assert all(layout[ground].y > other.y for other in others), (
            f"{model.stem}: {ground} is not below what it connects to"
        )


@pytest.mark.parametrize("model", DRAWN, ids=[m.stem for m in DRAWN])
def test_the_picture_is_not_mostly_empty(model: Path) -> None:
    """Ink over canvas. The measurement behind "there is a wrong distance".

    `TwoFalls` was **0.1 %**: three boxes in 3789 by 4306 pixels, because it declares three
    components and connects none of them, so 400 steps of repulsion had nothing pulling
    back. A centring term and a grid for the edgeless case took it to 27 %.

    8 % is the floor, and it is generous on purpose. A schematic is mostly wires and white,
    and a rule tight enough to force a good layout would also forbid a correct one; this
    catches the layout that has *diverged*, which is the failure that actually happened.
    """
    _, layout = _layout(model)
    ink = sum(box.width * box.height for box in layout.boxes.values())
    fill = ink / (layout.width * layout.height) * 100
    assert fill > 8.0, f"{model.stem}: {fill:.1f}% ink in {layout.width:.0f}x{layout.height:.0f}"


@pytest.mark.parametrize("model", DRAWN, ids=[m.stem for m in DRAWN])
def test_the_canvas_is_a_readable_shape(model: Path) -> None:
    """No page-sized picture of four boxes, and no ribbon.

    A diverged force layout does not only waste white -- it makes the drawing unreadable at
    any zoom that fits it on a screen. Both bounds were measured over the corpus: the widest
    honest picture today is under three to one.
    """
    _, layout = _layout(model)
    ratio = layout.width / layout.height
    assert 0.2 < ratio < 5.0, f"{model.stem}: {layout.width:.0f}x{layout.height:.0f}"
    assert max(layout.width, layout.height) < 1200, f"{model.stem}: too big to read"


@pytest.mark.parametrize("model", DRAWN, ids=[m.stem for m in DRAWN])
def test_a_connection_writes_each_port_name_once_and_legibly(model: Path) -> None:
    """Two names 13 px in from each end of a 40 px line print on top of each other.

    `DCMotor` drew ``flange`` over ``flange`` between the damper and the motor. The renderer
    now drops the far name when the line is too short to hold both, or when it says the same
    word as the near one.

    Compared as *boxes* and not as points, which is the second version of this test. The
    first asked whether the anchors were far apart and `DCMotor` passed it: the two
    ``flange`` anchors are 4 px apart across and 14.7 down, so by centres they are on
    separate lines, while the words are 42 px wide and lie on top of each other. A label is
    the rectangle it occupies.
    """
    import re  # noqa: PLC0415 - one test reads the output text

    graph, _ = _layout(model)
    texts = re.findall(
        r'<text class="small" x="([-\d.]+)" y="([-\d.]+)"[^>]*>([^<]*)</text>', svg(graph)
    )
    placed: list[tuple[float, float, float, str]] = []
    for x, y, name in texts:
        # `text-anchor="middle"`, and the small face is about 0.8 of the body character.
        half = len(name) * CHAR * 0.8 / 2
        here = (float(x) - half, float(x) + half, float(y), name)
        for other in placed:
            apart = here[0] > other[1] or here[1] < other[0] or abs(here[2] - other[2]) >= LINE
            assert apart, f"{model.stem}: {name!r} and {other[3]!r} overlap at {x},{y}"
        placed.append(here)
