"""What a picture of a model is, as a value.

A Modelica model is a graph before it is anything else. Components are the nodes and
``connect()`` gives the edges. The equations keep that graph after flattening, but they
keep it in a form no person can read.

This module holds the graph and nothing else. The passes that build one are in
:mod:`modelica.diagram.build`. The code that draws one is in
:mod:`modelica.diagram.render`. The split is the same as the split between
:mod:`modelica.ir.nodes` and its sibling modules, and for the same reason: a picture is a
value you can hold, print, compare in a test, and write to a file.

A graph has three shapes, and :attr:`Graph.kind` says which:

* ``"schematic"`` shows the components and their connections. This is the diagram.
* ``"incidence"`` shows which equation touches which variable. This is bipartite.
* ``"blocks"`` shows the sorted blocks and the order to solve them. This is a DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from modelica.diagram.icon import Frame

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType

    from modelica.ir import nodes as ir

__all__ = ["Edge", "EdgeKind", "Graph", "GraphKind", "Node", "NodeKind", "Port"]

type GraphKind = Literal["schematic", "incidence", "blocks"]

type NodeKind = Literal[
    "component",  # an instance of a model or a block
    "port",  # a connector on the boundary of the drawn model
    "junction",  # where more than two connectors meet
    "variable",  # one scalar of a flat model
    "equation",  # one scalar equation
    "block",  # one diagonal entry of the BLT form
    "state",  # a variable the solver integrates
]

type EdgeKind = Literal[
    "connection",  # connect(a, b) between acausal connectors
    "signal",  # connect(a, b) where one side is an output and the other an input
    "incidence",  # this equation contains this variable
    "order",  # this block must be solved before that one
]


@dataclass(frozen=True, slots=True)
class Port:
    """One connector on a component: the place a connection attaches."""

    name: str
    """The local name, so ``p`` and not ``r.p``."""
    type_name: str = ""
    causality: ir.Causality = "none"
    """``"none"`` for a physical connector. A signal connector is an input or an output."""
    comment: str = ""
    at: tuple[float, float] | None = None
    """Where the connector sits in the owner's icon frame, when the model says so.

    ``None`` when no ``Placement`` was written, and then a wire aims at the middle of the
    box like it always did.
    """


@dataclass(frozen=True, slots=True)
class Node:
    """One box in the picture."""

    id: str
    """Unique in the graph. For a component this is the path, so ``motor.shaft``."""
    label: str
    kind: NodeKind = "component"
    type_name: str = ""
    comment: str = ""
    ports: tuple[Port, ...] = ()
    detail: tuple[str, ...] = ()
    """Lines to show under the label: parameter values, an equation, the unknowns."""
    parent: str = ""
    """The id of the node that contains this one. Empty at the top level."""
    at: tuple[float, float] | None = None
    """Where the model says this instance sits, from its own ``Placement`` annotation.

    A diagram coordinate, not an icon one. When every component of a model has this, the
    picture is the one the modeller arranged; when none does, the layout works it out.
    """
    rotation: float = 0.0
    """Degrees counter-clockwise, from ``Placement(transformation(rotation = ...))``.

    A component on the vertical side of a loop is the same drawing stood up, not a second
    drawing. The pins turn with it, which is the part that has to be right: a wire that
    arrives where the pin *was* is a wire that misses.
    """
    icon: Frame = field(default_factory=Frame)
    """What the component's class draws for itself, from its ``Icon`` annotation.

    Empty for anything that has none, and then the renderer falls back to a labelled box.
    A picture that invents a shape for a component would be a drawing of a model nobody
    wrote, so there is no other fallback than the box.
    """


@dataclass(frozen=True, slots=True)
class Edge:
    """One line in the picture."""

    source: str
    """A node id."""
    target: str
    """A node id."""
    kind: EdgeKind = "connection"
    source_port: str = ""
    target_port: str = ""
    label: str = ""
    directed: bool = False
    """True when the line has a direction. A physical connection has none."""


def _pathlib() -> ModuleType:
    """``pathlib`` is only needed by :meth:`Graph.write`, so it is imported there."""
    import pathlib  # noqa: PLC0415 - a graph that is never written should not pay for this

    return pathlib


@dataclass(frozen=True)
class Graph:
    """A picture of one model, in one of the three shapes :data:`GraphKind` lists."""

    name: str
    kind: GraphKind
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...] = ()
    caption: str = ""
    """One line under the title. Say what the reader is looking at."""
    _index: dict[str, Node] = field(
        init=False, repr=False, compare=False, default_factory=dict[str, "Node"]
    )

    def __post_init__(self) -> None:
        self._index.update({node.id: node for node in self.nodes})

    def __getitem__(self, node_id: str) -> Node:
        try:
            return self._index[node_id]
        except KeyError:
            raise KeyError(f"{self.name} has no node {node_id!r}") from None

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._index

    def __len__(self) -> int:
        return len(self.nodes)

    def children(self, parent: str = "") -> tuple[Node, ...]:
        """The nodes directly inside ``parent``, or the top level when it is empty."""
        return tuple(node for node in self.nodes if node.parent == parent)

    def neighbours(self, node_id: str) -> Iterator[str]:
        """Every node id that an edge joins to ``node_id``."""
        for edge in self.edges:
            if edge.source == node_id:
                yield edge.target
            elif edge.target == node_id:
                yield edge.source

    @property
    def is_nested(self) -> bool:
        """True when some node contains another one."""
        return any(node.parent for node in self.nodes)

    # -- rendering ---------------------------------------------------------------------
    #
    # The renderers live in a sibling module and import this one, so these methods import
    # them at call time. The alternative is one large module, which the layout code alone
    # would make hard to read.

    def to_dot(self) -> str:
        """Graphviz DOT source. Use it when you have `dot` and want a good layout."""
        from modelica.diagram.render import dot  # noqa: PLC0415 - see the note above

        return dot(self)

    def to_mermaid(self) -> str:
        """Mermaid source. Use it in Markdown that a site renders, such as these docs."""
        from modelica.diagram.render import mermaid  # noqa: PLC0415 - see the note above

        return mermaid(self)

    def to_svg(self) -> str:
        """A standalone SVG image. This needs no other program and no other package."""
        from modelica.diagram.render import svg  # noqa: PLC0415 - see the note above

        return svg(self)

    def _repr_html_(self) -> str:
        """Show the SVG when a notebook displays the graph."""
        return self.to_svg()

    def _repr_svg_(self) -> str:
        return self.to_svg()

    def write(self, path: str | Path) -> Path:
        """Write the graph to ``path``. The suffix selects the format.

        ``.svg`` writes an image, ``.dot`` writes Graphviz source, and ``.mmd`` or ``.md``
        writes Mermaid source.
        """
        target = _pathlib().Path(path)
        writers = {".svg": self.to_svg, ".dot": self.to_dot, ".gv": self.to_dot}
        writers |= {".mmd": self.to_mermaid, ".mermaid": self.to_mermaid, ".md": self.to_mermaid}
        try:
            render = writers[target.suffix.lower()]
        except KeyError:
            raise ValueError(
                f"cannot write a graph to {target.suffix!r}; use .svg, .dot or .mmd"
            ) from None
        text = render()
        if target.suffix.lower() == ".md":
            text = f"```mermaid\n{text}\n```\n"
        target.write_text(text, encoding="utf-8")
        return target
