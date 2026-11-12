"""Pictures of a model: components, connections, equations and blocks.

A Modelica model is a graph. The components are the boxes and ``connect()`` gives the
lines. That graph is in the source text, but a person cannot see it there, and after
flattening it is gone: the equations keep the information and hide the shape.

This package gets the shape back. Give it a ``.mo`` file and it gives you a value:

    >>> from modelica import diagram
    >>> picture = diagram.of("examples/models/RLCCircuit.mo", "examples/library/Electrical.mo")
    >>> [node.label for node in picture.nodes]
    ['source', 'r', 'l', 'c', 'ground']
    >>> picture.write("rlc.svg")

There are three pictures of the same model, one for each stage of the toolchain:

===============  ===================================  ==========================
Function         Reads                                Shows
===============  ===================================  ==========================
:func:`of`       Modelica files                       components and connections
:func:`schematic`  a class, before flattening         components and connections
:func:`incidence`  a :class:`~modelica.ir.flat.FlatModel`  equations and variables
:func:`blocks`     a :class:`~modelica.ir.system.System`   the order to solve in
===============  ===================================  ==========================

Every one of them returns a :class:`~modelica.diagram.graph.Graph`, and a graph writes
itself as SVG, as Graphviz source, or as Mermaid source. A notebook shows the SVG when
you put the graph on the last line of a cell.

This package adds no dependency. It draws the SVG itself.
"""

from __future__ import annotations

from pathlib import Path

from modelica.diagram.build import blocks as blocks
from modelica.diagram.build import incidence as incidence
from modelica.diagram.build import schematic as schematic
from modelica.diagram.graph import Edge as Edge
from modelica.diagram.graph import Graph as Graph
from modelica.diagram.graph import Node as Node
from modelica.diagram.graph import Port as Port
from modelica.diagram.render import dot as dot
from modelica.diagram.render import mermaid as mermaid
from modelica.diagram.render import svg as svg
from modelica.ir.source import AmbiguousModelError as AmbiguousModelError
from modelica.ir.source import only_model
from modelica.lang import parse

__all__ = [
    "Edge",
    "Graph",
    "Node",
    "Port",
    "blocks",
    "dot",
    "incidence",
    "mermaid",
    "of",
    "schematic",
    "svg",
]


def of(*paths: str | Path, model: str | None = None, depth: int = 1) -> Graph:
    """Read Modelica files and draw one model from them.

    Give the libraries as well as the model. Name resolution needs them, exactly as the
    compiler does::

        diagram.of("models/RLCCircuit.mo", "library/Electrical.mo")

    ``model`` names the class to draw. Leave it out when the files hold one model and
    only one. ``depth`` opens the components: at ``1`` each component is a box, and at
    ``2`` a component that holds components becomes a frame around them.
    """
    programs = [parse(Path(path).read_text(encoding="utf-8"), str(path)) for path in paths]
    if model is None:
        model = only_model(programs, ", ".join(str(path) for path in paths))
    return schematic(programs, model, depth=depth)
