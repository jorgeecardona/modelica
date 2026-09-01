"""The passes that turn a model into a :class:`~modelica.diagram.graph.Graph`.

There are three, and they read the model at three different stages:

* :func:`schematic` reads the classes, before flattening. Nothing is expanded, so the
  components keep their names and the ``connect()`` statements are still there. This is
  the diagram a person draws on paper.
* :func:`incidence` reads a :class:`~modelica.ir.flat.FlatModel`. Flattening has removed
  the components, and what is left is equations and variables. The graph is bipartite.
* :func:`blocks` reads a :class:`~modelica.ir.system.System`. Causalization has sorted
  the equations, and the graph shows the order to solve them in.

:func:`schematic` uses the name resolution of :mod:`modelica.ir.flatten` instead of its
own. A second implementation of ``extends`` would drift from the first one, and then the
picture would show a model that the toolchain does not simulate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelica.diagram.graph import Edge, Graph, Node, Port
from modelica.diagram.icon import placement, rotation, shapes
from modelica.ir import nodes as ir
from modelica.ir.flatten import FlattenError, Found, Scope, effective
from modelica.ir.source import to_program

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from modelica.ir.flat import FlatModel
    from modelica.ir.source import Source
    from modelica.ir.system import System

__all__ = ["blocks", "incidence", "schematic"]

_DRAWN = frozenset({"model", "block"})
"""Restrictions that get a box. A `package`, a `record` and a `type` do not."""

PAIR = 2
"""Endpoints in a connection that is one line. More than this makes a junction."""

SMALL = 3
"""Unknowns a block can name before its label becomes a count."""


# --------------------------------------------------------------------------------------
# The schematic: components and connections
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Drawn:
    """One declared component, resolved far enough to know what to draw for it."""

    name: str
    component: ir.Component
    scope: Scope
    cls: ir.Class | None
    """The class it instantiates, or ``None`` when the type is a predefined scalar."""

    @property
    def restriction(self) -> str:
        return self.cls.restriction if self.cls is not None else "type"

    @property
    def is_connector(self) -> bool:
        return self.restriction == "connector"

    @property
    def is_drawn(self) -> bool:
        return self.restriction in _DRAWN


def _members(found: Found, scope: Scope) -> Iterator[_Drawn]:
    """Every component of ``found``, with the ``extends`` chain already folded in.

    An ``outer`` component is skipped, and that is the one exclusion. It declares nothing:
    it is a name for a declaration further out, and :mod:`modelica.ir.flatten` resolves
    every reference to it back to the matching ``inner``. Drawing it would put a box in the
    picture for an instance the flattener never creates -- the same class in three places
    would show three environments where the model has one -- which is precisely the way
    ``docs/architecture.md:217`` says a diagram starts lying about its model.
    """
    for name, member in effective(found, scope).members.items():
        if member.component.scoping == "outer":
            continue
        resolved = member.scope.resolve(member.component.type_name)
        cls = resolved.cls if isinstance(resolved, Found) else None
        yield _Drawn(name, member.component, member.scope, cls)


def _causality(found: Found, scope: Scope) -> ir.Causality:
    """Which way a signal connector points. A physical connector points no way.

    A connector carrying a ``flow`` variable is acausal: the connection is a shared
    potential plus a sum of flows, and neither side is the source. A connector whose
    scalars are all inputs, or all outputs, is a signal, and the arrow is worth drawing.
    """
    causalities: set[ir.Causality] = set()
    for member in _members(found, scope):
        if member.component.connector == "flow":
            return "none"
        causalities.add(member.component.causality)
    if len(causalities) == 1:
        return causalities.pop()
    return "none"


def _ports(found: Found, scope: Scope) -> tuple[Port, ...]:
    """The connectors of a class, in declaration order."""
    return tuple(
        Port(
            name=member.name,
            type_name=member.component.type_name,
            causality=_causality(_found_of(member), member.scope),
            comment=member.component.comment,
            at=placement(member.component.annotation),
        )
        for member in _members(found, scope)
        if member.is_connector
    )


def _found_of(member: _Drawn) -> Found:
    resolved = member.scope.resolve(member.component.type_name)
    if not isinstance(resolved, Found):
        raise FlattenError(f"{member.name!r} has no class to look inside")
    return resolved


def _settings(member: _Drawn) -> tuple[str, ...]:
    """The modifications written at the instance, as ``R = 100`` lines."""
    lines: list[str] = []
    for name, mod in member.component.modification.entries.items():
        if mod.value is not None:
            lines.append(f"{name} = {_text(mod.value)}")
    return tuple(lines)


def _text(node: ir.Expr) -> str:
    """An expression as short source text, for a label."""
    # Imported here and not at the top: a graph is built from names and connections, and
    # only the labels need the unparser.
    from modelica.lang.unparse import unparse_expr  # noqa: PLC0415 - see above

    return unparse_expr(node)


@dataclass
class _Schematic:
    """Walks a class and collects the boxes and the lines."""

    depth: int
    nodes: list[Node] = field(default_factory=list[Node])
    sets: list[list[tuple[str, str]]] = field(default_factory=list[list[tuple[str, str]]])
    """Connection sets, each a list of ``(node id, port name)`` endpoints."""
    ids: set[str] = field(default_factory=set[str])

    def add(self, node: Node) -> None:
        self.nodes.append(node)
        self.ids.add(node.id)

    def walk(self, found: Found, scope: Scope, path: str, level: int) -> None:
        """Draw the inside of ``found``: its ports, its components, its connections."""
        members = list(_members(found, scope))
        for member in members:
            if member.is_connector:
                self.add(
                    Node(
                        id=_join(path, member.name),
                        label=member.name,
                        kind="port",
                        type_name=member.component.type_name,
                        comment=member.component.comment,
                        ports=(
                            Port(
                                member.name,
                                member.component.type_name,
                                _causality(_found_of(member), member.scope),
                            ),
                        ),
                        parent=path,
                    )
                )
            elif member.is_drawn:
                self.component(member, path, level)
        self.connections(found, scope, members, path)

    def component(self, member: _Drawn, path: str, level: int) -> None:
        """One box. Expand it into a cluster while there is depth left to spend."""
        inner = _found_of(member)
        node_id = _join(path, member.name)
        inner_scope = member.scope.inside(inner)
        expand = level < self.depth and bool(
            [m for m in _members(inner, member.scope) if m.is_drawn]
        )
        self.add(
            Node(
                id=node_id,
                label=member.name,
                kind="component",
                type_name=member.component.type_name,
                comment=member.component.comment or inner.cls.comment,
                ports=_ports(inner, member.scope),
                detail=_settings(member),
                parent=path,
                at=placement(member.component.annotation),
                rotation=rotation(member.component.annotation),
                icon=shapes(inner.cls.annotation),
            )
        )
        if expand:
            self.walk(inner, inner_scope, node_id, level + 1)

    def connections(self, found: Found, scope: Scope, members: list[_Drawn], path: str) -> None:
        """Group the ``connect()`` statements into connection sets."""
        by_name = {member.name: member for member in members}
        union = _Union()
        for equation in effective(found, scope).equations:
            if isinstance(equation, ir.Connect):
                union.join(
                    self._endpoint(equation.a, by_name, path),
                    self._endpoint(equation.b, by_name, path),
                )
        self.sets.extend(union.sets())

    def _endpoint(
        self, ref: ir.Ref | ir.Subscript, by_name: dict[str, _Drawn], path: str
    ) -> tuple[str, str]:
        """Map ``r.p`` to the node that owns it and the port on that node.

        A subscripted endpoint draws to its base: the picture shows the port, and which
        element of it carries the wire is the flattener's business, not the reader's.

        A connector on the boundary of the drawn class is a node of its own, so
        ``connect(p, r.p)`` attaches to the node ``p`` with no port. A connector on a
        component that stayed closed attaches to the component with the port named.
        """
        if isinstance(ref, ir.Subscript):
            ref = ref.base
        if ref.parts[0] not in by_name:
            raise FlattenError(f"connect() endpoint {ref.name!r} names nothing in scope")
        full = _join(path, ref.name)
        if full in self.ids:
            return (full, "")
        owner, _, port = full.rpartition(".")
        if owner in self.ids:
            return (owner, port)
        return (full, "")


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


@dataclass
class _Union:
    """Union-find over connection endpoints, keeping first-seen order."""

    parent: dict[tuple[str, str], tuple[str, str]] = field(
        default_factory=dict[tuple[str, str], tuple[str, str]]
    )
    order: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def _root(self, item: tuple[str, str]) -> tuple[str, str]:
        if item not in self.parent:
            self.parent[item] = item
            self.order.append(item)
            return item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, a: tuple[str, str], b: tuple[str, str]) -> None:
        self.parent[self._root(b)] = self._root(a)

    def sets(self) -> list[list[tuple[str, str]]]:
        grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for item in self.order:
            grouped.setdefault(self._root(item), []).append(item)
        return list(grouped.values())


def schematic(source: Source | Sequence[Source], model: str, *, depth: int = 1) -> Graph:
    """Draw the components of ``model`` and the connections between them.

    ``depth`` says how many levels to open. At ``1`` each component is one box. At ``2``
    a component that contains components becomes a box with those inside it.

    The graph shows the model as written. It does not flatten, so a component keeps its
    name, and a connection of three connectors stays one junction instead of becoming
    the sum equation that flattening makes of it.
    """
    program = to_program(source)
    scope = Scope(program)
    found = scope.resolve(model)
    if not isinstance(found, Found):
        raise FlattenError(f"{model!r} is a predefined type, not a model")
    if found.cls.restriction not in _DRAWN:
        raise FlattenError(f"{model!r} is a {found.cls.restriction}, not a model")

    builder = _Schematic(depth=depth)
    builder.walk(found, scope, path="", level=1)

    edges: list[Edge] = []
    for index, endpoints in enumerate(builder.sets):
        if len(endpoints) == PAIR:
            edges.append(_edge(builder, endpoints[0], endpoints[1]))
            continue
        # Three connectors meeting is one junction, not three lines. That is what the
        # flattener does with them, and drawing it any other way would tell a lie about
        # where the flow sum is taken.
        junction = f"·{index}"
        builder.add(Node(id=junction, label="", kind="junction"))
        edges.extend(_edge(builder, endpoint, (junction, "")) for endpoint in endpoints)

    counts = f"{len(builder.nodes)} components, {len(builder.sets)} connections"
    return Graph(
        name=found.cls.name,
        kind="schematic",
        nodes=tuple(builder.nodes),
        edges=tuple(edges),
        caption=found.cls.comment or counts,
    )


def _edge(builder: _Schematic, a: tuple[str, str], b: tuple[str, str]) -> Edge:
    """One line, pointed the right way when both ends are signals."""
    source, target = a, b
    kind = "connection"
    directed = False
    if _direction(builder, a) == "output" or _direction(builder, b) == "input":
        kind, directed = "signal", True
    elif _direction(builder, a) == "input" or _direction(builder, b) == "output":
        source, target = b, a
        kind, directed = "signal", True
    return Edge(
        source=source[0],
        target=target[0],
        kind=kind,  # pyright: ignore[reportArgumentType]
        source_port=source[1],
        target_port=target[1],
        directed=directed,
    )


def _direction(builder: _Schematic, endpoint: tuple[str, str]) -> ir.Causality:
    node_id, port_name = endpoint
    node = next((n for n in builder.nodes if n.id == node_id), None)
    if node is None:
        return "none"
    if not port_name:
        return node.ports[0].causality if node.ports else "none"
    for port in node.ports:
        if port.name == port_name:
            return port.causality
    return "none"


# --------------------------------------------------------------------------------------
# The incidence graph: equations and the variables they touch
# --------------------------------------------------------------------------------------


def _names(node: ir.Expr) -> Iterator[str]:
    """Every variable name in an expression, ``der(x)`` reported as ``x``."""
    match node:
        case ir.Ref(parts=parts):
            yield ".".join(parts)
        case ir.Call(args=args):
            for arg in args:
                yield from _names(arg)
        case ir.Binary(lhs=lhs, rhs=rhs):
            yield from _names(lhs)
            yield from _names(rhs)
        case ir.Unary(operand=operand):
            yield from _names(operand)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            yield from _names(condition)
            yield from _names(then)
            yield from _names(otherwise)
        case _:
            return


def incidence(flat: FlatModel, *, parameters: bool = False) -> Graph:
    """Draw which equation touches which variable.

    This is the graph that matching runs on. An equation is one node, a variable is
    another, and an edge says the variable occurs in the equation. Parameters are left
    out unless you ask for them: they are known before the solver starts, so no equation
    has to be spent on one.
    """
    known = {v.name for v in flat.variables if v.is_constant}
    states = set(flat.states)
    nodes: list[Node] = []
    edges: list[Edge] = []
    drawn = {v.name for v in flat.variables if parameters or v.name not in known}

    for variable in flat.variables:
        if variable.name not in drawn:
            continue
        nodes.append(
            Node(
                id=f"v:{variable.name}",
                label=variable.name,
                kind="state" if variable.name in states else "variable",
                type_name=variable.base_type,
                comment=variable.comment,
                detail=(variable.unit,) if variable.unit else (),
            )
        )
    for index, equation in enumerate(flat.equations):
        text = f"{_text(equation.lhs)} = {_text(equation.rhs)}"
        nodes.append(Node(id=f"e:{index}", label=f"eq {index}", kind="equation", detail=(text,)))
        touched = set(_names(equation.lhs)) | set(_names(equation.rhs))
        edges.extend(
            Edge(source=f"e:{index}", target=f"v:{name}", kind="incidence")
            for name in sorted(touched & drawn)
        )
    caption = f"{len(flat.equations)} equations, {len(drawn)} unknowns"
    return Graph(
        name=flat.name, kind="incidence", nodes=tuple(nodes), edges=tuple(edges), caption=caption
    )


# --------------------------------------------------------------------------------------
# The block graph: what to solve, in what order
# --------------------------------------------------------------------------------------


def blocks(system: System) -> Graph:
    """Draw the sorted blocks and the order they must run in.

    Causalization gives each block the unknowns it answers. A block that reads an unknown
    another block answers must run after it, and that is the only edge here. The result is
    a DAG, because a cycle would mean the two blocks were one block.
    """
    mode = system.mode
    answers: dict[str, int] = {}
    for index, block in enumerate(mode.blocks):
        for unknown in block.unknowns:
            answers[unknown] = index

    nodes: list[Node] = []
    edges: list[Edge] = []
    seen: set[tuple[int, int]] = set()
    for index, block in enumerate(mode.blocks):
        size = len(block)
        label = ", ".join(block.unknowns) if size <= SMALL else f"{size} unknowns"
        detail = () if block.assignment is None else (f"= {_text(block.assignment)}",)
        nodes.append(
            Node(
                id=f"b:{index}",
                label=label,
                kind="block",
                type_name="assignment" if block.is_linear_scalar else f"{size}x{size} solve",
                detail=detail if size == 1 else tuple(block.unknowns),
            )
        )
        for equation in block.equations:
            for name in set(_names(equation.lhs)) | set(_names(equation.rhs)):
                producer = answers.get(name)
                if producer is None or producer == index or (producer, index) in seen:
                    continue
                seen.add((producer, index))
                edges.append(
                    Edge(source=f"b:{producer}", target=f"b:{index}", kind="order", directed=True)
                )
    largest = system.largest_block
    caption = f"{len(mode.blocks)} blocks, largest {largest}x{largest}, index {system.index}"
    return Graph(
        name=system.name, kind="blocks", nodes=tuple(nodes), edges=tuple(edges), caption=caption
    )
