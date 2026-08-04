"""Instantiation: a class tree in, a :class:`~modelica.ir.flat.FlatModel` out.

This pass spends the whole object-oriented layer of the language. ``extends`` chains are
merged, modifications are applied outermost-wins, component trees are walked down to
predefined-type leaves whose names are the dotted paths (``c.v``), and ``connect()``
equations are expanded into their potential/flow forms.

Two decisions worth stating:

* **Lookup is a scope chain, not a global table.** A type name resolves first among the
  nested classes of the class using it, then outward through its enclosing classes, then
  in the program. That is the slice of Modelica's lexical lookup the corpus needs, and
  it fails loudly on anything it cannot resolve rather than guessing.

* **Connection sets are sets of (path, role) pairs, not of paths.** A connector
  reference is *outside* when a class connects its own boundary connector, *inside* when
  it connects a component's. The same instance can appear in both roles -- its parent
  wiring it up, its own class wiring it through -- and those occurrences must stay in
  different sets, or the flow variable at the boundary drops out of the equations and
  the model goes underdetermined. Flow sums count inside connectors positive, outside
  negative; an inside connector left unconnected gets its flow variables pinned to zero.

:class:`Scope`, :class:`Found` and :func:`effective` are public because a second consumer
needs them: :mod:`modelica.diagram` draws the model *before* it is flattened, and to do
that it must resolve the same type names and fold the same ``extends`` chains. A second
implementation would drift, and then the picture would show a model that this pass does
not build. They resolve names and merge classes. They produce no equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import pairwise, product
from typing import TYPE_CHECKING, Literal

from modelica.ir import nodes as ir
from modelica.ir.evaluate import EvaluationError, evaluate
from modelica.ir.flat import BaseType, FlatModel, FlatVariable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "Effective",
    "FlattenError",
    "Found",
    "Member",
    "Scope",
    "effective",
    "element",
    "flatten",
]

_PREDEFINED: frozenset[BaseType] = frozenset(("Real", "Integer", "Boolean", "String"))

type _Path = tuple[str, ...]
type _Role = Literal["inside", "outside"]


class FlattenError(Exception):
    """The model cannot be flattened, and the message says which name is to blame."""


# --------------------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Found:
    """A resolved class plus the enclosing chain its own names resolve against."""

    cls: ir.Class
    chain: tuple[ir.Class, ...]
    """``cls`` first, then its enclosing classes outward."""


@dataclass(frozen=True)
class Scope:
    program: ir.Program
    chain: tuple[ir.Class, ...] = ()

    def resolve(self, name: str) -> Found | BaseType:
        head, *rest = name.split(".")
        if head in _PREDEFINED and not rest:
            return head
        found = self._find_head(head)
        if found is None:
            raise FlattenError(f"nothing named {head!r} is in scope here (resolving {name!r})")
        for part in rest:
            nested = next((c for c in found.cls.classes if c.name == part), None)
            if nested is None:
                raise FlattenError(f"{found.cls.name!r} has no nested class {part!r}")
            found = Found(nested, (nested, *found.chain))
        return found

    def _find_head(self, head: str) -> Found | None:
        for depth, cls in enumerate(self.chain):
            for nested in cls.classes:
                if nested.name == head:
                    return Found(nested, (nested, *self.chain[depth:]))
        for cls in self.program.classes:
            if cls.name == head:
                return Found(cls, (cls,))
        return None

    def inside(self, found: Found) -> Scope:
        return Scope(self.program, found.chain)


# --------------------------------------------------------------------------------------
# Modifications
# --------------------------------------------------------------------------------------


def _merge(inner: ir.Modification, outer: ir.Modification) -> ir.Modification:
    """Merge two modifications, the outer one winning where they collide."""
    if not inner:
        return outer
    if not outer:
        return inner
    entries = dict(inner.entries)
    for name, mod in outer.entries.items():
        entries[name] = _merge(entries[name], mod) if name in entries else mod
    return ir.Modification(
        value=outer.value if outer.value is not None else inner.value, entries=entries
    )


def _merged(*mods: ir.Modification) -> ir.Modification:
    """Merge weakest-first: later arguments override earlier ones."""
    total = ir.EMPTY
    for mod in mods:
        total = _merge(total, mod)
    return total


# --------------------------------------------------------------------------------------
# The effective class: extends chains merged
# --------------------------------------------------------------------------------------


@dataclass
class Member:
    component: ir.Component
    scope: Scope
    """Where the component was declared -- its type name resolves here."""
    mods: tuple[ir.Modification, ...] = ()
    """Modifications picked up from ``extends`` clauses, weakest first."""


@dataclass
class Effective:
    """One class with its whole ``extends`` chain folded in."""

    members: dict[str, Member] = field(default_factory=dict[str, "Member"])
    equations: list[ir.Equation] = field(default_factory=list[ir.Equation])
    initial_equations: list[ir.Equation] = field(default_factory=list[ir.Equation])


def effective(found: Found, scope: Scope) -> Effective:
    inner = scope.inside(found)
    result = Effective()
    for extends in found.cls.extends:
        base = inner.resolve(extends.type_name)
        if not isinstance(base, Found):
            raise FlattenError(
                f"{found.cls.name} extends predefined type {extends.type_name!r}; "
                f"only `type` classes may do that"
            )
        base_effective = effective(base, inner)
        for name, member in base_effective.members.items():
            if name in result.members:
                raise FlattenError(f"{found.cls.name} inherits {name!r} twice")
            picked_up = extends.modification.entries.get(name)
            result.members[name] = (
                member if picked_up is None else replace(member, mods=(*member.mods, picked_up))
            )
        result.equations += base_effective.equations
        result.initial_equations += base_effective.initial_equations
    for component in found.cls.components:
        if component.name in result.members:
            raise FlattenError(f"{found.cls.name} declares {component.name!r} twice")
        result.members[component.name] = Member(component, inner)
    result.equations += found.cls.equations
    result.initial_equations += found.cls.initial_equations
    return result


# --------------------------------------------------------------------------------------
# Type aliases down to their predefined base
# --------------------------------------------------------------------------------------


def _base_of(found: Found | BaseType, scope: Scope) -> tuple[BaseType, ir.Modification]:
    """Follow a ``type`` alias chain to its predefined base, collecting attributes."""
    collected = ir.EMPTY
    current = found
    while isinstance(current, Found):
        if current.cls.restriction != "type":
            raise FlattenError(f"{current.cls.name!r} is not a type")
        if len(current.cls.extends) != 1 or current.cls.components:
            raise FlattenError(f"type {current.cls.name!r} must be a single extends")
        extends = current.cls.extends[0]
        # The alias's own attributes are defaults under anything already collected.
        collected = _merge(extends.modification, collected)
        current = scope.inside(current).resolve(extends.type_name)
    return current, collected


# --------------------------------------------------------------------------------------
# Reference prefixing
# --------------------------------------------------------------------------------------


def _prefix_ref(
    ref: ir.Ref, prefix: _Path, local: frozenset[str], *, strict: bool = True
) -> ir.Ref:
    if ref.parts[0] in local:
        return ir.Ref((*prefix, *ref.parts))
    if ref.parts == ("time",) or not strict:
        return ref
    raise FlattenError(f"unknown name {ref.name!r} in an equation")


def _unary(op: ir.UnaryOp, operand: ir.Expr) -> ir.Expr:
    """Build a unary node, folding the two forms only one front-end can spell.

    ``-0.04`` is a unary minus applied to a literal in Modelica syntax, and a negative
    literal in Python; ``+x`` has no Python spelling at all. The folded form means exactly
    what the unfolded one means, so folding here is what lets a model written twice --
    once in ``.mo``, once with :mod:`modelica.build` -- flatten to the same value.
    """
    if op == "+":
        return operand
    if op == "-":
        match operand:
            case ir.RealLiteral(value=value):
                return ir.RealLiteral(-value)
            case ir.IntLiteral(value=value):
                return ir.IntLiteral(-value)
            case _:
                pass
    return ir.Unary(op, operand)


def _fold(node: ir.Expr | None) -> ir.Expr | None:
    """The same folding, for the places an expression is *not* walked.

    A declaration attribute -- `start`, `min`, a parameter's own binding -- is carried
    across untouched, so `start = -0.6` kept the parsed `Unary('-', 0.6)` while the Python
    front-end's `start=-0.6` arrived already negative. The two spell the same model and
    compared unequal, which `AtwoodPendulum` (the first example in the corpus with a
    negative start) is what found.
    """
    match node:
        case ir.Unary(op=op, operand=operand):
            return _unary(op, operand)
        case _:
            return node


def element(name: str, indices: Sequence[int]) -> str:
    """The scalar name of one element: ``T`` and ``(3,)`` give ``T[3]``.

    Public because three places have to agree on it -- the declaration that creates the
    variable, the subscript that reads it, and anything later that wants to recognise an
    element by its name. A second spelling of this would be a bug nobody could see.
    """
    return f"{name}[{', '.join(str(i) for i in indices)}]"


def _integer(node: ir.Expr, values: Mapping[str, float], what: str) -> int:
    """Evaluate ``node`` to a positive integer, or say why it is not one.

    A shape and a subscript are both structural: they decide how many equations exist, so
    they have to be known here and not at run time. That is what makes a parameter used as
    a dimension a *structural* parameter, and the error message says so rather than leaving
    the reader to work out why a parameter was not allowed.
    """
    try:
        value = evaluate(node, values)
    except EvaluationError as failure:
        raise FlattenError(f"{what} must be known when the model is flattened: {failure}") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise FlattenError(f"{what} must be a whole number, and this one is {value!r}")
    return int(value)


def _prefix_expr(  # noqa: PLR0911 -- one case per node type
    node: ir.Expr,
    prefix: _Path,
    local: frozenset[str],
    *,
    strict: bool = True,
    values: Mapping[str, float] | None = None,
) -> ir.Expr:
    def go(inner: ir.Expr) -> ir.Expr:
        return _prefix_expr(inner, prefix, local, strict=strict, values=values)

    match node:
        case ir.Ref():
            return _prefix_ref(node, prefix, local, strict=strict)
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, go(lhs), go(rhs))
        case ir.Unary(op=op, operand=operand):
            return _unary(op, go(operand))
        case ir.Call(name=name, args=args):
            return ir.Call(name, tuple(go(a) for a in args))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(go(condition), go(then), go(otherwise))
        case ir.Subscript(base=base, indices=indices):
            # The base is prefixed the ordinary way and then *renamed*: `T[3]` becomes the
            # scalar variable the declaration created. Everything below this module sees a
            # name, never an index -- which is the whole point of scalarising here.
            if values is None:
                raise FlattenError("subscripted references are not expanded here")
            path = _prefix_ref(base, prefix, local, strict=strict).parts
            resolved = [_integer(go(i), values, f"the subscript of {base.name!r}") for i in indices]
            return ir.Ref((*path[:-1], element(path[-1], resolved)))
        case ir.Range():
            raise FlattenError("a range is read only in the header of a 'for' equation")
        case _:
            return node


def _substitute(  # noqa: PLR0911 -- one case per node type
    node: ir.Expr, values: Mapping[str, ir.Expr]
) -> ir.Expr:
    """Replace whole single-part references by name. Used to put a loop index into a body."""

    def go(inner: ir.Expr) -> ir.Expr:
        return _substitute(inner, values)

    match node:
        case ir.Ref(parts=(name,)) if name in values:
            return values[name]
        case ir.Subscript(base=base, indices=indices):
            return ir.Subscript(base, tuple(go(i) for i in indices))
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, go(lhs), go(rhs))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, go(operand))
        case ir.Call(name=name, args=args, named=named):
            return ir.Call(name, tuple(go(a) for a in args), tuple((k, go(v)) for k, v in named))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(go(condition), go(then), go(otherwise))
        case ir.Array(elements=elements):
            return ir.Array(tuple(go(e) for e in elements))
        case ir.Range(start=start, stop=stop, step=step):
            return ir.Range(go(start), go(stop), None if step is None else go(step))
        case _:
            return node


def _substitute_equation(node: ir.Equation, values: Mapping[str, ir.Expr]) -> ir.Equation:
    """The same substitution, over an equation and anything nested inside it."""

    def go(inner: ir.Expr) -> ir.Expr:
        return _substitute(inner, values)

    match node:
        case ir.Equal(lhs=lhs, rhs=rhs):
            return ir.Equal(go(lhs), go(rhs))
        case ir.Connect():
            # A connection names components, and a loop index cannot be one of them: an
            # array of connectors is indexed, which is a `Subscript` and not a bare `Ref`.
            raise FlattenError("connect() inside a 'for' equation is not supported yet")
        case ir.When(branches=branches):
            return ir.When(
                tuple(
                    ir.WhenBranch(
                        go(branch.condition),
                        tuple(type(s)(s.target, go(s.value)) for s in branch.body),
                    )
                    for branch in branches
                )
            )
        case ir.IfEquation(branches=branches, otherwise=otherwise):
            return ir.IfEquation(
                tuple(
                    ir.IfBranch(
                        go(branch.condition),
                        tuple(_substitute_equation(e, values) for e in branch.body),
                    )
                    for branch in branches
                ),
                tuple(_substitute_equation(e, values) for e in otherwise),
            )
        case ir.ForEquation(index=index, over=over, body=body):
            # A nested loop shadows the outer index if it reuses the name, which is what
            # dropping the key here says.
            inner_values = {k: v for k, v in values.items() if k != index}
            return ir.ForEquation(
                index,
                go(over),
                tuple(_substitute_equation(e, inner_values) for e in body),
            )


def _prefix_modification(
    mod: ir.Modification, prefix: _Path, local: frozenset[str]
) -> ir.Modification:
    """Give a modification's expressions the path of the class that wrote them.

    ``winding(R = Rw)`` inside ``Motor`` means *this* motor's ``Rw``. One level down that
    variable is called ``motor.Rw``, and without this the binding names a variable that
    does not exist. At the top level the prefix is empty and nothing moves, which is why
    it stayed hidden until a component of the corpus held components of its own.

    The keys are left alone. ``R`` names a parameter of the *inner* class, and only the
    value beside it is written in the outer one. And an unknown name is kept rather than
    rejected: an equation must resolve every name, but an attribute may legally hold
    something this pass does not yet instantiate.
    """
    if not prefix or not mod:
        return mod
    return ir.Modification(
        value=None if mod.value is None else _prefix_expr(mod.value, prefix, local, strict=False),
        entries={
            name: _prefix_modification(entry, prefix, local) for name, entry in mod.entries.items()
        },
    )


def _prefix_statement(node: ir.Statement, prefix: _Path, local: frozenset[str]) -> ir.Statement:
    target = _prefix_ref(node.target, prefix, local)
    value = _prefix_expr(node.value, prefix, local)
    return ir.Assign(target, value) if isinstance(node, ir.Assign) else ir.Reinit(target, value)


# --------------------------------------------------------------------------------------
# Connection sets
# --------------------------------------------------------------------------------------

type _Endpoint = tuple[_Path, _Role]


@dataclass
class _Connections:
    """Union-find over (path, role) endpoints, in first-seen order for determinism."""

    parent: dict[_Endpoint, _Endpoint] = field(default_factory=dict[_Endpoint, "_Endpoint"])
    order: list[_Endpoint] = field(default_factory=list[_Endpoint])

    def _root(self, endpoint: _Endpoint) -> _Endpoint:
        if endpoint not in self.parent:
            self.parent[endpoint] = endpoint
            self.order.append(endpoint)
            return endpoint
        while self.parent[endpoint] != endpoint:
            self.parent[endpoint] = self.parent[self.parent[endpoint]]
            endpoint = self.parent[endpoint]
        return endpoint

    def join(self, a: _Endpoint, b: _Endpoint) -> None:
        self.parent[self._root(b)] = self._root(a)

    def sets(self) -> list[list[_Endpoint]]:
        grouped: dict[_Endpoint, list[_Endpoint]] = {}
        for endpoint in self.order:
            grouped.setdefault(self._root(endpoint), []).append(endpoint)
        return list(grouped.values())

    def connected(self, endpoint: _Endpoint) -> bool:
        return endpoint in self.parent


@dataclass(frozen=True)
class _ConnectorLeaf:
    path: _Path
    """Relative to the connector instance."""
    kind: ir.ConnectorKind
    causality: ir.Causality = "none"
    """``input`` or ``output`` on a signal connector, ``none`` on a physical one."""

    @property
    def shape(self) -> tuple[_Path, ir.ConnectorKind]:
        """What two connectors must agree on before they can be connected.

        The names and the flow prefixes, and not the causality: ``connect(a.y, b.u)``
        joins an output to an input, which is the whole point of a signal connector.
        """
        return (self.path, self.kind)


# --------------------------------------------------------------------------------------
# The flattener
# --------------------------------------------------------------------------------------

_ATTRIBUTES = frozenset({"start", "fixed", "unit", "min", "max", "nominal", "displayUnit"})

_FLOOR: dict[ir.Variability, int] = {
    "continuous": 0,
    "discrete": 1,
    "parameter": 2,
    "constant": 3,
}


def _tighten(declared: ir.Variability, floor: ir.Variability) -> ir.Variability:
    return declared if _FLOOR[declared] >= _FLOOR[floor] else floor


@dataclass
class _Flattener:
    program: ir.Program
    variables: list[FlatVariable] = field(default_factory=list[FlatVariable])
    equations: list[ir.Equal] = field(default_factory=list[ir.Equal])
    whens: list[ir.When] = field(default_factory=list[ir.When])
    initial_equations: list[ir.Equal] = field(default_factory=list[ir.Equal])
    connections: _Connections = field(default_factory=_Connections)
    connectors: dict[_Path, Found] = field(default_factory=dict[_Path, "Found"])
    values: dict[str, float] = field(default_factory=dict[str, float])
    """What every parameter and constant declared so far is worth, by flat name.

    Filled as the declarations go past, and read by the two things that have to be numbers
    before the model exists: an array's shape and a subscript. Nothing else uses it -- the
    solver reads bindings from the variables themselves.
    """

    # -- components -------------------------------------------------------------------

    def instantiate(
        self,
        found: Found,
        scope: Scope,
        prefix: _Path,
        outer: ir.Modification,
        floor: ir.Variability,
    ) -> None:
        merged = effective(found, scope)
        local = frozenset(merged.members)
        if outer.value is not None:
            raise FlattenError(f"{'.'.join(prefix)}: a composite component cannot have a value")
        for name, member in merged.members.items():
            self._component(
                member, (*prefix, name), outer.entries.get(name, ir.EMPTY), floor, local
            )
        for equation in merged.equations:
            self._equation(equation, prefix, local)
        for equation in merged.initial_equations:
            self._equation(equation, prefix, local, initial=True)

    def _component(
        self,
        member: Member,
        path: _Path,
        outer: ir.Modification,
        floor: ir.Variability,
        local: frozenset[str],
    ) -> None:
        component = member.component
        if component.dimensions:
            self._array(member, path, outer, floor, local)
            return
        # `outer` arrived from the enclosing class and was prefixed there. These two were
        # written *here*, so they are prefixed here, against the path of the class that
        # holds them -- everything in `path` except this component's own name.
        here = path[:-1]
        own = _prefix_modification(component.modification, here, local)
        inherited = tuple(_prefix_modification(mod, here, local) for mod in member.mods)
        total = _merged(own, *inherited, outer)
        resolved = member.scope.resolve(component.type_name)
        variability = _tighten(component.variability, floor)
        if isinstance(resolved, Found) and resolved.cls.restriction != "type":
            if resolved.cls.restriction == "connector":
                self.connectors[path] = resolved
            self.instantiate(resolved, member.scope, path, total, variability)
            return
        base, attributes = _base_of(resolved, member.scope)
        self._leaf(component, path, _merge(attributes, total), base, variability)

    def _array(
        self,
        member: Member,
        path: _Path,
        outer: ir.Modification,
        floor: ir.Variability,
        local: frozenset[str],
    ) -> None:
        """``Real T[n]`` becomes ``T[1] ... T[n]``, and nothing below sees a shape.

        The elements are ordinary components, so this declares each one by handing it back
        to :meth:`_component` with the shape removed and the element's own name. That keeps
        every rule about modifications, attributes and nested classes in one place: an array
        of connectors instantiates the same way an array of ``Real`` does, because neither
        knows it is in an array.

        The shape is evaluated *here*, against the parameters declared before this one. That
        is what makes a parameter used as a dimension structural -- the number of equations
        depends on it, so it cannot move once the model is built.
        """
        component = member.component
        here = path[:-1]
        sizes = [
            _integer(
                _prefix_expr(dimension, here, local, values=self.values),
                self.values,
                f"the shape of {'.'.join(path)!r}",
            )
            for dimension in component.dimensions
        ]
        if any(size < 0 for size in sizes):
            raise FlattenError(f"{'.'.join(path)!r} has a negative dimension: {sizes}")
        scalar = replace(component, dimensions=())
        for indices in product(*(range(1, size + 1) for size in sizes)):
            self._component(
                replace(member, component=scalar),
                (*here, element(component.name, indices)),
                outer,
                floor,
                local,
            )

    def _leaf(
        self,
        component: ir.Component,
        path: _Path,
        total: ir.Modification,
        base: BaseType,
        variability: ir.Variability,
    ) -> None:
        name = ".".join(path)
        unknown = set(total.entries) - _ATTRIBUTES
        if unknown:
            raise FlattenError(f"{name}: unknown attribute(s) {sorted(unknown)}")

        def attribute(key: str) -> ir.Expr | None:
            entry = total.entries.get(key)
            return _fold(entry.value) if entry is not None else None

        unit = attribute("unit")
        if unit is not None and not isinstance(unit, ir.StringLiteral):
            raise FlattenError(f"{name}: `unit` must be a string literal")
        fixed = attribute("fixed")
        if fixed is not None and not isinstance(fixed, ir.BoolLiteral):
            raise FlattenError(f"{name}: `fixed` must be a boolean literal")

        binding = _fold(total.value)
        variable = FlatVariable(
            name=name,
            base_type=base,
            variability=variability,
            causality=component.causality if len(path) == 1 else "none",
            binding=binding if variability in ("parameter", "constant") else None,
            start=attribute("start"),
            fixed=fixed.value if fixed is not None else None,
            unit=unit.value if unit is not None else "",
            nominal=attribute("nominal"),
            minimum=attribute("min"),
            maximum=attribute("max"),
            comment=component.comment,
        )
        self.variables.append(variable)
        if variability in ("parameter", "constant") and isinstance(
            binding, (ir.IntLiteral, ir.RealLiteral)
        ):
            self.values[name] = binding.value
        if binding is not None and variability not in ("parameter", "constant"):
            # A declaration equation on a non-constant is an ordinary equation.
            self.equations.append(ir.Equal(ir.Ref(path), binding))

    # -- equations --------------------------------------------------------------------

    def _equation(
        self, equation: ir.Equation, prefix: _Path, local: frozenset[str], *, initial: bool = False
    ) -> None:
        def carry(node: ir.Expr) -> ir.Expr:
            return _prefix_expr(node, prefix, local, values=self.values)

        if initial and not isinstance(equation, (ir.Equal, ir.ForEquation)):
            raise FlattenError("only `a = b` equations may appear in initial equation")

        match equation:
            case ir.Equal(lhs=lhs, rhs=rhs):
                sink = self.initial_equations if initial else self.equations
                sink.append(ir.Equal(carry(lhs), carry(rhs)))
            case ir.Connect(a=a, b=b):
                self.connections.join(
                    self._endpoint(a, prefix, local), self._endpoint(b, prefix, local)
                )
            case ir.When(branches=branches):
                self.whens.append(
                    ir.When(
                        tuple(
                            ir.WhenBranch(
                                carry(branch.condition),
                                tuple(_prefix_statement(s, prefix, local) for s in branch.body),
                            )
                            for branch in branches
                        )
                    )
                )
            case ir.IfEquation():
                raise FlattenError("if-equations are not supported yet")
            case ir.ForEquation():
                self._unroll(equation, prefix, local, initial=initial)

    def _unroll(
        self,
        loop: ir.ForEquation,
        prefix: _Path,
        local: frozenset[str],
        *,
        initial: bool = False,
    ) -> None:
        """Write the body once per value of the index, with the index a literal.

        The substitution happens *before* the prefixing, which is what keeps the index out of
        the model: by the time an equation reaches :func:`_prefix_expr` its `i` is a number,
        so nothing has to teach name resolution about a name that is not a component.

        A range whose bounds are not known here is refused rather than deferred. How many
        equations a model has is not something a solver can be told later.
        """
        index, over = loop.index, loop.over
        if not isinstance(over, ir.Range):
            raise FlattenError(f"the range of 'for {index}' must be written 'a:b' or 'a:s:b'")
        bounds = [
            _integer(
                _prefix_expr(part, prefix, local, values=self.values),
                self.values,
                f"the range of 'for {index}'",
            )
            for part in (over.start, over.stop, over.step or ir.IntLiteral(1))
        ]
        start, stop, step = bounds
        if step == 0:
            raise FlattenError(f"the range of 'for {index}' steps by zero")
        for value in range(start, stop + (1 if step > 0 else -1), step):
            replacement = {index: ir.IntLiteral(value)}
            for inner in loop.body:
                self._equation(
                    _substitute_equation(inner, replacement), prefix, local, initial=initial
                )

    def _endpoint(self, ref: ir.Ref, prefix: _Path, local: frozenset[str]) -> _Endpoint:
        path = _prefix_ref(ref, prefix, local).parts
        role: _Role = "outside" if len(ref.parts) == 1 else "inside"
        if path not in self.connectors:
            raise FlattenError(f"connect() endpoint {'.'.join(path)!r} is not a connector")
        return (path, role)

    # -- connect expansion ------------------------------------------------------------

    def _connector_leaves(self, found: Found, scope: Scope) -> list[_ConnectorLeaf]:
        leaves: list[_ConnectorLeaf] = []
        for name, member in effective(found, scope).members.items():
            resolved = member.scope.resolve(member.component.type_name)
            if isinstance(resolved, Found) and resolved.cls.restriction != "type":
                leaves.extend(
                    _ConnectorLeaf((name, *leaf.path), leaf.kind, leaf.causality)
                    for leaf in self._connector_leaves(resolved, member.scope)
                )
            else:
                leaves.append(
                    _ConnectorLeaf((name,), member.component.connector, member.component.causality)
                )
        return leaves

    def expand_connections(self) -> None:
        for endpoints in self.connections.sets():
            self._expand_set(endpoints)
        for path, found in self.connectors.items():
            # An inside connector nobody connected: its flows have nowhere to go.
            if len(path) > 1 and not self.connections.connected((path, "inside")):
                for leaf in self._connector_leaves(found, Scope(self.program)):
                    if leaf.kind == "flow":
                        self.equations.append(
                            ir.Equal(ir.Ref((*path, *leaf.path)), ir.RealLiteral(0.0))
                        )

    def _matched_leaves(self, endpoints: list[_Endpoint]) -> list[_ConnectorLeaf]:
        """The scalars a connection set joins, after checking every end has the same ones.

        Modelica compares connectors by *structure*, not by class name. Two connectors
        can be connected when they carry the same scalar names with the same flow
        prefixes. Causality is deliberately not part of that comparison: a signal wire
        goes from an ``output`` connector to an ``input`` one, so demanding one class
        would make every signal connection illegal.
        """
        scope = Scope(self.program)
        first_path = endpoints[0][0]
        leaves = self._connector_leaves(self.connectors[first_path], scope)
        reference = sorted(leaf.shape for leaf in leaves)
        for path, _ in endpoints[1:]:
            other = self._connector_leaves(self.connectors[path], scope)
            if sorted(leaf.shape for leaf in other) != reference:
                first = self.connectors[first_path].cls.name
                second = self.connectors[path].cls.name
                raise FlattenError(
                    f"connect({'.'.join(first_path)}, {'.'.join(path)}) joins {first} to "
                    f"{second}, and the two do not carry the same variables"
                )
        return leaves

    def _one_source_only(self, endpoints: list[_Endpoint]) -> None:
        """A signal has one source, and this finds a set that was given two.

        Which end drives depends on the role as well as the prefix. A component's
        ``output`` drives the wire from inside the model. The enclosing model's own
        ``input`` drives it from outside, because the value arrives through the boundary.
        """
        scope = Scope(self.program)
        driving: list[str] = []
        for path, role in endpoints:
            drives: ir.Causality = "output" if role == "inside" else "input"
            driving += [
                ".".join((*path, *leaf.path))
                for leaf in self._connector_leaves(self.connectors[path], scope)
                if leaf.causality == drives
            ]
        if len(driving) > 1:
            raise FlattenError(
                f"connect() joins {' and '.join(driving)}, which all write the signal; "
                f"a signal has one source"
            )

    def _expand_set(self, endpoints: list[_Endpoint]) -> None:
        leaves = self._matched_leaves(endpoints)
        self._one_source_only(endpoints)
        for leaf in leaves:
            refs = [ir.Ref((*path, *leaf.path)) for path, _ in endpoints]
            if leaf.kind == "flow":
                self._flow_sum(refs, [role for _, role in endpoints])
            else:
                self.equations.extend(ir.Equal(a, b) for a, b in pairwise(refs))

    def _flow_sum(self, refs: list[ir.Ref], roles: list[_Role]) -> None:
        total: ir.Expr | None = None
        for ref, role in zip(refs, roles, strict=True):
            term: ir.Expr = ref if role == "inside" else ir.Unary("-", ref)
            total = term if total is None else ir.Binary("+", total, term)
        if total is not None:
            self.equations.append(ir.Equal(total, ir.RealLiteral(0.0)))


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def flatten(program: ir.Program, model: str) -> FlatModel:
    """Flatten class ``model`` (a dotted name) as found in ``program``."""
    scope = Scope(program)
    found = scope.resolve(model)
    if not isinstance(found, Found):
        raise FlattenError(f"{model!r} is a predefined type, not a model")
    if found.cls.restriction not in ("model", "block"):
        raise FlattenError(f"{model!r} is a {found.cls.restriction}, not a model")
    flattener = _Flattener(program)
    flattener.instantiate(found, scope, (), ir.EMPTY, "continuous")
    flattener.expand_connections()
    return FlatModel(
        name=found.cls.name,
        variables=tuple(flattener.variables),
        equations=tuple(flattener.equations),
        whens=tuple(flattener.whens),
        initial_equations=tuple(flattener.initial_equations),
    )
