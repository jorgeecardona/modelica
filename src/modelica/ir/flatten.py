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

import math
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import cache
from importlib.resources import files
from itertools import pairwise, product
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, cast

from modelica.ir import nodes as ir
from modelica.ir.clocks import as_when, partition, unhold
from modelica.ir.evaluate import EvaluationError, Value, at, evaluate
from modelica.ir.flat import (
    ArrayEquation,
    BaseType,
    CellError,
    FlatModel,
    FlatVariable,
    IndexSet,
    Loop,
    at_cell,
    element,
)
from modelica.ir.rewrite import names, substitute

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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

# Enumerations the language predefines but no file declares. `StateSelect` is the one the
# standard library leans on: a hint, set as `stateSelect = StateSelect.prefer` or declared
# `parameter StateSelect`, on which variables a tool should prefer as states. The hint has no
# effect here -- state selection is not implemented -- but the name has to resolve like any
# other enumeration, and every electrical and mechanical model carries it. Its members map to
# 1..5 the way every enumeration's literals do.
_PRELUDE: tuple[ir.Class, ...] = (
    ir.Class(
        name="StateSelect",
        restriction="type",
        literals=("never", "avoid", "default", "prefer", "always"),
    ),
)

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
    bindings: tuple[tuple[str, Found], ...] = ()
    """Class names this *instance* has had replaced, by ``redeclare package Medium = Water``.

    Per-instance, which is the whole difficulty and the reason this lives on the scope rather
    than on the class. A component redeclaration changes one declaration, and the instantiator
    already holds it as a value. A class redeclaration changes what a name *means inside a
    whole instance* -- every ``Medium.p``, at every depth -- and the same class instantiated
    twice with different media has to resolve the name two ways at once. A lexical chain
    cannot say that; a binding carried down the instantiation can.

    A pair list and not a mapping so the scope stays a comparable value like everything else
    here. There is never more than a handful.
    """

    def bound(self, extra: Mapping[str, Found]) -> Scope:
        """The same scope with ``extra`` replacing class names inside it and everything below."""
        if not extra:
            return self
        kept = tuple((name, found) for name, found in self.bindings if name not in extra)
        return replace(self, bindings=(*kept, *extra.items()))

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
                if any(found.chain[-1] is cls for cls in _bundled()):
                    raise FlattenError(
                        f"{name!r} is not in the bundled slice of the Modelica Standard Library"
                        f" ({found.cls.name!r} has no {part!r}). The slice has Units.SI,"
                        " Electrical.Analog, Mechanics.Translational and Rotational,"
                        " Thermal.HeatTransfer and part of Blocks; pass the full library as a"
                        " source to use the rest."
                    )
                raise FlattenError(f"{found.cls.name!r} has no nested class {part!r}")
            found = Found(nested, (nested, *found.chain))
        return found

    def _find_head(self, head: str) -> Found | None:
        # A redeclaration replaces the declaration, so it is consulted before the class that
        # declared the name it replaces -- otherwise the default would always win.
        for name, target in self.bindings:
            if name == head:
                return target
        for depth, cls in enumerate(self.chain):
            for nested in cls.classes:
                if nested.name == head:
                    return Found(nested, (nested, *self.chain[depth:]))
            found = self._imported(cls, head)
            if found is not None:
                return found
        for cls in self.program.classes:
            if cls.name == head:
                return Found(cls, (cls,))
        return None

    def _imported(self, cls: ir.Class, head: str) -> Found | None:
        """What ``head`` means here because an ``import`` said so.

        Checked *after* the class's own nested classes and before the enclosing ones, which
        is Modelica's order: a name declared in a class is not shadowed by something the
        same class imported, and an import does shadow what an outer class declares.

        The path is resolved from the top of the program and never relative to ``cls``, so
        the lookup starts from an empty chain. That is the rule, and it is also what keeps
        this from recursing: the root scope has no imports to consult.
        """
        # From the top, and with no bindings: an import is resolved against the program, and
        # a class a redeclaration put in scope is not importable from anywhere else.
        root = Scope(self.program)
        for entry in cls.imports:
            target = f"{entry.path}.{head}" if entry.wildcard else entry.path
            if not entry.wildcard and entry.name != head:
                continue
            try:
                found = root.resolve(target)
            except FlattenError:
                if entry.wildcard:
                    continue
                raise FlattenError(
                    f"{cls.name} imports {entry.path!r}, which does not resolve"
                ) from None
            if isinstance(found, Found):
                return found
        return None

    def inside(self, found: Found) -> Scope:
        """Step into ``found``, carrying whatever this instance has had redeclared.

        The bindings travel because they are a property of the *instance* being built, not of
        the class: a medium redeclared at the top of a fluid model has to still mean the same
        thing four components down.
        """
        return Scope(self.program, found.chain, self.bindings)

    def function(self, name: str) -> Found | None:
        """The ``function`` class this name refers to, or ``None`` if it is not one.

        Non-raising, unlike :meth:`resolve`, because the caller is asking a question about
        a call it has already parsed: ``sqrt(x)`` and ``der(v)`` name no class and must not
        become errors here. Whether an unresolved name is a built-in or a typo is settled
        further down, by the emitter, which is the stage that knows the built-ins.
        """
        try:
            found = self.resolve(name)
        except FlattenError:
            return None
        if isinstance(found, str) or found.cls.restriction != "function":
            return None
        return found


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


@dataclass(frozen=True)
class Inherited:
    """A modification an ``extends`` clause writes on a base member, and where it was written."""

    modification: ir.Modification
    scope: Scope
    """The class holding the ``extends`` clause: a function or constant in the value resolves here."""


@dataclass
class Member:
    component: ir.Component
    scope: Scope
    """Where the component was declared -- its type name resolves here."""
    mods: tuple[Inherited, ...] = ()
    """Modifications picked up from ``extends`` clauses, weakest first."""


@dataclass
class Effective:
    """One class with its whole ``extends`` chain folded in."""

    members: dict[str, Member] = field(default_factory=dict[str, "Member"])
    equations: list[ir.Equation] = field(default_factory=list[ir.Equation])
    initial_equations: list[ir.Equation] = field(default_factory=list[ir.Equation])
    writers: dict[int, Scope] = field(default_factory=dict[int, Scope])
    """The scope of the base class that wrote an inherited equation, by ``id`` of the equation.

    ``Adaptors`` writes ``u = Functions.state1({f, u1}, time)``: ``Functions`` resolves in the
    package of the base, not of a derived class in another package. An equation the class
    itself writes is not here.
    """


def _same_declaration(a: ir.Component, b: ir.Component) -> bool:
    """Whether two component declarations are the same element written twice, not a name clash.

    A derived class may re-state a component its base already declares, identically -- MSL's
    ``TwoPin`` declares ``SI.AngularVelocity omega`` that its base ``TwoPinElementary`` already has.
    That is the same variable, not a conflict, so flattening accepts it. A description or a diagram
    placement is cosmetic and left out of the comparison; everything semantic (the type, the
    modification, the prefixes) is kept, so a re-statement that actually differs is still a clash.
    """
    blank: dict[str, object] = {"comment": "", "annotation": ir.EMPTY}
    return replace(a, **blank) == replace(b, **blank)


def _extends_member(
    member: Member, picked_up: ir.Modification | None, writer: Scope, name: str
) -> Member:
    """A base member as an ``extends`` clause leaves it, given the modification the clause writes on it.

    A component redeclaration (``extends Base(redeclare final T2 comp)``) replaces the member's type
    the same way a redeclaration written at instantiation does -- the concrete machines redeclare a
    ``replaceable`` thermal port to the type that carries their own heat ports this way, and without
    it the base type is instantiated and the added ports are missing. Any other modification (a value,
    an attribute) is carried on top of the member instead, applied when it is instantiated. A class
    redeclaration is left on the mods for ``_rebind`` to read, not applied here.
    """
    if picked_up is None:
        return member
    if picked_up.redeclare is not None and not isinstance(picked_up.redeclare, ir.Class):
        return _redeclared(member, picked_up, writer, (name,))
    return replace(member, mods=(*member.mods, Inherited(picked_up, writer)))


def effective(found: Found, scope: Scope) -> Effective:  # noqa: C901 -- extends, redeclare, own members
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
            picked_up = extends.modification.entries.get(name)
            existing = result.members.get(name)
            if existing is not None:
                # The same element inherited through two `extends` paths -- MSL's `CellData` reaches
                # an identical `constant String CellType` through both a base record and an icon
                # record. Identical declarations are the one element, kept once, so the record is
                # accepted; declarations that differ are a real conflict. A modification this
                # `extends` writes on the name still applies, on top of the member already placed.
                if not _same_declaration(existing.component, member.component):
                    raise FlattenError(f"{found.cls.name} inherits {name!r} twice")
                result.members[name] = _extends_member(existing, picked_up, inner, name)
                continue
            result.members[name] = _extends_member(member, picked_up, inner, name)
        result.equations += base_effective.equations
        result.initial_equations += base_effective.initial_equations
        base_inner = inner.inside(base)
        for equation in (*base_effective.equations, *base_effective.initial_equations):
            result.writers[id(equation)] = base_effective.writers.get(id(equation), base_inner)
    for replacement in found.cls.redeclarations:
        # Applied here and nowhere else: this is the first moment the base chain is folded,
        # so it is the first moment there is an inherited member to replace. The other
        # spelling waits until instantiation because a modification is written by whoever
        # instantiates -- but both end in `_redeclared`, so there is one rule and not two.
        member = result.members.get(replacement.name)
        if member is None:
            raise FlattenError(
                f"{found.cls.name} redeclares {replacement.name!r}, which it does not inherit"
            )
        result.members[replacement.name] = _redeclared(
            member, ir.Modification(redeclare=replacement), inner, (replacement.name,)
        )
    for component in found.cls.components:
        existing = result.members.get(component.name)
        if existing is not None:
            # An identical re-statement of an inherited component is the same element, not a clash,
            # so it is kept once (the inherited member, whose extends-chain modifications survive).
            # A declaration that differs is a genuine conflict and still refused.
            if _same_declaration(existing.component, component):
                continue
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
        if current.cls.literals:
            # The specification says a literal *is* its position counted from one, so an
            # enumeration is an Integer that has been told its range. Emitting it as such
            # means nothing below here needs a fifth base type: `emit` puts every variable
            # in a float workspace, and an ordinal survives that unchanged.
            span = ir.Modification(
                entries={
                    "min": ir.Modification(ir.IntLiteral(1)),
                    "max": ir.Modification(ir.IntLiteral(len(current.cls.literals))),
                }
            )
            return "Integer", _merge(span, collected)
        if current.cls.restriction != "type":
            raise FlattenError(f"{current.cls.name!r} is not a type")
        if len(current.cls.extends) != 1 or current.cls.components:
            raise FlattenError(f"type {current.cls.name!r} must be a single extends")
        extends = current.cls.extends[0]
        # The alias's own attributes are defaults under anything already collected.
        collected = _merge(extends.modification, collected)
        current = scope.inside(current).resolve(extends.type_name)
    return current, collected


def _signal_base(found: Found, scope: Scope) -> tuple[BaseType, ir.Modification] | None:
    """A signal connector's predefined base, or ``None`` if it is an ordinary record connector.

    ``connector RealInput = input Real`` is a connector that *is* a scalar: it aliases a
    predefined type (directly, or through a ``type`` alias) and carries a causality, but holds no
    fields of its own. It lays down as one leaf variable the way a ``type`` alias does, rather
    than instantiating a record with no members -- which would fold its ``extends Real`` and be
    refused, since only ``type`` classes may extend a predefined type. A connector that extends
    another connector, or declares its own fields, is a record and returns ``None`` so it
    instantiates the usual way.
    """
    cls = found.cls
    if cls.restriction != "connector" or cls.components or len(cls.extends) != 1:
        return None
    inner = scope.inside(found)
    extends = cls.extends[0]
    target = inner.resolve(extends.type_name)
    if isinstance(target, str):
        return target, extends.modification
    if target.cls.restriction == "type":
        base, attributes = _base_of(target, inner)
        return base, _merge(extends.modification, attributes)
    return None


# --------------------------------------------------------------------------------------
# Reference prefixing
# --------------------------------------------------------------------------------------


def _prefix_ref(  # noqa: PLR0913 - values and outer are optional context
    ref: ir.Ref,
    prefix: _Path,
    local: frozenset[str],
    *,
    strict: bool = True,
    outer: Mapping[str, _Path] | None = None,
    values: Mapping[str, float] | None = None,
) -> ir.Ref:
    """The flat name of a written one: qualified by the instance path that holds it.

    ``outer`` is checked first and is the exception to the rule. An ``outer`` component is
    a *name* for a declaration somewhere further out, so it does not take this instance's
    prefix -- it takes the path the matching ``inner`` was declared at, whole. Checking it
    after ``local`` would qualify it into a variable that no declaration ever created.
    """
    if ref.subscripts:
        if values is None:
            raise FlattenError(
                f"{ref.name!r}: a subscript on an inner component is not supported yet"
            )
        ref = _elements(ref, prefix, local, values, outer)
    head = ref.parts[0].split("[", 1)[0]
    if outer and head in outer:
        return ir.Ref((*outer[head], *ref.parts[1:]))
    if head in local:
        return ir.Ref((*prefix, *ref.parts))
    if ref.parts == ("time",) or not strict:
        return ref
    raise FlattenError(f"unknown name {ref.name!r} in an equation{at(ref.span)}")


def _elements(
    ref: ir.Ref,
    prefix: _Path,
    local: frozenset[str],
    values: Mapping[str, float],
    outer: Mapping[str, _Path] | None,
) -> ir.Ref:
    """``r[1].n`` as the parts the array expansion declared: ``("r[1]", "n")``.

    Each index is structural, like a dimension: it picks which element the name means, so it
    is evaluated now against the parameters declared so far.
    """
    parts = tuple(
        element(
            part,
            tuple(
                _integer(
                    _prefix_expr(i, prefix, local, values=values, outer=outer),
                    values,
                    f"the subscript of {part!r} in {ref.name!r}",
                )
                for i in indices
            ),
        )
        if indices
        else part
        for part, indices in zip(ref.parts, ref.subscripts, strict=True)
    )
    return ir.Ref(parts)


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


def _expand_cat(node: ir.Expr, shapes: Mapping[_Path, tuple[int, ...]]) -> ir.Expr:
    """``cat(1, a, b, ...)`` of 1-D parts as one literal array, where every part's length is known.

    A part is an array literal or a reference to a 1-D array in ``shapes``. Anything else (a
    matrix, a dimension other than 1, a part of unknown length) leaves the call unchanged.
    """
    match node:
        case ir.Call(name="cat", args=(ir.IntLiteral(value=1), *parts)) if parts:
            elements: list[ir.Expr] = []
            for part in parts:
                match part:
                    case ir.Array(elements=inner):
                        elements.extend(inner)
                    case ir.Ref(parts=path) if len(shapes.get(tuple(path), ())) == 1:
                        (length,) = shapes[tuple(path)]
                        elements.extend(
                            ir.Ref((*path[:-1], element(path[-1], (i,))))
                            for i in range(1, length + 1)
                        )
                    case _:
                        return node
            return ir.Array(tuple(elements))
        case _:
            return node


def _resolve_subscripts(  # noqa: PLR0911 - one case per node type, as its neighbours have
    node: ir.Expr, values: Mapping[str, float]
) -> ir.Expr:
    """A prefixed binding with ``A[i]`` renamed to the element ``A[3]`` it names (#3).

    Equations get this rename while they are prefixed. A binding is prefixed without the
    parameter values, so its subscripts are resolved here instead. An ``if`` whose condition
    is a constant is folded first, so the branch that is not taken may hold an index that is
    out of range (``if i == 1 then 0 else 1 / A[i - 1]``). A subscript that is not a
    constant is left as it is.
    """

    def go(inner: ir.Expr) -> ir.Expr:
        return _resolve_subscripts(inner, values)

    match node:
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            try:
                taken = evaluate(condition, values)
            except EvaluationError:
                return ir.IfExpr(go(condition), go(then), go(otherwise))
            if isinstance(taken, bool):
                return go(then) if taken else go(otherwise)
            return ir.IfExpr(go(condition), go(then), go(otherwise))
        case ir.Subscript(base=ir.Ref(parts=path), indices=indices) if path:
            try:
                resolved = [int(evaluate(go(i), values)) for i in indices]  # type: ignore[arg-type]
            except (EvaluationError, TypeError, ValueError):
                return node
            return ir.Ref((*path[:-1], element(path[-1], resolved)))
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, go(lhs), go(rhs))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, go(operand))
        case ir.Call(name=name, args=args, named=named):
            return ir.Call(name, tuple(go(a) for a in args), tuple((k, go(v)) for k, v in named))
        case _:
            return node


def _call_names(node: ir.Expr) -> set[str]:
    """The names of every function called anywhere in ``node``."""
    match node:
        case ir.Call(name=name, args=args, named=named):
            found = {name}
            for argument in (*args, *(value for _, value in named)):
                found |= _call_names(argument)
            return found
        case ir.Binary(lhs=lhs, rhs=rhs):
            return _call_names(lhs) | _call_names(rhs)
        case ir.Unary(operand=operand):
            return _call_names(operand)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return _call_names(condition) | _call_names(then) | _call_names(otherwise)
        case _:
            return set()


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


MATRIX_RANK: Final = 2  # a matrix is two-dimensional; `transpose` swaps its two indices

# The dotted elementwise operators become their plain sibling once an element is read out: the
# whole point of `x .* y` was to be elementwise, and one element of it is an ordinary scalar `*`.
_DOTTED: Final[dict[str, ir.BinaryOp]] = {".*": "*", "./": "/", ".^": "^"}


def _element(  # noqa: C901, PLR0911, PLR0912 -- one case per node type, as its neighbours have
    node: ir.Expr, indices: tuple[int, ...], shapes: Mapping[_Path, tuple[int, ...]]
) -> ir.Expr:
    """One scalar element of an array-valued expression, at ``indices``.

    The counterpart of :func:`element` for equations rather than declarations: an equation
    between arrays is one equation *per element*, and this reads out that element by
    subscripting every array-valued reference and leaving scalars (parameters, ``time``, a
    literal) to broadcast. A reference is array-valued exactly when its flat path is a key of
    ``shapes`` -- the same map the declaration filled when it scalarised the component -- so a
    coefficient like ``a`` in ``der(x) = a * x`` is left alone while ``x`` becomes ``x[i]``.

    The structure carries the broadcast for free: ``der(x)`` becomes ``der(x[i])``,
    ``-x`` becomes ``-x[i]``, ``a * x`` becomes ``a * x[i]``. A ``*`` between two arrays is the
    one place an element is *not* a subscript: it is a contraction -- ``(A * x)[i]`` is the sum
    over ``j`` of ``A[i, j] * x[j]`` -- so that case builds the sum out of the elements of each
    side (a dot product, a matrix-vector or a matrix-matrix product, all the same rule).
    """

    def go(inner: ir.Expr) -> ir.Expr:
        return _element(inner, indices, shapes)

    match node:
        case ir.Ref(parts=parts) if parts in shapes:
            return ir.Ref((*parts[:-1], element(parts[-1], indices)))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, go(operand))
        case ir.Binary(op="*", lhs=lhs, rhs=rhs) if _infer_shape(lhs, shapes) and _infer_shape(
            rhs, shapes
        ):
            # A product of two arrays contracts the last dimension of the left with the first of
            # the right: the output index splits into the part that stays on the left and the part
            # that stays on the right, and the element is the sum over the shared dimension.
            left_shape = _infer_shape(lhs, shapes)
            contracted = left_shape[-1]
            left_out = indices[: len(left_shape) - 1]
            right_out = indices[len(left_shape) - 1 :]
            terms = [
                ir.Binary(
                    "*",
                    _element(lhs, (*left_out, j), shapes),
                    _element(rhs, (j, *right_out), shapes),
                )
                for j in range(1, contracted + 1)
            ]
            summed = terms[0]
            for term in terms[1:]:
                summed = ir.Binary("+", summed, term)
            return summed
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            # A dotted operator (`.*`/`./`/`.^`) reads out as its plain scalar sibling.
            return ir.Binary(_DOTTED.get(op, op), go(lhs), go(rhs))
        case ir.Call(name="sum" | "product", args=(arg,)) if _infer_shape(arg, shapes):
            # A reduction folds the whole array into one scalar, so it does not depend on the
            # equation's own index: `sum(x)` is `x[1] + x[2] + ... + x[n]` in every element it
            # appears in. `product` is the same fold with `*`. A 2-D argument folds every element.
            terms = [_element(arg, idx, shapes) for idx in _indices(_infer_shape(arg, shapes))]
            return _reduce_binary("+" if node.name == "sum" else "*", terms)
        case ir.Call(name="min" | "max", args=(arg,)) if _infer_shape(arg, shapes):
            # `min`/`max` of a single array is a reduction too, folded through the pairwise
            # `min(a, b)` / `max(a, b)` the interpreter and the derivative rules already handle.
            # (The two-argument `min(a, b)` form is left for the general call arm below.)
            terms = [_element(arg, idx, shapes) for idx in _indices(_infer_shape(arg, shapes))]
            return _reduce_call(node.name, terms)
        case ir.Call(name="transpose", args=(arg,)) if (
            len(_infer_shape(arg, shapes)) == MATRIX_RANK
        ):
            # `transpose(A)[i, j]` is `A[j, i]`: the first two indices swap.
            i, j, *rest = indices
            return _element(arg, (j, i, *rest), shapes)
        case ir.Call(name=name, args=args):
            return ir.Call(name, tuple(go(a) for a in args))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            # The condition selects the same arm for every element, so it stays whole; only
            # the two array-valued arms are read out.
            return ir.IfExpr(condition, go(then), go(otherwise))
        case ir.Array(elements=elements):
            first, *rest = indices
            return _element(elements[first - 1], tuple(rest), shapes)
        case ir.Matrix(rows=rows):
            row, column, *rest = indices
            return _element(rows[row - 1][column - 1], tuple(rest), shapes)
        case _:
            return node


def _indices(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Every one-based multi-index of an array of ``shape``, row-major."""
    return list(product(*(range(1, size + 1) for size in shape)))


def _reduce_binary(op: ir.BinaryOp, terms: list[ir.Expr]) -> ir.Expr:
    """Left-fold ``terms`` with a binary operator: ``[a, b, c]`` becomes ``(a op b) op c``."""
    result = terms[0]
    for term in terms[1:]:
        result = ir.Binary(op, result, term)
    return result


def _reduce_call(name: str, terms: list[ir.Expr]) -> ir.Expr:
    """Fold ``terms`` through a two-argument call: ``[a, b, c]`` becomes ``name(a, name(b, c))``."""
    result = terms[-1]
    for term in reversed(terms[:-1]):
        result = ir.Call(name, (term, result))
    return result


def _fold_reductions(  # noqa: PLR0911 -- one case per node type, as its neighbours have
    node: ir.Expr, shapes: Mapping[_Path, tuple[int, ...]]
) -> ir.Expr:
    """Expand every array reduction in ``node``, leaving the rest of its structure untouched.

    :func:`_element` folds a reduction as it scalarises an *equation*; this is the same fold for
    the scalar contexts an equation split never reaches -- a ``when`` condition, an assert -- so
    ``when sum(x) < 1`` and ``assert(sum(x) < top, ...)`` mean what they say instead of reaching
    the emitter as an untranslated ``sum``. Unlike ``_element`` this is safe on an array-valued
    expression: it touches only a reduction call (which is scalar) and copies a bare array
    reference or a product through unchanged, so applying it to both sides of an array equation
    before that equation is split does no harm.
    """
    match node:
        case ir.Call(name="sum" | "product", args=(arg,)) if _infer_shape(arg, shapes):
            folded = _fold_reductions(arg, shapes)
            terms = [_element(folded, idx, shapes) for idx in _indices(_infer_shape(arg, shapes))]
            return _reduce_binary("+" if node.name == "sum" else "*", terms)
        case ir.Call(name="min" | "max", args=(arg,)) if _infer_shape(arg, shapes):
            folded = _fold_reductions(arg, shapes)
            terms = [_element(folded, idx, shapes) for idx in _indices(_infer_shape(arg, shapes))]
            return _reduce_call(node.name, terms)
        case ir.Call(name=name, args=args):
            return ir.Call(name, tuple(_fold_reductions(a, shapes) for a in args), node.named)
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, _fold_reductions(lhs, shapes), _fold_reductions(rhs, shapes))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, _fold_reductions(operand, shapes))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(
                _fold_reductions(condition, shapes),
                _fold_reductions(then, shapes),
                _fold_reductions(otherwise, shapes),
            )
        case _:
            return node


def _merge_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """The shape two operands share, broadcasting a scalar against an array.

    Equal shapes are that shape; a scalar takes the other's; two different array shapes are a
    modelling error caught here rather than passed on as a wrong equation count.
    """
    if right in (left, ()):
        return left
    if left == ():
        return right
    raise FlattenError(f"array shapes {left} and {right} do not match in an equation")


def _infer_shape(  # noqa: C901, PLR0911, PLR0912 -- one node arm each
    node: ir.Expr, shapes: Mapping[_Path, tuple[int, ...]]
) -> tuple[int, ...]:
    """The array shape of an already-flattened expression, ``()`` for a scalar.

    A reference is array-valued when its flat path is a key of ``shapes``; ``der``/``pre``/
    ``noEvent``/``smooth`` and a function call pass a shape through. ``+`` and ``-`` between two
    arrays are elementwise -- the two shapes must match -- and broadcast a scalar against an array.
    A ``*`` between two arrays is a contraction -- the last dimension of the left runs against the
    first of the right and both drop out, so a vector times a vector is a scalar, a matrix times a
    vector a vector, a matrix times a matrix a matrix. Array ``/`` and ``^`` have no elementwise
    reading (Modelica spells those ``./`` and ``.^``) and are refused.
    """
    match node:
        case ir.Ref(parts=parts):
            return shapes.get(parts, ())
        case ir.Unary(operand=operand):
            return _infer_shape(operand, shapes)
        case ir.Call(name="der" | "pre" | "noEvent" | "smooth", args=args) if args:
            return _infer_shape(args[-1], shapes)
        case ir.Call(name="sum" | "product", args=(arg,)) if _infer_shape(arg, shapes):
            return ()  # a reduction over an array is one scalar
        case ir.Call(name="min" | "max", args=(arg,)) if _infer_shape(arg, shapes):
            return ()  # min/max of a single array is a scalar reduction
        case ir.Call(name="transpose", args=(arg,)) if (
            len(_infer_shape(arg, shapes)) == MATRIX_RANK
        ):
            rows, columns = _infer_shape(arg, shapes)
            return (columns, rows)  # transpose swaps the two dimensions of a matrix
        case ir.Call(name="size", args=(_, _)):
            return ()  # `size(a, k)` is the scalar length of one dimension, not an array
        case ir.Call(name="size", args=(array,)):
            return (len(_infer_shape(array, shapes)),)  # `size(a)` is the vector of dim lengths
        case ir.Call(args=args):
            shape: tuple[int, ...] = ()
            for argument in args:
                shape = _merge_shape(shape, _infer_shape(argument, shapes))
            return shape
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            left, right = _infer_shape(lhs, shapes), _infer_shape(rhs, shapes)
            if left and right:
                # `*` between two arrays contracts; `+`/`-` and the dotted operators `.*`/`./`/`.^`
                # are elementwise (the shared shape, via the merge below); a plain `/`/`^` has no
                # elementwise reading and is refused, naming its dotted form.
                if op == "*":
                    if left[-1] != right[0]:
                        raise FlattenError(
                            f"array shapes {left} and {right} do not conform for a product"
                            f" ({left[-1]} against {right[0]})"
                        )
                    return (*left[:-1], *right[1:])
                if op not in ("+", "-", ".*", "./", ".^"):
                    raise FlattenError(
                        f"an array `{op}` is not supported; `+`/`-`/`.*`/`./`/`.^` are elementwise"
                        f" and `*` is a product -- write `.{op}` for an elementwise `{op}`"
                    )
            return _merge_shape(left, right)
        case ir.IfExpr(then=then, otherwise=otherwise):
            return _merge_shape(_infer_shape(then, shapes), _infer_shape(otherwise, shapes))
        case ir.Array(elements=elements):
            inner = _infer_shape(elements[0], shapes) if elements else ()
            return (len(elements), *inner)
        case ir.Matrix(rows=rows):
            return (len(rows), len(rows[0]) if rows else 0)
        case _:
            return ()


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


def _boolean(node: ir.Expr, values: Mapping[str, float], what: str) -> bool:
    """Evaluate ``node`` to a Boolean, or say why it is not one.

    The condition on a conditional component is structural the way a shape is: it decides
    whether the component's variables and equations exist at all, so it has to be known
    here and not at run time.
    """
    try:
        value = evaluate(node, values)
    except EvaluationError as failure:
        raise FlattenError(f"{what} must be known when the model is flattened: {failure}") from None
    if not isinstance(value, bool):
        raise FlattenError(f"{what} must be a Boolean, and this one is {value!r}")
    return value


def _range_values(
    over: ir.Range,
    resolve: Callable[[ir.Expr], ir.Expr],
    values: Mapping[str, float],
    what: str,
) -> list[int]:
    """The integers a literal range enumerates, its bounds resolved and read as whole numbers.

    A range decides how many equations or elements exist -- a ``for`` header, a comprehension
    iterator, an array slice -- so its bounds are structural and known here; ``resolve`` prefixes
    each bound before it is read, and a zero step is refused rather than looped forever.
    """
    lo = _integer(resolve(over.start), values, what)
    hi = _integer(resolve(over.stop), values, what)
    step = _integer(resolve(over.step or ir.IntLiteral(1)), values, what)
    if step == 0:
        raise FlattenError(f"{what} steps by zero")
    return list(range(lo, hi + (1 if step > 0 else -1), step))


def _expand_comprehension(
    node: ir.Comprehension, prefix: Callable[[ir.Expr], ir.Expr], values: Mapping[str, float]
) -> ir.Expr:
    """A comprehension as a plain array: one element per combination of its iterators.

    Expanded where the ranges can be read, each iterator a literal substituted into the body
    before it is prefixed -- the same move :meth:`_Flattener._unroll` makes for a ``for`` equation.
    Nested for more than one iterator, so a two-index comprehension is a matrix. Downstream then
    sees an array, so ``sum({f(i) for i in 1:n})`` folds through the ordinary reduction of one.
    """
    ranges: list[tuple[str, list[int]]] = []
    for name, over in node.iterators:
        if not isinstance(over, ir.Range):
            raise FlattenError(f"the range of {name!r} in a comprehension must be 'a:b'")
        what = f"the range of {name!r} in a comprehension"
        ranges.append((name, _range_values(over, prefix, values, what)))

    def build(bound: dict[str, ir.Expr], rest: list[tuple[str, list[int]]]) -> ir.Expr:
        if not rest:
            return prefix(_substitute(node.body, bound))
        name, indices = rest[0]
        return ir.Array(tuple(build({**bound, name: ir.IntLiteral(i)}, rest[1:]) for i in indices))

    return build({}, ranges)


def _ends(indices: tuple[ir.Expr, ...], path: _Path) -> tuple[ir.Expr, ...]:
    """``indices`` with each ``end`` replaced by the size of the dimension it indexes.

    An ``end`` inside a nested subscript belongs to that subscript and is left for it.
    """

    def fill(node: ir.Expr, size: int) -> ir.Expr:  # noqa: PLR0911 - one case per node
        match node:
            case ir.End():
                return ir.IntLiteral(size)
            case ir.Subscript():
                return node
            case ir.Binary(op=op, lhs=lhs, rhs=rhs):
                return ir.Binary(op, fill(lhs, size), fill(rhs, size))
            case ir.Unary(op=op, operand=operand):
                return ir.Unary(op, fill(operand, size))
            case ir.Range(start=start, stop=stop, step=step):
                return ir.Range(
                    fill(start, size), fill(stop, size), None if step is None else fill(step, size)
                )
            case ir.Call(name=name, args=args, named=named):
                return ir.Call(name, tuple(fill(a, size) for a in args), named)
            case _:
                return node

    if not any(_has_end(index) for index in indices):
        return indices
    shape = _SHAPES.get().get(path)
    if shape is None or len(shape) < len(indices):
        message = f"`end` in a subscript of {'.'.join(path)!r}, whose shape is not known here"
        raise FlattenError(message)
    return tuple(fill(index, size) for index, size in zip(indices, shape, strict=False))


def _has_end(node: ir.Expr) -> bool:  # noqa: PLR0911 - one case per node
    match node:
        case ir.End():
            return True
        case ir.Subscript():
            return False
        case ir.Binary(lhs=lhs, rhs=rhs):
            return _has_end(lhs) or _has_end(rhs)
        case ir.Unary(operand=operand):
            return _has_end(operand)
        case ir.Range(start=start, stop=stop, step=step):
            return _has_end(start) or _has_end(stop) or (step is not None and _has_end(step))
        case ir.Call(args=args):
            return any(_has_end(a) for a in args)
        case _:
            return False


def _slice(
    path: _Path,
    indices: tuple[ir.Expr, ...],
    resolve: Callable[[ir.Expr], ir.Expr],
    values: Mapping[str, float],
) -> ir.Expr:
    """An array slice as a nested array literal: one array level per range dimension.

    ``x[1:m]`` becomes ``{x[1], ..., x[m]}`` and ``A[1:2, 3]`` becomes ``{A[1, 3], A[2, 3]}`` -- a
    range dimension adds a level and enumerates, a scalar dimension is held fixed. So the slice has
    the shape of its ranges, and :func:`_element` reads it out exactly as it reads any array, which
    makes an equation ``x[1:m] = zeros(m)`` split into one scalar equation per element like the rest.
    """
    what = f"the subscript of {path[-1]!r}"

    def build(dim: int, fixed: tuple[int, ...]) -> ir.Expr:
        if dim == len(indices):
            return ir.Ref((*path[:-1], element(path[-1], fixed)))
        index = indices[dim]
        if isinstance(index, ir.Range):
            return ir.Array(
                tuple(
                    build(dim + 1, (*fixed, value))
                    for value in _range_values(index, resolve, values, what)
                )
            )
        return build(dim + 1, (*fixed, _integer(resolve(index), values, what)))

    return build(0, ())


def _subscripts(node: ir.Expr) -> Iterator[ir.Subscript]:
    """Every ``Subscript`` inside ``node``."""
    match node:
        case ir.Subscript():
            yield node
        case ir.Binary(lhs=lhs, rhs=rhs):
            yield from _subscripts(lhs)
            yield from _subscripts(rhs)
        case ir.Unary(operand=operand):
            yield from _subscripts(operand)
        case ir.Call(args=args):
            for arg in args:
                yield from _subscripts(arg)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            for part in (condition, then, otherwise):
                yield from _subscripts(part)
        case _:
            return


def _prefix_expr(  # noqa: C901, PLR0911, PLR0912, PLR0913 -- one case per node type, one flag per rule
    node: ir.Expr,
    prefix: _Path,
    local: frozenset[str],
    *,
    strict: bool = True,
    values: Mapping[str, float] | None = None,
    outer: Mapping[str, _Path] | None = None,
) -> ir.Expr:
    def go(inner: ir.Expr) -> ir.Expr:
        return _prefix_expr(inner, prefix, local, strict=strict, values=values, outer=outer)

    match node:
        case ir.Ref():
            return _prefix_ref(node, prefix, local, strict=strict, outer=outer, values=values)
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
            if values is None or (not strict and base.parts[0] not in local and not outer):
                # Not strict, an unknown base would be kept and renamed into a name nothing
                # declares (an inliner's leftover local `v_[1]`), so it is refused instead.
                raise FlattenError("subscripted references are not expanded here")
            path = _prefix_ref(base, prefix, local, strict=strict, outer=outer, values=values).parts
            indices = _ends(indices, path)
            if any(isinstance(i, ir.Range) for i in indices):
                # A range subscript is a slice -- a sub-array, not one element -- so it becomes an
                # array literal (one level per range dimension), which the equation split then reads
                # element by element like any other array. `i[1:m] = zeros(m)` is m scalar equations.
                return _slice(path, indices, go, values)
            try:
                resolved = [
                    _integer(go(i), values, f"the subscript of {base.name!r}") for i in indices
                ]
            except FlattenError:
                # An index that reads a loop index left local (`_unroll_once`) stays symbolic:
                # `at_cell` names the element once the index has a value.
                carried = tuple(go(i) for i in indices)
                if any(set(names(i)) & local for i in carried):
                    return ir.Subscript(ir.Ref(path), carried)
                raise
            return ir.Ref((*path[:-1], element(path[-1], resolved)))
        case ir.Array(elements=elements):
            # `when {sampleTrigger, initial()}`: each element is prefixed like any expression.
            return ir.Array(tuple(go(e) for e in elements))
        case ir.Comprehension():
            if values is None:
                raise FlattenError("a comprehension is not expanded here")
            return _expand_comprehension(node, go, values)
        case ir.Range():
            raise FlattenError("a range is read only in the header of a 'for' equation")
        case _:
            return node


def _substitute(  # noqa: C901, PLR0911 -- one case per node type
    node: ir.Expr, values: Mapping[str, ir.Expr]
) -> ir.Expr:
    """Replace whole references by their dotted name.

    Single-part keys are what a loop index needs; dotted ones are what an algorithm section
    needs, because a statement may assign a component's variable (``c.v := ...``). A
    single-part key still matches only a single-part reference, so the loop index case is
    unchanged by the generalisation.
    """

    def go(inner: ir.Expr) -> ir.Expr:
        return _substitute(inner, values)

    match node:
        case ir.Ref(parts=parts) if ".".join(parts) in values:
            return values[".".join(parts)]
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
        case ir.Matrix(rows=rows):
            return ir.Matrix(tuple(tuple(go(e) for e in row) for row in rows))
        case ir.Comprehension(body=body, iterators=iterators):
            return ir.Comprehension(
                go(body),
                tuple((n, None if over is None else go(over)) for n, over in iterators),
            )
        case ir.Range(start=start, stop=stop, step=step):
            return ir.Range(go(start), go(stop), None if step is None else go(step))
        case _:
            return node


def _substitute_statement(node: ir.Statement, values: Mapping[str, ir.Expr]) -> ir.Statement:
    """Substitution inside the body of a ``when``, which is where a statement can reach here.

    Only the two simple forms are handled, and that is not a shortcut: a ``when`` in an
    equation section takes assignments and ``reinit`` and nothing else, so the compound
    statements cannot occur. They are refused rather than ignored, because the day flattening
    learns to unroll an algorithm section this function will be the first thing to be wrong.
    """
    match node:
        case ir.Assign(target=target, value=value):
            return ir.Assign(target, _substitute(value, values))
        case ir.Reinit(target=target, value=value):
            return ir.Reinit(target, _substitute(value, values))
        case ir.CallStatement(call=ir.Call(name="terminate", args=args, named=named), span=span):
            return ir.CallStatement(
                (), ir.Call("terminate", tuple(_substitute(a, values) for a in args), named), span
            )
        case _:  # pragma: no cover - unreachable while `when` bodies are simple statements
            raise FlattenError(
                f"{type(node).__name__} inside a 'for' equation is not supported yet"
            )


def _substitute_equation(  # noqa: C901, PLR0911 -- one case per equation form
    node: ir.Equation, values: Mapping[str, ir.Expr]
) -> ir.Equation:
    """The same substitution, over an equation and anything nested inside it."""

    def go(inner: ir.Expr) -> ir.Expr:
        return _substitute(inner, values)

    match node:
        case ir.Equal(lhs=lhs, rhs=rhs, span=span):
            return ir.Equal(go(lhs), go(rhs), span)
        case ir.Assert(condition=condition, message=message, level=level, span=span):
            return ir.Assert(go(condition), go(message), None if level is None else go(level), span)
        case ir.CallEquation(targets=targets, call=call, span=span):
            # The targets are names of the enclosing class and the loop index is not one of
            # them, so only the arguments can carry it. A subscripted target -- `(a[i], b) =
            # f(i)` -- would be the case that needs more, and the parser reads a target as a
            # plain reference, so it cannot arise yet.
            substituted = go(call)
            if not isinstance(substituted, ir.Call):  # pragma: no cover - a call stays a call
                raise FlattenError("an output list is assigned from a function call")
            return ir.CallEquation(targets, substituted, span)
        case ir.Connect(a=a, b=b):
            # `connect(pin[i], other[i])` in a for-loop: the loop index rides on the endpoints, which
            # are indexed (a `Subscript`). Substituting it into each turns the loop into one connect
            # per value -- `connect(pin[1], other[1])` and its kin -- which the endpoint resolver reads
            # like any other. An endpoint that does not come out a component reference (a subscript on
            # a literal, say) is refused rather than passed on as something the resolver cannot use.
            new_a, new_b = go(a), go(b)
            if not isinstance(new_a, (ir.Ref, ir.Subscript)) or not isinstance(
                new_b, (ir.Ref, ir.Subscript)
            ):
                raise FlattenError("a connect() endpoint in a 'for' equation must name a connector")
            return ir.Connect(new_a, new_b)
        case ir.When(branches=branches):
            return ir.When(
                tuple(
                    ir.WhenBranch(
                        go(branch.condition),
                        tuple(_substitute_statement(s, values) for s in branch.body),
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


#: The one-argument and two-argument functions the interpreter knows, so a qualified call to one --
#: MSL writes ``Modelica.Math.asin`` -- can be read as the builtin when a constant's value is
#: evaluated here.
_BUILTIN_FUNCTIONS = frozenset(
    {
        "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "exp", "log", "log10", "abs", "sign", "ceil", "floor", "integer",
        "atan2", "mod", "div", "rem", "min", "max",
    }
)  # fmt: skip


def _debuiltin(node: ir.Expr) -> ir.Expr:
    """Read a qualified call to a builtin -- ``Modelica.Math.asin(x)`` -- as the builtin itself.

    A package constant's value is often an expression over the library's own maths, ``pi = 2 *
    Modelica.Math.asin(1.0)``; the interpreter knows ``asin``, not the qualified name, so the prefix
    is dropped where the last part is a function it knows. Anything else is left whole, so the
    evaluation that follows fails cleanly on a name that is not a builtin, and the constant is left
    unresolved rather than resolved wrong.
    """
    match node:
        case ir.Call(name=name, args=args, named=named):
            tail = name.rsplit(".", 1)[-1]
            resolved = tail if "." in name and tail in _BUILTIN_FUNCTIONS else name
            args = tuple(_debuiltin(a) for a in args)
            named = tuple((key, _debuiltin(value)) for key, value in named)
            return ir.Call(resolved, args, named)
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, _debuiltin(lhs), _debuiltin(rhs))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, _debuiltin(operand))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(_debuiltin(condition), _debuiltin(then), _debuiltin(otherwise))
        case _:
            return node


_CONSTANT_DEPTH: Final = 32
"""How many package hops a constant's value may take before resolution gives up.

A same-package chain is bounded by ``seen`` (a name resolved once is not chased again). A chain that
crosses into another package resets that name set -- ``Constants.eps`` and ``Machine.eps`` are two
constants with the same simple name, and one names the other -- so a separate hop count bounds the
crossing instead, against a pathological library whose packages name each other in a ring.
"""


def _resolve_local_constants(  # noqa: C901, PLR0911 -- one case per node type and lookup source
    node: ir.Expr, cls: ir.Class, seen: frozenset[str], program: ir.Program, depth: int
) -> ir.Expr:
    """Substitute a reference to a constant, in this package or another, with its value.

    ``D2R = pi / 180`` names ``pi``, another constant beside it, so the value cannot be evaluated
    until ``pi`` is put in. ``eps = ModelicaServices.Machine.eps`` names a constant of another
    package, reached by its qualified name from the top of the program. ``seen`` bounds a same-package
    chain (a constant that names itself, directly or in a ring, is left unresolved); ``depth`` bounds
    a chain that crosses packages, where the simple names may repeat and ``seen`` cannot.
    """

    def go(inner: ir.Expr) -> ir.Expr:
        return _resolve_local_constants(inner, cls, seen, program, depth)

    match node:
        case ir.Ref(parts=(single,)) if single not in seen:
            literal = _constant_literal(cls, single, program, seen, depth)
            return literal if literal is not None else node
        case ir.Ref(parts=parts) if len(parts) > 1 and depth > 0:
            try:
                found = Scope(program).resolve(".".join(parts[:-1]))
            except FlattenError:
                return node
            if not isinstance(found, Found):
                return node
            if parts[-1] in found.cls.literals:
                return ir.IntLiteral(found.cls.literals.index(parts[-1]) + 1)
            literal = _constant_literal(found.cls, parts[-1], program, frozenset(), depth - 1)
            return literal if literal is not None else node
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, go(lhs), go(rhs))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, go(operand))
        case ir.Call(name=fname, args=args, named=named):
            return ir.Call(fname, tuple(go(a) for a in args), tuple((k, go(v)) for k, v in named))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(go(condition), go(then), go(otherwise))
        case _:
            return node


def _constant_literal(  # noqa: PLR0911 -- one per kind
    cls: ir.Class,
    name: str,
    program: ir.Program,
    seen: frozenset[str] = frozenset(),
    depth: int = _CONSTANT_DEPTH,
) -> ir.Expr | None:
    """A package constant as its value, or ``None`` when it does not reduce to one here.

    ``Modelica.Constants.pi`` is a ``constant`` declared in a package, not a component of the model,
    so it cannot be prefixed into a variable; its value is read off the declaration, its qualified
    builtin calls stripped (``Modelica.Math.asin`` as ``asin``), any constant it names substituted
    -- one beside it (``D2R = pi / 180``) or one in another package (``eps =
    ModelicaServices.Machine.eps``) -- and evaluated to a number, then put in place. A value that
    will not reduce to a number here -- one that names a function that is not a builtin, or a
    constant that itself does not reduce -- is left alone, so the reference stays whatever it was and
    the ordinary "unknown name" report still catches a typo.
    """
    for component in cls.components:
        if component.name != name or component.variability != "constant":
            continue
        value = component.modification.value
        if value is None:
            return None
        resolved = _resolve_local_constants(_debuiltin(value), cls, seen | {name}, program, depth)
        try:
            result = evaluate(resolved, {})
        except EvaluationError:
            return None
        if isinstance(result, bool):
            return ir.BoolLiteral(result)
        if isinstance(result, int):
            return ir.IntLiteral(result)
        if isinstance(result, float):
            return ir.RealLiteral(result)
        return None
    return None


def _imported_constant(scope: Scope, head: str) -> ir.Expr | None:  # noqa: C901 -- named, wildcard, enclosing
    """A bare name an ``import`` (or an enclosing package) makes a package constant, as its value.

    ``import Modelica.Constants.pi;`` then ``pi`` in an equation, or ``import Modelica.Constants.*;``
    then ``pi``, names a ``constant`` declared in a package. The qualified case above reads it off a
    multi-part reference; this reads it off the simple name the import brought into scope. The
    package it lives in is resolved from the top of the program, the way an imported class is, and the
    constant is then read off it -- ``None`` when nothing in scope introduces the name or its value
    will not reduce to a number here, so the ordinary "unknown name" report still catches a typo.

    An enumeration literal brought in the same way (``import P.Colors.*;`` then ``red``) is its
    position, matching what the qualified case does. Imports are consulted innermost class first, so
    an inner import shadows an outer one, and a class's own constant only after its imports.
    """
    root = Scope(scope.program)

    def read(package: str, member: str) -> ir.Expr | None:
        try:
            found = root.resolve(package)
        except FlattenError:
            return None
        if not isinstance(found, Found):
            return None
        if member in found.cls.literals:
            return ir.IntLiteral(found.cls.literals.index(member) + 1)
        return _constant_literal(found.cls, member, scope.program)

    for cls in scope.chain:
        for entry in cls.imports:
            if entry.wildcard:
                package, member = entry.path, head
            elif entry.name == head:
                package, _, member = entry.path.rpartition(".")
            else:
                continue
            resolved = read(package, member)
            if resolved is not None:
                return resolved
    # An enclosing class's own constant, reachable by its simple name -- but not the innermost class,
    # whose own constants are components of the instance and are left for prefixing (they are in
    # `local`). `chain[0]` is that class, so start one out.
    for cls in scope.chain[1:]:
        resolved = _constant_literal(cls, head, scope.program)
        if resolved is not None:
            return resolved
    return None


def _literals(  # noqa: C901, PLR0911, PLR0912 -- one case per node type, as its neighbours have
    node: ir.Expr, scope: Scope, local: frozenset[str]
) -> ir.Expr:
    """``Mode.day`` becomes ``1`` -- an enumeration literal is its position, counted from one.

    Done here, before prefixing, because ``Mode.day`` is not a component reference and
    prefixing would qualify it into a variable that does not exist. A name that *is* a local
    component is left alone whatever else it might resolve to, which is Modelica's rule and
    also the only reading that cannot surprise anybody.
    """

    def go(inner: ir.Expr) -> ir.Expr:
        return _literals(inner, scope, local)

    match node:
        case ir.Ref(parts=parts) if len(parts) > 1 and parts[0] not in local:
            try:
                found = scope.resolve(".".join(parts[:-1]))
            except FlattenError:
                return node
            if isinstance(found, Found):
                if parts[-1] in found.cls.literals:
                    return ir.IntLiteral(found.cls.literals.index(parts[-1]) + 1)
                constant = _constant_literal(found.cls, parts[-1], scope.program)
                if constant is not None:
                    return constant
            return node
        case ir.Ref(parts=[head]) if head not in local:
            imported = _imported_constant(scope, head)
            return imported if imported is not None else node
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
        case ir.Matrix(rows=rows):
            return ir.Matrix(tuple(tuple(go(e) for e in row) for row in rows))
        case ir.Comprehension(body=body, iterators=iterators):
            return ir.Comprehension(
                go(body),
                tuple((n, None if over is None else go(over)) for n, over in iterators),
            )
        case _:
            return node


def _fold_ifs[T](node: T, known: Mapping[str, Value]) -> T:
    """``node`` with each if-expression whose condition ``known`` decides replaced by its branch.

    Walks any IR dataclass (equations, ``when`` bodies, expressions) by its fields.
    """
    if isinstance(node, ir.IfExpr):
        try:
            decided = evaluate(node.condition, known)
        except EvaluationError:
            decided = None
        if isinstance(decided, bool):
            return cast("T", _fold_ifs(node.then if decided else node.otherwise, known))
    if not is_dataclass(node) or isinstance(node, type):
        return node
    changes: dict[str, object] = {}
    for item in fields(node):
        value: object = getattr(node, item.name)
        if is_dataclass(value) and not isinstance(value, type):
            folded = _fold_ifs(value, known)
        elif isinstance(value, tuple) and value:
            folded = tuple(
                _fold_ifs(element, known) for element in cast("tuple[object, ...]", value)
            )
            if all(a is b for a, b in zip(folded, cast("tuple[object, ...]", value), strict=True)):
                continue
        else:
            continue
        if folded is not value:
            changes[item.name] = folded
    return cast("T", replace(node, **changes)) if changes else node  # pyright: ignore[reportArgumentType]


def _literal_modification(
    mod: ir.Modification, scope: Scope, local: frozenset[str]
) -> ir.Modification:
    """The same rewrite over a modification, which is where ``parameter Mode m = Mode.night`` is.

    User function calls in the value are inlined too, as they are in equations, so a binding
    ``parameter Real b = sq(a)`` reaches emit as an expression and not as a call (#2).
    """
    if not mod:
        return mod
    return ir.Modification(
        value=None
        if mod.value is None
        else _inline_calls(_literals(mod.value, scope, local), scope),
        entries={
            name: _literal_modification(entry, scope, local) for name, entry in mod.entries.items()
        },
        redeclare=mod.redeclare,
    )


# --------------------------------------------------------------------------------------
# Function calls
#
# A user-written `function` is inlined here rather than emitted as a callable, and that is
# the decision this section exists to record.
#
# The alternative -- keep the call, emit a Python function beside the model -- fails at
# `modelica.ir.derive`, which differentiates expressions and cannot differentiate a
# statement sequence. A model whose state derivative went through such a call would lose
# its analytic Jacobian, which is the one thing `docs/architecture.md:120` says this
# compiler does not give up. Inlining sidesteps that entirely: what comes back is an
# ordinary expression, and every stage below already knows how to differentiate one.
#
# The price is that only the inlinable subset of Modelica's function bodies is accepted --
# assignment and `if`, no loops, no early exit. Each refusal below says which one it hit.
# --------------------------------------------------------------------------------------

_INLINE_DEPTH: Final = 16

_SPELLING: Final[dict[type, str]] = {
    ir.Assert: "'assert'",
    ir.When: "a 'when'",
    ir.CallStatement: "an output list assignment",
    ir.ForStatement: "a 'for' loop",
    ir.WhileStatement: "a 'while' loop",
    ir.Break: "'break'",
    ir.Return: "'return'",
    ir.Reinit: "'reinit'",
}


def _homotopy_positional(call: ir.Call) -> ir.Call:
    """``homotopy(actual = A, simplified = S)`` as ``homotopy(A, S)``.

    The standard library writes homotopy with named arguments, but every consumer of the call -- the
    derivative, the emitter, the evaluator -- reads the actual off the first positional argument and
    the simplified off the second. The names are resolved to that order once, here, so none of them
    needs to know the call could be named. A call that is not the ``(actual, simplified)`` shape is
    left as it is, to be refused where it is used rather than reshaped into something wrong.
    """
    if not call.named:
        return call
    by_name = dict(call.named)
    actual = call.args[0] if call.args else by_name.get("actual")
    simplified = call.args[1] if len(call.args) > 1 else by_name.get("simplified")
    if actual is None or simplified is None:
        return call
    return ir.Call("homotopy", (actual, simplified), ())


def _inline_calls(  # noqa: C901, PLR0911 -- one case per node type, as its neighbours have
    node: ir.Expr, scope: Scope, depth: int = 0
) -> ir.Expr:
    """Replace every call to a user-written ``function`` with the expression it computes."""

    def go(inner: ir.Expr) -> ir.Expr:
        return _inline_calls(inner, scope, depth)

    match node:
        case ir.Call(name=name, args=args, named=named):
            call = ir.Call(name, tuple(go(a) for a in args), tuple((k, go(v)) for k, v in named))
            if name == "homotopy":
                call = _homotopy_positional(call)
            if name.startswith("."):
                # Global scope: a top-level function of that name, else the built-in.
                found = Scope(scope.program).function(name[1:])
                call = ir.Call(name[1:], call.args, call.named)
            else:
                found = scope.function(name)
            return call if found is None else _inline(found, scope, call, depth)
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, go(lhs), go(rhs))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, go(operand))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(go(condition), go(then), go(otherwise))
        case ir.Array(elements=elements):
            return ir.Array(tuple(go(e) for e in elements))
        case ir.Matrix(rows=rows):
            return ir.Matrix(tuple(tuple(go(e) for e in row) for row in rows))
        case ir.Comprehension(body=body, iterators=iterators):
            return ir.Comprehension(
                go(body),
                tuple((n, None if over is None else go(over)) for n, over in iterators),
            )
        case ir.Subscript(base=base, indices=indices):
            return ir.Subscript(base, tuple(go(i) for i in indices))
        case _:
            return node


_EXTERNAL_INTRINSICS: Final = frozenset({"Modelica.Utilities.Strings.substring"})
"""External functions kept as calls, by qualified name, for `evaluate` to compute."""


def _inline(found: Found, scope: Scope, call: ir.Call, depth: int) -> ir.Expr:
    """One call, replaced by the expression its body assigns to the output."""
    name = found.cls.name
    if depth >= _INLINE_DEPTH:
        raise FlattenError(f"{name!r} is called more than {_INLINE_DEPTH} deep; is it recursive?")
    inner = scope.inside(found)
    merged = effective(found, inner)
    if merged.equations or merged.initial_equations:
        raise FlattenError(f"function {name!r} has an equation section; a function has a body")
    qualified = ".".join(c.name for c in reversed(found.chain))
    if not found.cls.algorithms and qualified in _EXTERNAL_INTRINSICS:
        # Written in C, so there is no body; `evaluate` computes it once its arguments are known.
        return ir.Call(qualified, tuple(_inline_calls(a, scope, depth + 1) for a in call.args))
    if len(found.cls.algorithms) != 1:
        raise FlattenError(
            f"function {name!r} needs exactly one algorithm section, not {len(found.cls.algorithms)}"
        )

    components = [m.component for m in merged.members.values()]
    inputs = [c for c in components if c.causality == "input"]
    outputs = [c for c in components if c.causality == "output"]
    if not outputs:
        raise FlattenError(f"function {name!r} declares no output; there is nothing to inline")

    environment = _bind(name, inputs, call)
    for component in components:
        if component.causality != "input" and component.modification.value is not None:
            environment[component.name] = component.modification.value
    environment = _sequence(found.cls.algorithms[0], environment, f"function {name!r}")

    # A function with more than one output, called where a single value is expected, yields its
    # first output; the rest are computed by the same run of the body and then discarded. (The
    # tuple form `(a, b) = f(x)` reads every output instead, through `_inline_all`.) So a call like
    # `v_ = ToSpacePhasor(i)`, whose function returns a space phasor and a zero-sequence value, reads
    # the phasor and drops the zero sequence, rather than being refused for having two outputs.
    wanted = outputs[0]
    result = environment.get(wanted.name) or _assembled(wanted.name, environment)
    if result is None:
        raise FlattenError(f"function {name!r} never assigns its output {wanted.name!r}")
    # A name the body reads that is not a local -- an enumeration literal, a package constant, the
    # `Modelica.Constants.T_zero` the unit conversions add -- is resolved in the function's own scope,
    # the same rewrite an equation gets, before prefixing would try to make it a variable.
    result = _literals(result, inner, frozenset(merged.members))
    # The body may itself call a function, and it resolves in the function's own scope.
    return _inline_calls(result, inner, depth + 1)


def _inline_all(found: Found, scope: Scope, call: ir.Call, depth: int) -> dict[str, ir.Expr]:
    """One call, as an expression per output -- what an output expression list needs.

    The same symbolic run of the body as `_inline`, read out more than once. That it is *one*
    run is the point: two outputs of one call are two views of one computation, and running
    the body twice would compute the shared work twice and, for a body with a branch, could
    read two different arms.

    Kept beside `_inline` rather than replacing it. The single-output path is every call in
    every expression in the corpus and it returns an expression, not a mapping; making it go
    through here would put a dictionary lookup and a name in the middle of the hottest path
    in flattening to serve a form that appears at the top of an equation.
    """
    name = found.cls.name
    if depth >= _INLINE_DEPTH:
        raise FlattenError(f"{name!r} is called more than {_INLINE_DEPTH} deep; is it recursive?")
    inner = scope.inside(found)
    merged = effective(found, inner)
    if merged.equations or merged.initial_equations:
        raise FlattenError(f"function {name!r} has an equation section; a function has a body")
    if len(found.cls.algorithms) != 1:
        raise FlattenError(
            f"function {name!r} needs exactly one algorithm section, not {len(found.cls.algorithms)}"
        )
    components = [m.component for m in merged.members.values()]
    inputs = [c for c in components if c.causality == "input"]
    outputs = [c for c in components if c.causality == "output"]
    environment = _bind(name, inputs, call)
    for component in components:
        if component.causality != "input" and component.modification.value is not None:
            environment[component.name] = component.modification.value
    environment = _sequence(found.cls.algorithms[0], environment, f"function {name!r}")
    local = frozenset(merged.members)
    written: dict[str, ir.Expr] = {}
    for component in outputs:
        value = environment.get(component.name) or _assembled(component.name, environment)
        if value is None:
            raise FlattenError(f"function {name!r} never assigns its output {component.name!r}")
        # As in `_inline`: a package constant or an enumeration literal the body reads is resolved in
        # the function's own scope before prefixing sees it.
        written[component.name] = _inline_calls(_literals(value, inner, local), inner, depth + 1)
    return written


def _element_of(scalar: ir.Component, indices: tuple[int, ...], path: _Path) -> ir.Component:
    """One element of an array component, with its own share of every array-valued piece.

    ``parameter Real t[4] = {0, 1, 2, 3}`` declares four parameters with four values, and the
    expansion above makes four components -- so without this each of them carries the *whole*
    literal as its binding and the model ends at "array equations are not supported yet". The
    same is true of an array-valued attribute: ``Real x[2](start = {1, 2})`` gives its two
    elements two different starts, so the ``start`` entry is indexed exactly as the binding is.

    A scalar piece is left exactly as it is, because that is the broadcast every array in the
    corpus relies on: ``Real T[n](start = T_0)`` gives every cell the same start, and
    `HeatBar` would stop working if this indexed into it.
    """
    modification = _index_modification(scalar.modification, indices, path)
    if modification is scalar.modification:
        return scalar
    return replace(scalar, modification=modification)


def _index_modification(
    modification: ir.Modification, indices: tuple[int, ...], path: _Path
) -> ir.Modification:
    """The modification of one element: its binding and every attribute indexed to ``indices``."""
    value = _index_value(modification.value, indices, path)
    entries = {
        key: _index_modification(entry, indices, path)
        for key, entry in modification.entries.items()
    }
    if value is modification.value and all(
        entries[key] is entry for key, entry in modification.entries.items()
    ):
        return modification
    return replace(modification, value=value, entries=entries)


def _index_value(value: ir.Expr | None, indices: tuple[int, ...], path: _Path) -> ir.Expr | None:
    """Read out one element of an array literal, or leave a scalar to broadcast."""
    for index in indices:
        if not isinstance(value, ir.Array):
            return value  # a scalar broadcasts, which is what every array here relies on
        if len(value.elements) != 0 and index > len(value.elements):
            raise FlattenError(
                f"{'.'.join(path)!r} is declared with {index} or more elements and its value"
                f" lists {len(value.elements)}"
            )
        value = value.elements[index - 1]
    return value


def _filled(sizes: tuple[int, ...], element: ir.Expr) -> ir.Expr:
    """An array of ``sizes`` every entry of which is ``element`` (nested for a matrix)."""
    if not sizes:
        return element
    inner = _filled(sizes[1:], element)
    return ir.Array(tuple(inner for _ in range(sizes[0])))


def _shape_constructor(  # noqa: PLR0911 -- one case per constructor, as its neighbours have
    value: ir.Expr | None, sizes: tuple[int, ...], path: _Path, values: Mapping[str, float]
) -> ir.Expr | None:
    """Expand an array-constructor call into the literal array of the declared shape.

    ``zeros(n)``, ``ones(n)`` and ``fill(v, n)`` fill the shape; ``identity(n)`` is the square
    unit matrix; ``linspace(a, b, n)`` steps evenly from ``a`` to ``b``. A comprehension binding --
    ``{f(i) for i in 1:n}``, the way the machines set a per-phase parameter -- is the array it stands
    for, one element per iterator value. They are the ordinary way to give an array declaration its
    start or its value, and without expanding them here the whole call reached the emitter as an
    untranslated ``zeros``. The declared ``sizes`` are the shape (the constructor's own size argument
    only has to agree with it), so this needs no structural evaluation of its own. Anything else is
    left untouched.
    """
    where = ".".join(path)
    match value:
        case ir.Comprehension():
            # The iterator values are literals substituted into the body; the elements are prefixed
            # later, when the array reaches `_prefix_expr` element by element, so identity here.
            return _expand_comprehension(value, lambda expr: expr, values)
        case ir.Call(name="zeros"):
            return _filled(sizes, ir.RealLiteral(0.0))
        case ir.Call(name="ones"):
            return _filled(sizes, ir.RealLiteral(1.0))
        case ir.Call(name="fill", args=(entry, *_)):
            return _filled(sizes, entry)
        case ir.Call(name="identity"):
            if len(sizes) != 2 or sizes[0] != sizes[1]:  # noqa: PLR2004 -- a matrix is 2-D
                raise FlattenError(f"{where}: identity() is a square matrix, not shape {sizes}")
            n = sizes[0]
            return ir.Array(
                tuple(
                    ir.Array(tuple(ir.RealLiteral(1.0 if i == j else 0.0) for j in range(n)))
                    for i in range(n)
                )
            )
        case ir.Call(name="linspace", args=(lo, hi, _)):
            if len(sizes) != 1:
                raise FlattenError(f"{where}: linspace() makes a vector, not shape {sizes}")
            n = sizes[0]
            if n < 2:  # noqa: PLR2004 -- with fewer than two points there is nothing to space
                return ir.Array((lo,) * n)
            # a + (b - a) * i / (n - 1), one entry per point, keeping a and b as written so the
            # endpoints are exact.
            span = ir.Binary("-", hi, lo)
            middle = [
                ir.Binary("+", lo, ir.Binary("*", span, ir.RealLiteral(i / (n - 1))))
                for i in range(1, n - 1)
            ]
            return ir.Array((lo, *middle, hi))
        case _:
            return value


def _shape_modification(
    mod: ir.Modification, sizes: tuple[int, ...], path: _Path, values: Mapping[str, float]
) -> ir.Modification:
    """Expand an array constructor in a modification's value and each of its attributes.

    The binding (``A = identity(n)``) and the array-valued attributes (``x[n](start = zeros(n))``)
    share the component's shape, so the same ``sizes`` expand both, and everything below then sees
    the literal array it already knows how to read out element by element.
    """
    value = _shape_constructor(mod.value, sizes, path, values)
    entries = {
        key: _shape_modification(entry, sizes, path, values) for key, entry in mod.entries.items()
    }
    if value is mod.value and all(entries[key] is entry for key, entry in mod.entries.items()):
        return mod
    return replace(mod, value=value, entries=entries)


def _bind(name: str, inputs: Sequence[ir.Component], call: ir.Call) -> dict[str, ir.Expr]:
    """Arguments to inputs -- positional first, then by name, then declared defaults."""
    if len(call.args) > len(inputs):
        raise FlattenError(
            f"{name!r} takes {len(inputs)} argument(s), and {len(call.args)} were given"
        )
    given = dict(call.named)
    known = {c.name for c in inputs}
    for key in given:
        if key not in known:
            raise FlattenError(f"{name!r} has no input named {key!r}")
    environment: dict[str, ir.Expr] = {}
    for position, component in enumerate(inputs):
        if position < len(call.args):
            if component.name in given:
                raise FlattenError(f"{name!r}: {component.name!r} is given twice")
            environment[component.name] = call.args[position]
        elif component.name in given:
            environment[component.name] = given[component.name]
        elif component.modification.value is not None:
            environment[component.name] = component.modification.value
        else:
            raise FlattenError(f"{name!r}: no value for input {component.name!r}")
    return environment


def _run(  # noqa: PLR0911 - one case per statement form
    statement: ir.Statement, environment: Mapping[str, ir.Expr], where: str
) -> dict[str, ir.Expr]:
    """Execute one statement symbolically: an environment in, a new environment out.

    "Execute" is exact rather than a metaphor. Straight-line assignment is substitution,
    and an ``if`` is a *merge*: each branch runs against its own copy, and a variable the
    branches disagree about becomes an :class:`~modelica.ir.IfExpr` over the conditions
    that chose it. That is the whole trick, and it is also its own limit -- a loop has no
    such merge unless its trip count is known, and an early exit has no merge at all.
    """
    match statement:
        case ir.Assign(target=ir.Ref(subscripts=subscripts) as target) if subscripts:
            # The environment is keyed by dotted name, which has no slot for `lamp[2]`; keying it
            # as `lamp.p.v` would assign a variable that does not exist.
            raise FlattenError(
                f"{where}: an assignment to {target.name!r} through a subscripted component is"
                " not supported yet"
            )
        case ir.Assign(target=ir.Ref(name=name), value=value):
            return {
                **environment,
                name: _substitute(_read_elements(value, environment), environment),
            }
        case ir.Assign(
            target=ir.Subscript(base=ir.Ref(parts=(name,)), indices=indices), value=value
        ):
            known = _constant_indices(indices, environment)
            if known is None:
                return {**environment}
            computed = _substitute(_read_elements(value, environment), environment)
            return {**environment, _element_key(name, known): computed}
        case ir.Assign():
            # A subscripted target -- one element of an array, possibly at a run-time index -- names
            # no single variable to fold here; it is refused later, at `_prefix_statement`.
            return {**environment}
        case ir.IfStatement(branches=branches, otherwise=otherwise):
            return _merge_branches(branches, otherwise, environment, where)
        case ir.ForStatement(index=index, over=ir.Range() as over, body=body):
            return _unroll_for(index, over, body, environment, where)
        case ir.WhileStatement(condition=condition, body=body):
            return _unroll_while(condition, body, environment, where)
        case _:
            # Named in Modelica's words rather than the node's: the reader wrote `while`,
            # and `WhileStatement` is this toolchain's business, not theirs.
            raise FlattenError(
                f"{where} uses {_SPELLING[type(statement)]}, which cannot be turned into an"
                f" expression; statements here may assign and branch, and nothing else"
            )


_PARAMETERS: ContextVar[Mapping[str, float]] = ContextVar("_PARAMETERS", default={})  # noqa: B039
_SHAPES: ContextVar[Mapping[_Path, tuple[int, ...]]] = ContextVar("_SHAPES", default={})  # noqa: B039
"""The declared array shapes, by flat path, so ``end`` in a subscript can read its size."""
"""The parameter values known so far, so a loop bound in an inlined function can name one (#2).

Set by `flatten` for the duration of one flattening. Keys are flat dotted names, so only a
parameter whose name in the call is its flat name (a top-level parameter) is found.
"""


def _element_key(name: str, indices: Sequence[int]) -> str:
    """The environment key of one element of an array local: ``A[2]``, ``M[1,3]``."""
    return f"{name}[{','.join(str(i) for i in indices)}]"


def _constant_indices(
    indices: Sequence[ir.Expr], environment: Mapping[str, ir.Expr]
) -> tuple[int, ...] | None:
    """The subscripts as integers when they evaluate to constants here, else ``None``."""
    result: list[int] = []
    for index in indices:
        try:
            value = evaluate(_substitute(index, environment), _PARAMETERS.get())
        except EvaluationError:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float) or value != int(value):
            return None
        result.append(int(value))
    return tuple(result)


def _read_elements(  # noqa: PLR0911 - one case per node type, as its neighbours have
    node: ir.Expr, environment: Mapping[str, ir.Expr]
) -> ir.Expr:
    """Replace ``A[k]`` with the value the body assigned to that element, where it did."""

    def go(inner: ir.Expr) -> ir.Expr:
        return _read_elements(inner, environment)

    match node:
        case ir.Subscript(base=ir.Ref(parts=(name,)), indices=indices):
            known = _constant_indices(indices, environment)
            if known is not None and (key := _element_key(name, known)) in environment:
                return environment[key]
            # An input bound to an array literal (`state1({f, u1}, time)`): `u[2]` is `u1`.
            picked = _array_element(environment.get(name), known)
            if picked is not None:
                return picked
            return ir.Subscript(node.base, tuple(go(i) for i in indices))
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, go(lhs), go(rhs))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, go(operand))
        case ir.Call(name=fname, args=args, named=named):
            return ir.Call(fname, tuple(go(a) for a in args), tuple((k, go(v)) for k, v in named))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(go(condition), go(then), go(otherwise))
        case ir.Array(elements=elements):
            return ir.Array(tuple(go(e) for e in elements))
        case _:
            return node


def _array_element(value: ir.Expr | None, indices: tuple[int, ...] | None) -> ir.Expr | None:
    """The element of an array literal at 1-based ``indices``, or ``None`` if it is not one."""
    if value is None or indices is None:
        return None
    for index in indices:
        if not isinstance(value, ir.Array) or not 1 <= index <= len(value.elements):
            return None
        value = value.elements[index - 1]
    return value


def _assembled(name: str, environment: Mapping[str, ir.Expr]) -> ir.Expr | None:
    """A 1-D array output built from its assigned elements ``A[1]``, ``A[2]``, ..., or ``None``."""
    elements: list[ir.Expr] = []
    while (key := _element_key(name, (len(elements) + 1,))) in environment:
        elements.append(environment[key])
    return ir.Array(tuple(elements)) if elements else None


_WHILE_LIMIT: Final = 10_000
"""The most iterations a ``while`` loop may take before flattening refuses it."""


def _unroll_for(
    index: str,
    over: ir.Range,
    body: Sequence[ir.Statement],
    environment: Mapping[str, ir.Expr],
    where: str,
) -> dict[str, ir.Expr]:
    """Run a ``for`` loop once per index value, when its range is known here (#2).

    The bounds are read after the inputs are put in, so ``for j in 1:n`` works when ``n`` is
    a literal argument or an input default. A bound that is still a variable is refused.
    The index does not exist after the loop, so its earlier value, if any, comes back.
    """
    what = f"the range of {index!r} in {where}"
    values = _range_values(
        over, lambda bound: _substitute(bound, environment), _PARAMETERS.get(), what
    )
    result = dict(environment)
    for value in values:
        result[index] = ir.IntLiteral(value)
        result = _sequence(body, result, where)
    if index in environment:
        result[index] = environment[index]
    else:
        result.pop(index, None)
    return result


def _unroll_while(
    condition: ir.Expr,
    body: Sequence[ir.Statement],
    environment: Mapping[str, ir.Expr],
    where: str,
) -> dict[str, ir.Expr]:
    """Run a ``while`` loop while its condition evaluates to a constant ``true`` (#2).

    Only a loop whose condition depends on constants alone (an integer counter, for example)
    can be unrolled. A condition that reads a variable, or a loop longer than
    ``_WHILE_LIMIT`` iterations, is refused.
    """
    result = dict(environment)
    for _ in range(_WHILE_LIMIT):
        try:
            holds = evaluate(_substitute(condition, result), _PARAMETERS.get())
        except EvaluationError:
            raise FlattenError(
                f"{where} uses a 'while' loop whose condition depends on a variable;"
                f" only a loop with a constant trip count can be turned into an expression"
            ) from None
        if not holds:
            return result
        result = _sequence(body, result, where)
    raise FlattenError(f"{where} runs a 'while' loop more than {_WHILE_LIMIT} times")


def _merge_branches(
    branches: Sequence[ir.StatementBranch],
    otherwise: Sequence[ir.Statement],
    environment: Mapping[str, ir.Expr],
    where: str,
) -> dict[str, ir.Expr]:
    """Run every arm against its own copy, then reconcile what they disagree about.

    An arm whose condition is known (a constant, or a parameter value, as a loop bound is (#2))
    is decided here: a false one is dropped, and
    a true one is the answer with the arms after it. `symmetricOrientation(m)` assigns its
    output only in the arm that `mod(m, 2) == 0` picks, so running every arm refused it.
    """
    kept: list[ir.StatementBranch] = []
    for branch in branches:
        condition = _substitute(branch.condition, environment)
        try:
            known = evaluate(condition, _PARAMETERS.get())
        except EvaluationError:
            kept.append(ir.StatementBranch(condition, branch.body))
            continue
        if known:
            otherwise = branch.body
            break
    if not kept:
        return _sequence(otherwise, environment, where)
    conditions = [branch.condition for branch in kept]
    arms = [_sequence(branch.body, environment, where) for branch in kept]
    fallback = _sequence(otherwise, environment, where)

    def chosen(arm: Mapping[str, ir.Expr], key: str) -> ir.Expr:
        value = arm.get(key)
        if value is None:
            raise FlattenError(
                f"{where} assigns {key!r} on some paths and not on others; every branch,"
                f" and the `else`, has to give it a value"
            )
        return value

    result = dict(environment)
    touched = {key for arm in (*arms, fallback) for key in arm if arm[key] != environment.get(key)}
    for key in sorted(touched):
        value = chosen(fallback, key)
        for condition, arm in reversed(list(zip(conditions, arms, strict=True))):
            value = ir.IfExpr(condition, chosen(arm, key), value)
        result[key] = value
    return result


def _sequence(
    body: Sequence[ir.Statement], environment: Mapping[str, ir.Expr], where: str
) -> dict[str, ir.Expr]:
    """A run of statements, each against what the one before it left."""
    result = dict(environment)
    for statement in body:
        result = _run(statement, result, where)
    return result


def _section_equations(section: Sequence[ir.Statement], owner: str) -> list[ir.Equal]:
    """An ``algorithm`` section in a model, as the equations it is equivalent to.

    The same symbolic execution a function body gets, started from an *empty* environment:
    a name the section reads before assigning is left as a reference, so it means the
    model's own variable, and every name the section assigns comes out as one equation
    defining it. A section of ``n`` assignments becomes ``n`` equations, which is exactly
    the count the balance check expects of it.

    Doing it here rather than carrying the section down to causalization is a choice with
    a reason. A section carried down would have to match as a *group* -- all its outputs
    against one node -- and the emitter would need control flow, and `ir.derive` would have
    no way to differentiate it, so the Jacobian would fall back to differences. Converted
    here, none of that arises: what the later stages see is expressions, which they already
    know how to match, sort, differentiate and emit.

    The price is the subset. Statements that cannot become expressions are refused by
    :func:`_run`, and so is the one case that would silently mean the wrong thing.
    """
    where = f"the algorithm section of {owner!r}"
    environment = _sequence(section, {}, where)
    assigned = set(environment)
    equations: list[ir.Equal] = []
    for name, value in environment.items():
        read_back = sorted(assigned.intersection(names(value)))
        if read_back:
            # `y := y + 1` in an equation section's neighbour is not an increment: there is
            # no previous value of a continuous variable to add to. Refused rather than
            # emitted as `y = y + 1`, which has no solution and would be blamed on the solver.
            raise FlattenError(
                f"{where} reads {read_back[0]!r} before assigning it, and assigns it too;"
                f" a variable a section defines has no earlier value to read"
            )
        equations.append(ir.Equal(ir.Ref(tuple(name.split("."))), value))
    return equations


# --------------------------------------------------------------------------------------
# Redeclaration
# --------------------------------------------------------------------------------------


def _redeclared(member: Member, outer: ir.Modification, writer: Scope, path: _Path) -> Member:
    """The member a ``redeclare`` leaves behind: a new class, resolved where it was written.

    Two scopes meet here and confusing them is the whole difficulty. The *declaration* was
    written inside the class that owns the slot, and its ``constrainedby`` resolves there.
    The *redeclaration* was written by whoever instantiated that class, one level out, and
    its replacement type resolves there. Handing both to the same scope would find the
    wrong class, or fail to find one at all, in any model whose parts are packaged.

    The replacement's modification replaces the declaration's own rather than merging with
    it, because they describe different classes: ``replaceable Resistor e(R = 100)`` says
    nothing about the capacitor that arrives in its place, and carrying ``R = 100`` over
    would set an attribute that no longer exists.
    """
    name = ".".join(path)
    replacement = outer.redeclare
    original = member.component
    if isinstance(replacement, ir.Class):
        raise FlattenError(
            f"{name}: redeclaring a class is not supported yet; only a component can be replaced"
        )
    if replacement is None:  # pragma: no cover - the caller checks before it calls
        return member
    if not original.replaceable:
        raise FlattenError(
            f"{name} is not declared `replaceable`, so {replacement.type_name!r} cannot"
            f" replace {original.type_name!r} here"
        )
    _check_constraint(name, original, replacement, member.scope, writer)
    return Member(
        component=replace(
            original,
            type_name=replacement.type_name,
            modification=replacement.modification,
            dimensions=replacement.dimensions or original.dimensions,
        ),
        scope=writer,
        mods=(),
    )


def _rebind(found: Found, scope: Scope, outer: ir.Modification, prefix: _Path) -> Scope:
    """Apply this instance's class redeclarations, and hand back the scope they hold in.

    ``rig(redeclare package Medium = Water)`` does not change a declaration; it changes what
    the *name* ``Medium`` resolves to, everywhere inside this one instance. So it cannot be a
    value on a member the way a component redeclaration is -- it is a binding on the scope,
    and it travels down with the scope through every component the instance contains.

    The replacement resolves in ``scope``, which is where it was written: one level out from
    the class that owns the slot. Its own name, the one being replaced, is looked up in that
    class, which is where the ``replaceable`` was declared. Same two scopes as a component
    redeclaration, for the same reason.
    """
    replacements: dict[str, Found] = {}
    for name, entry in outer.entries.items():
        replacement = entry.redeclare
        if not isinstance(replacement, ir.Class):
            continue
        here = ".".join((*prefix, name))
        declared = next((c for c in found.cls.classes if c.name == name), None)
        if declared is None:
            raise FlattenError(f"{found.cls.name} has no class {name!r} to redeclare")
        if not declared.replaceable:
            raise FlattenError(f"{here} is not declared `replaceable`, so no class can replace it")
        if len(replacement.extends) != 1:
            raise FlattenError(f"{here}: a class redeclaration must name one class")
        target = scope.resolve(replacement.extends[0].type_name)
        if not isinstance(target, Found):
            raise FlattenError(
                f"{here}: {replacement.extends[0].type_name!r} is a predefined type,"
                f" and a class is wanted"
            )
        # The constraint is written inside the class that owns the slot, so it resolves
        # there; the replacement was written one level out, so it resolves in `scope`. The
        # same two scopes a component redeclaration meets, and the same way to get it wrong.
        _check_class_constraint(here, declared, target, scope.inside(found), scope)
        replacements[name] = target
    return scope.bound(replacements)


def _check_class_constraint(
    where: str, declared: ir.Class, target: Found, declared_in: Scope, writer: Scope
) -> None:
    """The replacement class has to provide what the constraining clause promises.

    The same structural, deliberately shallow rule a component redeclaration gets: every
    member the constraint declares must exist in the replacement under the same name. With no
    ``constrainedby``, the declared default is its own constraint.
    """
    wanted = declared.constraining or (declared.extends[0].type_name if declared.extends else "")
    if not wanted:
        return
    constraint = declared_in.resolve(wanted)
    if not isinstance(constraint, Found):
        return
    required = set(effective(constraint, declared_in.inside(constraint)).members)
    offered = set(effective(target, writer.inside(target)).members)
    missing = sorted(required - offered)
    if missing:
        raise FlattenError(
            f"{where}: {target.cls.name!r} does not satisfy {wanted!r} --"
            f" it has no {', '.join(repr(m) for m in missing)}"
        )


def _check_constraint(
    name: str, original: ir.Component, replacement: ir.Component, declared: Scope, writer: Scope
) -> None:
    """The replacement has to provide what the constraining clause promises.

    Structural, and deliberately shallow: every member the constraint declares must exist
    in the replacement under the same name. That is enough to catch the mistake this check
    is for -- an inertia dropped into an electrical slot -- and it does not pretend to be
    Modelica's full subtype relation, which also compares variability, causality and the
    flow prefix. What it does *not* do is checked nowhere else, so it is said here rather
    than implied by a passing test.

    With no ``constrainedby``, the declared type is its own constraint: that is what
    ``replaceable Resistor e`` means, and it is why a plain replaceable slot still refuses
    a capacitor.
    """
    wanted = original.constraining or original.type_name
    constraint = declared.resolve(wanted)
    provided = writer.resolve(replacement.type_name)
    if isinstance(constraint, str) or isinstance(provided, str):
        # A predefined type constrains by identity: `constrainedby Real` takes a Real.
        if constraint != provided:
            raise FlattenError(f"{name}: {replacement.type_name!r} does not satisfy {wanted!r}")
        return
    required = set(effective(constraint, declared.inside(constraint)).members)
    offered = set(effective(provided, writer.inside(provided)).members)
    missing = sorted(required - offered)
    if missing:
        raise FlattenError(
            f"{name}: {replacement.type_name!r} does not satisfy {wanted!r} --"
            f" it has no {', '.join(repr(m) for m in missing)}"
        )


def _prefix_modification(
    mod: ir.Modification,
    prefix: _Path,
    local: frozenset[str],
    values: Mapping[str, float] | None = None,
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
        value=None
        if mod.value is None
        else _prefix_expr(mod.value, prefix, local, strict=False, values=values),
        entries={
            name: _prefix_modification(entry, prefix, local, values)
            for name, entry in mod.entries.items()
        },
    )


def _prefix_statement(node: ir.Statement, prefix: _Path, local: frozenset[str]) -> ir.Statement:
    """Qualify the names in a ``when`` body with the instance path they were found under.

    The same two forms as :func:`_substitute_statement`, and for the same reason: a ``when``
    in an equation section holds nothing else, and the compound statements are refused here
    rather than silently returned unqualified -- which would leave a statement naming a
    variable that does not exist at this level.
    """
    match node:
        case ir.Assign(target=ir.Subscript(base=base)):
            # ``buf[iTick] := u`` -- an assignment to one element of an array. The standard
            # library's FFT and routing blocks write these; they parse now, but writing a single
            # element of a scalarised array (often at a run-time index) is not carried through the
            # rest of the pipeline yet, so it is refused here with the element it names.
            raise FlattenError(
                f"an assignment to the array element {base.name!r} in a 'when' or algorithm is not"
                " supported yet"
            )
        case ir.Assign(target=ir.Ref() as target, value=value):
            return ir.Assign(_prefix_ref(target, prefix, local), _prefix_expr(value, prefix, local))
        case ir.Reinit(target=target, value=value):
            return ir.Reinit(_prefix_ref(target, prefix, local), _prefix_expr(value, prefix, local))
        case ir.CallStatement(call=ir.Call(name="terminate", args=args, named=named), span=span):
            # `terminate("reason")` ends the run at the instant its `when` fires. It binds no
            # unknown and moves no state, so it is carried through as it was written, message
            # and all -- the emitter reads the message off it and the event loop stops there.
            return ir.CallStatement(
                (),
                ir.Call("terminate", tuple(_prefix_expr(a, prefix, local) for a in args), named),
                span,
            )
        case _:  # pragma: no cover - unreachable while `when` bodies are simple statements
            raise FlattenError(f"{type(node).__name__} inside a 'when' is not supported yet")


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

# The type attributes a declaration may set. `start`, `fixed`, `min`, `max` and `nominal` reach
# the numeric model; `unit`, `displayUnit` and `quantity` are descriptive and accepted but not
# used here (`quantity` names the physical quantity, e.g. `Real(quantity="ElectricPotential",
# unit="V")` -- the whole `Modelica.Units.SI` type set carries it, so refusing it blocked the
# electrical and machine libraries wholesale). An attribute outside this set is refused by name.
_ATTRIBUTES = frozenset(
    {"start", "fixed", "unit", "min", "max", "nominal", "displayUnit", "quantity", "stateSelect"}
)

_FLOOR: dict[ir.Variability, int] = {
    "continuous": 0,
    "discrete": 1,
    "parameter": 2,
    "constant": 3,
}


def _tighten(declared: ir.Variability, floor: ir.Variability) -> ir.Variability:
    return declared if _FLOOR[declared] >= _FLOOR[floor] else floor


class _DeferredIf(NamedTuple):
    """A structural if-equation held for the final pass, with the scope it was met in."""

    node: ir.IfEquation
    prefix: _Path
    local: frozenset[str]
    scope: Scope
    outer: Mapping[str, _Path] | None
    initial: bool


@dataclass
class _Flattener:
    program: ir.Program
    variables: list[FlatVariable] = field(default_factory=list[FlatVariable])
    equations: list[ir.Equal] = field(default_factory=list[ir.Equal])
    whens: list[ir.When] = field(default_factory=list[ir.When])
    asserts: list[ir.Assert] = field(default_factory=list[ir.Assert])
    initial_equations: list[ir.Equal] = field(default_factory=list[ir.Equal])
    loops: list[Loop] = field(default_factory=list[Loop])
    connections: _Connections = field(default_factory=_Connections)
    connectors: dict[_Path, Found] = field(default_factory=dict[_Path, "Found"])
    disabled: set[_Path] = field(default_factory=set[_Path])
    structural: set[str] = field(default_factory=set[str])
    """Constants and ``Evaluate = true`` parameters: an if-expression on them is folded."""
    """Conditional components whose condition came out false, by instance path.

    Read by the ``connect()`` handling: a connect whose endpoint sits inside a disabled
    component is removed with it, which is the one reference the specification allows to
    outlive the removal.
    """
    typed_arrays: set[_Path] = field(default_factory=set[_Path])
    """Components whose class added its shape (`type Axis = Real[3]`), so elements skip it."""
    deferred_ifs: list[_DeferredIf] = field(default_factory=list["_DeferredIf"])
    """Structural if-equations whose condition could not be evaluated where they were met.

    A condition names a parameter, and that parameter's value may be supplied by a sibling declared
    *after* the component the if sits in (the machine friction: ``if frictionParameters.PRef <= 0``,
    with ``frictionParameters`` set from a data record that comes later in the example). Such an if is
    held here, with the scope it needs, and resolved by :meth:`resolve_deferred_ifs` once the whole
    model's parameters have settled, when the condition is a number like any other.
    """
    values: dict[str, float] = field(default_factory=dict[str, float])
    """What every parameter and constant declared so far is worth, by flat name.

    Filled as the declarations go past, and read by the two things that have to be numbers
    before the model exists: an array's shape and a subscript. Nothing else uses it -- the
    solver reads bindings from the variables themselves.
    """
    shapes: dict[_Path, tuple[int, ...]] = field(default_factory=dict[_Path, "tuple[int, ...]"])
    """The size of every array component declared so far, by flat path.

    An array component is scalarised into its elements at declaration, but an equation
    written between whole arrays -- ``der(x) = -x`` for ``Real x[2]`` -- has to be scalarised
    too, and that needs to know a bare reference is an array and how big. Filled by
    :meth:`_array` and read by :meth:`_shape` / :func:`_element` when an equation is split.
    """

    # -- components -------------------------------------------------------------------

    def _resolve_parameter_values(self, started: int) -> None:
        """Record the value of any parameter added since ``started`` whose binding names another.

        The value of a parameter is read by a dimension and by the condition on a conditional
        component, and a literal binding lands in ``self.values`` as it is instantiated. A binding
        that names another parameter -- ``useHeatPort = useTemperatureDependency`` -- cannot, because
        the one it names may be instantiated after it. A fixed point over the parameters this class
        added closes that: each pass evaluates what it can against the values gathered so far, and it
        stops when a pass adds nothing. A parameter that never resolves is left for the ordinary
        "unknown name" report to catch where it is used.
        """
        changed = True
        while changed:
            changed = False
            for variable in self.variables[started:]:
                if variable.name in self.values or variable.binding is None:
                    continue
                if variable.variability not in ("parameter", "constant"):
                    continue
                try:
                    value = evaluate(variable.binding, self.values)
                except EvaluationError:
                    continue
                if isinstance(value, (bool, int, float)):
                    self.values[variable.name] = value
                    changed = True

    def _record_binding(
        self, outer: ir.Modification, merged: Effective, found: Found, prefix: _Path
    ) -> ir.Modification:
        """A record component given a value ``a = b`` is read as a modification of ``a``'s members.

        A record may be given another record as its value, in one of two shapes. The machine models
        write a *reference* -- ``frictionParameters = dcpmData.frictionParameters``, one record
        standing in for another -- and each member of ``a`` reads the matching member of ``b``: the
        value arrived from the enclosing class already prefixed into the flat namespace, so ``b.member``
        is written by appending the member's name to it. The flux-tube and fluid models write a
        *constructor* -- ``material = BaseData()`` for the record's own defaults, ``R(a = 5)`` to set a
        field -- and each argument becomes that member's modification, positional ones matched to the
        members in order. Either way the result is handed to the member as an ordinary modification,
        which ``_component`` takes as-is, and a member the caller modified explicitly wins over the copy.

        A constructor of a *different* record than the component's own type is a redeclaration (the
        flux-tube examples pick a specific magnetic material this way), not a plain binding, and is
        refused as one. So is a value of any other shape -- a literal, an array -- that names no members.
        """
        value = outer.value
        entries = dict(outer.entries)
        settable = [
            name for name, member in merged.members.items() if member.component.scoping != "outer"
        ]
        if isinstance(value, ir.Ref):
            for name in settable:
                entries.setdefault(
                    name,
                    ir.Modification(value=ir.Ref((*value.parts, name)), entries={}, redeclare=None),
                )
            return replace(outer, value=None, entries=entries)
        if isinstance(value, ir.Call):
            if value.name.rsplit(".", 1)[-1] != found.cls.name:
                raise FlattenError(
                    f"{'.'.join(prefix)}: a record constructor of a different type is a"
                    " redeclaration, not supported yet"
                )
            for name, argument in zip(settable, value.args, strict=False):
                entries.setdefault(
                    name, ir.Modification(value=argument, entries={}, redeclare=None)
                )
            for name, argument in value.named:
                entries.setdefault(
                    name, ir.Modification(value=argument, entries={}, redeclare=None)
                )
            return replace(outer, value=None, entries=entries)
        raise FlattenError(
            f"{'.'.join(prefix)}: a composite component can only be bound to another record,"
            " by name or by its constructor"
        )

    def instantiate(  # noqa: C901, PLR0912, PLR0913, PLR0917 -- one pass over one class, and the
        # `inner` declarations it can see have to travel with it: they are a property of
        # where the instance *sits*, which no other argument carries.
        self,
        found: Found,
        scope: Scope,
        prefix: _Path,
        outer: ir.Modification,
        floor: ir.Variability,
        inners: Mapping[str, _Path] | None = None,
    ) -> None:
        # Bound before anything is merged, because a redeclared class changes what the type
        # names in this instance's own members resolve to -- including the ones an `extends`
        # brings in, which `effective` is about to read.
        scope = _rebind(found, scope, outer, prefix)
        merged = effective(found, scope)
        local = frozenset(merged.members)
        visible, resolved_outer = self._scoping(merged, prefix, inners or {})
        if outer.value is not None:
            outer = self._record_binding(outer, merged, found, prefix)
        inner = scope.inside(found)
        for name, entry in outer.entries.items():
            # A redeclaration is applied where the component is instantiated, below. One
            # that names no component would never be reached there, and the model would run
            # as though it had never been written -- so the unmatched ones are caught here,
            # before anything else happens. A redeclared *class* always lands here, which is
            # where that refusal comes from: no component carries its name.
            if entry.redeclare is None or name in merged.members:
                continue
            if isinstance(entry.redeclare, ir.Class):
                continue  # a class redeclaration; `_rebind` above already applied it
            raise FlattenError(f"{found.cls.name} has no component {name!r} to redeclare")
        # The members that carry a condition are held back to a second pass. A condition names a
        # parameter, and that parameter may name another (`useHeatPort = useTemperatureDependency` on
        # a transistor's heat port), so every unconditional member has to be in -- and the parameter
        # values resolved among themselves -- before a condition can be evaluated.
        started = len(self.variables)
        conditional: list[tuple[str, Member]] = []
        for name, member in merged.members.items():
            if member.component.scoping == "outer":
                # An `outer` declares nothing. It named something further out, `_scoping`
                # found what, and every reference to it now resolves there instead.
                continue
            if member.component.condition is not None:
                conditional.append((name, member))
                continue
            self._component(
                member,
                (*prefix, name),
                outer.entries.get(name, ir.EMPTY),
                floor,
                local,
                writer=scope,
                inners=visible,
            )
        self._resolve_parameter_values(started)
        for name, member in conditional:
            self._component(
                member,
                (*prefix, name),
                outer.entries.get(name, ir.EMPTY),
                floor,
                local,
                writer=scope,
                inners=visible,
            )
        for equation in merged.equations:
            writer = merged.writers.get(id(equation), inner)
            self._equation(equation, prefix, local, scope=writer, outer=resolved_outer)
        for equation in merged.initial_equations:
            writer = merged.writers.get(id(equation), inner)
            self._equation(
                equation, prefix, local, scope=writer, outer=resolved_outer, initial=True
            )
        for section in found.cls.algorithms:
            for equation in _section_equations(section, found.cls.name):
                self._equation(equation, prefix, local, scope=inner, outer=resolved_outer)
        for section in found.cls.initial_algorithms:
            for equation in _section_equations(section, found.cls.name):
                self._equation(
                    equation, prefix, local, scope=inner, outer=resolved_outer, initial=True
                )

    def _scoping(
        self, merged: Effective, prefix: _Path, inherited: Mapping[str, _Path]
    ) -> tuple[dict[str, _Path], dict[str, _Path]]:
        """What this instance publishes as ``inner``, and where its ``outer`` names point.

        The two are computed together and in this order because a class may declare an
        ``inner`` and use an ``outer`` of a *different* name in the same body, and because
        an ``inner`` declared here shadows one further out for everything below it. Nothing
        binds an ``outer`` to an ``inner`` in the same class -- that is what `inner outer`
        is for, and it is refused in the parser.
        """
        visible = dict(inherited)
        for name, member in merged.members.items():
            if member.component.scoping == "inner":
                visible[name] = (*prefix, name)
        resolved: dict[str, _Path] = {}
        for name, member in merged.members.items():
            if member.component.scoping != "outer":
                continue
            target = inherited.get(name)
            if target is None:
                raise FlattenError(
                    f"{'.'.join((*prefix, name))} is `outer` and no enclosing class declares"
                    f" an `inner {name}`"
                )
            resolved[name] = target
        return visible, resolved

    def _component(  # noqa: PLR0913 -- the writer's scope is what resolves a redeclaration
        self,
        member: Member,
        path: _Path,
        outer: ir.Modification,
        floor: ir.Variability,
        local: frozenset[str],
        *,
        writer: Scope,
        inners: Mapping[str, _Path] | None = None,
    ) -> None:
        if outer.redeclare is not None:
            member = _redeclared(member, outer, writer, path)
            outer = replace(outer, redeclare=None)
        component = member.component
        if component.condition is not None:
            # Evaluated here for the same reason a dimension is: whether the component
            # exists decides how many variables and equations the model has. The condition takes
            # the enumeration-literal rewrite first, the same one an if-equation condition gets:
            # the power-converter blocks gate a component on `refType == SingleReferenceType.Sawtooth`,
            # and that literal has to become its ordinal before the names are resolved, or prefixing
            # reads it as a variable that does not exist.
            resolved = _inline_calls(
                _literals(component.condition, member.scope, local), member.scope
            )
            held = _boolean(
                _prefix_expr(resolved, path[:-1], local, values=self.values),
                self.values,
                f"the condition on {'.'.join(path)!r}",
            )
            if not held:
                self.disabled.add(path)
                return
        member = self._with_type_shape(member, path)
        component = member.component
        if component.dimensions:
            self._array(member, path, outer, floor, local, writer=writer, inners=inners)
            return
        # `outer` arrived from the enclosing class and was prefixed there. These two were
        # written *here*, so they are prefixed here, against the path of the class that
        # holds them -- everything in `path` except this component's own name.
        here = path[:-1]
        # An inlined call reads this instance's parameters by their local names.
        view = _LocalValues(self.values, here, local)
        token = _PARAMETERS.set(view)
        try:
            resolved_literals = _literal_modification(component.modification, member.scope, local)
            # `extends Base(final ViNominal = VaNominal - convertResistance(...))` (the DC
            # machines): the call and the constants resolve where the clause was written.
            inherited_literals = [
                _literal_modification(mod.modification, mod.scope, local) for mod in member.mods
            ]
        finally:
            _PARAMETERS.reset(token)
        resolved_literals = _expanded_comprehensions(resolved_literals, view)
        # With the values, a subscript in a binding (`u = if abs(n[1]) >= 0.99 then ...`,
        # MultiBody's `World`) names the element its index picks.
        own = _prefix_modification(resolved_literals, here, local, self.values)
        inherited = tuple(
            _prefix_modification(mod, here, local, self.values) for mod in inherited_literals
        )
        total = _merged(own, *inherited, outer)
        resolved = member.scope.resolve(component.type_name)
        if (
            isinstance(resolved, Found)
            and resolved.cls.causality != "none"
            and component.causality == "none"
        ):
            # `connector RealInput = input Real`: the class carries the causality, and
            # every instance is an input without writing the keyword at the declaration.
            component = replace(component, causality=resolved.cls.causality)
            member = replace(member, component=component)
        variability = _tighten(component.variability, floor)
        if isinstance(resolved, Found) and resolved.cls.restriction != "type":
            if resolved.cls.restriction == "connector":
                if resolved.cls.expandable:
                    raise FlattenError(
                        f"{'.'.join(path)}: expandable connectors are not supported yet",
                    )
                self.connectors[path] = resolved
                signal = _signal_base(resolved, member.scope)
                if signal is not None:
                    # A signal connector (`connector RealInput = input Real`) is a scalar, not a
                    # record: lay it down as one leaf, the causality already on the component. It
                    # is still a connector -- registered above -- so `connect()` joins two of them.
                    base, attributes = signal
                    self._leaf(component, path, _merge(attributes, total), base, variability)
                    return
            self.instantiate(resolved, member.scope, path, total, variability, inners)
            return
        base, attributes = _base_of(resolved, member.scope)
        self._leaf(component, path, _merge(attributes, total), base, variability)

    def _with_type_shape(self, member: Member, path: _Path) -> Member:
        """``type Axis = Real[3]; Axis a[2];`` declares ``a[2, 3]``: the type's shape goes last.

        The elements still name ``Axis``, so the path is recorded and the shape is added once.
        """
        component = member.component
        typed = member.scope.resolve(component.type_name)
        if (
            not isinstance(typed, Found)
            or not typed.cls.dimensions
            or (*path[:-1], path[-1].split("[", 1)[0]) in self.typed_arrays
        ):
            return member
        self.typed_arrays.add(path)
        shaped = replace(component, dimensions=(*component.dimensions, *typed.cls.dimensions))
        return replace(member, component=shaped)

    def _array(  # noqa: PLR0913 -- the writer's scope travels with every component
        self,
        member: Member,
        path: _Path,
        outer: ir.Modification,
        floor: ir.Variability,
        local: frozenset[str],
        *,
        writer: Scope,
        inners: Mapping[str, _Path] | None = None,
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
        # Remember the shape under the flat path, so an equation written between whole arrays
        # can be scalarised the way this declaration is. Keyed by the same tuple a reference
        # to the array prefixes to, which is what lets `_element` recognise it later.
        self.shapes[tuple(path)] = tuple(sizes)
        # An array constructor -- `start = zeros(n)`, `A = identity(n)` -- becomes the literal array
        # of this shape before the element loop reads it, so each element gets its own entry the way
        # it would from a written `{...}` (the whole call reached the emitter untranslated otherwise).
        # Function calls in the binding are inlined first, so a call that returns an array
        # (`A = areas(0.1, N)`) arrives here as the literal array it computes (#2).
        view = _LocalValues(self.values, here, local)
        token = _PARAMETERS.set(view)
        try:
            inlined = _literal_modification(component.modification, member.scope, local)
        finally:
            _PARAMETERS.reset(token)
        if inlined.value is not None and "cat" in _call_names(inlined.value):
            here_shapes = {
                key[len(here) :]: shape
                for key, shape in self.shapes.items()
                if len(key) > len(here) and key[: len(here)] == here
            }
            inlined = replace(inlined, value=_expand_cat(inlined.value, here_shapes))
        shaped = _shape_modification(inlined, tuple(sizes), path, view)
        scalar = replace(component, dimensions=(), modification=shaped)
        # A binding that is an expression over whole arrays -- `C = A * dx / k` -- is read out
        # element by element, as an array equation is (#3). The shapes are keyed by flat path,
        # and the binding names arrays relative to the class that declares it.
        local_shapes = {
            key[len(here) :]: shape
            for key, shape in self.shapes.items()
            if len(key) > len(here) and key[: len(here)] == here
        }
        array_binding = False
        if shaped.value is not None and not isinstance(shaped.value, ir.Array):
            try:
                array_binding = _infer_shape(shaped.value, local_shapes) == tuple(sizes)
            except FlattenError:
                array_binding = False  # a shape this pass cannot read keeps the old path
        # The enclosing class's modification (`A a(final v = fill(v0, m))`) is shaped and indexed
        # the same way, or every element would carry the whole array as its binding.
        outer_shaped = _shape_modification(outer, tuple(sizes), path, self.values)
        for indices in product(*(range(1, size + 1) for size in sizes)):
            if array_binding and shaped.value is not None:
                one = replace(shaped, value=_element(shaped.value, indices, local_shapes))
                scalar = replace(component, dimensions=(), modification=one)
            self._component(
                replace(member, component=_element_of(scalar, indices, path)),
                (*here, element(component.name, indices)),
                _index_modification(outer_shaped, indices, path),
                floor,
                local,
                writer=writer,
                inners=inners,
            )

    def _scalar_statements(self, statement: ir.Statement) -> tuple[ir.Statement, ...]:
        """A ``when`` assignment to a whole array, as one assignment per element (#3).

        ``h = A`` in a ``when`` sets every element of ``h``. Any other statement, and an
        assignment to a scalar, comes back unchanged.
        """
        match statement:
            case ir.Assign(target=ir.Ref(parts=parts) as target, value=value) if (
                shape := self.shapes.get(tuple(parts))
            ):
                scalars: list[ir.Statement] = []
                for indices in product(*(range(1, size + 1) for size in shape)):
                    element_target = _element(target, indices, self.shapes)
                    if not isinstance(element_target, ir.Ref | ir.Subscript):
                        return (statement,)
                    scalars.append(ir.Assign(element_target, _element(value, indices, self.shapes)))
                return tuple(scalars)
            case _:
                return (statement,)

    def _leaf(  # noqa: C901, PLR0912 -- one pass laying a scalar variable down, attribute by attribute
        self,
        component: ir.Component,
        path: _Path,
        total: ir.Modification,
        base: BaseType,
        variability: ir.Variability,
    ) -> None:
        name = ".".join(path)
        evaluate_entry = component.annotation.entries.get("Evaluate")
        if variability == "constant" or (
            evaluate_entry is not None and evaluate_entry.value == ir.BoolLiteral(value=True)
        ):
            self.structural.add(name)
        unknown = set(total.entries) - _ATTRIBUTES
        if unknown:
            raise FlattenError(f"{name}: unknown attribute(s) {sorted(unknown)}")

        def attribute(key: str) -> ir.Expr | None:
            entry = total.entries.get(key)
            folded = _fold(entry.value) if entry is not None else None
            return _fold_reductions(folded, self.shapes) if folded is not None else None

        unit = attribute("unit")
        if unit is not None and not isinstance(unit, ir.StringLiteral):
            raise FlattenError(f"{name}: `unit` must be a string literal")
        fixed = attribute("fixed")
        if fixed is not None and not isinstance(fixed, ir.BoolLiteral):
            # `fixed` may name a boolean parameter rather than spell the literal -- the fluid models
            # write `T(start = T0, fixed = T0fixed)` with `parameter Boolean T0fixed = false` -- so it
            # is evaluated against the parameter values gathered so far, the way a condition is. The
            # parameter was declared before this component, so its value is already in hand; anything
            # that still does not resolve to a boolean is the real error.
            try:
                resolved = evaluate(fixed, self.values)
            except EvaluationError:
                resolved = None
            if not isinstance(resolved, bool):
                raise FlattenError(f"{name}: `fixed` must be a boolean literal")
            fixed = ir.BoolLiteral(resolved)

        # A reduction in a scalar binding (`parameter Real total = sum(x0)`) or an attribute folds
        # here, where the array it sums has already been declared and its shape is known. A leaf
        # never passes through the equation split that folds the reductions in an equation, so
        # without this the call reached the emitter as an untranslated `sum`.
        folded_binding = _fold(total.value)
        if folded_binding is not None and "size" in _call_names(folded_binding):
            # `parameter Integer M = size(A, 1)` reads a shape, so it is resolved to the literal
            # here, before the value is recorded for a later dimension to use (#3).
            folded_binding = self._constructors(folded_binding)
        if folded_binding is not None:
            folded_binding = _resolve_subscripts(folded_binding, self.values)
        binding = (
            _fold_reductions(folded_binding, self.shapes) if folded_binding is not None else None
        )
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
        if variability in ("parameter", "constant") and binding is not None:
            # The value is read by a dimension and by the condition on a conditional component. A
            # literal is itself; a binding that names another parameter already recorded -- MSL's
            # `useHeatPort = useTemperatureDependency` on a transistor's heat port -- is evaluated
            # against those. One that names a parameter not yet in hand is left for a typo check.
            try:
                value = evaluate(binding, self.values)
            except EvaluationError:
                value = None
            if isinstance(value, (bool, int, float)):
                self.values[name] = value
        elif variability == "parameter":
            # A parameter with no binding takes its `start` value: the specification lets a parameter
            # stand on its start (with a tool warning), and the machines rely on it -- a synchronous
            # machine declares `parameter Boolean useDamperCage(start = true)` with no binding and
            # gates its damper-cage heat port on it, so `start` is the value the condition must read.
            start_value = attribute("start")
            if start_value is not None:
                try:
                    value = evaluate(start_value, self.values)
                except EvaluationError:
                    value = None
                if isinstance(value, (bool, int, float)):
                    self.values[name] = value
        if binding is not None and variability not in ("parameter", "constant"):
            # A declaration equation on a non-constant is an ordinary equation.
            self.equations.append(ir.Equal(ir.Ref(path), binding))

    # -- equations --------------------------------------------------------------------

    def _results(  # noqa: PLR0913, PLR0917 -- the targets, the call, and what a name needs
        self,
        targets: tuple[ir.Ref | None, ...],
        call: ir.Call,
        scope: Scope,
        prefix: _Path,
        local: frozenset[str],
        outer: Mapping[str, _Path] | None,
        span: ir.Span | None,
    ) -> list[ir.Equal]:
        """``(a, b) = f(x)`` as one ordinary equation per wanted output.

        The body is run **once** and read out per output. That is not an optimisation: two
        results of one call are two views of one computation, and a body with a branch run
        twice could take a different arm each time and return a pair that never existed.
        """
        resolved = _literals(call, scope, local)
        if not isinstance(resolved, ir.Call):  # pragma: no cover - the parser refuses this
            raise FlattenError("an output list is assigned from a function call")
        found = scope.function(resolved.name)
        if found is None:
            raise FlattenError(
                f"{resolved.name!r} is not a function this model can see; an output list needs one"
            )
        written = _inline_all(found, scope, resolved, 0)
        declared = list(written)
        if len(targets) > len(declared):
            raise FlattenError(
                f"{resolved.name!r} returns {len(declared)} value(s)"
                f" and {len(targets)} were asked for"
            )

        def qualified(node: ir.Expr) -> ir.Expr:
            return _prefix_expr(node, prefix, local, values=self.values, outer=outer)

        return [
            ir.Equal(qualified(target), qualified(written[output]), span)
            # `None` is a result the writer said they do not want: `(, b) = f(x)`.
            for target, output in zip(targets, declared, strict=False)
            if target is not None
        ]

    def _shape(self, node: ir.Expr) -> tuple[int, ...]:
        """The array shape of an already-flattened expression, against this model's arrays."""
        return _infer_shape(node, self.shapes)

    def _dim(self, node: ir.Expr) -> int:
        """One dimension of an array constructor, evaluated against the structural parameters."""
        return _integer(node, self.values, "an array constructor's size")

    def _constructors(self, node: ir.Expr) -> ir.Expr:  # noqa: C901, PLR0911, PLR0912 -- one case per constructor
        """Expand every array constructor in an expression to its literal array.

        The counterpart of :func:`_shape_constructor` for a constructor written in an *equation*
        rather than a declaration: the declaration knows its own shape, an equation reads the shape
        off the constructor's own size arguments, which are structural and known here. Only a
        constructor is rewritten; everything else is copied through, so applying it to an equation
        that has none does nothing.
        """
        match node:
            case ir.Call(name="zeros" | "ones", args=args) if args:
                return (
                    _shape_constructor(
                        node, tuple(self._dim(a) for a in args), ("<equation>",), self.values
                    )
                    or node
                )
            case ir.Call(name="fill", args=(value, *dims)) if dims:
                built = ir.Call("fill", (self._constructors(value), *dims))
                return (
                    _shape_constructor(
                        built, tuple(self._dim(a) for a in dims), ("<equation>",), self.values
                    )
                    or built
                )
            case ir.Call(name="identity", args=(n,)):
                size = self._dim(n)
                return _shape_constructor(node, (size, size), ("<equation>",), self.values) or node
            case ir.Call(name="linspace", args=(lo, hi, n)):
                built = ir.Call("linspace", (self._constructors(lo), self._constructors(hi), n))
                return (
                    _shape_constructor(built, (self._dim(n),), ("<equation>",), self.values)
                    or built
                )
            case ir.Call(name="size", args=(array, dim)):
                # `size(a, k)` is the k-th dimension of a: a structural number read from the shape
                # registry, not an element-wise value, so it is resolved to a literal here rather
                # than left to broadcast (which would make the element split subscript `a` *inside*
                # the call). Its argument is already prefixed, so its flat shape is known.
                shape = self._shape(self._constructors(array))
                index = _integer(dim, self.values, "the dimension argument of `size`")
                if not 1 <= index <= len(shape):
                    raise FlattenError(
                        f"`size` asks for dimension {index} of a value with"
                        f" {len(shape)} dimension(s)"
                    )
                return ir.IntLiteral(shape[index - 1])
            case ir.Call(name="cat"):
                built = ir.Call("cat", tuple(self._constructors(a) for a in node.args), node.named)
                return _expand_cat(built, self.shapes)
            case ir.Call(name="size", args=(array,)):
                # `size(a)` is the vector of a's dimension lengths, itself a literal array.
                return ir.Array(
                    tuple(ir.IntLiteral(d) for d in self._shape(self._constructors(array)))
                )
            case ir.Call(name=name, args=args):
                return ir.Call(name, tuple(self._constructors(a) for a in args), node.named)
            case ir.Binary(op=op, lhs=lhs, rhs=rhs):
                return ir.Binary(op, self._constructors(lhs), self._constructors(rhs))
            case ir.Unary(op=op, operand=operand):
                return ir.Unary(op, self._constructors(operand))
            case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
                return ir.IfExpr(
                    self._constructors(condition),
                    self._constructors(then),
                    self._constructors(otherwise),
                )
            case _:
                return node

    def _carry(
        self,
        node: ir.Expr,
        prefix: _Path,
        local: frozenset[str],
        scope: Scope,
        outer: Mapping[str, _Path] | None,
    ) -> ir.Expr:
        """The expression rewrite every equation form shares: literals, then names, then folding.

        Three rewrites, in the order their outputs feed each other. An enumeration literal becomes a
        number first, because it is not a reference and prefixing would turn it into one. Then a call
        is resolved in the scope of the class that wrote it, and what comes back is an expression in
        that class's own names -- which is exactly what prefixing then qualifies. A last pass folds
        any array reduction (`sum(x)` and its kin) now that the names are flat: an equation is split
        element by element after this, but a `when` condition, an `if`-equation condition or an assert
        is not, so this is where their reductions become the arithmetic the emitter can read.
        """
        written = _literals(node, scope, local)
        prefixed = _prefix_expr(
            _inline_calls(written, scope), prefix, local, values=self.values, outer=outer
        )
        # Then any array constructor (`zeros(n)`, `identity(n)`) becomes its literal array, the way a
        # declaration expands one -- its size arguments are structural and known here, so a forced
        # linear system `der(x) = A * x + ones(n)` reads as the arithmetic it stands for.
        return _fold_reductions(self._constructors(prefixed), self.shapes)

    def _equation(  # noqa: C901, PLR0913 -- one arm per equation form; the scope
        # resolves a call, the outer map a name
        self,
        equation: ir.Equation,
        prefix: _Path,
        local: frozenset[str],
        *,
        scope: Scope,
        outer: Mapping[str, _Path] | None = None,
        initial: bool = False,
    ) -> None:
        def carry(node: ir.Expr) -> ir.Expr:
            return self._carry(node, prefix, local, scope, outer)

        if initial and not isinstance(
            equation, (ir.Equal, ir.CallEquation, ir.ForEquation, ir.IfEquation)
        ):
            raise FlattenError("only `a = b` equations may appear in initial equation")

        match equation:
            case ir.CallEquation(targets=(), call=call):
                # The no-output form is the connection-graph operators and `terminate`,
                # and neither has a meaning here yet. Refused by name, so the work list
                # can count it, rather than fed to `_results` to fail as a lookup.
                raise FlattenError(
                    f"a call equation is not supported yet: {call.name}()",
                )
            case ir.CallEquation(targets=targets, call=call, span=span):
                # One equation in, one equation per wanted output out. Everything after
                # flattening then sees the shape it already knows -- one equation, one
                # unknown -- and causalization needs no idea that a call can return a pair.
                sink = self.initial_equations if initial else self.equations
                sink.extend(self._results(targets, call, scope, prefix, local, outer, span))
            case ir.Equal(lhs=lhs, rhs=rhs, span=span):
                sink = self.initial_equations if initial else self.equations
                # The span rides along. Flattening rewrites *names*, so the equation a block
                # holds is still the one the modeller wrote, and saying where it was written
                # is the difference between "block 7" and a line a reader can open.
                left, right = carry(lhs), carry(rhs)
                shape = _merge_shape(self._shape(left), self._shape(right))
                # One equation between arrays is one scalar equation per element, the same split
                # the declaration did for the variables. `_element` reads out element `indices` of
                # each side; the span is shared, so every scalar row points back at the one line the
                # modeller wrote. A scalar equation runs the loop once with no index -- `_element` is
                # then the identity, unless it holds a contraction (a dot product `s = a * x`), which
                # it still expands into the sum.
                for indices in product(*(range(1, size + 1) for size in shape)):
                    sink.append(
                        ir.Equal(
                            _element(left, indices, self.shapes),
                            _element(right, indices, self.shapes),
                            span,
                        )
                    )
            case ir.Assert(condition=condition, message=message, level=level, span=span):
                # No unknown is bound here, so it goes beside the equations, not among
                # them: causalization must never count an assert. The level is left as it
                # was written: it is an `AssertionLevel` enumeration literal, not a name in
                # this scope, so carrying it would try to resolve it as a variable and fail.
                self.asserts.append(ir.Assert(carry(condition), carry(message), level, span))
            case ir.Connect(a=a, b=b):
                # A connect whose endpoint sits inside a disabled conditional component
                # is removed together with the component, silently: that is the one
                # reference the specification allows to survive the removal.
                if not self._severed(a, prefix, local) and not self._severed(b, prefix, local):
                    for endpoint_a, endpoint_b in self._connect_pairs(a, b, prefix, local):
                        self.connections.join(endpoint_a, endpoint_b)
            case ir.When(branches=branches):
                self.whens.append(
                    ir.When(
                        tuple(
                            ir.WhenBranch(
                                carry(branch.condition),
                                tuple(
                                    scalar
                                    for s in branch.body
                                    for scalar in self._scalar_statements(
                                        _prefix_statement(s, prefix, local)
                                    )
                                ),
                            )
                            for branch in branches
                        )
                    )
                )
            case ir.IfEquation():
                self._if_equation(
                    equation, prefix, local, scope=scope, outer=outer, initial=initial
                )
            case ir.ForEquation():
                self._unroll(equation, prefix, local, scope=scope, outer=outer, initial=initial)

    def _unroll(  # noqa: PLR0913 -- everything `_equation` needs, plus the loop
        self,
        loop: ir.ForEquation,
        prefix: _Path,
        local: frozenset[str],
        *,
        scope: Scope,
        outer: Mapping[str, _Path] | None = None,
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
        if self._unroll_once(
            loop, (start, stop, step), prefix, local, scope, outer, initial=initial
        ):
            return
        for value in range(start, stop + (1 if step > 0 else -1), step):
            replacement = {index: ir.IntLiteral(value)}
            for inner in loop.body:
                self._equation(
                    _substitute_equation(inner, replacement),
                    prefix,
                    local,
                    scope=scope,
                    outer=outer,
                    initial=initial,
                )

    def _unroll_once(  # noqa: PLR0913, PLR0917 -- everything `_unroll` has
        self,
        loop: ir.ForEquation,
        bounds: tuple[int, int, int],
        prefix: _Path,
        local: frozenset[str],
        scope: Scope,
        outer: Mapping[str, _Path] | None,
        *,
        initial: bool = False,
    ) -> bool:
        """Write ``loop`` with each body equation carried once, not once per cell.

        For a loop (nested loops included) whose bodies hold only ``a = b`` equations over
        constant ranges: each equation is carried with the indices left as local names, and each
        cell then only fills in index values and element names (`at_cell`). The first and last
        cells are also written the eager way and must come out the same. Returns False, having
        written nothing, whenever any of that does not hold; `_unroll` then unrolls as before.
        """
        plan = self._loop_plan(loop, bounds, prefix, local)
        if plan is None:
            return False
        indices, cells = plan
        try:
            inner = local | frozenset(indices)
            templates = {
                id(equation): (
                    self._carry(equation.lhs, prefix, inner, scope, outer),
                    self._carry(equation.rhs, prefix, inner, scope, outer),
                )
                for equation in {id(e): e for _, e in cells}.values()
            }
            shapes = {".".join(path): shape for path, shape in self.shapes.items()}
            constants = {
                name: int(value) for name, value in self.values.items() if float(value).is_integer()
            }
            written = [
                ir.Equal(
                    at_cell(templates[id(equation)][0], cell, shapes, constants),
                    at_cell(templates[id(equation)][1], cell, shapes, constants),
                    equation.span,
                )
                for cell, equation in cells
            ]
        except (FlattenError, CellError, EvaluationError, KeyError, TypeError, ValueError):
            return False
        for k in {0, len(cells) - 1}:
            cell, equation = cells[k]
            values = {name: ir.IntLiteral(v) for name, v in cell.items()}
            sink = self.initial_equations if initial else self.equations
            size = len(sink)
            self._equation(
                _substitute_equation(equation, values),
                prefix,
                local,
                scope=scope,
                outer=outer,
                initial=initial,
            )
            eager = sink[size:]
            del sink[size:]
            if eager != [written[k]]:
                return False
        if not initial and len(templates) == 1:
            self._keep_loop(indices, cells, templates, shapes, constants)
        (self.initial_equations if initial else self.equations).extend(written)
        return True

    def _keep_loop(
        self,
        indices: list[str],
        cells: list[tuple[dict[str, int], ir.Equal]],
        templates: Mapping[int, tuple[ir.Expr, ir.Expr]],
        shapes: Mapping[str, tuple[int, ...]],
        constants: Mapping[str, int],
    ) -> None:
        """Record a one-equation loop over a full rectangle of cells as a `Loop`."""
        sets = tuple(
            IndexSet(name, min(c[name] for c, _ in cells), max(c[name] for c, _ in cells))
            for name in indices
        )
        if math.prod(len(s) for s in sets) != len(cells):
            return
        (left, right), equation = next(iter(templates.values())), cells[0][1]
        subscripts = [node for side in (left, right) for node in _subscripts(side)]
        read = {".".join(node.base.parts) for node in subscripts}
        used = {name for node in subscripts for i in node.indices for name in names(i)}
        self.loops.append(
            Loop(
                len(self.equations),
                ArrayEquation(
                    sets,
                    ir.Equal(left, right, equation.span),
                    tuple((name, shapes[name]) for name in sorted(read)),
                    tuple((name, constants[name]) for name in sorted(used) if name in constants),
                ),
            )
        )

    def _loop_plan(  # noqa: C901 -- one walk over the nesting
        self,
        loop: ir.ForEquation,
        bounds: tuple[int, int, int],
        prefix: _Path,
        local: frozenset[str],
    ) -> tuple[list[str], list[tuple[dict[str, int], ir.Equal]]] | None:
        """Every (cell, body equation) of ``loop`` in unroll order, or None.

        None when a body holds another kind of equation, a step is not 1, or an inner range is
        not constant.
        """
        indices: list[str] = []
        cells: list[tuple[dict[str, int], ir.Equal]] = []

        def walk(node: ir.ForEquation, span: tuple[int, int, int], cell: dict[str, int]) -> bool:
            start, stop, step = span
            if step != 1 or node.index in cell:
                return False
            if node.index not in indices:
                indices.append(node.index)
            for value in range(start, stop + 1):
                here = {**cell, node.index: value}
                for item in node.body:
                    if isinstance(item, ir.Equal):
                        cells.append((here, item))
                    elif isinstance(item, ir.ForEquation) and isinstance(item.over, ir.Range):
                        over = item.over
                        parts = (over.start, over.stop, over.step or ir.IntLiteral(1))
                        if any(set(names(part)) & set(here) for part in parts):
                            return False
                        try:
                            inner = tuple(
                                _integer(
                                    _prefix_expr(part, prefix, local, values=self.values),
                                    self.values,
                                    f"the range of 'for {item.index}'",
                                )
                                for part in parts
                            )
                        except FlattenError:
                            return False
                        if not walk(item, (inner[0], inner[1], inner[2]), here):
                            return False
                    else:
                        return False
            return True

        if not walk(loop, bounds, {}) or not cells:
            return None
        return indices, cells

    def _if_equation(  # noqa: PLR0913 -- everything `_equation` needs, plus the equation
        self,
        node: ir.IfEquation,
        prefix: _Path,
        local: frozenset[str],
        *,
        scope: Scope,
        outer: Mapping[str, _Path] | None = None,
        initial: bool = False,
        defer: bool = True,
    ) -> None:
        """Emit the body of the one branch whose condition holds -- a *structural* if.

        The condition has to be known when the model is flattened, the same rule a conditional
        component's ``if`` obeys: a parameter or a constant, evaluated here, chooses which equations
        exist. A condition on a continuous variable would make that a run-time choice -- a hybrid
        system whose active equations switch on an event -- which is a ``when``, a different machine
        and not this one; ``_boolean`` refuses such a condition by name, saying it must be known
        when the model is flattened.

        A parameter in the condition may not have its value yet: it can be set from a sibling
        declared later (the machine friction reads ``frictionParameters.PRef``, supplied by a data
        record that comes after the machine in the example). While ``defer`` is set, a condition that
        does not yet evaluate holds the whole if for :meth:`resolve_deferred_ifs`, run once every
        parameter is in. On that final pass ``defer`` is off, so a condition that *still* does not
        resolve is the genuine run-time one, and ``_boolean`` refuses it.
        """
        if not defer and (paired := self._run_time_if(node, prefix, local, scope, outer)):
            for equation in paired:
                self._equation(equation, prefix, local, scope=scope, outer=outer, initial=initial)
            return
        body = node.otherwise
        for branch in node.branches:
            # The full expression rewrite, not a bare prefix: the condition may name an enumeration
            # literal (`homotopyType == LimiterHomotopy.NoHomotopy`) or a package constant, which has
            # to become its number before the condition can be evaluated here -- the same treatment a
            # `when` condition and a top-level equation get.
            condition = self._carry(branch.condition, prefix, local, scope, outer)
            if defer:
                try:
                    evaluate(condition, self.values)
                except EvaluationError:
                    self.deferred_ifs.append(
                        _DeferredIf(node, prefix, local, scope, outer, initial)
                    )
                    return
            if _boolean(condition, self.values, "the condition of an if-equation"):
                body = branch.body
                break
        for inner in body:
            self._equation(inner, prefix, local, scope=scope, outer=outer, initial=initial)

    def _run_time_if(
        self,
        node: ir.IfEquation,
        prefix: _Path,
        local: frozenset[str],
        scope: Scope,
        outer: Mapping[str, _Path] | None,
    ) -> list[ir.Equal] | None:
        """An if-equation whose condition changes during the run, as if-expressions; else None.

        Each branch must hold the same number of plain equations and an ``else``. The equations
        are paired by position: ``if c then y = a; else y = b;`` is ``y = if c then a else b``,
        and a pair with different left sides becomes ``0 = if c then l1 - r1 else l2 - r2``.
        None when a condition is known now, so the structural reading above applies.
        """
        try:
            for branch in node.branches:
                evaluate(self._carry(branch.condition, prefix, local, scope, outer), self.values)
        except (EvaluationError, FlattenError):
            pass
        else:
            return None
        bodies = [branch.body for branch in node.branches]
        bodies.append(node.otherwise)
        if not node.otherwise or any(
            len(body) != len(node.otherwise) or not all(isinstance(e, ir.Equal) for e in body)
            for body in bodies
        ):
            return None
        paired: list[ir.Equal] = []
        for position in range(len(node.otherwise)):
            arms = [body[position] for body in bodies]
            if not all(isinstance(e, ir.Equal) for e in arms):
                return None
            equals = [e for e in arms if isinstance(e, ir.Equal)]
            same = all(e.lhs == equals[0].lhs for e in equals) and isinstance(equals[0].lhs, ir.Ref)
            values = [e.rhs if same else ir.Binary("-", e.lhs, e.rhs) for e in equals]
            combined = values[-1]
            for branch, value in zip(reversed(node.branches), reversed(values[:-1]), strict=True):
                combined = ir.IfExpr(branch.condition, value, combined)
            paired.append(ir.Equal(equals[0].lhs if same else ir.IntLiteral(0), combined))
        return paired

    def resolve_deferred_ifs(self) -> None:
        """Resolve the if-equations held because a parameter in their condition was not yet known.

        By now :meth:`instantiate` has returned and the model's parameters have all settled, so each
        held condition is a number. A branch's body may itself hold a further if that only resolves
        now, so the list is drained until empty; the final pass has ``defer`` off, so a condition that
        cannot be evaluated even here is refused rather than held again.
        """
        while self.deferred_ifs:
            pending = self.deferred_ifs
            self.deferred_ifs = []
            for held in pending:
                self._if_equation(
                    held.node,
                    held.prefix,
                    held.local,
                    scope=held.scope,
                    outer=held.outer,
                    initial=held.initial,
                    defer=False,
                )

    def fold_known_ifs(self) -> None:
        """Replace each if-expression whose condition is structural by its chosen branch.

        ``InductorDC`` writes ``v = if quasiStatic then 0 else L*der(i)`` with ``quasiStatic``
        marked ``Evaluate = true``. Kept, the if is solved for ``der(i)`` and divides by zero.
        Only constants and ``Evaluate = true`` parameters count: a plain parameter may be
        overridden at simulate time, so an if on it stays.
        """
        known = {name: value for name, value in self.values.items() if name in self.structural}
        if not known:
            return
        self.equations[:] = [_fold_ifs(e, known) for e in self.equations]
        self.initial_equations[:] = [_fold_ifs(e, known) for e in self.initial_equations]

    def _severed(self, end: ir.Ref | ir.Subscript, prefix: _Path, local: frozenset[str]) -> bool:
        """Whether this connect endpoint sits inside a disabled conditional component."""
        ref = end.base if isinstance(end, ir.Subscript) else end
        path = _prefix_ref(ref, prefix, local, values=self.values).parts
        return any(path[:length] in self.disabled for length in range(1, len(path) + 1))

    def _connect_pairs(
        self,
        a: ir.Ref | ir.Subscript,
        b: ir.Ref | ir.Subscript,
        prefix: _Path,
        local: frozenset[str],
    ) -> list[tuple[_Endpoint, _Endpoint]]:
        """The endpoint pairs a ``connect`` joins -- one, or one per element of an array of connectors.

        The polyphase and power-converter models connect whole arrays of connectors rather than
        writing the loop: a plug's pins to a replicator's outputs (`connect(internalEnable,
        booleanReplicator.y)`), or every phase's resistor to the plug (`connect(resistor.p,
        plug_p.pin)`, where `resistor` is an array of components so `resistor.p` is one `p` per
        element). Each side that names a one-dimensional array of connectors is expanded to its
        elements, and the two are joined element by element. A plain connect (neither side an array)
        is the single pair it already was; two sides of unequal array length cannot be joined.
        """
        elements_a = self._connector_array(a, prefix, local)
        elements_b = self._connector_array(b, prefix, local)
        if elements_a is None and elements_b is None:
            return [(self._endpoint(a, prefix, local), self._endpoint(b, prefix, local))]
        if elements_a is None or elements_b is None or len(elements_a) != len(elements_b):
            here_a = len(elements_a) if elements_a is not None else "a single connector"
            here_b = len(elements_b) if elements_b is not None else "a single connector"
            raise FlattenError(
                f"connect() joins connectors of unequal array length ({here_a} and {here_b});"
                " an array of connectors joins only to an array of the same length"
            )
        return list(zip(elements_a, elements_b, strict=True))

    def _connector_array(
        self, end: ir.Ref | ir.Subscript, prefix: _Path, local: frozenset[str]
    ) -> list[_Endpoint] | None:
        """The element endpoints if ``end`` names a whole one-dimensional array of connectors, else None.

        The array may sit on any segment of the reference: the last (`plug_p.pin`, where `pin[m]` is
        the array), or an earlier one (`resistor.p`, where `resistor[m]` is an array of components and
        `p` a connector of each). The segment whose flat path has a shape recorded and whose first
        element is a registered connector is the array; the elements are that path with the segment
        subscripted, the rest of the reference following each. A subscript already picks one element,
        so it is never a whole array.
        """
        if isinstance(end, ir.Subscript):
            return None
        parts = end.parts
        for split in range(len(parts)):
            head = _prefix_ref(
                ir.Ref(parts[: split + 1], subscripts=end.subscripts[: split + 1]),
                prefix,
                local,
                strict=False,
                values=self.values,
            ).parts
            shape = self.shapes.get(tuple(head))
            if shape is None or len(shape) != 1:
                continue
            tail = parts[split + 1 :]
            if (*head[:-1], element(head[-1], (1,)), *tail) not in self.connectors:
                continue
            paths = [
                (*head[:-1], element(head[-1], (index,)), *tail) for index in range(1, shape[0] + 1)
            ]
            return [(path, self._role(path, len(parts))) for path in paths]
        return None

    def _role(self, path: _Path, written: int) -> _Role:
        """``outside`` when the first written part names a connector of the class itself.

        ``connect(ra.heatPort, internal.armature)`` writes two parts on the right, and
        ``internal`` is the class's own connector, so ``internal.armature`` is an outside
        connector there (``PartialBasicMachine``), not a connector of a component.
        """
        return "outside" if path[: len(path) - written + 1] in self.connectors else "inside"

    def _endpoint(
        self, end: ir.Ref | ir.Subscript, prefix: _Path, local: frozenset[str]
    ) -> _Endpoint:
        ref = end.base if isinstance(end, ir.Subscript) else end
        path = _prefix_ref(ref, prefix, local, values=self.values).parts
        if isinstance(end, ir.Subscript):
            # A subscript picks one element of an array of connectors, and the element it
            # names is the one `_array` already declared -- so the index is structural,
            # evaluated here the way a dimension is.
            indices = tuple(
                _integer(
                    _prefix_expr(index, prefix, local, values=self.values),
                    self.values,
                    f"a subscript on connect() endpoint {'.'.join(ref.parts)!r}",
                )
                for index in end.indices
            )
            path = (*path[:-1], element(path[-1], indices))
        role = self._role(path, len(ref.parts))
        if path not in self.connectors:
            if (*path[:-1], element(path[-1], (1,))) in self.connectors:
                raise FlattenError(
                    f"connect() over a whole array of connectors is not supported yet"
                    f" ({'.'.join(path)!r}; connect the elements)",
                )
            raise FlattenError(f"connect() endpoint {'.'.join(path)!r} is not a connector")
        return (path, role)

    # -- connect expansion ------------------------------------------------------------

    def _connector_leaves(self, found: Found, scope: Scope) -> list[_ConnectorLeaf]:
        if _signal_base(found, scope) is not None:
            # A signal connector is its own single scalar leaf: no fields to walk, the connector
            # instance *is* the variable (an empty relative path). Its causality is the class's,
            # so `_one_source_only` can tell the driving end from the driven one.
            return [_ConnectorLeaf((), "none", found.cls.causality)]
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
        """The connection equations, one set per scalar leaf.

        Sets are joined per leaf, not per connector: ``connect(ambient.port, internal)`` and
        ``connect(ra.heatPort, internal.armature)`` put ``internal.armature.Q_flow`` in one
        set, which a per-connector set split in two.
        """
        leaves = _Connections()
        kinds: dict[_Endpoint, ir.ConnectorKind] = {}
        for endpoints in self.connections.sets():
            matched = self._matched_leaves(endpoints)
            self._one_source_only(endpoints)
            for leaf in matched:
                ends: list[_Endpoint] = [((*path, *leaf.path), role) for path, role in endpoints]
                for end in ends:
                    leaves.join(ends[0], end)
                    kinds[end] = leaf.kind
        for joined in leaves.sets():
            refs = [ir.Ref(path) for path, _ in joined]
            if kinds[joined[0]] == "flow":
                self._flow_sum(refs, [role for _, role in joined])
            else:
                self.equations.extend(ir.Equal(a, b) for a, b in pairwise(refs))
        seen: set[_Path] = set()
        for path, found in self.connectors.items():
            # A component's connector nobody connected from outside: its flows have nowhere to
            # go. A connector of the model itself (and the connectors nested in it) is not one.
            if (path[0],) in self.connectors:
                continue
            for leaf in self._connector_leaves(found, Scope(self.program)):
                flat = (*path, *leaf.path)
                if leaf.kind != "flow" or flat in seen or leaves.connected((flat, "inside")):
                    continue
                seen.add(flat)
                self.equations.append(ir.Equal(ir.Ref(flat), ir.RealLiteral(0.0)))

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
        Only the top-level model's ``input`` does: a component's own ``input`` seen from its
        equations (``connect(r, rInt)`` in ``Potentiometer``) is driven by whatever drives
        the component, which is already in the set.
        """
        scope = Scope(self.program)
        driving: list[str] = []
        for path, role in endpoints:
            if role == "outside" and len(path) > 1:
                continue
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


def _with_prelude(program: ir.Program) -> ir.Program:
    """The program with the predefined enumerations and the bundled ``Modelica`` package added.

    Each is added only where the program does not define a class of that name, so a user's own
    ``Modelica`` package (a full MSL, say) is the one that resolves.
    """
    defined = {cls.name for cls in program.classes}
    missing = tuple(cls for cls in (*_PRELUDE, *_bundled()) if cls.name not in defined)
    return ir.Program(missing + program.classes) if missing else program


@cache
def _bundled() -> tuple[ir.Class, ...]:
    """The bundled slice of the Modelica Standard Library (``modelica/stdlib/Modelica.mo``).

    Parsed once per process. It holds real MSL names (``Modelica.Electrical.Analog.Basic.
    Resistor``), so a model written against the MSL flattens after ``pip install modelica``.
    """
    from modelica.lang import parse  # noqa: PLC0415 - the parser imports this package

    text = (files("modelica") / "stdlib" / "Modelica.mo").read_text(encoding="utf-8")
    return parse(text, "Modelica.mo").classes


_CROSSING_OPS = frozenset({"<", "<=", ">", ">="})


def _resolve_boolean_when_conditions(
    equations: Sequence[ir.Equation], whens: Sequence[ir.When]
) -> tuple[ir.When, ...]:
    """Inline a Boolean variable's defining relation into a ``when`` condition.

    ``Boolean b = x > 2.5; when b then ... end when;`` writes the crossing once and names it,
    which the event machinery cannot read: a ``when`` fires on a threshold crossing and needs the
    relation itself, not a variable standing for it. Every continuous equation ``b = <relation>``
    whose right-hand side is such a crossing is collected, and a ``when`` condition mentioning ``b``
    gets the relation in its place, so ``when b`` becomes ``when x > 2.5`` -- the form the crossing
    detector already handles, so the whole event pipeline (causalize, emit, the Rust loop) sees a
    relation and nothing new. The defining equation stays, so ``b`` is still a variable of the model,
    observable like any other. Only a single relation counts: a compound definition (``b and c``, an
    equality ``==``) is not collected, so a condition that is not one crossing is left untouched and
    still refused downstream. A ``b`` set inside a ``when`` body (a discrete latch) is not a
    continuous equation and is not collected either -- that is edge/change territory, not this.
    """
    relations: dict[str, ir.Expr] = {
        equation.lhs.parts[0]: equation.rhs
        for equation in equations
        if isinstance(equation, ir.Equal)
        and isinstance(equation.lhs, ir.Ref)
        and len(equation.lhs.parts) == 1
        and isinstance(equation.rhs, ir.Binary)
        and equation.rhs.op in _CROSSING_OPS
    }
    if not relations:
        return tuple(whens)
    return tuple(
        ir.When(
            tuple(
                ir.WhenBranch(substitute(branch.condition, relations), branch.body)
                for branch in when.branches
            )
        )
        for when in whens
    )


def _lower_samples(
    whens: Sequence[ir.When], taken: set[str], initialized: set[str] | frozenset[str] = frozenset()
) -> tuple[tuple[ir.When, ...], tuple[FlatVariable, ...], tuple[ir.Equal, ...]]:
    """Rewrite ``when sample(start, interval)`` as a time event on a hidden discrete variable.

    ``sample`` (specification 3.7.5) is true at ``start + k * interval``. Each branch whose
    whole condition is a ``sample`` call gets a new discrete variable ``next`` that starts at
    ``start``. The condition becomes ``time >= next``, and the body gets
    ``next = pre(next) + interval``. This is the relation form that the event code already
    handles. A ``sample`` inside a larger condition (``sample(...) or b``) is not rewritten,
    and emit refuses it as before.

    A ``start`` that reads a name in ``initialized`` (a value an initial equation sets) is not
    known when the start attributes are stored, so ``next`` gets an initial equation instead,
    ordered after the one it reads.
    """
    added: list[FlatVariable] = []
    seeded: list[ir.Equal] = []
    lowered: list[ir.When] = []
    for when in whens:
        branches: list[ir.WhenBranch] = []
        for branch in when.branches:
            match branch.condition:
                case ir.Call(name="sample", args=(start, interval)):
                    index = len(added) + 1
                    while f"_sample{index}" in taken:
                        index += 1
                    name = f"_sample{index}"
                    taken.add(name)
                    ref = ir.Ref((name,))
                    if initialized.intersection(names(start)):
                        added.append(FlatVariable(name, variability="discrete"))
                        seeded.append(ir.Equal(ref, start))
                    else:
                        added.append(
                            FlatVariable(name, variability="discrete", start=start, fixed=True)
                        )
                    advance = ir.Assign(ref, ir.Binary("+", ir.Call("pre", (ref,)), interval))
                    branches.append(
                        ir.WhenBranch(
                            ir.Binary(">=", ir.Ref(("time",)), ref), (*branch.body, advance)
                        )
                    )
                case _:
                    branches.append(branch)
        lowered.append(ir.When(tuple(branches)))
    return tuple(lowered), tuple(added), tuple(seeded)


def _expand_inside(node: ir.Expr, values: Mapping[str, float]) -> ir.Expr:  # noqa: PLR0911 - one case per node type
    """``node`` with each comprehension whose ranges can be read expanded; the rest unchanged."""
    match node:
        case ir.Comprehension():
            try:
                return _expand_comprehension(node, lambda expr: expr, values)
            except (FlattenError, EvaluationError):
                return node
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, _expand_inside(lhs, values), _expand_inside(rhs, values))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, _expand_inside(operand, values))
        case ir.Call(name=name, args=args, named=named):
            return ir.Call(name, tuple(_expand_inside(a, values) for a in args), named)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(
                condition, _expand_inside(then, values), _expand_inside(otherwise, values)
            )
        case _:
            return node


def _expanded_comprehensions(mod: ir.Modification, values: Mapping[str, float]) -> ir.Modification:
    """Each comprehension in a modification's values, as the array it stands for.

    An inlined call can return one (``symmetricOrientation`` writes ``{... for k in 1:m}``), and
    the modification is prefixed next, where a comprehension is not expanded. A comprehension
    whose range cannot be read yet is left as it is.
    """
    value = None if mod.value is None else _expand_inside(mod.value, values)
    entries = {key: _expanded_comprehensions(entry, values) for key, entry in mod.entries.items()}
    if value is mod.value and all(entries[k] is e for k, e in mod.entries.items()):
        return mod
    return replace(mod, value=value, entries=entries)


class _LocalValues(Mapping[str, float]):
    """The known parameter values, read by the names the enclosing class writes.

    A call in a binding is inlined before its names are prefixed, so ``so(m)`` inside component
    ``c`` reads ``m`` while the value is stored as ``c.m``. A key whose first segment is a local
    name is looked up under the instance path; any other key is looked up as written.
    """

    def __init__(self, values: Mapping[str, float], here: _Path, local: frozenset[str]) -> None:
        self.known = values
        self.here = here
        self.local = local

    def _key(self, key: str) -> str:
        return ".".join((*self.here, key)) if key.split(".", 1)[0] in self.local else key

    def __getitem__(self, key: str) -> float:
        return self.known[self._key(key)]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._key(key) in self.known

    def __iter__(self) -> Iterator[str]:
        return iter(self.known)

    def __len__(self) -> int:
        return len(self.known)


def _fold_strings(node: ir.Expr) -> ir.Expr:  # noqa: PLR0911 - one case per node type
    """``"Dy" == "Dy"`` as ``true``, and an ``if`` on a literal as the branch it picks."""
    match node:
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            left, right = _fold_strings(lhs), _fold_strings(rhs)
            if (
                op in ("==", "<>")
                and isinstance(left, ir.StringLiteral)
                and isinstance(right, ir.StringLiteral)
            ):
                return ir.BoolLiteral((left.value == right.value) == (op == "=="))
            return ir.Binary(op, left, right)
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, _fold_strings(operand))
        case ir.Call(name=name, args=args, named=named):
            return ir.Call(name, tuple(_fold_strings(a) for a in args), named)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            test = _fold_strings(condition)
            if isinstance(test, ir.BoolLiteral):
                return _fold_strings(then if test.value else otherwise)
            return ir.IfExpr(test, _fold_strings(then), _fold_strings(otherwise))
        case _:
            return node


def _resolve_strings(flat: FlatModel, values: Mapping[str, float]) -> FlatModel:
    """Each String parameter or constant as its literal, substituted and folded away.

    A String cannot enter the parameter vector, which holds floats. Its uses are comparisons
    (``C1 == "D"`` in the transformers), so once the value is known they fold to a Boolean and
    the String variable is no longer needed. A String whose value cannot be found stays.
    The value is fixed here: a String parameter cannot be changed at simulate time.
    """
    candidates = [
        v
        for v in flat.variables
        if v.base_type == "String"
        and v.variability in ("parameter", "constant")
        and v.binding is not None
    ]
    if not candidates:
        return flat
    strings: dict[str, str] = {}
    progress = True
    while progress:
        progress = False
        for variable in candidates:
            if variable.name in strings or variable.binding is None:
                continue
            try:
                value = evaluate(variable.binding, {**values, **strings})
            except EvaluationError:
                continue
            if isinstance(value, str):
                strings[variable.name] = value
                progress = True
    if not strings:
        return flat
    mapping: dict[str, ir.Expr] = {k: ir.StringLiteral(v) for k, v in strings.items()}

    def fold(node: ir.Expr | None) -> ir.Expr | None:
        return None if node is None else _fold_strings(substitute(node, mapping))

    variables = tuple(
        replace(v, binding=fold(v.binding), start=fold(v.start))
        for v in flat.variables
        if v.name not in strings
    )
    return replace(
        flat,
        variables=variables,
        equations=tuple(
            replace(
                e,
                lhs=_fold_strings(substitute(e.lhs, mapping)),
                rhs=_fold_strings(substitute(e.rhs, mapping)),
            )
            for e in flat.equations
        ),
    )


def _assigned(body: Sequence[ir.Statement]) -> set[str]:
    """The names a statement list assigns, through ``if`` branches and loops."""
    found: set[str] = set()
    for statement in body:
        match statement:
            case ir.Assign(target=ir.Ref(name=name)):
                found.add(name)
            case ir.CallStatement(targets=targets):
                found.update(t.name for t in targets if isinstance(t, ir.Ref))
            case ir.IfStatement(branches=branches, otherwise=otherwise):
                for branch in branches:
                    found |= _assigned(branch.body)
                found |= _assigned(otherwise)
            case ir.ForStatement(body=inner) | ir.WhileStatement(body=inner):
                found |= _assigned(inner)
            case _:
                pass
    return found


def _der_arguments(node: ir.Expr) -> Iterator[ir.Ref]:
    """The references ``der`` is applied to inside ``node``."""
    match node:
        case ir.Call(name="der", args=(ir.Ref() as argument,)):
            yield argument
        case ir.Binary(lhs=lhs, rhs=rhs):
            yield from _der_arguments(lhs)
            yield from _der_arguments(rhs)
        case ir.Unary(operand=operand):
            yield from _der_arguments(operand)
        case ir.Call(args=args):
            for argument in args:
                yield from _der_arguments(argument)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            for part in (condition, then, otherwise):
                yield from _der_arguments(part)
        case _:
            pass


def _when_assigned(whens: Sequence[ir.When]) -> set[str]:
    """Every name a ``when`` assigns: discrete, with the prefix or without it.

    A ``reinit`` is not an assignment, so a state it resets stays continuous.
    """
    return {name for when in whens for branch in when.branches for name in _assigned(branch.body)}


def _inline_sample_triggers(flat: FlatModel) -> FlatModel:
    """``b = sample(...); when b then`` as ``when sample(...) then``, with ``b`` false between.

    Also ``when {a, b}`` as one ``when`` per element, and ``when initial()`` as initial
    equations, the forms the zero-order hold writes (``when {sampleTrigger, initial()}``).

    ``Blocks.Interfaces.DiscreteBlock`` writes ``sampleTrigger = sample(startTime,
    samplePeriod)`` and then ``when sampleTrigger``. A ``sample`` is true only at its instants,
    so between events ``b`` is false, and the ``when`` is the sample's time event.
    """
    triggers = {
        e.lhs.name: e.rhs
        for e in flat.equations
        if isinstance(e.lhs, ir.Ref) and isinstance(e.rhs, ir.Call) and e.rhs.name == "sample"
    }

    def swap(node: ir.Expr) -> ir.Expr:
        if isinstance(node, ir.Ref) and node.name in triggers:
            return triggers[node.name]
        return node

    whens: list[ir.When] = []
    initial: list[ir.Equal] = []
    for when in flat.whens:
        first = when.branches[0].condition
        conditions = (
            first.elements if len(when.branches) == 1 and isinstance(first, ir.Array) else (None,)
        )
        for condition in conditions:
            # `when {sampleTrigger, initial()}` fires on any element: one `when` per element.
            branches = (
                (ir.WhenBranch(condition, when.branches[0].body),)
                if condition is not None
                else when.branches
            )
            only = branches[0]
            if (
                len(branches) == 1
                and only.condition == ir.Call("initial", ())
                and all(
                    isinstance(st, ir.Assign) and isinstance(st.target, ir.Ref) for st in only.body
                )
            ):
                # `when initial()`: its assignments are the values at the start.
                initial.extend(
                    ir.Equal(st.target, st.value)
                    for st in only.body
                    if isinstance(st, ir.Assign) and isinstance(st.target, ir.Ref)
                )
                continue
            whens.append(ir.When(tuple(ir.WhenBranch(swap(b.condition), b.body) for b in branches)))
    equations = tuple(
        replace(e, rhs=ir.BoolLiteral(value=False))
        if isinstance(e.lhs, ir.Ref) and e.lhs.name in triggers
        else e
        for e in flat.equations
    )
    return replace(
        flat,
        whens=tuple(whens),
        equations=equations,
        initial_equations=(*flat.initial_equations, *initial),
    )


def flatten(program: ir.Program, model: str) -> FlatModel:
    """Flatten class ``model`` (a dotted name) as found in ``program``."""
    program = _with_prelude(program)
    scope = Scope(program)
    found = scope.resolve(model)
    if not isinstance(found, Found):
        raise FlattenError(f"{model!r} is a predefined type, not a model")
    if found.cls.restriction not in ("model", "block"):
        raise FlattenError(f"{model!r} is a {found.cls.restriction}, not a model")
    flattener = _Flattener(program)
    token = _PARAMETERS.set(flattener.values)
    shapes_token = _SHAPES.set(flattener.shapes)
    try:
        flattener.instantiate(found, scope, (), ir.EMPTY, "continuous")
        flattener.resolve_deferred_ifs()
        flattener.fold_known_ifs()
        flattener.expand_connections()
    finally:
        _PARAMETERS.reset(token)
        _SHAPES.reset(shapes_token)
    resolved = _resolve_boolean_when_conditions(flattener.equations, flattener.whens)
    equations = tuple(flattener.equations)
    flat = partition(
        _resolve_strings(
            FlatModel(
                loops=tuple(loop for loop in flattener.loops if loop.matches(equations)),
                name=found.cls.name,
                variables=tuple(flattener.variables),
                equations=equations,
                whens=tuple(resolved),
                asserts=tuple(flattener.asserts),
                initial_equations=tuple(flattener.initial_equations),
            ),
            flattener.values,
        )
    )
    # A clocked partition runs as a `when sample(0, interval)` (`clocks.as_when`), so it takes
    # the time-event path below like any other sample.
    # `parameter Real t0(fixed = false); initial equation t0 = time;` (Blocks.Math.Mean): a
    # value fixed at the initial point. It lives in the workspace as a discrete variable that
    # nothing changes after the start, so the parameter vector stays what the user set.
    targets = {e.lhs.name for e in flat.initial_equations if isinstance(e.lhs, ir.Ref)}
    demoted = {
        v.name
        for v in flat.variables
        if v.variability == "parameter" and v.fixed is False and v.name in targets
    }
    if demoted:
        flat = replace(
            flat,
            variables=tuple(
                replace(v, variability="discrete", binding=None) if v.name in demoted else v
                for v in flat.variables
            ),
        )
    flat = _inline_sample_triggers(flat)
    whens, samplers, seeded = _lower_samples(
        (*flat.whens, *(as_when(p) for p in flat.partitions)),
        {variable.name for variable in flat.variables},
        demoted,
    )
    if seeded:
        flat = replace(flat, initial_equations=(*flat.initial_equations, *seeded))
    # A state is reset with `reinit`, and a name an equation also defines is a conflict the
    # emitter refuses with its remedy; neither becomes discrete here.
    defined = {e.lhs.name for e in flat.equations if isinstance(e.lhs, ir.Ref)}
    states = {
        argument.name
        for e in flat.equations
        for side in (e.lhs, e.rhs)
        for argument in _der_arguments(side)
    }
    assigned = _when_assigned(whens) - defined - states
    variables = tuple(
        replace(v, variability="discrete")
        if v.name in assigned and v.variability == "continuous"
        else v
        for v in flat.variables
    )
    return replace(
        flat,
        variables=(*variables, *samplers),
        equations=tuple(unhold(e) for e in flat.equations) if flat.partitions else flat.equations,
        whens=whens,
    )
