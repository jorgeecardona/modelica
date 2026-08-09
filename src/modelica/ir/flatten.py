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
from typing import TYPE_CHECKING, Final, Literal

from modelica.ir import nodes as ir
from modelica.ir.evaluate import EvaluationError, evaluate
from modelica.ir.flat import BaseType, FlatModel, FlatVariable
from modelica.ir.rewrite import names

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
        case _:  # pragma: no cover - unreachable while `when` bodies are simple statements
            raise FlattenError(
                f"{type(node).__name__} inside a 'for' equation is not supported yet"
            )


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
    ir.ForStatement: "a 'for' loop",
    ir.WhileStatement: "a 'while' loop",
    ir.Break: "'break'",
    ir.Return: "'return'",
    ir.Reinit: "'reinit'",
}


def _inline_calls(  # noqa: PLR0911 -- one case per node type, as its neighbours have
    node: ir.Expr, scope: Scope, depth: int = 0
) -> ir.Expr:
    """Replace every call to a user-written ``function`` with the expression it computes."""

    def go(inner: ir.Expr) -> ir.Expr:
        return _inline_calls(inner, scope, depth)

    match node:
        case ir.Call(name=name, args=args, named=named):
            call = ir.Call(name, tuple(go(a) for a in args), tuple((k, go(v)) for k, v in named))
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
        case ir.Subscript(base=base, indices=indices):
            return ir.Subscript(base, tuple(go(i) for i in indices))
        case _:
            return node


def _inline(found: Found, scope: Scope, call: ir.Call, depth: int) -> ir.Expr:
    """One call, replaced by the expression its body assigns to the output."""
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
    if len(outputs) != 1:
        raise FlattenError(
            f"function {name!r} declares {len(outputs)} outputs; only one can be inlined"
        )

    environment = _bind(name, inputs, call)
    for component in components:
        if component.causality != "input" and component.modification.value is not None:
            environment[component.name] = component.modification.value
    environment = _sequence(found.cls.algorithms[0], environment, f"function {name!r}")

    result = environment.get(outputs[0].name)
    if result is None:
        raise FlattenError(f"function {name!r} never assigns its output {outputs[0].name!r}")
    # The body may itself call a function, and it resolves in the function's own scope.
    return _inline_calls(result, inner, depth + 1)


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


def _run(
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
        case ir.Assign(target=target, value=value):
            return {**environment, target.name: _substitute(value, environment)}
        case ir.IfStatement(branches=branches, otherwise=otherwise):
            return _merge_branches(branches, otherwise, environment, where)
        case _:
            # Named in Modelica's words rather than the node's: the reader wrote `while`,
            # and `WhileStatement` is this toolchain's business, not theirs.
            raise FlattenError(
                f"{where} uses {_SPELLING[type(statement)]}, which cannot be turned into an"
                f" expression; statements here may assign and branch, and nothing else"
            )


def _merge_branches(
    branches: Sequence[ir.StatementBranch],
    otherwise: Sequence[ir.Statement],
    environment: Mapping[str, ir.Expr],
    where: str,
) -> dict[str, ir.Expr]:
    """Run every arm against its own copy, then reconcile what they disagree about."""
    conditions = [_substitute(branch.condition, environment) for branch in branches]
    arms = [_sequence(branch.body, environment, where) for branch in branches]
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
            raise FlattenError(
                f"{name}: {replacement.type_name!r} does not satisfy {wanted!r}"
            )
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
    """Qualify the names in a ``when`` body with the instance path they were found under.

    The same two forms as :func:`_substitute_statement`, and for the same reason: a ``when``
    in an equation section holds nothing else, and the compound statements are refused here
    rather than silently returned unqualified -- which would leave a statement naming a
    variable that does not exist at this level.
    """
    match node:
        case ir.Assign(target=target, value=value):
            return ir.Assign(_prefix_ref(target, prefix, local), _prefix_expr(value, prefix, local))
        case ir.Reinit(target=target, value=value):
            return ir.Reinit(_prefix_ref(target, prefix, local), _prefix_expr(value, prefix, local))
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
                raise FlattenError(
                    f"{'.'.join((*prefix, name))}: redeclaring a class is not supported yet"
                )
            raise FlattenError(f"{found.cls.name} has no component {name!r} to redeclare")
        for name, member in merged.members.items():
            self._component(
                member,
                (*prefix, name),
                outer.entries.get(name, ir.EMPTY),
                floor,
                local,
                writer=scope,
            )
        for equation in merged.equations:
            self._equation(equation, prefix, local, scope=inner)
        for equation in merged.initial_equations:
            self._equation(equation, prefix, local, scope=inner, initial=True)
        for section in found.cls.algorithms:
            for equation in _section_equations(section, found.cls.name):
                self._equation(equation, prefix, local, scope=inner)

    def _component(  # noqa: PLR0913 -- the writer's scope is what resolves a redeclaration
        self,
        member: Member,
        path: _Path,
        outer: ir.Modification,
        floor: ir.Variability,
        local: frozenset[str],
        *,
        writer: Scope,
    ) -> None:
        if outer.redeclare is not None:
            member = _redeclared(member, outer, writer, path)
            outer = replace(outer, redeclare=None)
        component = member.component
        if component.dimensions:
            self._array(member, path, outer, floor, local, writer=writer)
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

    def _array(  # noqa: PLR0913 -- the writer's scope travels with every component
        self,
        member: Member,
        path: _Path,
        outer: ir.Modification,
        floor: ir.Variability,
        local: frozenset[str],
        *,
        writer: Scope,
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
                writer=writer,
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
        self,
        equation: ir.Equation,
        prefix: _Path,
        local: frozenset[str],
        *,
        scope: Scope,
        initial: bool = False,
    ) -> None:
        def carry(node: ir.Expr) -> ir.Expr:
            # Inlining comes first: a call is resolved in the scope of the class that wrote
            # it, and what comes back is an expression in that class's own names, which is
            # exactly what prefixing then qualifies.
            return _prefix_expr(
                _inline_calls(node, scope), prefix, local, values=self.values
            )

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
                self._unroll(equation, prefix, local, scope=scope, initial=initial)

    def _unroll(
        self,
        loop: ir.ForEquation,
        prefix: _Path,
        local: frozenset[str],
        *,
        scope: Scope,
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
                    _substitute_equation(inner, replacement),
                    prefix,
                    local,
                    scope=scope,
                    initial=initial,
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
