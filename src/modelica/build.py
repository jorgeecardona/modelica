"""The Python front-end: write a Modelica model as Python, get :mod:`modelica.ir` back.

This is a peer of :mod:`modelica.lang`, not a wrapper around it. A ``.mo`` file and the
Python model below produce the same :class:`modelica.ir.Class`, and the example suite
asserts exactly that::

    model NewtonCooling "A hot object cooling toward ambient"
      parameter Real k(unit = "1/s") = 0.5 "Cooling coefficient";
      Real T(unit = "degC", start = 90, fixed = true) "Object temperature";
    equation
      der(T) = -k * T;
    end NewtonCooling;

::

    class NewtonCooling(Model):
        '''A hot object cooling toward ambient'''

        k = parameter(Real, 0.5, unit="1/s", doc="Cooling coefficient")
        T = Real(unit="degC", start=90, fixed=True, doc="Object temperature")

        def equations(self):
            yield der(self.T) == -self.k * self.T

Four things make that work, and they are the whole module:

* **Declarations are descriptors.** ``T`` is a :class:`Declaration`; ``__set_name__``
  hands it its own name when the class body is executed, and ``__get__`` hands back a
  plain :class:`Expression` referring to it. So ``self.T`` inside ``equations`` already
  reads as ``Real T`` without any model instance ever existing.
* **``==`` builds an equation, not a bool.** Modelica's ``=`` is acausal -- it states a
  relation rather than assigning -- so mapping it to Python's ``==`` is the honest
  reading. Comparisons (``<``, ``<=``) still produce ordinary boolean expressions.
* **Python inheritance is ``extends``.** ``class Resistor(OnePort)`` emits
  ``extends OnePort;`` and contributes only its own equations, exactly as Modelica
  accumulates them.
* **A Python module is a ``package``.** Define the classes, then call :func:`package` at
  the bottom of the file.

Every value here is a frozen dataclass, in the same spirit as :mod:`modelica.ir`:
``flow(...)`` and friends return a new declaration rather than mutating one, so a builder
expression means the same thing wherever it appears.

One subtlety worth stating, because it dictated the design. Python's rich-comparison rule
gives priority to the *right* operand when its type is a proper subclass of the left's.
Had ``Declaration`` subclassed ``Expression``, ``der(self.h) == self.v`` would have run
``self.v.__eq__`` and quietly produced ``v = der(h)``. Acausally that is the same
equation, but it is not the same text, and the ``.mo``-equivalence test would have caught
it as a mismatch with no obvious cause. Keeping ``Declaration`` outside the ``Expression``
hierarchy removes the possibility rather than documenting it.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast, overload

from modelica import ir
from modelica.lang.unparse import unparse, unparse_expr

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from types import ModuleType

__all__ = [
    "Block",
    "Boolean",
    "Connector",
    "Declaration",
    "Expression",
    "Function",
    "Integer",
    "Model",
    "Modifier",
    "Outputs",
    "Package",
    "Real",
    "Record",
    "Redeclaration",
    "String",
    "Type",
    "alias",
    "and_",
    "array",
    "assert_",
    "assign",
    "atan2",
    "break_",
    "call",
    "connect",
    "constant",
    "cos",
    "der",
    "discrete",
    "enumeration",
    "equal",
    "exp",
    "flow",
    "for_",
    "for_statement",
    "icon",
    "if_",
    "if_statement",
    "import_",
    "inner",
    "input_",
    "log",
    "maximum",
    "minimum",
    "mod",
    "no_event",
    "not_",
    "not_equal",
    "or_",
    "outer",
    "output",
    "package",
    "parameter",
    "place",
    "pre",
    "range_",
    "redeclare",
    "redeclared",
    "ref",
    "reinit",
    "replaceable",
    "results",
    "return_",
    "shape",
    "sign",
    "sin",
    "sqrt",
    "tan",
    "time",
    "to_ir",
    "to_modelica",
    "when",
    "while_statement",
]

# Unquoted despite naming things defined further down: a `type` alias is evaluated lazily,
# so a forward reference needs no quotes here.
type Operand = Expression | int | float | bool | str
type TypeRef = Type | _ClassMeta


def _lift(value: object) -> ir.Expr:
    """Turn a Python value into an IR expression node.

    A negative number comes back as a *negated literal*, not a negative one, because that is
    what the same text parses to: Modelica has no negative literals, only a unary minus in
    front of one, and `-a` already builds `Unary` here through `__neg__`. Python folds the
    sign before this function ever sees it, so the folding is undone. Without that the two
    front ends build different trees for `R = -1` -- equal in value, unequal as IR -- and the
    printer renders both as `-1`, which is why a text comparison never noticed.
    """
    if isinstance(value, Expression):
        return value.node
    # bool before int: bool is a subclass of int, and `true` is not `1` in Modelica.
    if isinstance(value, bool):
        return ir.BoolLiteral(value)
    if isinstance(value, (int, float)) and value < 0:
        return ir.Unary("-", _lift(-value))
    if isinstance(value, int):
        return ir.IntLiteral(value)
    if isinstance(value, float):
        return ir.RealLiteral(value)
    if isinstance(value, str):
        return ir.StringLiteral(value)
    raise TypeError(f"cannot use {value!r} in a Modelica expression")


def _as_ref(value: Operand) -> ir.Ref:
    node = _lift(value)
    if not isinstance(node, ir.Ref):
        raise TypeError(f"expected a variable reference, got {unparse_expr(node)}")
    return node


# --------------------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Expression:
    """A handle on an IR expression, with Modelica's operators bound to Python's.

    ``eq=False`` keeps the generated structural ``__eq__`` away: here ``==`` has to build
    :class:`modelica.ir.Equal`, which is what makes ``der(self.h) == self.v`` read as an
    equation rather than a test. Comparing two expressions structurally means comparing
    their ``node`` values, which are ordinary dataclasses with ordinary equality.
    """

    node: ir.Expr

    def __repr__(self) -> str:
        return f"<{unparse_expr(self.node)}>"

    def __getattr__(self, name: str) -> Expression:
        # Only reached when normal lookup fails, so this never shadows a real attribute.
        if name.startswith("_"):
            raise AttributeError(name)
        base = self.node
        if not isinstance(base, ir.Ref):
            # Must stay AttributeError: anything else breaks getattr/hasattr protocol.
            raise AttributeError(f"{self!r} has no member {name!r}")  # noqa: TRY004
        return Expression(ir.Ref((*base.parts, name)))

    def __getitem__(self, index: Operand | tuple[Operand, ...]) -> Expression:
        """``self.T[i]`` and ``self.A[i, j]`` -- a subscript, the same node the parser builds.

        Only a plain reference can be subscripted, which is also true in Modelica: the index
        goes on a component, not on the result of an expression.
        """
        base = self.node
        if not isinstance(base, ir.Ref):
            raise TypeError(f"{self!r} is not a component, so it cannot be subscripted")
        indices = index if isinstance(index, tuple) else (index,)
        return Expression(ir.Subscript(base, tuple(_lift(i) for i in indices)))

    # Defining `__eq__` would otherwise set `__hash__` to None. Identity is the only
    # sensible hash here, since `==` does not answer "are these the same expression?".
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Operand) -> ir.Equal:  # pyright: ignore[reportIncompatibleMethodOverride]
        """``a == b`` is the Modelica equation ``a = b``."""
        return ir.Equal(self.node, _lift(other))

    def _binary(self, op: ir.BinaryOp, other: Operand) -> Expression:
        return Expression(ir.Binary(op, self.node, _lift(other)))

    def _rbinary(self, op: ir.BinaryOp, other: Operand) -> Expression:
        return Expression(ir.Binary(op, _lift(other), self.node))

    def __add__(self, other: Operand) -> Expression:
        return self._binary("+", other)

    def __radd__(self, other: Operand) -> Expression:
        return self._rbinary("+", other)

    def __sub__(self, other: Operand) -> Expression:
        return self._binary("-", other)

    def __rsub__(self, other: Operand) -> Expression:
        return self._rbinary("-", other)

    def __mul__(self, other: Operand) -> Expression:
        return self._binary("*", other)

    def __rmul__(self, other: Operand) -> Expression:
        return self._rbinary("*", other)

    def __truediv__(self, other: Operand) -> Expression:
        return self._binary("/", other)

    def __rtruediv__(self, other: Operand) -> Expression:
        return self._rbinary("/", other)

    def __pow__(self, other: Operand) -> Expression:
        return self._binary("^", other)

    def __rpow__(self, other: Operand) -> Expression:
        return self._rbinary("^", other)

    def __lt__(self, other: Operand) -> Expression:
        return self._binary("<", other)

    def __le__(self, other: Operand) -> Expression:
        return self._binary("<=", other)

    def __gt__(self, other: Operand) -> Expression:
        return self._binary(">", other)

    def __ge__(self, other: Operand) -> Expression:
        return self._binary(">=", other)

    def __neg__(self) -> Expression:
        return Expression(ir.Unary("-", self.node))

    def __pos__(self) -> Expression:
        return Expression(ir.Unary("+", self.node))


def call(name: str, *args: Operand) -> Expression:
    """Any Modelica function application, built-in or user-defined."""
    return Expression(ir.Call(name, tuple(_lift(a) for a in args)))


def der(x: Operand) -> Expression:
    """The time derivative. The presence of ``der(x)`` is what makes ``x`` a state."""
    return call("der", x)


def pre(x: Operand) -> Expression:
    """The value ``x`` held immediately before the current event."""
    return call("pre", x)


def sqrt(x: Operand) -> Expression:
    return call("sqrt", x)


def sin(x: Operand) -> Expression:
    return call("sin", x)


def cos(x: Operand) -> Expression:
    return call("cos", x)


def tan(x: Operand) -> Expression:
    return call("tan", x)


def exp(x: Operand) -> Expression:
    return call("exp", x)


def log(x: Operand) -> Expression:
    return call("log", x)


def sign(x: Operand) -> Expression:
    return call("sign", x)


def minimum(a: Operand, b: Operand) -> Expression:
    """``min(a, b)`` -- the smaller of the two. Named in full so it never shadows ``min``."""
    return call("min", a, b)


def maximum(a: Operand, b: Operand) -> Expression:
    """``max(a, b)`` -- the larger of the two, spelled out for the same reason as `minimum`."""
    return call("max", a, b)


def atan2(y: Operand, x: Operand) -> Expression:
    """``atan2(y, x)`` -- the angle of the point ``(x, y)``, quadrant included."""
    return call("atan2", y, x)


def no_event(x: Operand) -> Expression:
    """``noEvent(x)`` -- evaluate without generating a zero crossing."""
    return call("noEvent", x)


def not_equal(a: Operand, b: Operand) -> Expression:
    """``a <> b``.

    Spelled out because ``!=`` is deliberately left alone: ``==`` is already diverted to
    build an equation, and diverting only one of the pair is less surprising than both.
    """
    return Expression(ir.Binary("<>", _lift(a), _lift(b)))


def equal(a: Operand, b: Operand) -> Expression:
    """The comparison ``a == b``, which the operator itself cannot build here.

    ``==`` builds an equation here, so a condition that asks whether two values are equal --
    ``if schedule == Schedule.day then`` -- has no operator left to spell it. This is that
    spelling, and it is the mirror of :func:`not_equal`.
    """
    return Expression(ir.Binary("==", _lift(a), _lift(b)))


def and_(a: Operand, *rest: Operand) -> Expression:
    """``a and b and ...``.

    A function and not an operator, because Python's ``and`` cannot be overloaded: it
    short-circuits on the *truth* of its left side, so it would ask an `Expression` whether
    it is true and answer with one of the two operands rather than with a conjunction.
    ``&`` was the other candidate and is not taken -- it reads as bitwise, and a model that
    says ``x & y`` when it means ``x and y`` is a model that has to be decoded.

    Found missing by `PhaseChange`, whose pearlite reaction runs in a temperature *window*.
    """
    return _chain("and", a, rest)


def or_(a: Operand, *rest: Operand) -> Expression:
    """``a or b or ...``. See :func:`and_` for why this is not an operator."""
    return _chain("or", a, rest)


def not_(a: Operand) -> Expression:
    """``not a``. Python's ``not`` cannot be overloaded either, for the same reason."""
    return Expression(ir.Unary("not", _lift(a)))


def _chain(operator: ir.BinaryOp, first: Operand, rest: tuple[Operand, ...]) -> Expression:
    """Fold left, so ``and_(a, b, c)`` is ``(a and b) and c`` -- what the parser builds."""
    node = _lift(first)
    for other in rest:
        node = ir.Binary(operator, node, _lift(other))
    return Expression(node)


@dataclass(frozen=True)
class Outputs:
    """The left side of an output expression list, waiting for the call it reads.

    ``==`` builds an equation everywhere else in this module, and it does here too -- the
    difference is only which equation. `results(a, b) == call("split", x)` renders
    ``(a, b) = split(x);``, which is the spelling `modelica.lang` parses.
    """

    targets: tuple[ir.Ref | None, ...]

    def __eq__(self, other: object) -> ir.CallEquation:  # type: ignore[override]
        node = _lift(other)  # pyright: ignore[reportArgumentType]
        if not isinstance(node, ir.Call):
            message = "an output list is assigned from a function call, not an expression"
            raise TypeError(message)
        return ir.CallEquation(self.targets, node)

    def __hash__(self) -> int:
        return hash(self.targets)


def results(*targets: Operand | None) -> Outputs:
    """``(a, b) = f(x)`` -- the only way to read a function that returns two things.

    ``None`` for a result the caller does not want, which is Modelica's ``(, b) = f(x)``.
    """
    return Outputs(tuple(None if t is None else _as_ref(t) for t in targets))


def array(*elements: Operand) -> Expression:
    """``{a, b, c}`` -- an array literal, which is how an array component is given values.

    ``parameter Real t[4] = {0, 1, 2, 3}`` from the other front end. A Python list cannot
    stand in for it: `_lift` turns a value into an expression and a list is not one, so the
    literal needs a name.
    """
    return Expression(ir.Array(tuple(_lift(e) for e in elements)))


def range_(start: Operand, stop: Operand, step: Operand | None = None) -> Expression:
    """``a:b``, and ``a:s:b`` when a step is given -- the header of a :func:`for_`."""
    return Expression(ir.Range(_lift(start), _lift(stop), None if step is None else _lift(step)))


def for_(index: str, over: Operand, *body: ir.Equation) -> ir.ForEquation:
    """``for i in 1:n loop ... end for;`` -- one index, one range.

    The index is named as a string and referred to with :func:`ref`, because it is not a
    component: no declaration introduces it and ``self`` cannot answer for it. Writing the
    loop out in Python instead would produce the same *flat* model and a different `.mo`,
    and the corpus holds both front ends to the same text.
    """
    return ir.ForEquation(index, _lift(over), body)


def if_(condition: Operand, then: Operand, otherwise: Operand) -> Expression:
    """A conditional *expression*.

    Generates a state event unless wrapped in :func:`no_event`.
    """
    return Expression(ir.IfExpr(_lift(condition), _lift(then), _lift(otherwise)))


def ref(name: str) -> Expression:
    """A bare reference to another declaration of the same class.

    Inside ``equations`` a name is reached as ``self.x``. A *parameter* computed from
    other parameters, and a component modification that reads one, are written in the
    class body instead -- and there no instance exists yet, so ``self`` is out of reach.
    A declaration does not even know its own name at that point: Python assigns it
    afterwards, through ``__set_name__``.

    Modelica writes these as plain references::

        parameter Real m_dry = m_green * (1 - moisture);
        Electrical.Resistor winding(R = Rw);

    and this is how to spell one from Python::

        m_dry = parameter(Real, ref("m_green") * (1 - ref("moisture")), unit="kg")
        winding = Electrical.Resistor(R=ref("Rw"))
    """
    return Expression(ir.Ref((name,)))


time = Expression(ir.Ref(("time",)))
"""The built-in simulation time variable."""


# --------------------------------------------------------------------------------------
# Equations and statements
# --------------------------------------------------------------------------------------


def connect(a: Operand, b: Operand) -> ir.Connect:
    """``connect(a, b)`` -- equate the potential variables, sum the flow variables."""
    return ir.Connect(_as_ref(a), _as_ref(b))


def reinit(target: Operand, value: Operand) -> ir.Reinit:
    """Restart a state at a new value. Only legal inside :func:`when`."""
    return ir.Reinit(_as_ref(target), _lift(value))


def assign(target: Operand, value: Operand) -> ir.Assign:
    """Assign a discrete variable. Only legal inside :func:`when`."""
    return ir.Assign(_as_ref(target), _lift(value))


def when(condition: Operand, *body: ir.Statement) -> ir.When:
    """``when condition then body end when;`` -- fires at the instant it becomes true."""
    return ir.When((ir.WhenBranch(_lift(condition), body),))


def assert_(condition: Operand, message: str, *, warning: bool = False) -> ir.Assert:
    """``assert(condition, message);`` -- a runtime check the simulator evaluates.

    A false ``condition`` stops the run with ``message``; ``warning=True`` downgrades it to
    ``AssertionLevel.warning``, which reports the message once and lets the run finish. It binds
    no unknown, so it never changes what the model computes -- it only guards the model's range.
    """
    level = ir.Ref(("AssertionLevel", "warning")) if warning else None
    return ir.Assert(_lift(condition), _lift(message), level)


def break_() -> ir.Break:
    """``break;`` -- leave the innermost loop of an ``algorithm`` section."""
    return ir.Break()


def return_() -> ir.Return:
    """``return;`` -- leave the enclosing function."""
    return ir.Return()


def if_statement(
    condition: Operand,
    *body: ir.Statement,
    elseif: Sequence[tuple[Operand, Sequence[ir.Statement]]] = (),
    otherwise: Sequence[ir.Statement] = (),
) -> ir.IfStatement:
    """``if c then ... else ... end if;`` as a *statement*, which :func:`if_` is not.

    The two are different constructs that Modelica spells with the same word. :func:`if_`
    builds an expression and has a value; this builds control flow and has none, so its
    branches hold statements and its ``else`` may be absent altogether. One name for both
    would make the difference invisible at the call site, which is where it matters.

    ``elseif`` takes ``(condition, body)`` pairs and is not sugar for nesting: an
    :class:`~modelica.ir.IfStatement` holds its branches in a list, so ``elseif b then``
    and ``else if b then`` are *different values* here -- unlike the expression form, where
    nesting is the only representation there is. Passing the pairs keeps the value flat and
    keeps the printed source the same shape as what was written.
    """
    branches = [ir.StatementBranch(_lift(condition), body)]
    branches += [ir.StatementBranch(_lift(c), tuple(b)) for c, b in elseif]
    return ir.IfStatement(tuple(branches), tuple(otherwise))


def for_statement(index: str, over: Operand, *body: ir.Statement) -> ir.ForStatement:
    """``for i in 1:n loop ... end for;`` over statements -- the counterpart of :func:`for_`."""
    return ir.ForStatement(index, _lift(over), body)


def while_statement(condition: Operand, *body: ir.Statement) -> ir.WhileStatement:
    """``while c loop ... end while;`` -- no equation counterpart exists."""
    return ir.WhileStatement(_lift(condition), body)


# --------------------------------------------------------------------------------------
# Modifications
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Modifier:
    """The result of :func:`mod` -- a modification aimed at a nested component."""

    modification: ir.Modification


@dataclass(frozen=True, slots=True)
class Redeclaration:
    """The result of :func:`redeclare` -- a replacement waiting for the name it replaces.

    A redeclaration is written as a keyword argument, so the component it replaces is the
    *key*, and a value cannot see its own key. :func:`_modification` fills the name in when
    it files the entry, which is the only place both halves are in hand at once.
    """

    type_name: str
    modification: ir.Modification


def _entry(name: str, attribute: object) -> ir.Modification:
    if isinstance(attribute, Redeclaration):
        return ir.Modification(
            redeclare=ir.Component(
                name=name, type_name=attribute.type_name, modification=attribute.modification
            )
        )
    if isinstance(attribute, Modifier):
        return attribute.modification
    return ir.Modification(value=_lift(attribute))


def _modification(attributes: Mapping[str, object], value: object = None) -> ir.Modification:
    entries = {name: _entry(name, attribute) for name, attribute in attributes.items()}
    return ir.Modification(value=None if value is None else _lift(value), entries=entries)


def _shape(shape: object) -> tuple[ir.Expr, ...]:
    """``shape=n`` and ``shape=(2, 3)`` both give the dimensions of a declaration."""
    if shape is None:
        return ()
    dimensions: tuple[object, ...] = shape if isinstance(shape, tuple) else (shape,)  # pyright: ignore[reportUnknownVariableType]
    return tuple(_lift(d) for d in dimensions)


def redeclared(declaration: Declaration) -> Declaration:
    """``redeclare Capacitor element;`` -- an inherited component is now a different class.

    Distinct from :func:`redeclare`, and the difference is where it is written rather than
    what it means. :func:`redeclare` goes inside a modification, as an argument to the
    component that owns the slot; this one stands in the class body of a subclass, under the
    ``extends`` that brought the slot in. Both end at the same rule in the flattener.
    """
    return replace(declaration, redeclaring=True)


def inner(declaration: Declaration) -> Declaration:
    """``inner Environment environment;`` -- the one every ``outer`` below this will find."""
    return replace(declaration, scoping="inner")


def outer(declaration: Declaration) -> Declaration:
    """``outer Environment environment;`` -- a name for the ``inner`` further out.

    Declares nothing. The declaration handed in supplies only the *type*, which is what
    Modelica asks for and what the match is checked against.
    """
    return replace(declaration, scoping="outer")


def replaceable(declaration: Declaration, constrainedby: TypeRef | None = None) -> Declaration:
    """Offer this component's class as a slot: ``replaceable Resistor e constrainedby OnePort``.

    Without a constraint the declared class is its own, which is what Modelica means by a
    bare ``replaceable``: the slot takes a resistor and nothing else.
    """
    return replace(declaration, replaceable=True, constrainedby=constrainedby)


def redeclare(type_ref: TypeRef, **modifications: object) -> Redeclaration:
    """Fill a slot with a different class: ``rig(redeclare Capacitor element(C = 0.01))``.

    The replacement is named as it is seen from *outside* its package, because a
    redeclaration is written by whoever instantiates the slot and that is nearly always
    somewhere else. A model inside the same package as its replacement would want the bare
    name; none exists here, and this is where to look when one does.
    """
    return Redeclaration(_qualified(type_ref, ""), _modification(modifications))


def mod(**attributes: object) -> Modifier:
    """Modify a nested component: ``Capacitor(C=0.01, v=mod(start=0, fixed=True))``.

    That is Modelica's ``Capacitor c(C = 0.01, v(start = 0, fixed = true));``.
    """
    return Modifier(_modification(attributes))


# --------------------------------------------------------------------------------------
# Graphics -- what a component looks like
# --------------------------------------------------------------------------------------


def _coordinates(points: Iterable[tuple[float, float]]) -> ir.Array:
    """``{{x, y}, ...}`` -- a point list in the icon's own -100..100 frame."""
    return ir.Array(tuple(ir.Array((_lift(x), _lift(y))) for x, y in points))


def shape(name: str, **arguments: object) -> Expression:
    """One graphical primitive: ``shape("Line", points=[(-90, 0), (90, 0)])``.

    A thin wrapper on purpose. The Modelica graphics language has dozens of attributes and
    they are all ordinary named arguments, so spelling them out here would be a second,
    smaller, drifting copy of the specification. ``points`` and ``extent`` are the only ones
    given special treatment, because a list of pairs is what a person wants to write and
    ``{{-90, 0}, {90, 0}}`` is what Modelica reads.
    """
    named: list[tuple[str, ir.Expr]] = []
    for key, value in arguments.items():
        if key in ("points", "extent") and isinstance(value, list):
            named.append((key, _coordinates(cast("list[tuple[float, float]]", value))))
        elif isinstance(value, str) and key.endswith(("Pattern", "Style", "Color")):
            # `FillPattern.Solid` is a reference, not the string "FillPattern.Solid".
            named.append((key, ir.Ref(tuple(value.split(".")))))
        else:
            named.append((key, _lift(value)))
    return Expression(ir.Call(name, (), tuple(named)))


def place(extent: list[tuple[float, float]], rotation: float = 0.0) -> ir.Modification:
    """``annotation(Placement(transformation(extent = {{...}, {...}})))``.

    Where a component or a connector sits, in the frame that holds it. Without it a diagram
    has to guess where a wire should attach, and every tool guesses differently.

    ``rotation`` turns the instance, counter-clockwise, in degrees -- how a component ends
    up standing on the vertical side of a loop. Omitted when it is zero, because that is
    what the parser reads back and the two front ends are compared as text.
    """
    entries = {"extent": ir.Modification(value=_coordinates(extent))}
    if rotation:
        entries["rotation"] = ir.Modification(value=_lift(rotation))
    inner = ir.Modification(entries=entries)
    return ir.Modification(
        entries={"Placement": ir.Modification(entries={"transformation": inner})}
    )


def icon(*shapes: Expression) -> ir.Modification:
    """``annotation(Icon(graphics = {...}))`` -- assign it to ``ICON`` in a model body."""
    graphics = ir.Modification(value=ir.Array(tuple(s.node for s in shapes)))
    return ir.Modification(entries={"Icon": ir.Modification(entries={"graphics": graphics})})


# --------------------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Declaration:
    """One declared component, and the descriptor that names it.

    ``__set_name__`` learns the declaration's own name when the enclosing class body is
    executed; ``__get__`` then answers with a reference to it, which is why ``self.T``
    works inside ``equations`` without any instance of the model.
    """

    type_ref: TypeRef
    doc: str = ""
    modification: ir.Modification = ir.EMPTY
    variability: ir.Variability = "continuous"
    causality: ir.Causality = "none"
    connector: ir.ConnectorKind = "none"
    name: str = ""
    dimensions: tuple[ir.Expr, ...] = ()
    annotation: ir.Modification = ir.EMPTY
    scoping: ir.Scoping = "none"
    redeclaring: bool = False
    """Set by :func:`redeclared`. Files this under a class's redeclarations, not its components."""
    replaceable: bool = False
    constrainedby: TypeRef | None = None
    """Held as the class itself, not as its name, because a name depends on who is asking.

    ``constrainedby OnePort`` inside the electrical package and ``constrainedby
    Electrical.OnePort`` outside it are the same constraint, and only
    :meth:`component` knows which is being written.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        # The one mutation in the module, and it is unavoidable: a descriptor learns its
        # name only after the object exists. Everything else returns a new value.
        object.__setattr__(self, "name", name)

    @overload
    def __get__(self, instance: None, owner: type) -> Expression: ...
    @overload
    def __get__(self, instance: object, owner: type) -> Expression: ...

    def __get__(self, instance: object | None, owner: type) -> Expression:
        del instance, owner
        return Expression(self.reference)

    @property
    def reference(self) -> ir.Ref:
        if not self.name:
            raise RuntimeError(
                "this declaration was never assigned to a name in a model body, "
                "so it cannot be referred to"
            )
        return ir.Ref((self.name,))

    def component(self, within: str) -> ir.Component:
        return ir.Component(
            name=self.name,
            type_name=_qualified(self.type_ref, within),
            dimensions=self.dimensions,
            annotation=self.annotation,
            modification=self.modification,
            variability=self.variability,
            causality=self.causality,
            connector=self.connector,
            scoping=self.scoping,
            replaceable=self.replaceable,
            constraining=(
                "" if self.constrainedby is None else _qualified(self.constrainedby, within)
            ),
            comment=self.doc,
        )


def flow(declaration: Declaration) -> Declaration:
    """Mark a connector variable as a flow variable -- one that sums to zero at a node."""
    return replace(declaration, connector="flow")


def input_(declaration: Declaration) -> Declaration:
    return replace(declaration, causality="input")


def output(declaration: Declaration) -> Declaration:
    return replace(declaration, causality="output")


def discrete(declaration: Declaration) -> Declaration:
    """Mark a variable as piecewise constant -- it changes only at events."""
    return replace(declaration, variability="discrete")


def parameter(
    type_ref: TypeRef, value: object = None, *, doc: str = "", **attributes: object
) -> Declaration:
    """``parameter Real k(unit = "1/s") = 0.5 "Cooling coefficient";``.

    Reads in the same order as the Modelica line: prefix, type, value, comment.
    """
    return Declaration(
        type_ref,
        doc=doc,
        modification=_modification(attributes, value),
        variability="parameter",
    )


def constant(
    type_ref: TypeRef, value: object = None, *, doc: str = "", **attributes: object
) -> Declaration:
    return Declaration(
        type_ref,
        doc=doc,
        modification=_modification(attributes, value),
        variability="constant",
    )


@dataclass(frozen=True, eq=False)
class Type:
    """A predefined type (``Real``, ``Integer``, ``Boolean``) or an alias of one.

    Calling it declares a component: ``Real(start=1, fixed=True, doc="Height")``.
    """

    name: str
    base: Type | None = None
    attributes: Mapping[str, object] = field(default_factory=dict[str, object])
    module: str = ""
    doc: str = ""
    literals: tuple[str, ...] = ()
    """The literals of an enumeration, in order. Empty for every other kind of type."""

    def __getattr__(self, literal: str) -> Expression:
        """``Schedule.night`` -- a reference to one literal of this enumeration.

        Only reached when ordinary lookup fails, so it shadows nothing. It answers with a
        dotted reference rather than a number: flattening is what turns a literal into its
        ordinal, and doing it here would put the same rule in two places.
        """
        if literal.startswith("_") or literal not in self.literals:
            raise AttributeError(f"{self.name} has no literal {literal!r}")
        # Named from outside its package, for the reason `redeclare` is: whoever writes
        # `Schedule.night` is nearly always somewhere else, and only the class body being
        # written knows whether it is. See :func:`redeclare` for the same caveat.
        return Expression(ir.Ref((*_qualified(self, "").split("."), literal)))

    def __repr__(self) -> str:
        return f"<Type {self.name}>"

    def __call__(self, doc: str = "", *, shape: object = None, **attributes: object) -> Declaration:
        return Declaration(
            self, doc=doc, modification=_modification(attributes), dimensions=_shape(shape)
        )


Real = Type("Real")
Integer = Type("Integer")
Boolean = Type("Boolean")
String = Type("String")


def enumeration(name: str, *literals: str, doc: str = "") -> Type:
    """A new type whose values are names, written in Modelica as ``type Schedule = ...``.

    Not an alias of anything -- an enumeration defines a type rather than renaming one, so
    it has no base and no attributes. Its literals are reached as attributes,
    ``Schedule.night``, which reads like the Modelica and, more usefully, fails loudly on a
    literal that was never declared. A bare string would not.
    """
    caller = sys._getframe(1)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    return Type(name, literals=literals, doc=doc, module=str(caller.f_globals["__name__"]))


def alias(name: str, base: Type, **attributes: object) -> Type:
    """``type Voltage = Real(unit = "V");``.

    The name is passed explicitly because a module-level assignment, unlike a class
    attribute, never calls ``__set_name__``.
    """
    caller = sys._getframe(1)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    return Type(name, base=base, attributes=attributes, module=str(caller.f_globals["__name__"]))


# --------------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------------


class _ClassMeta(type):
    """Makes a model class callable as a component declaration.

    ``Electrical.Resistor(R=100)`` does not instantiate anything -- it declares a
    component of that type, which is exactly what the same text means in Modelica.
    """

    __modelica_restriction__: ir.Restriction
    __modelica_partial__: bool
    __modelica_marker__: bool

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        *,
        restriction: ir.Restriction | None = None,
        partial: bool = False,
    ) -> _ClassMeta:
        cls = super().__new__(mcls, name, bases, namespace)
        if restriction is not None:
            cls.__modelica_restriction__ = restriction
        # A class that names its own restriction is one of the marker bases below, not a
        # user model: it defines what `model` or `connector` means rather than being one.
        cls.__modelica_marker__ = restriction is not None
        cls.__modelica_partial__ = partial
        return cls

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        *,
        restriction: ir.Restriction | None = None,
        partial: bool = False,
    ) -> None:
        # `type.__init__` has no use for the keywords `__new__` consumed; they stop here.
        del restriction, partial
        super().__init__(name, bases, namespace)

    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        doc: str = "",
        *,
        shape: object = None,
        annotation: ir.Modification = ir.EMPTY,
        **modifications: object,
    ) -> Declaration:
        return Declaration(
            cls,
            doc=doc,
            modification=_modification(modifications),
            dimensions=_shape(shape),
            annotation=annotation,
        )


class Model(metaclass=_ClassMeta, restriction="model"):
    """Base class for a Modelica ``model``. Subclassing one of these is ``extends``."""

    def equations(self) -> Iterator[ir.Equation]:
        """Override and ``yield`` the equations this class contributes."""
        return iter(())

    def initial_equations(self) -> Iterator[ir.Equation]:
        """Override and ``yield`` equations that hold only at initialisation."""
        return iter(())

    def algorithm(self) -> Iterator[ir.Statement]:
        """Override and ``yield`` the statements of this class's ``algorithm`` section.

        One method, therefore one section. A class with two sections is expressible in
        ``.mo`` and not here, and that asymmetry is deliberate: the second section exists
        to re-run statements after the first has finished, which is a thing to write on
        purpose in a file, not a thing to reach for from Python.
        """
        return iter(())


class Connector(Model, restriction="connector"):
    """Base class for a ``connector`` -- the interface ``connect()`` joins."""


class Record(Model, restriction="record"):
    """Base class for a ``record``."""


class Block(Model, restriction="block"):
    """Base class for a ``block`` -- a model whose connectors all have causality."""


class Function(Model, restriction="function"):
    """Base class for a ``function`` -- ``input``/``output`` components and one algorithm.

    A function has no equations; it has a body. Declare its arguments with
    :func:`input_` and its results with :func:`output`, and write the body by overriding
    :meth:`Model.algorithm`.
    """


# --------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------

_PACKAGE_OF_MODULE: dict[str, str] = {}
"""Which Modelica package each Python module became. Filled in by :func:`package`.

Kept beside the classes rather than on them: a module maps to exactly one package, so the
membership is a property of the module, and recording it here leaves every class, type and
declaration immutable.
"""


@dataclass(frozen=True, eq=False)
class Package:
    """A Modelica ``package``, built from the classes defined in a Python module."""

    name: str
    doc: str
    members: tuple[TypeRef, ...]

    def __repr__(self) -> str:
        return f"<Package {self.name}>"

    def __getattr__(self, name: str) -> TypeRef:
        for member in self.members:
            if _member_name(member) == name:
                return member
        raise AttributeError(f"package {self.name!r} has no class {name!r}")


def _member_name(member: TypeRef) -> str:
    return member.name if isinstance(member, Type) else member.__name__


def _module_of(member: TypeRef) -> str:
    return member.module if isinstance(member, Type) else member.__module__


def package(name: str, module: str, doc: str = "") -> Package:
    """Collect everything defined in ``module`` into a package.

    Call it at the bottom of the file: ``Electrical = package("Electrical", __name__)``.
    Members are collected in definition order -- Python dictionaries preserve insertion
    order, so the emitted package reads in the order the file was written. The module
    docstring becomes the package comment unless ``doc`` says otherwise.
    """
    namespace: ModuleType = sys.modules[module]
    members = tuple(_members_defined_in(namespace, module))
    _PACKAGE_OF_MODULE[module] = name
    return Package(name, doc or _docstring(namespace.__doc__), members)


def _members_defined_in(namespace: ModuleType, module: str) -> Iterator[TypeRef]:
    """Yield the types and classes ``module`` itself defines, skipping imported ones."""
    for value in vars(namespace).values():
        if (isinstance(value, Type) and value.module == module) or (
            isinstance(value, _ClassMeta)
            and value.__module__ == module
            and not value.__modelica_marker__
        ):
            yield value


# --------------------------------------------------------------------------------------
# IR construction
# --------------------------------------------------------------------------------------


def _docstring(text: str | None) -> str:
    return inspect.cleandoc(text).strip() if text else ""


def _qualified(target: TypeRef, within: str) -> str:
    """Name ``target`` as seen from inside package ``within``.

    Same package: the bare name, which is what Modelica lookup finds. Different package:
    dotted -- ``Electrical.Resistor`` when the referring model is not itself electrical.
    """
    name = _member_name(target)
    home = _PACKAGE_OF_MODULE.get(_module_of(target), "")
    return name if not home or home == within else f"{home}.{name}"


def _type_to_ir(target: Type, within: str) -> ir.Class:
    if target.literals:
        return ir.Class(
            name=target.name,
            restriction="type",
            comment=target.doc,
            literals=target.literals,
        )
    if target.base is None:
        raise ValueError(f"{target.name} is a predefined type and has no definition")
    return ir.Class(
        name=target.name,
        restriction="type",
        extends=(ir.Extends(_qualified(target.base, within), _modification(target.attributes)),),
    )


def _class_to_ir(target: _ClassMeta, within: str) -> ir.Class:
    own = target.__dict__
    imports = tuple(cast("Iterable[ir.Import]", own.get("IMPORTS", ())))
    declared = [
        (
            value.redeclaring,
            replace(
                value.component(within),
                type_name=_shorten(value.component(within).type_name, imports),
            ),
        )
        for value in own.values()
        if isinstance(value, Declaration)
    ]
    components = tuple(component for redeclaring, component in declared if not redeclaring)
    redeclarations = tuple(component for redeclaring, component in declared if redeclaring)
    # `Model`/`Connector`/... are how Python spells the restriction, not superclasses to
    # extend; every other base is a real Modelica `extends`.
    extends = tuple(
        ir.Extends(_qualified(base, within))
        for base in target.__bases__
        if isinstance(base, _ClassMeta) and not base.__modelica_marker__
    )
    return ir.Class(
        name=target.__name__,
        restriction=target.__modelica_restriction__,
        partial=target.__modelica_partial__,
        comment=_docstring(cast("str | None", own.get("__doc__"))),
        imports=imports,
        extends=extends,
        redeclarations=redeclarations,
        components=components,
        equations=_section(target, "equations"),
        initial_equations=_section(target, "initial_equations"),
        algorithms=_algorithm(target),
        annotation=cast("ir.Modification", own.get("ICON", ir.EMPTY)),
    )


def _algorithm(target: _ClassMeta) -> tuple[tuple[ir.Statement, ...], ...]:
    """The class's own ``algorithm`` body, as the one-element tuple of sections it makes."""
    body = target.__dict__.get("algorithm")
    if body is None:
        return ()
    return (tuple(cast("Iterable[ir.Statement]", body(target))),)


def _shorten(type_name: str, imports: Sequence[ir.Import]) -> str:
    """The name a type is written by, given what this class imported.

    The Python front end always knows a class by its package, so ``Electrical.Resistor`` is
    what :func:`_qualified` produces. A ``.mo`` file that imported ``E = Electrical`` writes
    ``E.Resistor``, and the two front ends have to say the same thing -- so the imports are
    applied here, in the one place that has both the full name and the class's own imports.
    """
    for entry in imports:
        if entry.wildcard and type_name.startswith(f"{entry.path}."):
            return type_name[len(entry.path) + 1 :]
        if entry.alias and type_name.startswith(f"{entry.path}."):
            return f"{entry.alias}.{type_name[len(entry.path) + 1 :]}"
        if not entry.alias and not entry.wildcard and type_name == entry.path:
            return entry.name
    return type_name


def import_(path: str, alias: str = "", *, wildcard: bool = False) -> ir.Import:
    """One ``import`` for a class body, listed in its ``IMPORTS``.

    The path is a string and not a class, unlike everywhere else in this module, because an
    import names a *package* -- and a package here is a module, which cannot be referred to
    as a value from inside the class body that imports it.

    ``IMPORTS`` is a *tuple*, like every other class-level constant here: a list would be a
    mutable class attribute shared by every subclass, which is a real hazard and which the
    linter is right to refuse.
    """
    return ir.Import(path, alias, wildcard=wildcard)


def _section(target: _ClassMeta, name: str) -> tuple[ir.Equation, ...]:
    """Run a class's own ``equations``/``initial_equations`` body, if it defines one.

    ``target.__dict__`` rather than ``getattr``: a subclass contributes only the equations
    it writes, because the rest arrive through ``extends``.
    """
    body = target.__dict__.get(name)
    if body is None:
        return ()
    return tuple(cast("Iterable[ir.Equation]", body(target)))


def to_ir(target: TypeRef | Package) -> ir.Class:
    """Build the :class:`modelica.ir.Class` for a model, connector, type or package."""
    if isinstance(target, Package):
        return ir.Class(
            name=target.name,
            restriction="package",
            comment=target.doc,
            classes=tuple(_member_to_ir(m, target.name) for m in target.members),
        )
    return _member_to_ir(target, _PACKAGE_OF_MODULE.get(_module_of(target), ""))


def _member_to_ir(target: TypeRef, within: str) -> ir.Class:
    if isinstance(target, Type):
        return _type_to_ir(target, within)
    return _class_to_ir(target, within)


def to_modelica(target: TypeRef | Package | ir.Class | ir.Program) -> str:
    """Render a model, package or IR value as Modelica source text."""
    if isinstance(target, (ir.Class, ir.Program)):
        return unparse(target)
    return unparse(to_ir(target))
