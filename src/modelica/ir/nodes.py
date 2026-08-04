"""The intermediate representation: what a Modelica model *is*, as plain data.

This module is the hinge of the project. Everything upstream produces these values and
everything downstream consumes them, so it deliberately contains no logic beyond the
dataclasses themselves -- no parsing, no evaluation, no formatting. The passes that *do*
things to these values (flattening, differentiation, causalization) live beside it in
the :mod:`modelica.ir` package.

Two front-ends build the same values:

* :mod:`modelica.lang` parses ``.mo`` source into them;
* :mod:`modelica.build` lets you write them directly in Python.

Neither is privileged. A model written either way is the same model, and the test suite
enforces that by comparing the values with ``==``.

Expression nodes are hashable (frozen, slotted, no containers but tuples) so that later
stages can use them as dictionary keys for common-subexpression elimination and for
caching symbolic derivatives. Declaration nodes hold dictionaries and are therefore
compared but not hashed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "EMPTY",
    "Array",
    "Assign",
    "Binary",
    "BinaryOp",
    "BoolLiteral",
    "Call",
    "Causality",
    "Class",
    "Component",
    "Connect",
    "ConnectorKind",
    "Equal",
    "Equation",
    "Expr",
    "Extends",
    "ForEquation",
    "IfBranch",
    "IfEquation",
    "IfExpr",
    "IntLiteral",
    "Modification",
    "Program",
    "Range",
    "RealLiteral",
    "Ref",
    "Reinit",
    "Restriction",
    "Statement",
    "StringLiteral",
    "Subscript",
    "Unary",
    "UnaryOp",
    "Variability",
    "When",
    "WhenBranch",
]


# --------------------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------------------

type BinaryOp = Literal["+", "-", "*", "/", "^", "==", "<>", "<", "<=", ">", ">=", "and", "or"]
type UnaryOp = Literal["+", "-", "not"]


@dataclass(frozen=True, slots=True)
class Ref:
    """A reference to a variable, possibly qualified: ``c.v`` is ``Ref(("c", "v"))``."""

    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("a reference needs at least one name part")

    @property
    def name(self) -> str:
        return ".".join(self.parts)


@dataclass(frozen=True, slots=True)
class RealLiteral:
    value: float


@dataclass(frozen=True, slots=True)
class IntLiteral:
    value: int


@dataclass(frozen=True, slots=True)
class BoolLiteral:
    value: bool


@dataclass(frozen=True, slots=True)
class StringLiteral:
    value: str


@dataclass(frozen=True, slots=True)
class Array:
    """``{1, 2}``, and ``{{-100, -100}, {100, 100}}`` when the elements are arrays.

    The toolchain does not solve array equations, and flattening still refuses one. This
    node exists because *annotations* are full of arrays -- every point, every extent and
    every colour is one -- and an annotation has to survive the parser to be drawn.
    """

    elements: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Subscript:
    """``T[i]``, ``A[i, j]`` -- a reference with indices, before flattening scalarises it.

    A separate node rather than indices on :class:`Ref`, because it is a *temporary* shape.
    Flattening expands an array declaration into one scalar variable per element and rewrites
    each subscript to the name of the element it selects, so nothing below
    :mod:`modelica.ir.flatten` ever sees one. Putting the indices on ``Ref`` would have made
    every later stage carry a field that is always empty by the time it looks.
    """

    base: Ref
    indices: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Range:
    """``1:n``, and ``1:2:n`` when a step is given -- the step is the *middle* term."""

    start: Expr
    stop: Expr
    step: Expr | None = None


@dataclass(frozen=True, slots=True)
class Binary:
    op: BinaryOp
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True, slots=True)
class Unary:
    op: UnaryOp
    operand: Expr


@dataclass(frozen=True, slots=True)
class Call:
    """A function application. ``der``, ``pre``, ``sqrt`` and friends are all calls here.

    Keeping the built-in operators as ordinary calls means later stages match on one node
    type instead of a dozen, and a user-defined ``function`` needs no new IR.
    """

    name: str
    args: tuple[Expr, ...]
    named: tuple[tuple[str, Expr], ...] = ()
    """``Rectangle(extent = ..., lineColor = ...)``: the arguments given by name.

    A pair list rather than a dict, so the node stays hashable and the order the modeller
    wrote survives -- which is what lets ``parse(unparse(x)) == x`` hold over a file full
    of graphical annotations.
    """


@dataclass(frozen=True, slots=True)
class IfExpr:
    condition: Expr
    then: Expr
    otherwise: Expr


type Expr = (
    Ref
    | RealLiteral
    | IntLiteral
    | BoolLiteral
    | StringLiteral
    | Array
    | Subscript
    | Range
    | Binary
    | Unary
    | Call
    | IfExpr
)


# --------------------------------------------------------------------------------------
# Modifications and annotations
#
# Both are the same construct, and both are defined here -- above the equations and the
# declarations -- because every one of those can carry an annotation.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Modification:
    """A modification, which in Modelica is one recursive shape used for everything.

    ``R = 100`` is a value. ``v(start = 0, fixed = true)`` is entries. ``c(C = 0.01,
    v(start = 0))`` is both, nested. Built-in attributes (``start``, ``unit``, ``min``)
    and component bindings are the same construct and are stored the same way -- the
    difference is only what the name resolves to, which is a later stage's problem.
    """

    value: Expr | None = None
    entries: dict[str, Modification] = field(default_factory=dict[str, "Modification"])

    def __bool__(self) -> bool:
        return self.value is not None or bool(self.entries)


EMPTY = Modification()


# --------------------------------------------------------------------------------------
# Statements (the body of a `when`)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Assign:
    target: Ref
    value: Expr


@dataclass(frozen=True, slots=True)
class Reinit:
    """``reinit(x, expr)`` -- restart a state at a new value without changing the model."""

    target: Ref
    value: Expr


type Statement = Assign | Reinit


# --------------------------------------------------------------------------------------
# Equations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Equal:
    """``lhs = rhs``. Acausal: which side is "the answer" is decided by causalization."""

    lhs: Expr
    rhs: Expr


@dataclass(frozen=True, slots=True)
class Connect:
    """``connect(a, b)``.

    Kept as its own node rather than expanded on sight, because expanding it needs the
    whole connection set -- three flanges meeting at a point produce one sum equation,
    not three pairwise ones.
    """

    a: Ref
    b: Ref
    annotation: Modification = EMPTY
    """Where the line was drawn. ``Line(points = {{0, 0}, {10, 0}}, color = {0, 0, 255})``."""


@dataclass(frozen=True, slots=True)
class WhenBranch:
    condition: Expr
    body: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class When:
    branches: tuple[WhenBranch, ...]


@dataclass(frozen=True, slots=True)
class IfBranch:
    condition: Expr
    body: tuple[Equation, ...]


@dataclass(frozen=True, slots=True)
class IfEquation:
    branches: tuple[IfBranch, ...]
    otherwise: tuple[Equation, ...] = ()


@dataclass(frozen=True, slots=True)
class ForEquation:
    """``for i in 1:n loop ... end for;`` -- one index, one range, a body of equations.

    The index is a name and not a declaration: it exists only inside the body, it is an
    ``Integer``, and no component clause introduces it. Flattening evaluates the range,
    substitutes each value for the index, and emits the body once per value -- so this node,
    like :class:`Subscript`, does not survive into a :class:`~modelica.ir.flat.FlatModel`.
    """

    index: str
    over: Expr
    body: tuple[Equation, ...]


type Equation = Equal | Connect | When | IfEquation | ForEquation


# --------------------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------------------

type Variability = Literal["continuous", "discrete", "parameter", "constant"]
type Causality = Literal["none", "input", "output"]
type ConnectorKind = Literal["none", "flow", "stream"]
type Restriction = Literal["model", "connector", "package", "record", "block", "type", "function"]


@dataclass(frozen=True)
class Component:
    """One declared component: ``parameter Real k(unit = "1/s") = 0.5 "Cooling rate";``."""

    name: str
    type_name: str
    modification: Modification = EMPTY
    variability: Variability = "continuous"
    causality: Causality = "none"
    connector: ConnectorKind = "none"
    comment: str = ""
    annotation: Modification = EMPTY
    """Where the component sits in the enclosing diagram: ``Placement(transformation(...))``."""
    dimensions: tuple[Expr, ...] = ()
    """``Real T[n]`` is ``dimensions=(Ref(("n",)),)``. Empty for a scalar, which is most of them.

    The shape is an *expression* rather than a number because Modelica writes it as one, and
    it is usually a parameter. Flattening is what evaluates it, which is also what makes such
    a parameter structural: the number of equations depends on it, so it cannot change after
    the model is built.
    """


@dataclass(frozen=True)
class Extends:
    type_name: str
    modification: Modification = EMPTY


@dataclass(frozen=True)
class Class:
    """A class definition -- ``model``, ``connector``, ``package``, ``type``, ...

    ``type Voltage = Real(unit = "V")`` is represented as a ``type`` class with a single
    ``extends``, which is what a short class definition means.
    """

    name: str
    restriction: Restriction
    partial: bool = False
    comment: str = ""
    extends: tuple[Extends, ...] = ()
    components: tuple[Component, ...] = ()
    equations: tuple[Equation, ...] = ()
    initial_equations: tuple[Equation, ...] = ()
    classes: tuple[Class, ...] = ()
    annotation: Modification = EMPTY
    """What the class looks like: ``Icon(graphics = {...})`` and ``Diagram(...)``."""


@dataclass(frozen=True)
class Program:
    """The top level of one source file or one Python module."""

    classes: tuple[Class, ...] = ()

    def find(self, name: str) -> Class | None:
        """Look up a class by dotted name: ``prog.find("Electrical.Resistor")``."""
        parts = name.split(".")
        candidates = self.classes
        found: Class | None = None
        for part in parts:
            found = next((c for c in candidates if c.name == part), None)
            if found is None:
                return None
            candidates = found.classes
        return found
