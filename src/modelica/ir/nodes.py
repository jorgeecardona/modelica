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
    "Break",
    "Call",
    "CallEquation",
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
    "ForStatement",
    "IfBranch",
    "IfEquation",
    "IfExpr",
    "IfStatement",
    "Import",
    "IntLiteral",
    "Modification",
    "Program",
    "Range",
    "RealLiteral",
    "Ref",
    "Reinit",
    "Restriction",
    "Return",
    "Scoping",
    "Span",
    "Statement",
    "StatementBranch",
    "StringLiteral",
    "Subscript",
    "Unary",
    "UnaryOp",
    "Variability",
    "When",
    "WhenBranch",
    "WhileStatement",
]


# --------------------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------------------

type BinaryOp = Literal["+", "-", "*", "/", "^", "==", "<>", "<", "<=", ">", ">=", "and", "or"]
type UnaryOp = Literal["+", "-", "not"]


@dataclass(frozen=True, slots=True)
class Span:
    """Where a value was written: a file, a line and a column, all one-based but the file.

    Its own type rather than three fields on everything that has one, so that "carries a
    position" is one thing to add and one thing to look for. Compared and hashed like any
    other value; what keeps a span out of a *node's* equality is the node's field, not this.
    """

    file: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


# Every ``span`` field below is written ``field(default=None, compare=False, repr=False)``,
# and the ``compare=False`` is doing real work rather than tidying. ``parse(unparse(x)) == x``
# compares IR trees, and `tests/test_examples.py` compares flat models built by the two front
# ends -- one of which is Python, where there is no source text and every span is ``None``. A
# span that counted towards equality would fail both, and because a frozen dataclass builds
# its ``__hash__`` from the same fields, it would also change the content hash that keys a
# compiled kernel. Excluded from both, a span is metadata in the strict sense: present or
# absent, the value means the same thing.


@dataclass(frozen=True, slots=True)
class Ref:
    """A reference to a variable, possibly qualified: ``c.v`` is ``Ref(("c", "v"))``."""

    parts: tuple[str, ...]
    span: Span | None = field(default=None, compare=False, repr=False)

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
class Matrix:
    """``[1, 2; 3, 4]`` -- rows split by semicolons, elements by commas.

    Read and carried for the same reason :class:`Array` is: the construct must survive the
    parser for the file holding it to load at all. Nothing gives a matrix a meaning yet --
    the semantic layers refuse the node by name where one would be needed.
    """

    rows: tuple[tuple[Expr, ...], ...]


@dataclass(frozen=True, slots=True)
class Colon:
    """The bare ``:`` subscript -- every index of its dimension, as in ``A[:, 1]``.

    Carried the way :class:`Array` and :class:`Matrix` are, so the file that writes one
    loads; the semantic layers refuse it beside them at the point of use.
    """


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
    | Matrix
    | Colon
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
    redeclare: Component | Class | None = None
    """``M(redeclare Capacitor element)`` -- the declaration that replaces one.

    A redeclaration is not a value and not an entry: it says *which class* a component is,
    which is a question settled before any equation exists. It rides on the modification
    because that is where Modelica writes it, and it holds a whole declaration rather than
    a type name because the replacement may carry modifications of its own -- ``redeclare
    Capacitor element(C = 0.01)`` is one redeclaration, not a redeclaration and a binding.
    """

    def __bool__(self) -> bool:
        return self.value is not None or bool(self.entries) or self.redeclare is not None


EMPTY = Modification()


# --------------------------------------------------------------------------------------
# Statements (the body of a `when`, and the body of an `algorithm` section)
#
# A statement is *causal* in a way an equation is not: it runs in order, and a variable it
# assigns may be assigned again below. That is the whole difference between the two lists,
# and it is why they are separate node types rather than one with a flag.
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


@dataclass(frozen=True, slots=True)
class Break:
    """``break;`` -- leave the innermost loop. No fields; the node *is* the information."""


@dataclass(frozen=True, slots=True)
class Return:
    """``return;`` -- leave the enclosing function."""


@dataclass(frozen=True, slots=True)
class StatementBranch:
    """One ``if``/``elseif`` arm of a statement, distinct from :class:`IfBranch`.

    Same shape as the equation-side branch and a different body type, which is the point:
    an arm of an if-*statement* holds statements, and the type system says so rather than
    a later pass discovering it.
    """

    condition: Expr
    body: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class IfStatement:
    branches: tuple[StatementBranch, ...]
    otherwise: tuple[Statement, ...] = ()


@dataclass(frozen=True, slots=True)
class ForStatement:
    """``for i in 1:n loop ... end for;`` in an algorithm section.

    Unlike :class:`ForEquation`, this one does **not** have to be unrolled: a statement
    sequence may keep its loop, because the emitted code has loops. Whether flattening
    unrolls it anyway is a decision for the pass, not for the node.
    """

    index: str
    over: Expr
    body: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class WhileStatement:
    """``while c loop ... end while;`` -- the one construct with no equation counterpart.

    An equation section cannot express it at all: the number of times a loop runs is a
    property of the values, and an equation says nothing about order.
    """

    condition: Expr
    body: tuple[Statement, ...]


type Statement = Assign | Reinit | Break | Return | IfStatement | ForStatement | WhileStatement


# --------------------------------------------------------------------------------------
# Equations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Equal:
    """``lhs = rhs``. Acausal: which side is "the answer" is decided by causalization."""

    lhs: Expr
    rhs: Expr
    span: Span | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class CallEquation:
    """``(a, b) = f(x)`` -- specification 8.3.1, the only way to read two results.

    Its own node rather than an `Equal` with a tuple on the left, because there is no tuple
    *value* in this IR and adding one would reach every expression walker in the project to
    serve a form that only ever appears at the top of an equation. A function with two
    results is a fact about the equation, not a new kind of expression.

    ``targets`` may hold ``None`` where the source skipped an output -- ``(, b) = f(x)`` is
    legal, and it means the first result is not wanted.
    """

    targets: tuple[Ref | None, ...]
    call: Call
    span: Span | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class Connect:
    """``connect(a, b)``.

    Kept as its own node rather than expanded on sight, because expanding it needs the
    whole connection set -- three flanges meeting at a point produce one sum equation,
    not three pairwise ones.
    """

    a: Ref | Subscript
    b: Ref | Subscript
    """An endpoint is a component reference, subscripted when it names one element of an
    array of connectors: ``connect(plug.pin[2], x)``. The subscript sits on the last name
    only -- flattening resolves it to the element the array expansion already declared."""
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


@dataclass(frozen=True, slots=True)
class Assert:
    """``assert(condition, message);`` -- a runtime check, not a binding.

    It constrains no unknown, so matching and causalization never see it: flattening
    carries it into ``FlatModel.asserts`` instead of the equation list. The simulator does
    not evaluate asserts yet -- the node is kept so a model's diagnostics stay part of its
    value instead of being dropped on the floor.
    """

    condition: Expr
    message: Expr
    level: Expr | None = None
    span: Span | None = field(default=None, compare=False, repr=False)


type Equation = Equal | CallEquation | Assert | Connect | When | IfEquation | ForEquation


# --------------------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------------------

type Variability = Literal["continuous", "discrete", "parameter", "constant"]
type Causality = Literal["none", "input", "output"]
type Scoping = Literal["none", "inner", "outer"]
type ConnectorKind = Literal["none", "flow", "stream"]
type Restriction = Literal["model", "connector", "package", "record", "block", "type", "function"]


@dataclass(frozen=True)
class Component:
    """One declared component: ``parameter Real k(unit = "1/s") = 0.5 "Cooling rate";``."""

    name: str
    type_name: str
    span: Span | None = field(default=None, compare=False, repr=False)
    """Where the component was declared. See the note above :class:`Ref` for why it does not
    take part in equality."""
    modification: Modification = EMPTY
    variability: Variability = "continuous"
    causality: Causality = "none"
    connector: ConnectorKind = "none"
    scoping: Scoping = "none"
    """``inner`` declares the one everybody shares; ``outer`` refers to it.

    An ``outer`` component declares no variable of its own. It is a *name* for a declaration
    further out -- the ``World`` object, the ambient conditions -- and flattening resolves it
    to the nearest enclosing ``inner`` of the same name.
    """
    condition: Expr | None = None
    """``Sensor s if useSensor;`` -- the component exists only when this holds.

    The expression must be evaluable when the model is flattened (it decides how many
    variables and equations exist, the same way a dimension does). A disabled component
    disappears together with every ``connect()`` that touches it, which is the one kind of
    reference the specification allows to survive the removal.
    """
    comment: str = ""
    annotation: Modification = EMPTY
    """Where the component sits in the enclosing diagram: ``Placement(transformation(...))``."""
    replaceable: bool = False
    """Whether a modification may replace this component's class -- ``replaceable Resistor e;``."""
    constraining: str = ""
    """``constrainedby TwoPin``: the interface a replacement has to provide.

    Empty when none was written, which means the declared type is its own constraint. The
    two are not the same claim: ``replaceable Resistor e`` accepts only a resistor, and
    ``replaceable Resistor e constrainedby TwoPin`` accepts anything with a resistor's pins.
    """
    dimensions: tuple[Expr, ...] = ()
    """``Real T[n]`` is ``dimensions=(Ref(("n",)),)``. Empty for a scalar, which is most of them.

    The shape is an *expression* rather than a number because Modelica writes it as one, and
    it is usually a parameter. Flattening is what evaluates it, which is also what makes such
    a parameter structural: the number of equations depends on it, so it cannot change after
    the model is built.
    """


@dataclass(frozen=True, slots=True)
class Import:
    """``import A.B.C;``, ``import D = A.B.C;`` and ``import A.B.*;`` -- the three forms.

    All three say the same kind of thing: *this simple name, here, means that full path*.
    They differ only in which simple name. The qualified form takes the last segment of the
    path, the renaming form takes the name given, and the wildcard form takes every class in
    the package. Storing the path as written keeps the distinction, which the printer needs
    and which an error message reads better for.

    An import is always resolved from the top of the program, never relative to the class
    that wrote it. That is Modelica's rule and it is the reason this node carries no scope.
    """

    path: str
    alias: str = ""
    """The name given by ``import D = A.B.C``. Empty for the other two forms."""
    wildcard: bool = False
    """``import A.B.*`` -- every class in ``path`` is visible by its own simple name."""

    @property
    def name(self) -> str:
        """The simple name this import introduces; empty for a wildcard, which introduces many."""
        if self.wildcard:
            return ""
        return self.alias or self.path.rsplit(".", 1)[-1]


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
    expandable: bool = False
    """``expandable connector`` -- a bus whose members exist by being connected.

    Only a connector can carry it. It is read and written faithfully so a library that
    declares one still parses; *instantiating* one is refused in flattening until the
    expansion semantics exist, the same accept-then-refuse two-step the array syntax uses.
    """
    replaceable: bool = False
    """``replaceable package Medium = PartialMedium`` -- a class a modification may replace."""
    constraining: str = ""
    """The ``constrainedby`` of a replaceable class; empty when none was written."""
    comment: str = ""
    imports: tuple[Import, ...] = ()
    """``import`` elements, in the order written. They bind names and nothing else."""
    literals: tuple[str, ...] = ()
    """``type Mode = enumeration(day, night)`` -- the literals, in the order declared.

    Order is the whole content of an enumeration beyond the names: the specification says a
    literal *is* its position, counted from one, and that is what the flattener emits. So
    this is a tuple and not a set, and reordering it changes the model.

    Non-empty only on a ``type``, and mutually exclusive with ``extends``: an enumeration
    defines a new type rather than renaming one.
    """
    extends: tuple[Extends, ...] = ()
    redeclarations: tuple[Component, ...] = ()
    """``redeclare Capacitor element;`` written as an element, under an ``extends``.

    Kept apart from :attr:`components` because it declares nothing new: it says that a
    component an ``extends`` brought in is now a different class. Filed with the components
    it would be indistinguishable from one, and the class would gain a member the base
    already has.

    The other spelling of the same idea rides on a modification
    (:attr:`Modification.redeclare`), and the two mean exactly the same thing. They differ
    only in *when* they can be applied: a modification at instantiation, this one as soon as
    the ``extends`` chain is folded.
    """
    components: tuple[Component, ...] = ()
    equations: tuple[Equation, ...] = ()
    initial_equations: tuple[Equation, ...] = ()
    initial_algorithms: tuple[tuple[Statement, ...], ...] = ()
    """The ``initial algorithm`` sections: assignments that hold at initialization.

    Kept sectioned for the same reason :attr:`algorithms` is, and folded into the
    initial equations by flattening the same way the plain sections fold into the
    continuous ones.
    """
    algorithms: tuple[tuple[Statement, ...], ...] = ()
    """The ``algorithm`` sections, each kept whole rather than merged into one list.

    Two sections are not one section written twice. Each runs as a unit, and a variable
    assigned in the first and read in the second is a different model from the same
    statements in one section -- so the boundary is information, and concatenating them
    here would throw it away before any pass could use it. The equations *are* merged,
    because an equation section has no order for a boundary to interrupt.

    A ``function`` is a class whose body is one of these over its ``input`` and ``output``
    components; no separate node is needed for it.
    """
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
