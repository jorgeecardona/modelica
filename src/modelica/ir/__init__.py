"""AST-level model data and the passes that turn it into a solvable system.

The package splits along the line the architecture doc draws: :mod:`modelica.ir.nodes`
is *what a model says* (classes, components, equations -- pure data, two front-ends
build it), and the sibling modules are *what the toolchain does to it*:

* :mod:`modelica.ir.flatten`   -- instantiate a class into a :class:`FlatModel` of
  scalar variables and equations (``extends`` resolved, modifications applied,
  ``connect()`` expanded).
* :mod:`modelica.ir.evaluate`  -- evaluate an expression numerically in an environment.
* :mod:`modelica.ir.derive`    -- symbolic differentiation and constant folding.
* :mod:`modelica.ir.causalize` -- matching + Tarjan BLT: which equation answers which
  unknown, and in what order.

Importing ``modelica.ir`` gives the node types directly (``ir.Equal``, ``ir.Ref``, ...),
so code written against the old single-module layout keeps working unchanged.
"""

from modelica.ir.nodes import (
    EMPTY as EMPTY,
)
from modelica.ir.nodes import (
    Array as Array,
)
from modelica.ir.nodes import (
    Assign as Assign,
)
from modelica.ir.nodes import (
    Binary as Binary,
)
from modelica.ir.nodes import (
    BinaryOp as BinaryOp,
)
from modelica.ir.nodes import (
    BoolLiteral as BoolLiteral,
)
from modelica.ir.nodes import (
    Break as Break,
)
from modelica.ir.nodes import (
    Call as Call,
)
from modelica.ir.nodes import (
    Causality as Causality,
)
from modelica.ir.nodes import (
    Class as Class,
)
from modelica.ir.nodes import (
    Component as Component,
)
from modelica.ir.nodes import (
    Connect as Connect,
)
from modelica.ir.nodes import (
    ConnectorKind as ConnectorKind,
)
from modelica.ir.nodes import (
    Equal as Equal,
)
from modelica.ir.nodes import (
    Equation as Equation,
)
from modelica.ir.nodes import (
    Expr as Expr,
)
from modelica.ir.nodes import (
    Extends as Extends,
)
from modelica.ir.nodes import (
    ForEquation as ForEquation,
)
from modelica.ir.nodes import (
    ForStatement as ForStatement,
)
from modelica.ir.nodes import (
    IfBranch as IfBranch,
)
from modelica.ir.nodes import (
    IfEquation as IfEquation,
)
from modelica.ir.nodes import (
    IfExpr as IfExpr,
)
from modelica.ir.nodes import (
    IfStatement as IfStatement,
)
from modelica.ir.nodes import (
    Import as Import,
)
from modelica.ir.nodes import (
    IntLiteral as IntLiteral,
)
from modelica.ir.nodes import (
    Modification as Modification,
)
from modelica.ir.nodes import (
    Program as Program,
)
from modelica.ir.nodes import (
    Range as Range,
)
from modelica.ir.nodes import (
    RealLiteral as RealLiteral,
)
from modelica.ir.nodes import (
    Ref as Ref,
)
from modelica.ir.nodes import (
    Reinit as Reinit,
)
from modelica.ir.nodes import (
    Restriction as Restriction,
)
from modelica.ir.nodes import (
    Return as Return,
)
from modelica.ir.nodes import (
    Scoping as Scoping,
)
from modelica.ir.nodes import (
    Statement as Statement,
)
from modelica.ir.nodes import (
    StatementBranch as StatementBranch,
)
from modelica.ir.nodes import (
    StringLiteral as StringLiteral,
)
from modelica.ir.nodes import (
    Subscript as Subscript,
)
from modelica.ir.nodes import (
    Unary as Unary,
)
from modelica.ir.nodes import (
    UnaryOp as UnaryOp,
)
from modelica.ir.nodes import (
    Variability as Variability,
)
from modelica.ir.nodes import (
    When as When,
)
from modelica.ir.nodes import (
    WhenBranch as WhenBranch,
)
from modelica.ir.nodes import (
    WhileStatement as WhileStatement,
)

__all__ = [
    "EMPTY",
    "Array",
    "Assign",
    "Binary",
    "BinaryOp",
    "BoolLiteral",
    "Break",
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
