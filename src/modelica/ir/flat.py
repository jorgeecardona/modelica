"""What flattening produces: scalar variables and scalar equations, nothing else.

A :class:`FlatModel` is the end of the object-oriented part of Modelica. Classes,
inheritance, modifications and ``connect()`` are all gone -- spent, by
:mod:`modelica.ir.flatten`, on producing plain lists -- and what remains is the
mathematical object the rest of the toolchain works on: a set of named scalars, a set of
equations over them, and the ``when``-clauses that fire at events.

Like :mod:`modelica.ir.nodes`, this module is data only. The questions one asks of a
flat model (which variables are states? is it balanced?) are cheap enough to answer on
demand, so they are properties here rather than fields filled in by a pass.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

from modelica.ir import nodes as ir

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from modelica.ir.clocks import ClockedPartition

__all__ = ["BaseType", "FlatModel", "FlatVariable"]

type BaseType = Literal["Real", "Integer", "Boolean", "String"]


@dataclass(frozen=True)
class FlatVariable:
    """One scalar of the flattened model, attributes merged and name fully qualified."""

    name: str
    base_type: BaseType = "Real"
    variability: ir.Variability = "continuous"
    causality: ir.Causality = "none"
    binding: ir.Expr | None = None
    """The declaration equation: a parameter's value, or an equation for a continuous
    variable declared with one. Still an expression -- it may name other parameters."""
    start: ir.Expr | None = None
    fixed: bool | None = None
    unit: str = ""
    nominal: ir.Expr | None = None
    minimum: ir.Expr | None = None
    maximum: ir.Expr | None = None
    comment: str = ""

    @property
    def is_constant(self) -> bool:
        return self.variability in ("parameter", "constant")


def _der_targets(node: ir.Expr) -> Iterator[str]:
    """Names ``x`` for every ``der(x)`` inside ``node``."""
    match node:
        case ir.Call(name="der", args=(ir.Ref() as ref,)):
            yield ref.name
        case ir.Call(args=args):
            for arg in args:
                yield from _der_targets(arg)
        case ir.Binary(lhs=lhs, rhs=rhs):
            yield from _der_targets(lhs)
            yield from _der_targets(rhs)
        case ir.Unary(operand=operand):
            yield from _der_targets(operand)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            yield from _der_targets(condition)
            yield from _der_targets(then)
            yield from _der_targets(otherwise)
        case _:
            return


def element(name: str, indices: Sequence[int]) -> str:
    """The scalar name of one element: ``T`` and ``(3,)`` give ``T[3]``.

    Public because three places have to agree on it -- the declaration that creates the
    variable, the subscript that reads it, and anything later that wants to recognise an
    element by its name. A second spelling of this would be a bug nobody could see.
    """
    return f"{name}[{', '.join(str(i) for i in indices)}]"


class CellError(ValueError):
    """An array equation reads an element outside its array at some cell."""


@dataclass(frozen=True)
class IndexSet:
    """A loop index and its inclusive integer range, step 1: ``for i in start:stop``."""

    name: str
    start: int
    stop: int

    def __len__(self) -> int:
        return max(0, self.stop - self.start + 1)


@dataclass(frozen=True)
class ArrayEquation:
    """One equation written once over index sets, as a ``for`` loop writes it.

    ``equation`` is the body with each index a ``Ref`` to its name and each array read an
    ``ir.Subscript`` on the array's flat name. ``shapes`` gives the size of every array it
    subscripts. `scalarize` writes one scalar equation per cell, the outer index slowest, the
    order a ``for`` loop unrolls in.
    """

    indices: tuple[IndexSet, ...]
    equation: ir.Equal
    shapes: tuple[tuple[str, tuple[int, ...]], ...]
    constants: tuple[tuple[str, int], ...] = ()
    """Integer parameters an index reads, ``n`` in ``mod(i, n)``, with their values."""

    def cells(self) -> Iterator[dict[str, int]]:
        """Every combination of index values, outer index slowest."""
        names = [index.name for index in self.indices]
        ranges = [range(index.start, index.stop + 1) for index in self.indices]
        for values in itertools.product(*ranges):
            yield dict(zip(names, values, strict=True))

    def scalarize(self) -> tuple[ir.Equal, ...]:
        """One scalar equation per cell. Raises `CellError` for a read outside its array.

        An index the source wraps itself (``mod(i - 2, n) + 1``) is evaluated like any other
        expression, so a periodic stencil needs no special case, and ``x[i + 1]`` at the last
        cell is an error instead of a silent wrap.
        """
        return tuple(self.at(cell) for cell in self.cells())

    def at(self, cell: Mapping[str, int]) -> ir.Equal:
        """The scalar equation at one ``cell``. Raises `CellError` for a read outside its array."""
        shapes, constants = dict(self.shapes), dict(self.constants)
        return ir.Equal(
            at_cell(self.equation.lhs, cell, shapes, constants),
            at_cell(self.equation.rhs, cell, shapes, constants),
            self.equation.span,
        )


def _integer(node: ir.Expr, cell: Mapping[str, int]) -> int:  # noqa: C901, PLR0911 - one case per node
    """The integer value of an index expression at ``cell``."""
    match node:
        case ir.IntLiteral(value=value):
            return value
        case ir.Ref() if node.name in cell:
            return cell[node.name]
        case ir.Unary(op="-", operand=operand):
            return -_integer(operand, cell)
        case ir.Binary(op="+", lhs=lhs, rhs=rhs):
            return _integer(lhs, cell) + _integer(rhs, cell)
        case ir.Binary(op="-", lhs=lhs, rhs=rhs):
            return _integer(lhs, cell) - _integer(rhs, cell)
        case ir.Binary(op="*", lhs=lhs, rhs=rhs):
            return _integer(lhs, cell) * _integer(rhs, cell)
        case ir.Call(name="mod", args=(a, b)):
            return _integer(a, cell) % _integer(b, cell)
        case ir.Call(name="div", args=(a, b)):
            return int(_integer(a, cell) / _integer(b, cell))
        case ir.Call(name="max", args=(a, b)):
            return max(_integer(a, cell), _integer(b, cell))
        case ir.Call(name="min", args=(a, b)):
            return min(_integer(a, cell), _integer(b, cell))
        case _:
            message = f"an array index must be an integer expression of the loop indices: {node}"
            raise CellError(message)


def index_at(index: ir.Expr, known: Mapping[str, int]) -> int:
    """The integer value of one index expression, with ``known`` giving every name it reads."""
    return _integer(index, known)


def indices_at(
    node: ir.Subscript, cell: Mapping[str, int], constants: Mapping[str, int] | None = None
) -> tuple[int, ...]:
    """The 1-based element ``node`` reads at ``cell``, one integer per axis."""
    known = cell if not constants else {**constants, **cell}
    return tuple(_integer(index, known) for index in node.indices)


def at_cell(  # noqa: PLR0911 - one case per node
    node: ir.Expr,
    cell: Mapping[str, int],
    shapes: Mapping[str, tuple[int, ...]],
    constants: Mapping[str, int] | None = None,
) -> ir.Expr:
    """``node`` at ``cell``: indices replaced by their values, subscripts by element names.

    ``constants`` gives integer parameters an index may read (``n`` in ``mod(i, n)``); they are
    used inside indices only. Raises `CellError` for a read outside its array or an index it
    cannot evaluate.
    """
    known = cell if not constants else {**constants, **cell}

    def go(inner: ir.Expr) -> ir.Expr:
        return at_cell(inner, cell, shapes, constants)

    match node:
        case ir.Subscript(base=base, indices=indices):
            at = tuple(_integer(index, known) for index in indices)
            shape = shapes[".".join(base.parts)]
            if len(at) != len(shape) or not all(
                1 <= k <= n for k, n in zip(at, shape, strict=True)
            ):
                where = ", ".join(f"{name} = {value}" for name, value in cell.items())
                message = f"{base.name}[{', '.join(map(str, at))}] is outside {shape} at {where}"
                raise CellError(message)
            return ir.Ref((*base.parts[:-1], element(base.parts[-1], at)), base.span)
        case ir.Ref() if node.name in cell:
            return ir.IntLiteral(cell[node.name])
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, go(lhs), go(rhs))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, go(operand))
        case ir.Call(name=name, args=args, named=named):
            return ir.Call(name, tuple(go(a) for a in args), named)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(
                go(condition),
                go(then),
                go(otherwise),
            )
        case _:
            return node


@dataclass(frozen=True)
class Loop:
    """Where one `ArrayEquation` sits in `FlatModel.equations`.

    Its cells, in order, from ``first``. Later stages may treat those equations as one; the
    scalar ones stay the truth.
    """

    first: int
    equation: ArrayEquation

    def matches(self, equations: Sequence[ir.Equal]) -> bool:
        """Whether ``equations`` still hold this loop's cells at its ends.

        A cheap check that a ``dataclasses.replace`` of the equations, or a later stage that
        moved them, has not left the loop pointing at something else.
        """
        loop = self.equation
        size = math.prod(len(index) for index in loop.indices)
        last = self.first + size - 1
        if size == 0 or last >= len(equations):
            return False
        ends = (
            {index.name: index.start for index in loop.indices},
            {index.name: index.stop for index in loop.indices},
        )
        try:
            first_cell, last_cell = (loop.at(cell) for cell in ends)
        except CellError:
            return False
        return equations[self.first] == first_cell and equations[last] == last_cell


@dataclass(frozen=True)
class FlatModel:
    """A model flattened to scalars: the input to causalization and simulation."""

    name: str
    variables: tuple[FlatVariable, ...]
    equations: tuple[ir.Equal, ...]
    """The continuous equations, ``connect()`` already expanded."""
    whens: tuple[ir.When, ...] = ()
    """Event clauses, kept apart from the continuous part they interrupt."""
    asserts: tuple[ir.Assert, ...] = ()
    """Runtime checks, carried but not yet evaluated by the simulator.

    Apart from the equations because they bind nothing: an assert is a diagnostic on the
    solution, not a constraint that shapes it."""
    initial_equations: tuple[ir.Equal, ...] = ()
    loops: tuple[Loop, ...] = ()
    """The ``for`` loops `equations` came from, where flattening kept them (`Loop`)."""
    partitions: tuple[ClockedPartition, ...] = ()
    """The clocked equations, one partition per periodic clock (`modelica.ir.clocks`)."""

    def __getitem__(self, name: str) -> FlatVariable:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"{self.name} has no variable {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    @cached_property
    def _by_name(self) -> dict[str, FlatVariable]:
        return {v.name: v for v in self.variables}

    @cached_property
    def states(self) -> tuple[str, ...]:
        """Every ``x`` with a ``der(x)`` somewhere, in declaration order."""
        seen: set[str] = set()
        for equation in self.equations:
            seen.update(_der_targets(equation.lhs))
            seen.update(_der_targets(equation.rhs))
        return tuple(v.name for v in self.variables if v.name in seen)

    @property
    def parameters(self) -> tuple[FlatVariable, ...]:
        return tuple(v for v in self.variables if v.is_constant)

    @property
    def unknowns(self) -> tuple[FlatVariable, ...]:
        """The variables the equations must determine.

        Everything that is neither constant nor fed from outside.
        """
        return tuple(v for v in self.variables if not v.is_constant and v.causality != "input")

    @property
    def algebraics(self) -> tuple[FlatVariable, ...]:
        states = set(self.states)
        return tuple(v for v in self.unknowns if v.name not in states)

    def is_balanced(self) -> bool:
        """As many equations as unknowns -- the first structural sanity check.

        Discrete variables are counted against ``when``-assignments rather than
        continuous equations: a ``when`` body is what defines them.
        """
        continuous = [v for v in self.unknowns if v.variability == "continuous"]
        discrete = [v for v in self.unknowns if v.variability == "discrete"]
        assigned: set[str] = {
            statement.target.name
            for clause in self.whens
            for branch in clause.branches
            for statement in branch.body
            if isinstance(statement, ir.Assign) and isinstance(statement.target, ir.Ref)
        }
        return len(self.equations) == len(continuous) and all(v.name in assigned for v in discrete)
