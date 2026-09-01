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

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

from modelica.ir import nodes as ir

if TYPE_CHECKING:
    from collections.abc import Iterator

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
            if isinstance(statement, ir.Assign)
        }
        return len(self.equations) == len(continuous) and all(v.name in assigned for v in discrete)
