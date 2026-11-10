"""Clocked partitions (Modelica 3.3 chapter 16), the minimal scope of plan step 6b.

An equation is clocked when it calls ``sample(u, Clock(interval))`` or ``previous(x)``, or
reads or defines a variable another clocked equation defines; the rest is the continuous part.
A clocked variable is read in the continuous part only through ``hold(x)``. Each partition has
exactly one clock. Refused: two clocks meeting in one partition, and every other clock form
(an inferred ``Clock()``, ``subSample``, ``superSample``, ``shiftSample``, ``interval()``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from modelica.ir import nodes as ir

if TYPE_CHECKING:
    from collections.abc import Iterator

    from modelica.ir.flat import FlatModel

_REFUSED = frozenset({"subSample", "superSample", "shiftSample", "backSample", "interval"})


class ClockError(Exception):
    """A clocked construct outside the supported scope, or two clocks in one partition."""


@dataclass(frozen=True)
class ClockedPartition:
    """Equations that run together at each tick of one periodic clock."""

    interval: ir.Expr
    """The clock's period, as ``Clock(interval)`` wrote it."""
    equations: tuple[ir.Equal, ...]
    variables: tuple[str, ...]
    """The variables these equations define."""


def _calls(node: ir.Expr) -> Iterator[ir.Call]:
    """Every call under ``node``, ``node`` included, outermost first."""
    match node:
        case ir.Call(args=args):
            yield node
            for arg in args:
                yield from _calls(arg)
        case ir.Binary(lhs=lhs, rhs=rhs):
            yield from _calls(lhs)
            yield from _calls(rhs)
        case ir.Unary(operand=operand):
            yield from _calls(operand)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            yield from _calls(condition)
            yield from _calls(then)
            yield from _calls(otherwise)
        case _:
            return


def _equation_calls(equation: ir.Equal) -> Iterator[ir.Call]:
    yield from _calls(equation.lhs)
    yield from _calls(equation.rhs)


def _reads(node: ir.Expr) -> set[str]:  # noqa: PLR0911 - one case per node type
    """The variable names ``node`` reads, leaving out what it reads inside ``hold(...)``."""
    match node:
        case ir.Call(name="hold"):
            return set()
        case ir.Ref():
            return {node.name}
        case ir.Binary(lhs=lhs, rhs=rhs):
            return _reads(lhs) | _reads(rhs)
        case ir.Unary(operand=operand):
            return _reads(operand)
        case ir.Call(args=args):
            found: set[str] = set()
            for arg in args:
                found |= _reads(arg)
            return found
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return _reads(condition) | _reads(then) | _reads(otherwise)
        case _:
            return set()


def _touches(equation: ir.Equal) -> set[str]:
    return _reads(equation.lhs) | _reads(equation.rhs)


def _clocks(equation: ir.Equal) -> list[ir.Expr]:
    """The intervals of the ``Clock(...)`` this equation samples with, in order."""
    found: list[ir.Expr] = []
    for node in _equation_calls(equation):
        if node.name in _REFUSED:
            message = f"{node.name}() is not supported; only a periodic Clock(interval) is"
            raise ClockError(message)
        if node.name == "Clock":
            if len(node.args) != 1:
                message = "only Clock(interval) is supported, not an inferred or event clock"
                raise ClockError(message)
            found.append(node.args[0])
    return found


def _seeds(equation: ir.Equal) -> bool:
    return any(
        node.name == "previous" or (node.name == "sample" and not _is_time_sample(node))
        for node in _equation_calls(equation)
    )


def _is_time_sample(node: ir.Call) -> bool:
    """``sample(start, interval)``, the time-event form, rather than ``sample(u, Clock(...))``."""
    if len(node.args) != 2:  # noqa: PLR2004 - sample takes two arguments
        return True
    clock = node.args[1]
    return not (isinstance(clock, ir.Call) and clock.name == "Clock")


def _defines(equation: ir.Equal) -> str | None:
    return equation.lhs.name if isinstance(equation.lhs, ir.Ref) else None


def partition(model: FlatModel) -> FlatModel:  # noqa: C901 - seed, spread, group, check
    """``model`` with its clocked equations moved into `ClockedPartition`s.

    The clocked variables become ``discrete``: the continuous part holds them between ticks.

    Raises:
        ClockError: an unsupported clock form, a partition with two clocks, or a clocked
            equation that does not define one variable on its left-hand side.
    """
    equations = list(model.equations)
    clocked = {k for k, equation in enumerate(equations) if _seeds(equation)}
    if not clocked:
        return model
    changed = True
    while changed:
        changed = False
        names = {_defines(equations[k]) for k in clocked} - {None}
        for k, equation in enumerate(equations):
            if k in clocked:
                continue
            if _touches(equation) & names and _defines(equation) is not None:
                clocked.add(k)
                changed = True
    for k in clocked:
        if _defines(equations[k]) is None:
            message = f"a clocked equation must define one variable: {equations[k]}"
            raise ClockError(message)
    # Group by connection: equations sharing a variable are one partition.
    groups: list[set[int]] = []
    for k in sorted(clocked):
        touched = _touches(equations[k])
        joined = [g for g in groups if any(touched & _touches(equations[j]) for j in g)]
        merged = {k}.union(*joined)
        groups = [g for g in groups if g not in joined] + [merged]
    partitions: list[ClockedPartition] = []
    for group in groups:
        intervals = {interval for k in group for interval in _clocks(equations[k])}
        if len(intervals) != 1:
            message = (
                "a clocked partition needs exactly one Clock(interval); found "
                f"{len(intervals)}: {sorted(map(str, intervals))}"
            )
            raise ClockError(message)
        ordered = tuple(equations[k] for k in sorted(group))
        partitions.append(
            ClockedPartition(
                next(iter(intervals)),
                ordered,
                tuple(name for e in ordered if (name := _defines(e)) is not None),
            )
        )
    variables = {name for p in partitions for name in p.variables}
    return replace(
        model,
        equations=tuple(e for k, e in enumerate(equations) if k not in clocked),
        variables=tuple(
            replace(v, variability="discrete") if v.name in variables else v
            for v in model.variables
        ),
        partitions=tuple(partitions),
    )
