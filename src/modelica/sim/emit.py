"""A causalized system as Python source text.

This is where the architecture's one hard performance rule is paid for: *the IR must be
compilable*. Nothing here builds a tree of closures to walk at every step. A
:class:`~modelica.ir.system.System` comes in and a string of ordinary Python comes out --
scalars, arithmetic and array indexing, and nothing else -- which the CPython bytecode
compiler will run as-is and Numba will compile to machine code without leaving nopython
mode. The same emitter, pointed at a different set of templates, is what will produce C
for FMU export.

The generated module holds nine functions, all of them filling a caller-supplied array
rather than returning a new one, so that the step loop allocates as little as it can:

``parameters(p)``
    Evaluate the parameter bindings, in dependency order.
``initial(p, y, w)``
    Fill the state vector from the ``start`` attributes, and seed the workspace.
``latch(t, y, p, w)``
    Record which relations hold at the initial point -- see below.
``rhs(t, y, p, w, dy)``
    The state derivatives: the blocks, in the order causalization sorted them.
``jacobian(t, y, p, w, out)``
    ``d(dy)/dy``, exactly -- the same blocks differentiated, not sampled.
``observe(t, y, p, w, out)``
    Every variable of the model, aliases included -- what a result is read from.
``observe_jacobian(t, y, p, w, out)``
    ``d(observed)/dy``, exactly -- what a state estimator calls ``H``.
``indicators(t, y, p, w, z)``
    One zero-crossing function per event, so the integrator can stop at it.
``transition(event, t, y, p, w, out)``
    The post-event state, given which indicator fired.

``w`` is a workspace holding everything that persists between steps without being
integrated. Its first slots hold the last known value of every variable: only implicit
blocks read those -- Newton needs a starting guess, and the previous step's answer is the
best one available -- but they are written by ``observe``, so a solver that observes each
accepted step keeps them warm for free.

The slots after those hold the *latched relations*. A relation written inside an equation
-- ``if level > 0 then ... else 0`` -- is not re-decided every time the right-hand side is
evaluated; Modelica freezes it between events, and the integrator is told to stop when it
would change. Freezing is not pedantry: an integrator that re-reads such a branch mid-step
is integrating a different function on either side of it, and the tank in the example set
drains to a level of minus one machine epsilon and then oscillates about zero forever. So
each such relation gets a workspace slot holding ``1.0`` or ``0.0``, ``latch`` writes them
at the initial point, ``indicators`` watches the relation for a crossing in either
direction, and ``transition`` flips the slot when one happens.

A model with more than one admissible state set gets one slot more, holding which set is
in use, and ``rhs``, ``observe``, ``indicators`` and ``transition`` are each written once
per set with a one-line dispatcher in front. That keeps the calling convention above this
line exactly as it was -- the mode is discrete state, so it belongs in ``w`` beside the
latches rather than in a signature SciPy has no room for. Switching is an event like any
other: :func:`_better` watches one state set's conditioning overtake another's, and the
transition re-reads the state vector in the new set's variables.

``jacobian`` is ``rhs`` differentiated by source transformation, forward mode, one block at
a time. It is not a faster difference quotient -- for the small systems here a difference
quotient is two right-hand sides and hard to beat -- it is an *exact* one. Two places make
that the difference between right and approximately right. A block that Newton iterates on
has no derivative a difference quotient can see: perturbing the input re-runs the loop, so
the answer is only as smooth as the tolerance the loop stopped at, while differentiating
the block's own residual gives the derivative of the exact solution from one linear solve
against the matrix the block was already solved with. And a latched relation is frozen by
construction, so an exact Jacobian steps *across* a switch without the difference quotient
ever straddling it. Both are why FMI asks for `fmi3GetDirectionalDerivative` rather than
offering to difference the outputs itself.

It is the one function that allocates. Its intermediates are rows of length ``n`` rather
than scalars, so the source stays the size of the model instead of the size of the model
squared, and a row has to come from somewhere. That is a deliberate trade and an affordable
one: an implicit solver asks for a Jacobian once every several steps and for the right-hand
side several times per step.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from modelica.ir import nodes as ir
from modelica.ir.derive import ONE, ZERO, d_dx, mul, simplify, sub
from modelica.ir.rewrite import contains, names, residual, substitute
from modelica.ir.share import share, substituted
from modelica.lang.unparse import unparse_expr
from modelica.sim.dialect import (
    CONDITIONAL as _CONDITIONAL,
)
from modelica.sim.dialect import (
    NON_ASSOCIATIVE as _NON_ASSOCIATIVE,
)
from modelica.sim.dialect import (
    PRIMARY as _PRIMARY,
)
from modelica.sim.dialect import PYTHON, Dialect
from modelica.sim.dialect import (
    SIGN as _SIGN,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from modelica.ir.flat import FlatVariable
    from modelica.ir.system import Block, Mode, System

__all__ = ["HELD", "Code", "EmitError", "Emitter", "Trigger", "emit", "iterated"]

_ITERATIONS: Final = 50
"""Newton iterations before an implicit block gives up and returns what it has."""

_TOLERANCE: Final = 1e-12
"""Step size below which a Newton iteration counts as converged."""


class EmitError(Exception):
    """A system that cannot be written as source, and why."""


# --------------------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------------------

_PRODUCT: Final = 6


_TRANSPARENT: Final = frozenset({"noEvent", "pre", "smooth"})
"""Calls that mean something to the *solver* and nothing to arithmetic.

``noEvent`` and ``smooth`` are hints about event detection, so an evaluator drops them.
``pre`` is the value before the current event, which is exactly what the transition
function has in hand -- it reads the pre-event state vector -- so it drops there too.
"""

_RESERVED: Final = frozenset(
    {
        "_A",
        "_D",
        "_M",
        "_R",
        "_b",
        "_null",
        "_seed",
        "_step",
        "_x",
        "abs",
        "dy",
        "event",
        "max",
        "min",
        "np",
        "out",
        "p",
        "range",
        "t",
        "w",
        "y",
        "z",
    }
)
"""Names the generated code uses itself, which no variable may be renamed onto.

The underscored ones are the scratch locals of a solve or a tangent. Modelica lets an
identifier start with an underscore, so ``_A`` is a name a model may legally use, and it
would otherwise be renamed straight onto the matrix the block is being solved in.
"""


def _real(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):  # noqa: PLR0124 - NaN check
        return {"nan": "np.nan", "inf": "np.inf", "-inf": "-np.inf"}[repr(value)]
    return repr(float(value))


def _define(lines: list[str], name: str, definition: str) -> list[str]:
    """Put ``definition`` immediately before the first line that reads ``name``.

    Textual, and safe to be: the names are generated with a prefix no symbol in the module
    starts with, so a match is a use and never part of a longer word. The indentation is
    copied from the line it goes in front of, which is what puts a name first used inside a
    Newton loop inside that loop.
    """
    for position, line in enumerate(lines):
        if name not in line:
            continue
        pad = line[: len(line) - len(line.lstrip())]
        return [*lines[:position], f"{pad}{definition}", *lines[position:]]
    return lines  # pragma: no cover - every name came from an expression that was printed


def _identifier(name: str, taken: set[str]) -> str:
    """A legal, unused Python name for the Modelica variable ``name``."""
    text = re.sub(r"\W", "_", name)
    if not text or text[0].isdigit():
        text = f"_{text}"
    if keyword.iskeyword(text) or text in _RESERVED:
        text = f"{text}_"
    candidate, suffix = text, 1
    while candidate in taken:
        suffix += 1
        candidate = f"{text}_{suffix}"
    taken.add(candidate)
    return candidate


def _unwrap(node: ir.Expr) -> ir.Expr:
    """See through the wrappers that carry no arithmetic."""
    while isinstance(node, ir.Call) and node.name in _TRANSPARENT and node.args:
        node = node.args[-1]
    return node


def _precedence(node: ir.Expr, dialect: Dialect) -> int:
    match _unwrap(node):
        case ir.IfExpr():
            return _CONDITIONAL
        case ir.Binary(op=op):
            return dialect.binary[op][1]
        case ir.Unary(op="not"):
            return dialect.negation_level
        case ir.Unary():
            return _SIGN
        case _:
            return _PRIMARY


def emit_expr(
    node: ir.Expr,
    symbols: Mapping[str, str],
    *,
    minimum: int = _CONDITIONAL,
    dialect: Dialect = PYTHON,
) -> str:
    """Render ``node`` in ``dialect``, parenthesised if it binds looser than ``minimum``."""
    text = _render(_unwrap(node), symbols, dialect)
    return f"({text})" if _precedence(node, dialect) < minimum else text


def _call(node: ir.Call, symbols: Mapping[str, str], dialect: Dialect) -> str:
    if node.name == "div":
        left, right = (emit_expr(a, symbols, dialect=dialect) for a in node.args)
        return f"{dialect.truncate}(({left}) / ({right}))"
    if node.name in ("initial", "terminal"):
        return dialect.false
    if node.name not in dialect.functions:
        raise EmitError(f"no translation for the function {node.name!r} in {dialect.name}")
    arguments = ", ".join(emit_expr(a, symbols, dialect=dialect) for a in node.args)
    return f"{dialect.functions[node.name]}({arguments})"


def _render(  # noqa: C901, PLR0911, PLR0912 - one case per node type, two of them dialect-split
    node: ir.Expr, symbols: Mapping[str, str], dialect: Dialect
) -> str:
    match node:
        case ir.Ref():
            try:
                return symbols[node.name]
            except KeyError:
                raise EmitError(f"nothing computes {node.name!r}") from None
        case ir.RealLiteral(value=value):
            return _real(value)
        case ir.IntLiteral(value=value):
            return dialect.integer(value)
        case ir.BoolLiteral(value=value):
            return dialect.true if value else dialect.false
        case ir.StringLiteral(value=value):
            return repr(value)
        case ir.Array() | ir.Subscript() | ir.Range():
            raise EmitError("array equations are not supported yet")
        case ir.Binary(op="^", lhs=lhs, rhs=rhs) if dialect.power is not None:
            return dialect.power(
                emit_expr(lhs, symbols, dialect=dialect),
                emit_expr(rhs, symbols, dialect=dialect),
            )
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            symbol, level = dialect.binary[op]

            def floor(side: ir.Expr, base: int, op: str = op) -> int:
                inner = _unwrap(side)
                wanted = dialect.clarify.get(op, frozenset())
                return _PRIMARY if isinstance(inner, ir.Binary) and inner.op in wanted else base

            left_at = level + (op in _NON_ASSOCIATIVE)
            left = emit_expr(lhs, symbols, minimum=floor(lhs, left_at), dialect=dialect)
            right = emit_expr(rhs, symbols, minimum=floor(rhs, level + 1), dialect=dialect)
            return f"{left} {symbol} {right}"
        case ir.Unary(op="not", operand=operand):
            inner = emit_expr(operand, symbols, minimum=dialect.negation_level, dialect=dialect)
            return f"{dialect.negation}{inner}"
        case ir.Unary(op=op, operand=operand):
            return f"{op}{emit_expr(operand, symbols, minimum=_SIGN, dialect=dialect)}"
        case ir.Call():
            return _call(node, symbols, dialect)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return dialect.conditional(
                emit_expr(condition, symbols, minimum=1, dialect=dialect),
                emit_expr(then, symbols, minimum=1, dialect=dialect),
                emit_expr(otherwise, symbols, minimum=_CONDITIONAL, dialect=dialect),
            )


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------


_CROSSINGS: Final = frozenset({"<", "<=", ">", ">="})
"""The relations that can be watched. ``==`` on reals is not one of them, deliberately."""

HELD: Final = 0.5
"""Above this, a latch slot counts as holding a true relation.

The slots carry ``1.0`` or ``0.0`` and nothing between, so any threshold in the open
interval would do; what matters is that everyone reading a latch reads it the same way, and
three modules do (this one writes them, :mod:`modelica.sim.solve` iterates on them,
:mod:`modelica.verify` enumerates them).
"""


@dataclass(frozen=True)
class Trigger:
    """A zero crossing the integrator has to stop at, and what it means."""

    condition: str
    """The relation, in Modelica, for reporting."""

    direction: float
    """``1.0`` for a ``when`` branch, which fires only as its condition becomes true;
    ``0.0`` for a relation inside an equation, which is an event whichever way it turns."""

    latch: int | None = None
    """The workspace slot this crossing owns, for a relation inside an equation.

    ``None`` for a ``when`` branch, which reinitialises states instead of holding a value.
    This is the discrete structure of the model, and it is on the *public* record rather
    than buried in the emitter because two things above this line need it: the event
    iteration in :mod:`modelica.sim.solve` has to know which slot went stale, and
    :mod:`modelica.verify` builds the model checker's state out of exactly these slots.
    """

    switch: int | None = None
    """The state set this crossing moves the run into, for a mode switch.

    Deliberately not a latch. A latch is a question the model keeps asking and this is not
    one: nothing has to be re-decided after a switch, and the event iteration that chases
    stale latches has nothing to chase here, which is why it skips anything whose
    :attr:`latch` is ``None``.
    """

    def __str__(self) -> str:
        if self.switch is not None:
            return f"switch to {self.condition}"
        return f"when {self.condition}" if self.direction else f"{self.condition} changed"


@dataclass(frozen=True)
class _Event:
    """One crossing, as an expression to watch and what to do when it is reached."""

    indicator: ir.Expr
    """Positive exactly while the condition holds, so it rises as the condition becomes
    true and falls as it stops being true."""

    trigger: Trigger
    body: tuple[ir.Statement, ...] = ()
    """The ``when`` branch's statements, if it is one."""

    origin: int | None = None
    """The state set whose ``indicators`` this crossing is live in, for a mode switch.

    It reads ``-1`` in every other one: a set the run is not currently using is not a set it
    can leave, and a crossing that fires from nowhere would corrupt the state vector. ``None``
    means the crossing is about the model rather than about the arrangement, so it is watched
    the same way whatever the run is integrating.
    """

    @property
    def latch(self) -> int | None:
        """The workspace slot to flip, for a relation inside an equation."""
        return self.trigger.latch


_HYSTERESIS: Final = 1.2
"""How much better conditioned an alternative state set has to be before the run moves.

Not a knob to tune, a condition for the switch to terminate. Without a margin the two
crossings of a pair sit at the same point facing opposite ways, so leaving a set is
immediately grounds for going back to it and the run chatters where it should pass through.
Applied to the squared pivots, so 1.2 is about five percent on the pivots themselves.
"""


def _better(destination: Mode, origin: Mode) -> ir.Expr:
    """Positive once ``destination`` is the better conditioned of the two.

    The squares, for two reasons. A pivot's sign says nothing about how near singular it is,
    and squaring keeps the comparison smooth exactly where a root finder has to work -- at a
    pivot passing through zero, which is the event this exists to catch.
    """
    if destination.pivot is None or origin.pivot is None:  # pragma: no cover - defensive
        message = "a state set was offered as an alternative without a pivot to compare"
        raise EmitError(message)
    ahead = mul(destination.pivot, destination.pivot)
    behind = mul(origin.pivot, origin.pivot)
    return simplify(sub(ahead, mul(ir.RealLiteral(_HYSTERESIS), behind)))


def _jacobian(block: Block) -> list[list[ir.Expr]]:
    """``d residual / d unknown``, one row per equation."""
    return [
        [simplify(d_dx(residual(equation), unknown)) for unknown in block.unknowns]
        for equation in block.equations
    ]


def iterated(block: Block) -> bool:
    """True when the block's Jacobian moves with its unknowns, so Newton is the only way.

    Asked here rather than left inside the code that writes the loop because the answer is
    needed twice: once to choose between a linear solve and an iteration, and once to know
    which unknowns carry a starting guess in the workspace and therefore need one put there
    before a mode that iterates on them is entered.
    """
    return block.assignment is None and any(
        contains(entry, unknown)
        for row in _jacobian(block)
        for entry in row
        for unknown in block.unknowns
    )


def _indicator(condition: ir.Expr) -> ir.Expr:
    """Turn a relation into an expression that is positive exactly while it holds."""
    match condition:
        case ir.Binary(op="<" | "<=", lhs=lhs, rhs=rhs):
            return simplify(sub(rhs, lhs))
        case ir.Binary(op=">" | ">=", lhs=lhs, rhs=rhs):
            return simplify(sub(lhs, rhs))
        case _:
            raise EmitError(
                f"a when-condition has to be a relation, not {unparse_expr(condition)!r}"
            )


def _relations(node: ir.Expr) -> Iterator[ir.Binary]:
    """Every relation in ``node`` whose value has to be frozen between events.

    Not the ones under ``noEvent`` or ``smooth`` -- that is what those are *for*: the
    modeller asserting that the discontinuity is not worth stopping the integrator over.
    A relation's operands are not searched either, since a relation between relations is
    not something Modelica lets you write about reals.
    """
    match node:
        case ir.Call(name=name) if name in _TRANSPARENT:
            return
        case ir.Binary(op=op) if op in _CROSSINGS:
            yield node
        case ir.Binary(lhs=lhs, rhs=rhs):
            yield from _relations(lhs)
            yield from _relations(rhs)
        case ir.Unary(operand=operand):
            yield from _relations(operand)
        case ir.Call(args=args):
            for argument in args:
                yield from _relations(argument)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            yield from _relations(condition)
            yield from _relations(then)
            yield from _relations(otherwise)
        case _:
            return


def _rendered(system: System) -> Iterator[ir.Expr]:
    """The expressions that end up in the generated code, and only those.

    Every mode's blocks, not just the chosen one's. The latch slots have to mean the same
    thing on both sides of a switch, and they would not if each mode counted its own
    relations: the run would come out of the switch reading a latch that was written for a
    relation it no longer has.
    """
    for mode in system.modes:
        for block in mode.blocks:
            if block.assignment is not None:
                yield block.assignment
            else:
                for equation in block.equations:
                    yield equation.lhs
                    yield equation.rhs
    yield from system.aliases.values()


def _watched(system: System) -> tuple[ir.Binary, ...]:
    """The relations in the equations, deduplicated, in the order they are first read.

    Relations between constants are left alone: they cannot cross, so watching them would
    hand the integrator a function that never changes sign and cost a slot to say so.
    """
    constant = {variable.name for variable in system.parameters}
    found: dict[ir.Binary, None] = {}
    for expression in _rendered(system):
        for relation in _relations(expression):
            if any(name not in constant for name in names(relation)):
                found.setdefault(relation)
    return tuple(found)


def _latched(node: ir.Expr, held: Mapping[ir.Expr, ir.Expr]) -> ir.Expr:  # noqa: PLR0911 - one case per node type
    """Replace each watched relation by a read of the value latched at the last event."""
    if not held:
        return node
    if (frozen := held.get(node)) is not None:
        return frozen
    match node:
        case ir.Call(name=name) if name in _TRANSPARENT:
            return node
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, _latched(lhs, held), _latched(rhs, held))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, _latched(operand, held))
        case ir.Call(name=name, args=args):
            return ir.Call(name, tuple(_latched(argument, held) for argument in args))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(
                _latched(condition, held), _latched(then, held), _latched(otherwise, held)
            )
        case _:
            return node


# --------------------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Code:
    """Generated source, plus the layout of every array it expects."""

    name: str
    source: str
    states: tuple[str, ...]
    parameters: tuple[str, ...]
    observed: tuple[str, ...]
    """Every variable the model has, in the order ``observe`` and ``w`` use."""

    events: tuple[Trigger, ...]
    """What each zero-crossing function watches, in indicator order."""

    workspace: int
    """How long ``w`` has to be: one slot per observed variable, then the latches, then
    the mode if there is one to keep."""

    switching: int | None = None
    """The workspace slot holding which state set is in use, or ``None`` for a model with
    only one -- which is most of them, and for those the generated source is unchanged."""

    def __str__(self) -> str:
        return self.source


def _function(name: str, arguments: str, body: Iterable[str]) -> str:
    lines = [f"def {name}({arguments}):"]
    lines.extend(f"    {line}" if line else "" for line in body)
    return "\n".join(lines)


@dataclass
class Emitter:
    # Public since the C backend arrived. It was `_Emitter` while one module used it, and
    # renaming it is the honest half of that change: a second dialect asks it for the same
    # lines, and a leading underscore would have said the opposite.
    system: System
    symbols: dict[str, str] = field(default_factory=dict[str, str])
    slots: dict[str, int] = field(default_factory=dict[str, int])
    observed: tuple[str, ...] = ()
    watched: tuple[ir.Binary, ...] = ()
    held: dict[ir.Expr, ir.Expr] = field(default_factory=dict[ir.Expr, ir.Expr])
    """Each watched relation, mapped to the test on its latch slot that replaces it."""

    tangents: dict[str, str] = field(default_factory=dict[str, str])
    """Each variable, mapped to the local holding its row of the tangent."""

    switching: int | None = None
    """The workspace slot holding which state set is in use, when there is a choice."""

    shared: dict[ir.Expr, ir.Ref] = field(default_factory=dict[ir.Expr, "ir.Ref"])
    """Subexpressions this function has already named, and the name each one now reads as."""

    collecting: list[ir.Expr] | None = None
    """Where :meth:`expr` records what it is asked to print, during the first of two passes."""

    prefix: str = "_s"
    """What a named subexpression is called, chosen so it cannot collide with a variable."""

    dialect: Dialect = PYTHON
    """Which language the expressions come out in. The *decisions* are the same either way."""

    def __post_init__(self) -> None:
        self.observed = (
            *(variable.name for variable in self.system.variables),
            *self.system.aliases,
        )
        taken = set(_RESERVED)
        self.symbols = {name: _identifier(name, taken) for name in self.observed}
        # A prefix no symbol already starts with, so a named subexpression can never land on
        # a variable's name. `_s0` would do for every model anybody would write, and a model
        # with a variable called `_s0` is exactly the one that would be silently wrong.
        self.prefix = "_s"
        while any(symbol.startswith(self.prefix) for symbol in taken):
            self.prefix = f"_{self.prefix}"
        self.tangents = {name: _identifier(f"$d_{name}", taken) for name in self.observed}
        self.symbols["time"] = "t"
        self.slots = {name: slot for slot, name in enumerate(self.observed)}
        self.watched = _watched(self.system)
        for position, relation in enumerate(self.watched):
            held = f"$relation{position}"  # `$` keeps it out of reach of Modelica names
            self.symbols[held] = f"w[{self.holding(position)}]"
            self.held[relation] = ir.Binary(">", ir.Ref((held,)), ir.RealLiteral(0.5))
        if len(self.system.modes) > 1:
            self.switching = len(self.observed) + len(self.watched)

    def holding(self, position: int) -> int:
        """The workspace slot holding the ``position``-th watched relation."""
        return len(self.observed) + position

    # -- pieces ------------------------------------------------------------------------

    def expr(self, node: ir.Expr, *, minimum: int = _CONDITIONAL) -> str:
        # Latched *first*, and then named. The other order defeats the latch: `_latched`
        # matches a watched relation as a whole node, so naming a subexpression inside one
        # leaves a relation that no longer matches, and it goes back to being re-decided on
        # every evaluation -- which is the exact failure the freezing exists to prevent, and
        # it showed up as `BumpStop` losing 2.3e-8 of a conserved energy.
        node = _latched(node, self.held)
        if self.collecting is not None:
            self.collecting.append(node)
        if self.shared:
            node = substituted(node, self.shared)
        return emit_expr(node, self.symbols, minimum=minimum, dialect=self.dialect)

    def sharing(self, body: Callable[[], list[str]]) -> list[str]:
        """Emit ``body`` with every twice-written subexpression named once.

        The body is emitted **twice**: once to find out what it prints, and once for real
        with the names in place. That is deliberate, and cheaper than the alternative. Which
        expressions a block prints is decided inside `solve`, `linear` and `newton` -- a
        residual substituted at zero, a Newton block printing its Jacobian, a linear one
        printing neither unchanged -- and predicting that set from outside would mean a
        second copy of those three methods with nothing keeping them in step.

        **Each name is defined immediately before the first line that reads it**, and that
        placement is the whole correctness argument. What this function emits is not one
        expression but a *sequence*: block two reads what block one assigned, and a Newton
        block reads an unknown that only exists inside its own loop. Hoisting a name to the
        top of the function would put it before the assignment it depends on -- silently,
        because the variable exists as a Python local either way, holding whatever the last
        step left in it. That was measured, not reasoned about: it moved `BumpStop`'s
        conserved energy by 2.3e-8 and broke `AtwoodPendulum`'s Jacobian outright.

        Defining a name just before its first use cannot do that. Everything the definition
        reads was read by that line too, so it was already assigned. A name whose first use
        is inside a loop is defined inside the loop and recomputed per iteration, which is
        no worse than the repetition it replaced.
        """
        outer, self.collecting = self.collecting, []
        try:
            body()
            plan = share(self.collecting, self.prefix)
        finally:
            self.collecting = outer
        if not plan:
            return body()
        for name, _ in plan.definitions:
            self.symbols[name] = name
        held, self.shared = self.shared, dict(plan.names)
        try:
            lines = body()
        finally:
            self.shared = held
        # Latest first: inserting `_s1` before `_s0` means `_s1`'s own definition line is in
        # place to count as a use of `_s0`, which is where `_s0` then has to go.
        for name, node in reversed(plan.definitions):
            lines = _define(lines, name, f"{name} = {self.expr(node)}")
        return lines

    def live(self, mode: Mode) -> list[str]:
        """The blocks with every relation read directly, for setting the latches."""
        held, self.held = self.held, {}
        try:
            return [*self.unpack(mode), *self.blocks(mode)]
        finally:
            self.held = held

    def value(self, variable: FlatVariable) -> str:
        """A constant's binding, or its start attribute, or zero."""
        node = variable.binding if variable.binding is not None else variable.start
        return self.expr(node) if node is not None else "0.0"

    def ordered(self) -> list[FlatVariable]:
        """The constants, each after the ones its binding reads."""
        constants = {variable.name: variable for variable in self.system.parameters}
        order: list[FlatVariable] = []
        seen: set[str] = set()

        def visit(name: str) -> None:
            variable = constants.get(name)
            if variable is None or name in seen:
                return
            seen.add(name)
            if variable.binding is not None:
                for other in names(variable.binding):
                    visit(other)
            order.append(variable)

        for name in constants:
            visit(name)
        return order

    def unpack(self, mode: Mode) -> list[str]:
        """Read the parameter and state vectors into locals, in ``mode``'s variables."""
        return [
            *(
                f"{self.symbols[variable.name]} = p[{slot}]"
                for slot, variable in enumerate(self.system.parameters)
            ),
            *(f"{self.symbols[name]} = y[{slot}]" for slot, name in enumerate(mode.states)),
        ]

    def solve(self, block: Block) -> list[str]:
        """Compute a block's variables, then the torn ones.

        An assignment, a linear solve or a Newton iteration for what the block still has to
        solve; then the variables tearing moved out of it, which follow by substitution once
        the rest are known.
        """
        if block.assignment is not None:
            return [f"{self.symbols[block.unknowns[0]]} = {self.expr(block.assignment)}"]
        residuals = [residual(equation) for equation in block.equations]
        jacobian = _jacobian(block)
        lines = (
            self.newton(block, jacobian)
            if iterated(block)
            else (self.linear(block, residuals, jacobian))
        )
        # Recovered after the solve and in order, each one written over the torn unknowns and
        # whatever the earlier ones produced. They are outside the Newton loop on purpose:
        # the residuals already have them folded in, so iterating on them would be computing
        # the same numbers twice per step.
        lines.extend(f"{self.symbols[name]} = {self.expr(value)}" for name, value in block.torn)
        return lines

    def system_lines(
        self,
        block: Block,
        jacobian: list[list[ir.Expr]],
        right: Sequence[ir.Expr],
        indent: str = "",
    ) -> list[str]:
        """Fill ``_A`` and ``_b`` for one linear system, and solve it into ``_x``."""
        size = len(block.unknowns)
        dialect = self.dialect
        lines = [
            f"{indent}{dialect.square('_A', size)}",
            f"{indent}{dialect.column('_b', size)}",
        ]
        for row, entries in enumerate(jacobian):
            lines.extend(
                f"{indent}{dialect.entry('_A', row, column, size)} = {self.expr(entry)}"
                for column, entry in enumerate(entries)
                if entry != ZERO
            )
            lines.append(f"{indent}_b[{row}] = {self.expr(right[row])}")
        lines.append(f"{indent}{dialect.linear_solve('_A', '_b', '_x', size)}")
        return lines

    def linear(
        self, block: Block, residuals: list[ir.Expr], jacobian: list[list[ir.Expr]]
    ) -> list[str]:
        """A block linear in its unknowns: assemble ``J u = -c`` and solve it once."""
        at_zero = dict.fromkeys(block.unknowns, ZERO)
        right = [simplify(substitute(r, at_zero)) for r in residuals]
        if len(block.unknowns) == 1:
            slope, intercept = self.expr(jacobian[0][0]), self.expr(right[0])
            return [f"{self.symbols[block.unknowns[0]]} = -({intercept}) / ({slope})"]
        lines = self.system_lines(block, jacobian, [ir.Unary("-", r) for r in right])
        lines.extend(
            f"{self.symbols[unknown]} = _x[{position}]"
            for position, unknown in enumerate(block.unknowns)
        )
        return lines

    def newton(self, block: Block, jacobian: list[list[ir.Expr]]) -> list[str]:
        """A block that has to be iterated on, starting from the last value it held."""
        guess = [f"{self.symbols[u]} = w[{self.slots[u]}]" for u in block.unknowns]
        keep = [f"w[{self.slots[u]}] = {self.symbols[u]}" for u in block.unknowns]
        return [*guess, *self.iteration(block, jacobian), *keep]

    def iteration(self, block: Block, jacobian: list[list[ir.Expr]]) -> list[str]:
        residuals = [residual(equation) for equation in block.equations]
        negated = [ir.Unary("-", r) for r in residuals]
        header = f"for _ in range({_ITERATIONS}):"
        if len(block.unknowns) == 1:
            name = self.symbols[block.unknowns[0]]
            return [
                header,
                f"    _step = {self.expr(negated[0])} / ({self.expr(jacobian[0][0])})",
                f"    {name} = {name} + _step",
                f"    if abs(_step) < {_TOLERANCE}:",
                "        break",
            ]
        lines = [header, *self.system_lines(block, jacobian, negated, indent="    ")]
        lines.extend(
            f"    {self.symbols[u]} = {self.symbols[u]} + _x[{position}]"
            for position, u in enumerate(block.unknowns)
        )
        lines.append(f"    if np.max(np.abs(_x)) < {_TOLERANCE}:")
        lines.append("        break")
        return lines

    def blocks(self, mode: Mode) -> list[str]:
        lines: list[str] = []
        for block in mode.blocks:
            lines.extend(self.solve(block))
        return lines

    def rebase(self, mode: Mode) -> list[str]:
        """Write the state vector in ``mode``'s variables, indented for a transition.

        Nothing is solved here, and nothing moves. Every state of every mode is an unknown
        of the model, so at this point it has either been read out of ``y`` or computed by
        the blocks the transition just ran: the trajectory stays exactly where it is and
        only the description of it changes, which is why a switch costs no accuracy and
        never leaves the constraint it took index reduction to impose.

        The guesses matter as much as the states. A variable the destination iterates on is
        one the *origin* had no reason to touch, so its workspace slot still holds whatever
        ``initial`` put there -- and for the pendulum that is ``y = 0``, which is exactly
        the value the destination's Newton would divide by. The same argument that makes
        the states free makes the guesses exact: they are in hand, so they are written down.
        """
        lines = [f"    out[{slot}] = {self.symbols[name]}" for slot, name in enumerate(mode.states)]
        lines.extend(
            f"    w[{self.slots[unknown]}] = {self.symbols[unknown]}"
            for block in mode.blocks
            if iterated(block)
            for unknown in block.unknowns
        )
        return lines

    # -- derivatives ---------------------------------------------------------------------

    def combination(self, node: ir.Expr, carried: Mapping[str, str]) -> str | None:
        """``sum_v (d node / d v) . dv``, or ``None`` when nothing ``node`` reads moves.

        ``None`` rather than a string of zeros: a variable whose tangent is structurally zero
        is left out of ``carried`` entirely, so the sums downstream of it get shorter instead
        of longer. In a model of any size most variables do not depend on most states, and
        this is the only thing keeping the emitted source proportional to the sparsity.
        """
        terms: list[str] = []
        for name in dict.fromkeys(names(node)):
            row = carried.get(name)
            if row is None:
                continue
            partial = simplify(d_dx(node, name))
            if partial == ZERO:
                continue
            terms.append(
                row if partial == ONE else f"{self.expr(partial, minimum=_PRODUCT)} * {row}"
            )
        return " + ".join(terms) if terms else None

    def chain(self, unknown: str, node: ir.Expr, carried: dict[str, str]) -> list[str]:
        """The tangent of an explicit assignment: the chain rule, written out."""
        total = self.combination(node, carried)
        if total is None:
            carried.pop(unknown, None)
            return []
        carried[unknown] = row = self.tangents[unknown]
        return [f"{row} = {total}"]

    def implicit(self, block: Block, carried: dict[str, str], width: int) -> list[str]:
        """The tangent of a solved block, without solving it again.

        ``g(u, v) = 0`` differentiates to ``(dg/du) u' = -(dg/dv) v'``, so the tangent is one
        solve against the very matrix the block was solved with, evaluated at the answer. The
        iteration is not differentiated -- its *fixed point* is -- which is what makes this
        exact where a difference quotient over the same block cannot be.
        """
        rows = [self.combination(residual(equation), carried) for equation in block.equations]
        if all(row is None for row in rows):
            for unknown in block.solved:
                carried.pop(unknown, None)
            return []
        jacobian = _jacobian(block)
        if len(block.unknowns) == 1:
            carried[block.unknowns[0]] = row = self.tangents[block.unknowns[0]]
            single = f"{row} = -({rows[0]}) / ({self.expr(jacobian[0][0])})"
            return [single, *self._torn_tangents(block, carried)]
        size = len(block.unknowns)
        lines = [f"_M = np.zeros(({size}, {size}))", f"_R = np.zeros(({size}, {width}))"]
        for position, entries in enumerate(jacobian):
            lines.extend(
                f"_M[{position}, {column}] = {self.expr(entry)}"
                for column, entry in enumerate(entries)
                if entry != ZERO
            )
            if rows[position] is not None:
                lines.append(f"_R[{position}] = {rows[position]}")
        lines.append("_D = np.linalg.solve(_M, -_R)")
        for position, unknown in enumerate(block.unknowns):
            carried[unknown] = row = self.tangents[unknown]
            lines.append(f"{row} = _D[{position}]")
        return [*lines, *self._torn_tangents(block, carried)]

    def _torn_tangents(self, block: Block, carried: dict[str, str]) -> list[str]:
        """The chain rule over the variables a torn block recovers by substitution.

        Each is an explicit function of the torn unknowns, which have just been given their
        tangents, so this is the same chain rule an ordinary 1x1 assignment gets -- run in
        the order the substitutions were made, because a later one may read an earlier.
        Without it the recovered variables would have no tangent at all, and the Jacobian
        would come out short by exactly the variables tearing moved.
        """
        lines: list[str] = []
        for name, value in block.torn:
            lines.extend(self.chain(name, value, carried))
        return lines

    def tangent(self, mode: Mode) -> tuple[list[str], dict[str, str]]:
        """Forward mode over the very blocks the right-hand side runs, and what it reached.

        One row vector per variable rather than one scalar per (variable, state) pair. The
        scalars are quicker for the two-state models in the corpus -- no array to allocate --
        but they put a line in the source for every pair, so a hundred-state model emits a
        hundred times the text and spends its life in the bytecode compiler. The row vectors
        keep the source the size of the model.

        The seed rows and the zero row are module-level constants, read and never written,
        so the only arrays this allocates are the tangents it actually computes. A state
        whose derivative is another state costs nothing at all: its row *is* a row of the
        identity, and it is handed straight to ``out``.

        The mapping that comes back says which variables have a tangent at all. A name that
        is missing from it is structurally independent of every state, which is a different
        thing from having a tangent that happens to be zero, and the two callers below both
        turn it into the shared zero row rather than into an allocation.
        """
        width = len(mode.states)
        head = list(self.unpack(mode))
        carried: dict[str, str] = {}
        for slot, state in enumerate(mode.states):
            head.append(f"{self.tangents[state]} = _seed[{slot}]")

        def body() -> list[str]:
            # Rebuilt from scratch on each pass, because `sharing` runs this twice and the
            # second run is the one whose `carried` the callers read.
            carried.clear()
            carried.update({state: self.tangents[state] for state in mode.states})
            lines: list[str] = []
            for block in mode.blocks:
                lines.extend(self.solve(block))
                if block.assignment is not None:
                    lines.extend(self.chain(block.unknowns[0], block.assignment, carried))
                else:
                    lines.extend(self.implicit(block, carried, width))
            return lines

        return [*head, *self.sharing(body)], carried

    def differentiate(self, mode: Mode) -> list[str]:
        """``d(dy)/dy`` -- the Jacobian every implicit solver wants."""
        lines, carried = self.tangent(mode)
        lines.extend(
            f"out[{slot}] = {carried.get(mode.derivative[state]) or '_null'}"
            for slot, state in enumerate(mode.states)
        )
        return [*lines, "return out"]

    def differentiate_observed(self, mode: Mode) -> list[str]:
        """``d(observed)/dy`` -- the same walk, read off at the variables instead.

        An estimator needs this and cannot compute it: ``H`` is what maps a state to what a
        sensor sees, and differencing ``observe`` costs one extra evaluation per state and
        injects truncation error straight into the Kalman gain. It is the same forward mode
        as :meth:`differentiate` and it stops one step earlier -- at the variables rather
        than at their derivatives -- so there is no second transformation to keep in step.

        Emitted as ``observe_jacobian`` and not as ``sensitivity``. The word *sensitivity*
        has an owner in this field: it is the derivative with respect to a **parameter**,
        which is what identification and uncertainty propagation ask for. That is the same
        transformation again with a different seed, it will want the name, and taking it
        here for a different derivative would cost more than the shorter word is worth.

        The aliases are differentiated here and not in :meth:`differentiate`, because they
        are exactly what a model is usually observed *through*: ``i``, ``v``, ``lag`` and the
        rest are eliminated before the solver sees them, so a row for them exists nowhere
        else. Their values are emitted beside their tangents, in the order
        :meth:`observe` writes them, so an alias defined over an earlier one still reads.
        """
        lines, carried = self.tangent(mode)
        for name, node in self.system.aliases.items():
            lines.append(f"{self.symbols[name]} = {self.expr(node)}")
            lines.extend(self.chain(name, node, carried))
        lines.extend(
            f"out[{slot}] = {carried.get(name) or '_null'}"
            for slot, name in enumerate(self.observed)
        )
        return [*lines, "return out"]

    # -- functions ---------------------------------------------------------------------

    def dispatch(self, name: str, arguments: str, bodies: Sequence[list[str]]) -> str:
        """One function per state set, and the picker that reads the mode slot.

        With a single set -- which is every model that did not need index reduction -- this
        writes exactly the one function the emitter has always written, so nothing pays for
        a choice it does not have. The picker compares against ``k + 0.5`` rather than
        converting the slot to an integer: ``w`` is a float array, and a chain of float
        comparisons is something Numba compiles without a thought.
        """
        if len(bodies) == 1:
            return _function(name, arguments, bodies[0])
        parts = [_function(f"{name}_{k}", arguments, body) for k, body in enumerate(bodies)]
        picker: list[str] = []
        for k in range(len(bodies) - 1):
            picker.append(f"if w[{self.switching}] < {k + 0.5}:")
            picker.append(f"    return {name}_{k}({arguments})")
        picker.append(f"return {name}_{len(bodies) - 1}({arguments})")
        return "\n\n\n".join([*parts, _function(name, arguments, picker)])

    def parameters(self) -> str:
        body = [f"{self.symbols[v.name]} = {self.value(v)}" for v in self.ordered()]
        body.extend(
            f"p[{slot}] = {self.symbols[variable.name]}"
            for slot, variable in enumerate(self.system.parameters)
        )
        return _function("parameters", "p", [*body, "return p"])

    def initial(self) -> str:
        constant = {v.name for v in self.system.parameters}
        body = [f"{self.symbols[v.name]} = p[{i}]" for i, v in enumerate(self.system.parameters)]
        body.extend(
            f"y[{slot}] = {self.expr(self.system[name].start or ir.RealLiteral(0.0))}"
            for slot, name in enumerate(self.system.states)
        )
        body.extend(
            f"w[{self.slots[name]}] = "
            + (
                self.symbols[name]
                if name in constant
                else self.expr(self.system[name].start or ir.RealLiteral(0.0))
                if name in self.system and self.system[name].start is not None
                else "0.0"
            )
            for name in self.observed
        )
        if self.switching is not None:
            body.append(f"w[{self.switching}] = 0.0")
        return _function("initial", "p, y, w", [*body, "return y"])

    def latches(self) -> str:
        """Freeze each watched relation at whatever it says at the initial point.

        The blocks are evaluated with the relations read directly here, and only here:
        there is no previous event to inherit a value from, so the live answer *is* the
        latched one.
        """
        body = self.live(self.system.mode) if self.watched else []
        body.extend(
            f"w[{self.holding(position)}] = 1.0 if {emit_expr(relation, self.symbols)} else 0.0"
            for position, relation in enumerate(self.watched)
        )
        return _function("latch", "t, y, p, w", [*body, "return w"])

    def rhs(self, mode: Mode) -> list[str]:
        body = [*self.unpack(mode), *self.sharing(lambda: self.blocks(mode))]
        body.extend(
            f"dy[{slot}] = {self.symbols[mode.derivative[state]]}"
            for slot, state in enumerate(mode.states)
        )
        return [*body, "return dy"]

    def observe(self, mode: Mode) -> list[str]:
        body = [*self.unpack(mode), *self.blocks(mode)]
        body.extend(
            f"{self.symbols[name]} = {self.expr(node)}"
            for name, node in self.system.aliases.items()
        )
        body.extend(
            f"out[{slot}] = {self.symbols[name]}" for slot, name in enumerate(self.observed)
        )
        return [*body, "return out"]

    def events(self) -> tuple[_Event, ...]:
        """Every crossing to watch: ``when`` branches, then relations, then the switches."""
        branches = [
            _Event(
                indicator=_indicator(branch.condition),
                trigger=Trigger(unparse_expr(branch.condition), 1.0),
                body=branch.body,
            )
            for when in self.system.whens
            for branch in when.branches
        ]
        crossings = [
            _Event(
                indicator=_indicator(relation),
                trigger=Trigger(unparse_expr(relation), 0.0, self.holding(position)),
            )
            for position, relation in enumerate(self.watched)
        ]
        return (*branches, *crossings, *self.switches())

    def switches(self) -> list[_Event]:
        """One crossing per ordered pair of state sets: leave this one for that one.

        Pairwise rather than one crossing per set watching the best of the others, because
        a pair compares with a single smooth expression and a maximum does not -- and
        because the set being moved *into* has to be named when the crossing is written,
        not worked out when it fires.
        """
        modes = self.system.modes
        found: list[_Event] = []
        for origin, leaving in enumerate(modes):
            for destination, entering in enumerate(modes):
                if destination == origin:
                    continue
                named = f"({', '.join(entering.states)})"
                found.append(
                    _Event(
                        indicator=_better(entering, leaving),
                        trigger=Trigger(named, 1.0, switch=destination),
                        origin=origin,
                    )
                )
        return found

    def indicators(self, position: int, mode: Mode, events: tuple[_Event, ...]) -> list[str]:
        body = [*self.unpack(mode), *(self.blocks(mode) if events else [])]
        for k, event in enumerate(events):
            live = event.origin in (None, position)
            body.append(f"z[{k}] = {self.expr(event.indicator) if live else '-1.0'}")
        return [*body, "return z"]

    def statement(self, statement: ir.Statement, states: Mapping[str, int]) -> str:
        match statement:
            case ir.Reinit(target=target, value=value) if target.name in states:
                return f"    out[{states[target.name]}] = {self.expr(value)}"
            case ir.Reinit(target=target):
                raise EmitError(f"reinit({target.name}) -- only a state can be reinitialised")
            case _:
                raise EmitError(f"no translation for {statement!r} inside a when-clause")

    def transition(self, position: int, mode: Mode, events: tuple[_Event, ...]) -> list[str]:
        """What one event does: a branch's statements, a latch flipping, or a switch."""
        states = {name: slot for slot, name in enumerate(mode.states)}
        body = [*self.unpack(mode), *(self.blocks(mode) if events else []), "out[:] = y"]
        for k, event in enumerate(events):
            lines = [self.statement(s, states) for s in event.body]
            if event.latch is not None:
                lines.append(f"    w[{event.latch}] = 1.0 - w[{event.latch}]")
            if (moving := event.trigger.switch) is not None and event.origin == position:
                lines.extend(self.rebase(self.system.modes[moving]))
                lines.append(f"    w[{self.switching}] = {float(moving)}")
            body.append(f"if event == {k}:")
            body.extend(lines or ["    pass"])
        return [*body, "return out"]

    def code(self) -> Code:
        events = self.events()
        modes = self.system.modes
        width = len(self.system.states)
        parts = [
            "import numpy as np",
            f"_seed = np.eye({width})\n_null = np.zeros({width})",
            self.parameters(),
            self.initial(),
            self.latches(),
            self.dispatch("rhs", "t, y, p, w, dy", [self.rhs(m) for m in modes]),
            self.dispatch("jacobian", "t, y, p, w, out", [self.differentiate(m) for m in modes]),
            self.dispatch("observe", "t, y, p, w, out", [self.observe(m) for m in modes]),
            self.dispatch(
                "observe_jacobian",
                "t, y, p, w, out",
                [self.differentiate_observed(m) for m in modes],
            ),
            self.dispatch(
                "indicators",
                "t, y, p, w, z",
                [self.indicators(k, m, events) for k, m in enumerate(modes)],
            ),
            self.dispatch(
                "transition",
                "event, t, y, p, w, out",
                [self.transition(k, m, events) for k, m in enumerate(modes)],
            ),
        ]
        return Code(
            name=self.system.name,
            source="\n\n\n".join(parts) + "\n",
            states=self.system.states,
            parameters=tuple(v.name for v in self.system.parameters),
            observed=self.observed,
            events=tuple(event.trigger for event in events),
            workspace=len(self.observed) + len(self.watched) + (0 if self.switching is None else 1),
            switching=self.switching,
        )


def emit(system: System) -> Code:
    """Write ``system`` out as a Python module, ready to be compiled."""
    return Emitter(system).code()
