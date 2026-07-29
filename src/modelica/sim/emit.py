"""A causalized system as Python source text.

This is where the architecture's one hard performance rule is paid for: *the IR must be
compilable*. Nothing here builds a tree of closures to walk at every step. A
:class:`~modelica.ir.system.System` comes in and a string of ordinary Python comes out --
scalars, arithmetic and array indexing, and nothing else -- which the CPython bytecode
compiler will run as-is and Numba will compile to machine code without leaving nopython
mode. The same emitter, pointed at a different set of templates, is what will produce C
for FMU export.

The generated module holds seven functions, all of them filling a caller-supplied array so
that no allocation happens inside the integrator:

``parameters(p)``
    Evaluate the parameter bindings, in dependency order.
``initial(p, y, w)``
    Fill the state vector from the ``start`` attributes, and seed the workspace.
``latch(t, y, p, w)``
    Record which relations hold at the initial point -- see below.
``rhs(t, y, p, w, dy)``
    The state derivatives: the blocks, in the order causalization sorted them.
``observe(t, y, p, w, out)``
    Every variable of the model, aliases included -- what a result is read from.
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
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from modelica.ir import nodes as ir
from modelica.ir.derive import ZERO, d_dx, simplify, sub
from modelica.ir.rewrite import contains, names, residual, substitute
from modelica.lang.unparse import unparse_expr

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from modelica.ir.flat import FlatVariable
    from modelica.ir.system import Block, System

__all__ = ["HELD", "Code", "EmitError", "Trigger", "emit"]

_ITERATIONS: Final = 50
"""Newton iterations before an implicit block gives up and returns what it has."""

_TOLERANCE: Final = 1e-12
"""Step size below which a Newton iteration counts as converged."""


class EmitError(Exception):
    """A system that cannot be written as source, and why."""


# --------------------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------------------

_CONDITIONAL: Final = 0
_NOT: Final = 3
_SIGN: Final = 7
_PRIMARY: Final = 9

_BINARY: Final[dict[str, tuple[str, int]]] = {
    "or": ("or", 1),
    "and": ("and", 2),
    "==": ("==", 4),
    "<>": ("!=", 4),
    "<": ("<", 4),
    "<=": ("<=", 4),
    ">": (">", 4),
    ">=": (">=", 4),
    "+": ("+", 5),
    "-": ("-", 5),
    "*": ("*", 6),
    "/": ("/", 6),
    "^": ("**", 8),
}

_NON_ASSOCIATIVE: Final = frozenset({"^", "==", "<>", "<", "<=", ">", ">="})

_FUNCTIONS: Final[dict[str, str]] = {
    "abs": "abs",
    "acos": "np.arccos",
    "asin": "np.arcsin",
    "atan": "np.arctan",
    "atan2": "np.arctan2",
    "ceil": "np.ceil",
    "cos": "np.cos",
    "cosh": "np.cosh",
    "exp": "np.exp",
    "floor": "np.floor",
    "log": "np.log",
    "log10": "np.log10",
    "max": "max",
    "min": "min",
    "mod": "np.mod",
    "rem": "np.fmod",
    "sign": "np.sign",
    "sin": "np.sin",
    "sinh": "np.sinh",
    "sqrt": "np.sqrt",
    "tan": "np.tan",
    "tanh": "np.tanh",
}

_TRANSPARENT: Final = frozenset({"noEvent", "pre", "smooth"})
"""Calls that mean something to the *solver* and nothing to arithmetic.

``noEvent`` and ``smooth`` are hints about event detection, so an evaluator drops them.
``pre`` is the value before the current event, which is exactly what the transition
function has in hand -- it reads the pre-event state vector -- so it drops there too.
"""

_RESERVED: Final = frozenset(
    {"abs", "dy", "event", "max", "min", "np", "out", "p", "range", "t", "w", "y", "z"}
)
"""Names the generated code uses itself, which no variable may be renamed onto."""


def _real(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):  # noqa: PLR0124 - NaN check
        return {"nan": "np.nan", "inf": "np.inf", "-inf": "-np.inf"}[repr(value)]
    return repr(float(value))


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


def _precedence(node: ir.Expr) -> int:
    match _unwrap(node):
        case ir.IfExpr():
            return _CONDITIONAL
        case ir.Binary(op=op):
            return _BINARY[op][1]
        case ir.Unary(op="not"):
            return _NOT
        case ir.Unary():
            return _SIGN
        case _:
            return _PRIMARY


def emit_expr(node: ir.Expr, symbols: Mapping[str, str], *, minimum: int = _CONDITIONAL) -> str:
    """Render ``node`` as Python, parenthesised if it binds looser than ``minimum``."""
    text = _render(_unwrap(node), symbols)
    return f"({text})" if _precedence(node) < minimum else text


def _call(node: ir.Call, symbols: Mapping[str, str]) -> str:
    if node.name == "div":
        left, right = (emit_expr(a, symbols) for a in node.args)
        return f"np.trunc(({left}) / ({right}))"
    if node.name in ("initial", "terminal"):
        return "False"
    if node.name not in _FUNCTIONS:
        raise EmitError(f"no translation for the function {node.name!r}")
    arguments = ", ".join(emit_expr(argument, symbols) for argument in node.args)
    return f"{_FUNCTIONS[node.name]}({arguments})"


def _render(node: ir.Expr, symbols: Mapping[str, str]) -> str:  # noqa: C901, PLR0911 - one case per node type
    match node:
        case ir.Ref():
            try:
                return symbols[node.name]
            except KeyError:
                raise EmitError(f"nothing computes {node.name!r}") from None
        case ir.RealLiteral(value=value):
            return _real(value)
        case ir.IntLiteral(value=value):
            return str(value)
        case ir.BoolLiteral(value=value):
            return "True" if value else "False"
        case ir.StringLiteral(value=value):
            return repr(value)
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            symbol, level = _BINARY[op]
            left = emit_expr(lhs, symbols, minimum=level + (op in _NON_ASSOCIATIVE))
            right = emit_expr(rhs, symbols, minimum=level + 1)
            return f"{left} {symbol} {right}"
        case ir.Unary(op="not", operand=operand):
            return f"not {emit_expr(operand, symbols, minimum=_NOT)}"
        case ir.Unary(op=op, operand=operand):
            return f"{op}{emit_expr(operand, symbols, minimum=_SIGN)}"
        case ir.Call():
            return _call(node, symbols)
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return (
                f"{emit_expr(then, symbols, minimum=1)} if "
                f"{emit_expr(condition, symbols, minimum=1)} else "
                f"{emit_expr(otherwise, symbols, minimum=_CONDITIONAL)}"
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

    def __str__(self) -> str:
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

    @property
    def latch(self) -> int | None:
        """The workspace slot to flip, for a relation inside an equation."""
        return self.trigger.latch


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
    """The expressions that end up in the generated code, and only those."""
    for block in system.blocks:
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
    """How long ``w`` has to be: one slot per observed variable, then the latches."""

    def __str__(self) -> str:
        return self.source


def _function(name: str, arguments: str, body: Iterable[str]) -> str:
    lines = [f"def {name}({arguments}):"]
    lines.extend(f"    {line}" if line else "" for line in body)
    return "\n".join(lines)


@dataclass
class _Emitter:
    system: System
    symbols: dict[str, str] = field(default_factory=dict[str, str])
    slots: dict[str, int] = field(default_factory=dict[str, int])
    observed: tuple[str, ...] = ()
    watched: tuple[ir.Binary, ...] = ()
    held: dict[ir.Expr, ir.Expr] = field(default_factory=dict[ir.Expr, ir.Expr])
    """Each watched relation, mapped to the test on its latch slot that replaces it."""

    def __post_init__(self) -> None:
        self.observed = (
            *(variable.name for variable in self.system.variables),
            *self.system.aliases,
        )
        taken = set(_RESERVED)
        self.symbols = {name: _identifier(name, taken) for name in self.observed}
        self.symbols["time"] = "t"
        self.slots = {name: slot for slot, name in enumerate(self.observed)}
        self.watched = _watched(self.system)
        for position, relation in enumerate(self.watched):
            held = f"$relation{position}"  # `$` keeps it out of reach of Modelica names
            self.symbols[held] = f"w[{self.holding(position)}]"
            self.held[relation] = ir.Binary(">", ir.Ref((held,)), ir.RealLiteral(0.5))

    def holding(self, position: int) -> int:
        """The workspace slot holding the ``position``-th watched relation."""
        return len(self.observed) + position

    # -- pieces ------------------------------------------------------------------------

    def expr(self, node: ir.Expr) -> str:
        return emit_expr(_latched(node, self.held), self.symbols)

    def live(self) -> list[str]:
        """The blocks with every relation read directly, for setting the latches."""
        held, self.held = self.held, {}
        try:
            return [*self.unpack(), *self.blocks()]
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

    def unpack(self) -> list[str]:
        """Read the parameter and state vectors into locals."""
        return [
            *(
                f"{self.symbols[variable.name]} = p[{slot}]"
                for slot, variable in enumerate(self.system.parameters)
            ),
            *(f"{self.symbols[name]} = y[{slot}]" for slot, name in enumerate(self.system.states)),
        ]

    def solve(self, block: Block) -> list[str]:
        """Compute a block's unknowns: an assignment, a linear solve, or Newton."""
        if block.assignment is not None:
            return [f"{self.symbols[block.unknowns[0]]} = {self.expr(block.assignment)}"]
        residuals = [residual(equation) for equation in block.equations]
        jacobian = [[simplify(d_dx(r, u)) for u in block.unknowns] for r in residuals]
        linear = not any(
            contains(entry, unknown)
            for row in jacobian
            for entry in row
            for unknown in block.unknowns
        )
        return self.linear(block, residuals, jacobian) if linear else self.newton(block, jacobian)

    def system_lines(
        self,
        block: Block,
        jacobian: list[list[ir.Expr]],
        right: Sequence[ir.Expr],
        indent: str = "",
    ) -> list[str]:
        """Fill ``_A`` and ``_b`` for one linear system, and solve it into ``_x``."""
        size = len(block.unknowns)
        lines = [f"{indent}_A = np.zeros(({size}, {size}))", f"{indent}_b = np.empty({size})"]
        for row, entries in enumerate(jacobian):
            lines.extend(
                f"{indent}_A[{row}, {column}] = {self.expr(entry)}"
                for column, entry in enumerate(entries)
                if entry != ZERO
            )
            lines.append(f"{indent}_b[{row}] = {self.expr(right[row])}")
        lines.append(f"{indent}_x = np.linalg.solve(_A, _b)")
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

    def blocks(self) -> list[str]:
        lines: list[str] = []
        for block in self.system.blocks:
            lines.extend(self.solve(block))
        return lines

    # -- functions ---------------------------------------------------------------------

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
        return _function("initial", "p, y, w", [*body, "return y"])

    def latches(self) -> str:
        """Freeze each watched relation at whatever it says at the initial point.

        The blocks are evaluated with the relations read directly here, and only here:
        there is no previous event to inherit a value from, so the live answer *is* the
        latched one.
        """
        body = self.live() if self.watched else []
        body.extend(
            f"w[{self.holding(position)}] = 1.0 if {emit_expr(relation, self.symbols)} else 0.0"
            for position, relation in enumerate(self.watched)
        )
        return _function("latch", "t, y, p, w", [*body, "return w"])

    def rhs(self) -> str:
        body = [*self.unpack(), *self.blocks()]
        body.extend(
            f"dy[{slot}] = {self.symbols[self.system.derivative[state]]}"
            for slot, state in enumerate(self.system.states)
        )
        return _function("rhs", "t, y, p, w, dy", [*body, "return dy"])

    def observe(self) -> str:
        body = [*self.unpack(), *self.blocks()]
        body.extend(
            f"{self.symbols[name]} = {self.expr(node)}"
            for name, node in self.system.aliases.items()
        )
        body.extend(
            f"out[{slot}] = {self.symbols[name]}" for slot, name in enumerate(self.observed)
        )
        return _function("observe", "t, y, p, w, out", [*body, "return out"])

    def events(self) -> tuple[_Event, ...]:
        """Every crossing to watch: the ``when`` branches first, then the relations."""
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
        return (*branches, *crossings)

    def indicators(self, events: tuple[_Event, ...]) -> str:
        body = [*self.unpack(), *(self.blocks() if events else [])]
        body.extend(f"z[{k}] = {self.expr(event.indicator)}" for k, event in enumerate(events))
        return _function("indicators", "t, y, p, w, z", [*body, "return z"])

    def statement(self, statement: ir.Statement, states: Mapping[str, int]) -> str:
        match statement:
            case ir.Reinit(target=target, value=value) if target.name in states:
                return f"    out[{states[target.name]}] = {self.expr(value)}"
            case ir.Reinit(target=target):
                raise EmitError(f"reinit({target.name}) -- only a state can be reinitialised")
            case _:
                raise EmitError(f"no translation for {statement!r} inside a when-clause")

    def transition(self, events: tuple[_Event, ...]) -> str:
        """What one event does: the branch's statements, or a latch flipping over."""
        states = {name: slot for slot, name in enumerate(self.system.states)}
        body = [*self.unpack(), *(self.blocks() if events else []), "out[:] = y"]
        for position, event in enumerate(events):
            lines = [self.statement(s, states) for s in event.body]
            if event.latch is not None:
                lines.append(f"    w[{event.latch}] = 1.0 - w[{event.latch}]")
            body.append(f"if event == {position}:")
            body.extend(lines or ["    pass"])
        return _function("transition", "event, t, y, p, w, out", [*body, "return out"])

    def code(self) -> Code:
        events = self.events()
        parts = [
            "import numpy as np",
            self.parameters(),
            self.initial(),
            self.latches(),
            self.rhs(),
            self.observe(),
            self.indicators(events),
            self.transition(events),
        ]
        return Code(
            name=self.system.name,
            source="\n\n\n".join(parts) + "\n",
            states=self.system.states,
            parameters=tuple(v.name for v in self.system.parameters),
            observed=self.observed,
            events=tuple(event.trigger for event in events),
            workspace=len(self.observed) + len(self.watched),
        )


def emit(system: System) -> Code:
    """Write ``system`` out as a Python module, ready to be compiled."""
    return _Emitter(system).code()
