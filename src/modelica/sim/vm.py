"""Lower a causalized model to a register tape: the bytecode a precompiled engine runs.

The design is `docs/issues/08-11-2026-a-modelica-vm-runs-the-model-as-bytecode-in-a-precompiled-rust-engine.md`.
`lower` reads the same structural decisions `modelica.sim.emit.Emitter` makes (the observed
layout, the latches, the events, the modes, the blocks) and writes each kernel function as a flat
list of three-address instructions over one ``f64`` register file, instead of as source text. No
text is reparsed: every instruction comes from an `ir.Expr` or a `Block`.

`evaluate` runs a tape in Python. It is the reference the Rust engine is tested against, and the
proof that the lowering means what the interpreter means; it is not a fast path.

A truth value is ``1.0`` or ``0.0``. An if-expression is a `Op.SELECT`: both branches are
computed and one is kept, so a tape never branches. A Newton block is the one structured
instruction (`Op.NEWTON`): the engine runs its body until the step is below `TOLERANCE`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from modelica.ir import nodes as ir
from modelica.ir.derive import simplify
from modelica.ir.flat import at_cell
from modelica.ir.rewrite import names, residual, substitute
from modelica.sim.emit import (
    ITERATIONS,
    TOLERANCE,
    EmitError,
    Emitter,
    Trigger,
    _latched,  # pyright: ignore[reportPrivateUsage] - the same latching rule, not a copy
    _unwrap,  # pyright: ignore[reportPrivateUsage]
    block_jacobian,
    is_terminate,
    iterated,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from modelica.ir.system import ArrayBlock, Block, Mode, System

__all__ = ["Function", "Instr", "NewtonBlock", "Op", "Program", "evaluate", "lower"]


class Op(IntEnum):
    """One opcode. Operands are register numbers unless the comment says otherwise."""

    CONST = 0  # dst, a = index into the constant pool
    TIME = 1  # dst
    LOAD_Y = 2  # dst, a = slot of y
    LOAD_P = 3  # dst, a = slot of p
    LOAD_W = 4  # dst, a = slot of w
    STORE_OUT = 5  # a = slot of out, b = source register
    STORE_W = 6  # a = slot of w, b = source register
    MOVE = 7  # dst, a
    ADD = 8
    SUB = 9
    MUL = 10
    DIV = 11
    POW = 12
    LT = 13
    LE = 14
    GT = 15
    GE = 16
    EQ = 17
    NE = 18
    AND = 19
    OR = 20
    NEG = 21  # dst, a
    NOT = 22  # dst, a
    CALL1 = 23  # dst, a, c = builtin
    CALL2 = 24  # dst, a, b, c = builtin
    SELECT = 25  # dst, a = condition, b = then, c = otherwise
    SOLVE = 26  # dst = first of n x registers, a = first of n*n A registers, b = first of n b
    NEWTON = 27  # a = index into Function.newton
    COPY_Y_OUT = 28  # out[:] = y
    CLEAR_OUT = 29  # out[:a] = 0
    HOIST = 30  # no operands: what comes before reads only time and constants (`_hoist`)
    # Array opcodes over runs of ``n`` consecutive registers (`_ARRAY`). A table is a run of
    # register numbers in `Program.tables`, named by its offset.
    VGATHER = 31  # dst[k] = r[tables[a + k]], k < c
    VSCATTER = 32  # r[tables[a + k]] = b[k], k < c
    VBROADCAST = 33  # dst[k] = r[a], k < c
    VADD = 34  # dst[k] = a[k] + b[k], k < c
    VSUB = 35
    VMUL = 36
    VDIV = 37
    VNEG = 38  # dst[k] = -a[k], k < c
    VCALL1 = 39  # dst[k] = builtin c of a[k], k < b
    # Written only by `_fuse` (in ``rhs`` and ``observe``), after every pass that reads registers.
    VLOAD_Y = 40  # dst[k] = y[a + k], k < c
    VSTORE_OUT = 41  # out[a + k] = b[k], k < c


class Builtin(IntEnum):
    """The functions a ``CALL1``/``CALL2`` names, with Modelica's meaning (`dialect.RUST`)."""

    ABS = 0
    ACOS = 1
    ASIN = 2
    ATAN = 3
    CEIL = 4
    COS = 5
    COSH = 6
    EXP = 7
    FLOOR = 8
    LOG = 9
    LOG10 = 10
    SIGN = 11
    SIN = 12
    SINH = 13
    SQRT = 14
    TAN = 15
    TANH = 16
    TRUNC = 17
    ATAN2 = 32
    MAX = 33
    MIN = 34
    MOD = 35
    REM = 36


_UNARY_CALLS: Final = {
    "abs": Builtin.ABS,
    "acos": Builtin.ACOS,
    "asin": Builtin.ASIN,
    "atan": Builtin.ATAN,
    "ceil": Builtin.CEIL,
    "cos": Builtin.COS,
    "cosh": Builtin.COSH,
    "exp": Builtin.EXP,
    "floor": Builtin.FLOOR,
    "integer": Builtin.FLOOR,
    "log": Builtin.LOG,
    "log10": Builtin.LOG10,
    "sign": Builtin.SIGN,
    "sin": Builtin.SIN,
    "sinh": Builtin.SINH,
    "sqrt": Builtin.SQRT,
    "tan": Builtin.TAN,
    "tanh": Builtin.TANH,
}
_BINARY_CALLS: Final = {
    "atan2": Builtin.ATAN2,
    "max": Builtin.MAX,
    "min": Builtin.MIN,
    "mod": Builtin.MOD,
    "rem": Builtin.REM,
}
_BINARY_OPS: Final = {
    "+": Op.ADD,
    "-": Op.SUB,
    "*": Op.MUL,
    "/": Op.DIV,
    "^": Op.POW,
    ".*": Op.MUL,
    "./": Op.DIV,
    ".^": Op.POW,
    "<": Op.LT,
    "<=": Op.LE,
    ">": Op.GT,
    ">=": Op.GE,
    "==": Op.EQ,
    "<>": Op.NE,
    "and": Op.AND,
    "or": Op.OR,
}


@dataclass(frozen=True, slots=True)
class Instr:
    """One instruction: an opcode and up to four operands, unused ones zero."""

    op: Op
    dst: int = 0
    a: int = 0
    b: int = 0
    c: int = 0


@dataclass(frozen=True, slots=True)
class NewtonBlock:
    """An implicit block: ``body`` fills ``A`` and ``b`` from the current unknowns.

    The engine repeats: run ``body``, solve ``A x = b`` into the ``x`` registers, add ``x`` to
    the unknowns, stop when ``max |x| < TOLERANCE`` or after `ITERATIONS` rounds.
    """

    unknowns: tuple[int, ...]
    body: tuple[Instr, ...]
    matrix: int
    """First register of the row-major ``n x n`` matrix."""
    right: int
    """First register of the right-hand side."""
    step: int
    """First register of the solution."""


@dataclass(frozen=True, slots=True)
class Function:
    """One kernel function for one mode: straight-line code plus its Newton bodies."""

    code: tuple[Instr, ...]
    newton: tuple[NewtonBlock, ...] = ()
    tables: tuple[int, ...] = ()
    """Register numbers that ``VGATHER`` and ``VSCATTER`` read through, by offset."""


@dataclass(frozen=True)
class Layout:
    """The names and sizes a caller needs around the tapes, as `Code` records them."""

    states: tuple[str, ...]
    parameters: tuple[str, ...]
    observed: tuple[str, ...]
    events: tuple[Trigger, ...]
    workspace: int


@dataclass(frozen=True)
class Program:
    """Every kernel function of one model, one entry per mode where a model has modes.

    ``transition[mode][event]`` is the code for one event. The function the event loop calls is
    that entry run after ``transition_prefix[mode]``, which reads the variables and copies ``y``
    to ``out``.
    """

    registers: int
    constants: tuple[float, ...]
    rhs: tuple[Function, ...]
    jacobian: tuple[Function, ...]
    """``d rhs / d y``, row-major, built from ``rhs`` by forward mode (`_Tangents`)."""
    observe: tuple[Function, ...]
    indicators: tuple[Function, ...]
    transition_prefix: tuple[Function, ...]
    transition: tuple[tuple[Function, ...], ...]
    switching: int | None
    """The workspace slot holding the mode, as `Code.switching`."""
    parameters: Function
    """Writes the parameter vector into ``out``, as the emitted ``parameters``."""
    initial: Function
    """Reads ``p``; writes the initial state into ``out`` and the workspace, as ``initial``."""
    latch: Function
    """Sets every latch from the relation's live value, as the emitted ``latch``."""
    layout: Layout
    order: tuple[int, ...] = ()
    """The state order Radau factors its matrices in, `bandwidth_order` of the first mode.
    Empty for the natural order."""
    times: tuple[int, ...] = ()
    """Per event, the workspace slot of its instant for a time event (`Trigger.at`), else -1.
    Empty when no event is a time event."""


@dataclass
class _Scope:
    """What one instruction stream is being written into, and what it already computed."""

    code: list[Instr]
    values: dict[ir.Expr, int]
    names: dict[str, int]
    newton: list[NewtonBlock]
    tables: list[int] = field(default_factory=list[int])

    def child(self) -> _Scope:
        """A scope for a Newton body: it sees what is known, and leaks nothing back."""
        return _Scope([], dict(self.values), dict(self.names), self.newton)


@dataclass
class _Lowering:
    """Writes the tapes, reading every decision from an `Emitter` over the same system."""

    emitter: Emitter
    registers: int = 0
    constants: dict[float, int] = field(default_factory=dict[float, int])
    cells: dict[tuple[int, ir.Expr], tuple[ir.Expr, ...]] = field(
        default_factory=dict[tuple[int, ir.Expr], tuple[ir.Expr, ...]]
    )
    """A subscript of an `ArrayBlock` at each cell, by the block's first index and the node."""

    def register(self, count: int = 1) -> int:
        first = self.registers
        self.registers += count
        return first

    def constant(self, scope: _Scope, value: float) -> int:
        key = ir.RealLiteral(value)
        if key in scope.values:
            return scope.values[key]
        index = self.constants.setdefault(value, len(self.constants))
        dst = self.register()
        scope.code.append(Instr(Op.CONST, dst, index))
        scope.values[key] = dst
        return dst

    # -- expressions -----------------------------------------------------------------

    def expr(self, scope: _Scope, node: ir.Expr) -> int:
        """The register holding ``node`` after latching, as `Emitter.expr` renders it."""
        return self.value(scope, _latched(node, self.emitter.held))

    def value(self, scope: _Scope, node: ir.Expr) -> int:
        node = _unwrap(node)
        if (known := scope.values.get(node)) is not None:
            return known
        dst = self.compute(scope, node)
        scope.values[node] = dst
        return dst

    def compute(self, scope: _Scope, node: ir.Expr) -> int:  # noqa: C901, PLR0911 - one case per node type
        match node:
            case ir.Ref(name=name):
                return self.ref(scope, name)
            case ir.RealLiteral(value=value):
                return self.constant(scope, value)
            case ir.IntLiteral(value=value):
                return self.constant(scope, float(value))
            case ir.BoolLiteral(value=value):
                return self.constant(scope, 1.0 if value else 0.0)
            case ir.Binary(op=op, lhs=lhs, rhs=rhs) if op in _BINARY_OPS:
                a, b = self.value(scope, lhs), self.value(scope, rhs)
                return self.emit(scope, _BINARY_OPS[op], a, b)
            case ir.Unary(op="+", operand=operand):
                return self.value(scope, operand)
            case ir.Unary(op="-", operand=operand):
                return self.emit(scope, Op.NEG, self.value(scope, operand))
            case ir.Unary(op="not", operand=operand):
                return self.emit(scope, Op.NOT, self.value(scope, operand))
            case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
                c, a, b = (self.value(scope, n) for n in (condition, then, otherwise))
                return self.emit(scope, Op.SELECT, c, a, b)
            case ir.Call():
                return self.call(scope, node)
            case _:
                message = f"no tape instruction for {type(node).__name__}"
                raise EmitError(message)

    def call(self, scope: _Scope, node: ir.Call) -> int:
        name = node.name
        if name in ("initial", "terminal"):
            return self.constant(scope, 0.0)
        if name == "div":
            a, b = (self.value(scope, argument) for argument in node.args)
            quotient = self.emit(scope, Op.DIV, a, b)
            return self.emit(scope, Op.CALL1, quotient, 0, Builtin.TRUNC)
        if name not in _UNARY_CALLS and name not in _BINARY_CALLS and "." in name:
            name = name.rsplit(".", 1)[-1]
        args = [self.value(scope, argument) for argument in node.args]
        if name in _UNARY_CALLS and len(args) == 1:
            return self.emit(scope, Op.CALL1, args[0], 0, _UNARY_CALLS[name])
        if name in _BINARY_CALLS and len(args) == 2:  # noqa: PLR2004 - two arguments
            return self.emit(scope, Op.CALL2, args[0], args[1], _BINARY_CALLS[name])
        message = f"no translation for the function {node.name!r} in the tape"
        raise EmitError(message)

    def emit(self, scope: _Scope, op: Op, a: int, b: int = 0, c: int = 0) -> int:
        dst = self.register()
        scope.code.append(Instr(op, dst, a, b, c))
        return dst

    def ref(self, scope: _Scope, name: str) -> int:
        if name in scope.names:
            return scope.names[name]
        emitter = self.emitter
        if name == "time":
            dst = self.emit(scope, Op.TIME, 0)
        elif name.startswith("$relation"):
            slot = emitter.holding(int(name.removeprefix("$relation")))
            dst = self.emit(scope, Op.LOAD_W, slot)
        else:
            message = f"nothing computes {name!r}"
            raise EmitError(message)
        scope.names[name] = dst
        return dst

    # -- blocks ----------------------------------------------------------------------

    def unpack(self, scope: _Scope, mode: Mode) -> None:
        """Read parameters, states and remembered variables, as `Emitter.unpack`."""
        system = self.emitter.system
        for slot, variable in enumerate(system.parameters):
            scope.names[variable.name] = self.emit(scope, Op.LOAD_P, slot)
        for slot, name in enumerate(mode.states):
            scope.names[name] = self.emit(scope, Op.LOAD_Y, slot)
        for name in self.emitter.remembered:
            if name not in mode.states:
                scope.names[name] = self.emit(scope, Op.LOAD_W, self.emitter.slots[name])

    def assign(self, scope: _Scope, name: str, node: ir.Expr) -> None:
        source = self.expr(scope, node)
        # A fresh register per variable: a later block must not see a value through a
        # register some other expression shares and the Newton loop then overwrites.
        dst = self.emit(scope, Op.MOVE, source)
        scope.names[name] = dst

    def blocks(self, scope: _Scope, mode: Mode, *, arrays: bool = False) -> None:
        """Every block in order; with ``arrays``, each `ArrayBlock` it can as array opcodes."""
        runs = {array.first: array for array in mode.arrays} if arrays else {}
        k = 0
        while k < len(mode.blocks):
            array = runs.get(k)
            if array is not None and self.array(scope, mode, array):
                k += len(array)
                continue
            self.solve(scope, mode.blocks[k])
            k += 1

    def array(self, scope: _Scope, mode: Mode, array: ArrayBlock) -> bool:  # noqa: C901 - one case per node
        """One `ArrayBlock` as array opcodes; False, having written nothing, when it cannot.

        The loop's right-hand side is written once over all cells: a subscript is a
        ``VGATHER``, a part that reads no subscript is computed once and broadcast, and the
        arithmetic is elementwise. Each cell does the operations its scalar block would, in
        the same order, so the results are the same bits. A ``VSCATTER`` then gives each cell's
        unknown its own register, as `assign` does.
        """
        equation = array.equation
        cells = list(equation.cells())
        run = mode.blocks[array.first : array.first + len(cells)]
        shapes, constants = dict(equation.shapes), dict(equation.constants)
        template = _reals(equation.equation.rhs)
        indices = frozenset(index.name for index in equation.indices)
        if not _vectorizable(template, indices):
            return False
        for k in {0, len(cells) - 1}:
            if run[k].assignment != at_cell(template, cells[k], shapes, constants):
                return False
        n = len(cells)

        def go(node: ir.Expr) -> int:
            node = _unwrap(node)
            if not _reads_array(node):
                dst = self.register(n)
                scope.code.append(Instr(Op.VBROADCAST, dst, self.expr(scope, node), 0, n))
                return dst
            dst = self.register(n)
            match node:
                case ir.Subscript():
                    key = (array.first, node)
                    if key not in self.cells:  # `rhs` and `observe` gather the same cells
                        self.cells[key] = tuple(
                            at_cell(node, cell, shapes, constants) for cell in cells
                        )
                    offset = len(scope.tables)
                    scope.tables.extend(self.value(scope, cell) for cell in self.cells[key])
                    scope.code.append(Instr(Op.VGATHER, dst, offset, 0, n))
                case ir.Binary(op=op, lhs=lhs, rhs=rhs):
                    left, right = go(lhs), go(rhs)
                    scope.code.append(Instr(_VECTOR_OPS[op], dst, left, right, n))
                case ir.Unary(op="-", operand=operand):
                    scope.code.append(Instr(Op.VNEG, dst, go(operand), 0, n))
                case ir.Call(name=name, args=(argument,)):
                    scope.code.append(Instr(Op.VCALL1, dst, go(argument), n, _UNARY_CALLS[name]))
                case _:
                    raise AssertionError(node)
            return dst

        result = go(template)
        targets = [self.register() for _ in cells]
        offset = len(scope.tables)
        scope.tables.extend(targets)
        scope.code.append(Instr(Op.VSCATTER, 0, offset, result, n))
        for block, register in zip(run, targets, strict=True):
            scope.names[block.unknowns[0]] = register
        return True

    def solve(self, scope: _Scope, block: Block) -> None:
        """One block, as `Emitter.solve`: assignment, linear solve, or Newton."""
        if block.assignment is not None:
            self.assign(scope, block.unknowns[0], block.assignment)
            return
        residuals = [residual(equation) for equation in block.equations]
        jacobian = block_jacobian(block)
        if iterated(block):
            self.newton(scope, block, residuals, jacobian)
        else:
            self.linear(scope, block, residuals, jacobian)
        for name, value in block.torn:
            self.assign(scope, name, value)

    def system_into(
        self,
        scope: _Scope,
        jacobian: list[list[ir.Expr]],
        right: Sequence[ir.Expr],
    ) -> tuple[int, int, int]:
        """Fill contiguous ``A`` and ``b`` registers; return their first registers and ``x``'s."""
        size = len(right)
        entries = [self.expr(scope, entry) for row in jacobian for entry in row]
        values = [self.expr(scope, r) for r in right]
        matrix = self.register(size * size)
        for offset, source in enumerate(entries):
            scope.code.append(Instr(Op.MOVE, matrix + offset, source))
        column = self.register(size)
        for offset, source in enumerate(values):
            scope.code.append(Instr(Op.MOVE, column + offset, source))
        return matrix, column, self.register(size)

    def linear(
        self,
        scope: _Scope,
        block: Block,
        residuals: list[ir.Expr],
        jacobian: list[list[ir.Expr]],
    ) -> None:
        at_zero = dict.fromkeys(block.unknowns, ir.RealLiteral(0.0))
        right = [ir.Unary("-", simplify(substitute(r, at_zero))) for r in residuals]
        matrix, column, step = self.system_into(scope, jacobian, right)
        scope.code.append(Instr(Op.SOLVE, step, matrix, column, len(right)))
        for position, unknown in enumerate(block.unknowns):
            scope.names[unknown] = self.emit(scope, Op.MOVE, step + position)

    def newton(
        self,
        scope: _Scope,
        block: Block,
        residuals: list[ir.Expr],
        jacobian: list[list[ir.Expr]],
    ) -> None:
        slots = self.emitter.slots
        unknowns = tuple(self.emit(scope, Op.LOAD_W, slots[u]) for u in block.unknowns)
        inner = scope.child()
        for name, register in zip(block.unknowns, unknowns, strict=True):
            inner.names[name] = register
            inner.values.pop(ir.Ref((name,)), None)
        negated = [ir.Unary("-", r) for r in residuals]
        matrix, column, step = self.system_into(inner, jacobian, negated)
        scope.newton.append(
            NewtonBlock(unknowns, tuple(inner.code), matrix=matrix, right=column, step=step)
        )
        scope.code.append(Instr(Op.NEWTON, 0, len(scope.newton) - 1))
        for name, register in zip(block.unknowns, unknowns, strict=True):
            scope.names[name] = register
            scope.code.append(Instr(Op.STORE_W, 0, slots[name], register))

    # -- functions -------------------------------------------------------------------

    def function(self, body: Callable[[_Scope], None]) -> Function:
        scope = _Scope([], {}, {}, [])
        body(scope)
        return Function(tuple(scope.code), tuple(scope.newton), tuple(scope.tables))

    def parameters(self) -> Function:
        """The parameter vector, each binding after the ones it reads (`Emitter.ordered`)."""

        def body(scope: _Scope) -> None:
            for variable in self.emitter.ordered():
                node = variable.binding if variable.binding is not None else variable.start
                self.assign(scope, variable.name, node if node is not None else ir.RealLiteral(0.0))
            for slot, variable in enumerate(self.emitter.system.parameters):
                source = self.ref(scope, variable.name)
                scope.code.append(Instr(Op.STORE_OUT, 0, slot, source))

        return self.function(body)

    def initial(self) -> Function:
        """The start values into ``y`` and the workspace, as `Emitter.initial`."""
        emitter = self.emitter
        system = emitter.system
        constant = {variable.name for variable in system.parameters}

        def start(scope: _Scope, name: str) -> int:
            return self.expr(scope, system[name].start or ir.RealLiteral(0.0))

        def body(scope: _Scope) -> None:
            for slot, variable in enumerate(system.parameters):
                scope.names[variable.name] = self.emit(scope, Op.LOAD_P, slot)
            for slot, name in enumerate(system.states):
                scope.code.append(Instr(Op.STORE_OUT, 0, slot, start(scope, name)))
            for name in emitter.observed:
                if name in constant:
                    source = self.ref(scope, name)
                elif name in system and system[name].start is not None:
                    source = start(scope, name)
                else:
                    source = self.constant(scope, 0.0)
                scope.code.append(Instr(Op.STORE_W, 0, emitter.slots[name], source))
            states = {name: slot for slot, name in enumerate(system.states)}
            for name, value in emitter.conditions():
                self.assign(scope, name, value)
                source = self.ref(scope, name)
                scope.code.append(Instr(Op.STORE_OUT, 0, states[name], source))
                scope.code.append(Instr(Op.STORE_W, 0, emitter.slots[name], source))
            if emitter.switching is not None:
                zero = self.constant(scope, 0.0)
                scope.code.append(Instr(Op.STORE_W, 0, emitter.switching, zero))

        return self.function(body)

    def latch(self) -> Function:
        """Each watched relation, read live, into its latch slot (`Emitter.latches`)."""
        emitter = self.emitter

        def body(scope: _Scope) -> None:
            if not emitter.watched:
                return
            held, emitter.held = emitter.held, {}
            try:
                self.unpack(scope, emitter.system.mode)
                self.blocks(scope, emitter.system.mode)
            finally:
                emitter.held = held
            for position, relation in enumerate(emitter.watched):
                source = self.value(scope, relation)
                scope.code.append(Instr(Op.STORE_W, 0, emitter.holding(position), source))

        return self.function(body)

    def rhs(self, mode: Mode, *, arrays: bool = False) -> Function:
        def body(scope: _Scope) -> None:
            self.unpack(scope, mode)
            self.blocks(scope, mode, arrays=arrays)
            for slot, state in enumerate(mode.states):
                source = self.ref(scope, mode.derivative[state])
                scope.code.append(Instr(Op.STORE_OUT, 0, slot, source))

        return self.function(body)

    def observe(self, mode: Mode, *, arrays: bool = False) -> Function:
        emitter = self.emitter

        def body(scope: _Scope) -> None:
            self.unpack(scope, mode)
            self.blocks(scope, mode, arrays=arrays)
            for name, node in emitter.system.aliases.items():
                self.assign(scope, name, node)
            for slot, name in enumerate(emitter.observed):
                scope.code.append(Instr(Op.STORE_OUT, 0, slot, self.ref(scope, name)))

        return self.function(body)

    def indicators(self, position: int, mode: Mode) -> Function:
        events = self.emitter.events()

        def body(scope: _Scope) -> None:
            self.unpack(scope, mode)
            if events:
                self.blocks(scope, mode)
            for k, event in enumerate(events):
                live = event.origin in (None, position)
                source = self.expr(scope, event.indicator) if live else self.constant(scope, -1.0)
                scope.code.append(Instr(Op.STORE_OUT, 0, k, source))

        return self.function(body)

    def transition(self, position: int, mode: Mode) -> tuple[Function, tuple[Function, ...]]:
        """The shared prefix, and one function per event, as `Emitter.transition`."""
        emitter = self.emitter
        events = emitter.events()
        prefix = _Scope([], {}, {}, [])
        self.unpack(prefix, mode)
        if events:
            self.blocks(prefix, mode)
        prefix.code.append(Instr(Op.COPY_Y_OUT))
        states = {name: slot for slot, name in enumerate(mode.states)}
        branches: list[Function] = []
        for event in events:
            scope = _Scope([], dict(prefix.values), dict(prefix.names), [])
            for statement in event.body:
                if not is_terminate(statement):
                    self.statement(scope, statement, states)
            if event.latch is not None:
                held = self.emit(scope, Op.LOAD_W, event.latch)
                flipped = self.emit(scope, Op.SUB, self.constant(scope, 1.0), held)
                scope.code.append(Instr(Op.STORE_W, 0, event.latch, flipped))
            if (moving := event.trigger.switch) is not None and event.origin == position:
                self.rebase(scope, emitter.system.modes[moving])
                slot = emitter.switching
                if slot is None:  # a switch implies a mode slot
                    message = "a switch with no mode slot"
                    raise EmitError(message)
                scope.code.append(Instr(Op.STORE_W, 0, slot, self.constant(scope, float(moving))))
            branches.append(Function(tuple(scope.code), tuple(scope.newton)))
        return Function(tuple(prefix.code), tuple(prefix.newton)), tuple(branches)

    def statement(self, scope: _Scope, statement: ir.Statement, states: Mapping[str, int]) -> None:
        """One line of a ``when`` body, with `Emitter.statement`'s refusals."""
        emitter = self.emitter
        match statement:
            case ir.Reinit(target=target, value=value) if target.name in states:
                source = self.expr(scope, value)
                scope.code.append(Instr(Op.STORE_OUT, 0, states[target.name], source))
            case ir.Assign(target=ir.Ref(name=name), value=value) if name in emitter.remembered:
                source = self.expr(scope, value)
                scope.code.append(Instr(Op.STORE_W, 0, emitter.slots[name], source))
            case _:
                emitter.statement(statement, states)  # raises the emitter's own message
                message = f"no tape instruction for {statement!r} inside a when-clause"
                raise EmitError(message)

    def rebase(self, scope: _Scope, mode: Mode) -> None:
        for slot, name in enumerate(mode.states):
            scope.code.append(Instr(Op.STORE_OUT, 0, slot, self.ref(scope, name)))
        for block in mode.blocks:
            if iterated(block):
                for unknown in block.unknowns:
                    source = self.ref(scope, unknown)
                    scope.code.append(Instr(Op.STORE_W, 0, self.emitter.slots[unknown], source))


_LN10: Final = math.log(10.0)


#: Partials of a run of ``n`` registers: for each pattern (the state cell ``k`` differentiates
#: by, ``None`` for a zero partial) the first register of the run holding them.
_Runs = dict[tuple[int | None, ...], int]


class _Scalar(Exception):  # noqa: N818 - a signal, not an error
    """The tangents of an array opcode cannot be written as runs; use the scalar ``rhs``."""


def _distinct(runs: _Runs) -> None:
    """Raise `_Scalar` when two patterns in ``runs`` name the same state in one cell."""
    if len(runs) < 2:  # noqa: PLR2004
        return
    for column in zip(*runs, strict=True):
        named = [j for j in column if j is not None]
        if len(named) != len(set(named)):
            raise _Scalar


@dataclass
class _Tangents:
    """Forward mode over a tape: ``d out / d y`` for every state at once, sparse by seed.

    Each register's tangent is a map from a state index to the register holding that partial;
    a missing index is zero. The primal instructions are kept, and after each one the partials
    it implies are written. A linear solve differentiates as ``A dx = db - dA x``. A Newton block
    is re-evaluated once at its converged point and differentiated implicitly: with the unknowns'
    own partials held at zero, ``b = -F`` gives ``db = -F_v dv``, and ``du = A^-1 db``.
    """

    lowering: _Lowering
    width: int
    code: list[Instr] = field(default_factory=list[Instr])
    dots: dict[int, dict[int, int]] = field(default_factory=dict[int, dict[int, int]])
    runs: dict[int, _Runs] = field(default_factory=dict[int, "_Runs"])
    tables: list[int] = field(default_factory=list[int])
    scope: _Scope = field(default_factory=lambda: _Scope([], {}, {}, []))

    def emit(self, op: Op, a: int, b: int = 0, c: int = 0) -> int:
        dst = self.lowering.register()
        self.code.append(Instr(op, dst, a, b, c))
        return dst

    def constant(self, value: float) -> int:
        self.scope.code = self.code
        return self.lowering.constant(self.scope, value)

    def scaled(self, factor: int, tangent: Mapping[int, int]) -> dict[int, int]:
        return {j: self.emit(Op.MUL, factor, d) for j, d in tangent.items()}

    def combine(self, op: Op, left: Mapping[int, int], right: Mapping[int, int]) -> dict[int, int]:
        """``left + right`` or ``left - right``, seed by seed."""
        out: dict[int, int] = {}
        for j in left.keys() | right.keys():
            if j in left and j in right:
                out[j] = self.emit(op, left[j], right[j])
            elif j in left:
                out[j] = left[j]
            else:
                out[j] = right[j] if op is Op.ADD else self.emit(Op.NEG, right[j])
        return out

    def function(self, rhs: Function) -> Function:
        self.code = [Instr(Op.CLEAR_OUT, 0, self.width * self.width)]
        self.tables = list(rhs.tables)
        self.run(rhs, rhs.code)
        return Function(tuple(self.code), rhs.newton, tuple(self.tables))

    def vemit(self, op: Op, n: int, a: int, b: int = 0) -> int:
        dst = self.lowering.register(n)
        self.code.append(Instr(op, dst, a, b, n))
        return dst

    def vcombine(self, op: Op, left: _Runs, right: _Runs) -> _Runs:
        """``left + right`` or ``left - right`` over runs, pattern by pattern.

        Raises `_Scalar` when two different patterns name the same state in one cell: the
        scalar tangent would add them here, and the vector form cannot.
        """
        out: _Runs = {}
        for key in left.keys() | right.keys():
            if key in left and key in right:
                out[key] = self.vemit(op, len(key), left[key], right[key])
            elif key in left:
                out[key] = left[key]
            else:
                out[key] = right[key] if op is Op.ADD else self.vemit(Op.VNEG, len(key), right[key])
        _distinct(out)
        return out

    def vstep(self, i: Instr) -> None:  # noqa: C901, PLR0912 - one case per array opcode
        """The partials of one array opcode, as runs keyed by the state each cell reads."""
        runs, n = self.runs, i.c
        a, b = runs.get(i.a, {}), runs.get(i.b, {})
        out: _Runs
        if i.op is Op.VGATHER:
            out = self.gather([self.dots.get(r, {}) for r in self.tables[i.a : i.a + n]])
        elif i.op is Op.VSCATTER:
            for k, target in enumerate(self.tables[i.a : i.a + n]):
                cell = {key[k]: run + k for key, run in self.runs.get(i.b, {}).items()}
                cell.pop(None, None)  # type: ignore[call-overload]
                if cell:
                    self.dots[target] = cell  # type: ignore[assignment]
                else:
                    self.dots.pop(target, None)
            return
        elif i.op is Op.VBROADCAST:
            out = {
                (j,) * n: self.vemit(Op.VBROADCAST, n, d) for j, d in self.dots.get(i.a, {}).items()
            }
        elif i.op is Op.VNEG:
            out = {key: self.vemit(Op.VNEG, n, d) for key, d in a.items()}
        elif i.op in (Op.VADD, Op.VSUB):
            out = self.vcombine(_SCALAR_OF[i.op], a, b)
        elif i.op is Op.VMUL:
            left = {key: self.vemit(Op.VMUL, n, i.b, d) for key, d in a.items()}
            right = {key: self.vemit(Op.VMUL, n, i.a, d) for key, d in b.items()}
            out = self.vcombine(Op.ADD, left, right)
        elif i.op is Op.VDIV:
            scaled = {key: self.vemit(Op.VMUL, n, i.dst, d) for key, d in b.items()}
            inner = self.vcombine(Op.SUB, a, scaled)
            out = {key: self.vemit(Op.VDIV, n, d, i.b) for key, d in inner.items()}
        elif a:  # VCALL1 of something with partials: not written as runs yet
            raise _Scalar
        else:
            out = {}
        if out:
            runs[i.dst] = out
        else:
            runs.pop(i.dst, None)

    def gather(self, cells: Sequence[Mapping[int, int]]) -> _Runs:
        """Runs for a gather whose cell ``k`` has the scalar partials ``cells[k]``.

        Run ``q`` takes each cell's ``q``-th partial by state order; a cell with fewer is
        padded with a zero partial, keyed ``None``.
        """
        n = len(cells)
        width = max((len(c) for c in cells), default=0)
        out: _Runs = {}
        for q in range(width):
            key: list[int | None] = []
            registers: list[int] = []
            for cell in cells:
                items = sorted(cell.items())
                if q < len(items):
                    key.append(items[q][0])
                    registers.append(items[q][1])
                else:
                    key.append(None)
                    registers.append(self.constant(0.0))
            offset = len(self.tables)
            self.tables.extend(registers)
            dst = self.lowering.register(n)
            self.code.append(Instr(Op.VGATHER, dst, offset, 0, n))
            out[tuple(key)] = dst
        _distinct(out)
        return out

    def run(self, function: Function, code: Sequence[Instr]) -> None:
        for i in code:
            if i.op is not Op.STORE_OUT:  # out holds the Jacobian here, not ``dy``
                self.code.append(i)
            self.step(function, i)

    def step(self, function: Function, i: Instr) -> None:  # noqa: C901, PLR0912 - one case per opcode
        dots = self.dots
        op = i.op
        a, b = dots.get(i.a, {}), dots.get(i.b, {})
        if op is Op.LOAD_Y:
            dots[i.dst] = {i.a: self.constant(1.0)}
        elif op is Op.MOVE:
            if a:
                dots[i.dst] = dict(a)
            else:
                dots.pop(i.dst, None)
        elif op is Op.STORE_OUT:
            for j, d in sorted(dots.get(i.b, {}).items()):
                self.code.append(Instr(Op.STORE_OUT, 0, i.a * self.width + j, d))
        elif op in (Op.ADD, Op.SUB):
            dots[i.dst] = self.combine(op, a, b)
        elif op is Op.NEG:
            dots[i.dst] = {j: self.emit(Op.NEG, d) for j, d in a.items()}
        elif op is Op.MUL:
            dots[i.dst] = self.combine(Op.ADD, self.scaled(i.b, a), self.scaled(i.a, b))
        elif op is Op.DIV:
            # d(a/b) = (da - (a/b) db) / b
            inner = self.combine(Op.SUB, a, self.scaled(i.dst, b))
            dots[i.dst] = {j: self.emit(Op.DIV, d, i.b) for j, d in inner.items()}
        elif op is Op.POW:
            self.power(i, a, b)
        elif op is Op.SELECT:
            self.select(i)
        elif op is Op.CALL1:
            self.call1(i, a)
        elif op is Op.CALL2:
            self.call2(i, a, b)
        elif op is Op.SOLVE:
            self.solve(i.c, i.a, i.b, i.dst)
        elif op is Op.NEWTON:
            self.newton(function, function.newton[i.a])
        elif op in _ARRAY:
            self.vstep(i)
        else:
            # Loads of p, w, t, constants, comparisons and logic: no state in their partials.
            dots.pop(i.dst, None)

    def power(self, i: Instr, a: Mapping[int, int], b: Mapping[int, int]) -> None:
        out: dict[int, int] = {}
        if a:
            # b * a^(b-1) = b * (a^b / a) would divide by zero at a = 0; use pow directly.
            exponent = self.emit(Op.SUB, i.b, self.constant(1.0))
            factor = self.emit(Op.MUL, i.b, self.emit(Op.POW, i.a, exponent))
            out = self.scaled(factor, a)
        if b:
            logarithm = self.emit(Op.CALL1, i.a, 0, Builtin.LOG)
            out = self.combine(Op.ADD, out, self.scaled(self.emit(Op.MUL, i.dst, logarithm), b))
        self.set(i.dst, out)

    def select(self, i: Instr) -> None:
        then, otherwise = self.dots.get(i.b, {}), self.dots.get(i.c, {})
        out: dict[int, int] = {}
        for j in then.keys() | otherwise.keys():
            zero = self.constant(0.0)
            out[j] = self.emit(Op.SELECT, i.a, then.get(j, zero), otherwise.get(j, zero))
        self.set(i.dst, out)

    def call1(self, i: Instr, a: Mapping[int, int]) -> None:  # noqa: C901, PLR0912 - one rule per builtin
        if not a:
            self.set(i.dst, {})
            return
        x, y, f = i.a, i.dst, Builtin(i.c)
        one = self.constant(1.0)
        if f in (Builtin.CEIL, Builtin.FLOOR, Builtin.SIGN, Builtin.TRUNC):
            self.set(i.dst, {})
            return
        if f is Builtin.ABS:
            factor = self.emit(Op.CALL1, x, 0, Builtin.SIGN)
        elif f is Builtin.EXP:
            factor = y
        elif f is Builtin.LOG:
            factor = self.emit(Op.DIV, one, x)
        elif f is Builtin.LOG10:
            factor = self.emit(Op.DIV, one, self.emit(Op.MUL, x, self.constant(_LN10)))
        elif f is Builtin.SIN:
            factor = self.emit(Op.CALL1, x, 0, Builtin.COS)
        elif f is Builtin.COS:
            factor = self.emit(Op.NEG, self.emit(Op.CALL1, x, 0, Builtin.SIN))
        elif f is Builtin.TAN:
            factor = self.emit(Op.ADD, one, self.emit(Op.MUL, y, y))
        elif f is Builtin.SINH:
            factor = self.emit(Op.CALL1, x, 0, Builtin.COSH)
        elif f is Builtin.COSH:
            factor = self.emit(Op.CALL1, x, 0, Builtin.SINH)
        elif f is Builtin.TANH:
            factor = self.emit(Op.SUB, one, self.emit(Op.MUL, y, y))
        elif f is Builtin.SQRT:
            factor = self.emit(Op.DIV, self.constant(0.5), y)
        elif f is Builtin.ATAN:
            factor = self.emit(Op.DIV, one, self.emit(Op.ADD, one, self.emit(Op.MUL, x, x)))
        else:  # ASIN, ACOS
            root = self.emit(
                Op.CALL1, self.emit(Op.SUB, one, self.emit(Op.MUL, x, x)), 0, Builtin.SQRT
            )
            factor = self.emit(Op.DIV, one, root)
            if f is Builtin.ACOS:
                factor = self.emit(Op.NEG, factor)
        self.set(i.dst, self.scaled(factor, a))

    def call2(self, i: Instr, a: Mapping[int, int], b: Mapping[int, int]) -> None:
        f = Builtin(i.c)
        if f in (Builtin.MAX, Builtin.MIN):
            pick = self.emit(Op.GT if f is Builtin.MAX else Op.LT, i.a, i.b)
            zero = self.constant(0.0)
            out = {
                j: self.emit(Op.SELECT, pick, a.get(j, zero), b.get(j, zero))
                for j in a.keys() | b.keys()
            }
        elif f is Builtin.ATAN2:
            # d atan2(a, b) = (b da - a db) / (a^2 + b^2)
            square = self.emit(Op.ADD, self.emit(Op.MUL, i.a, i.a), self.emit(Op.MUL, i.b, i.b))
            inner = self.combine(Op.SUB, self.scaled(i.b, a), self.scaled(i.a, b))
            out = {j: self.emit(Op.DIV, d, square) for j, d in inner.items()}
        else:  # MOD, REM: a - q b with q piecewise constant
            rounding = Builtin.FLOOR if f is Builtin.MOD else Builtin.TRUNC
            quotient = self.emit(Op.CALL1, self.emit(Op.DIV, i.a, i.b), 0, rounding)
            out = self.combine(Op.SUB, a, self.scaled(quotient, b))
        self.set(i.dst, out)

    def set(self, register: int, tangent: dict[int, int]) -> None:
        if tangent:
            self.dots[register] = tangent
        else:
            self.dots.pop(register, None)

    def solve(self, n: int, matrix: int, right: int, step: int) -> None:
        """``A dx = db - dA x``, one solve per seed that reaches the system."""
        dots = self.dots
        seeds: set[int] = set()
        for register in range(matrix, matrix + n * n):
            seeds |= dots.get(register, {}).keys()
        for register in range(right, right + n):
            seeds |= dots.get(register, {}).keys()
        for k in range(n):
            dots.pop(step + k, None)
        for j in sorted(seeds):
            column = self.lowering.register(n)
            for row in range(n):
                total = dots.get(right + row, {}).get(j, self.constant(0.0))
                for k in range(n):
                    entry = dots.get(matrix + row * n + k, {}).get(j)
                    if entry is not None:
                        total = self.emit(Op.SUB, total, self.emit(Op.MUL, entry, step + k))
                self.code.append(Instr(Op.MOVE, column + row, total))
            partial = self.lowering.register(n)
            self.code.append(Instr(Op.SOLVE, partial, matrix, column, n))
            for k in range(n):
                dots.setdefault(step + k, {})[j] = partial + k

    def newton(self, function: Function, block: NewtonBlock) -> None:
        for unknown in block.unknowns:
            self.dots.pop(unknown, None)
        self.run(function, block.body)
        n = len(block.unknowns)
        seeds: set[int] = set()
        for register in range(block.right, block.right + n):
            seeds |= self.dots.get(register, {}).keys()
        for j in sorted(seeds):
            column = self.lowering.register(n)
            zero = self.constant(0.0)
            for row in range(n):
                source = self.dots.get(block.right + row, {}).get(j, zero)
                self.code.append(Instr(Op.MOVE, column + row, source))
            partial = self.lowering.register(n)
            self.code.append(Instr(Op.SOLVE, partial, block.matrix, column, n))
            for k, unknown in enumerate(block.unknowns):
                self.dots.setdefault(unknown, {})[j] = partial + k


def _bandwidth(sparsity: Sequence[Sequence[int]], position: Sequence[int]) -> int:
    """The larger of the lower and upper bandwidth with state ``k`` at row ``position[k]``."""
    return max(
        (abs(position[i] - position[j]) for i, row in enumerate(sparsity) for j in row),
        default=0,
    )


def _neighbours(sparsity: Sequence[Sequence[int]]) -> list[set[int]]:
    """Each state's neighbours in the Jacobian made symmetric, the diagonal left out."""
    neighbours: list[set[int]] = [set() for _ in sparsity]
    for i, row in enumerate(sparsity):
        for j in row:
            if i != j:
                neighbours[i].add(j)
                neighbours[j].add(i)
    return neighbours


def bandwidth_order(sparsity: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """A state order that makes the Jacobian banded: reverse Cuthill-McKee, or ``()``.

    Breadth-first from a state with the fewest neighbours, neighbours visited by ascending
    degree, the whole order reversed; every connected part in turn. Returns ``()`` when the
    natural order is as narrow, so a model that is banded already is factored as before.
    """
    n = len(sparsity)
    neighbours = _neighbours(sparsity)
    order: list[int] = []
    seen = [False] * n
    for root in sorted(range(n), key=lambda k: len(neighbours[k])):
        if seen[root]:
            continue
        seen[root] = True
        queue = [root]
        head = 0
        while head < len(queue):
            k = queue[head]
            head += 1
            for j in sorted(neighbours[k], key=lambda m: (len(neighbours[m]), m)):
                if not seen[j]:
                    seen[j] = True
                    queue.append(j)
        order.extend(queue)
    order.reverse()
    position = [0] * n
    for row, k in enumerate(order):
        position[k] = row
    if _bandwidth(sparsity, position) >= _bandwidth(sparsity, range(n)):
        return ()
    return tuple(order)


def lower(system: System, *, hoist: bool = True, arrays: bool = True) -> Program:
    """Write ``system``'s kernel functions as tapes.

    ``hoist=False`` leaves out `_hoist` and ``arrays=False`` writes every `ArrayBlock` as
    scalar blocks, so a test can check each changes no result.

    Raises `EmitError` for what the tape cannot express yet, with the emitter's message where
    the emitter would refuse too.
    """
    lowering = _Lowering(Emitter(system))
    modes = system.modes
    transitions = [lowering.transition(k, mode) for k, mode in enumerate(modes)]
    rhs = tuple(lowering.rhs(mode, arrays=arrays) for mode in modes)
    program = Program(
        registers=0,
        constants=(),
        rhs=rhs,
        jacobian=tuple(_jacobian(lowering, mode, f) for mode, f in zip(modes, rhs, strict=True)),
        observe=tuple(lowering.observe(mode, arrays=arrays) for mode in modes),
        indicators=tuple(lowering.indicators(k, mode) for k, mode in enumerate(modes)),
        transition_prefix=tuple(prefix for prefix, _ in transitions),
        transition=tuple(branches for _, branches in transitions),
        switching=lowering.emitter.switching,
        parameters=lowering.parameters(),
        initial=lowering.initial(),
        latch=lowering.latch(),
        layout=Layout(
            states=system.states,
            parameters=tuple(variable.name for variable in system.parameters),
            observed=lowering.emitter.observed,
            events=tuple(event.trigger for event in lowering.emitter.events()),
            workspace=len(lowering.emitter.observed)
            + len(lowering.emitter.watched)
            + (0 if lowering.emitter.switching is None else 1),
        ),
    )
    at = [
        -1 if event.trigger.at is None else event.trigger.at for event in lowering.emitter.events()
    ]
    program = replace(
        program,
        order=bandwidth_order(system.modes[0].sparsity),
        times=tuple(at) if any(slot >= 0 for slot in at) else (),
    )
    program = _fold(program, lowering.constants)
    pool = sorted(lowering.constants, key=lowering.constants.__getitem__)
    program = _preload(program, lowering.registers)
    if hoist:
        # Only what the integrator calls many times at one `t`: the others run once per event.
        base = lowering.registers
        program = replace(
            program,
            rhs=tuple(_hoist(f, base) for f in program.rhs),
            jacobian=tuple(_hoist(f, base) for f in program.jacobian),
            observe=tuple(_hoist(f, base) for f in program.observe),
            indicators=tuple(_hoist(f, base) for f in program.indicators),
        )
    program, base = _compact(program, lowering.registers)
    if arrays:
        program = replace(
            program,
            rhs=tuple(_fuse(f) for f in program.rhs),
            observe=tuple(_fuse(f) for f in program.observe),
        )
    return replace(program, registers=base + len(pool), constants=tuple(pool))


def _jacobian(lowering: _Lowering, mode: Mode, rhs: Function) -> Function:
    """``mode``'s Jacobian tape from ``rhs``, array opcodes and all.

    When an array opcode's partials cannot be runs (`_Scalar`), it is built from a scalar
    lowering of ``rhs`` instead.
    """
    try:
        return _Tangents(lowering, len(mode.states)).function(rhs)
    except _Scalar:
        return _Tangents(lowering, len(mode.states)).function(lowering.rhs(mode, arrays=False))


def _fuse(function: Function) -> Function:
    """``function`` with runs of adjacent ``LOAD_Y`` or ``STORE_OUT`` as one vector opcode.

    A run is instructions next to each other whose ``y`` or ``out`` slot and register both go up
    by one. It runs last, so no pass that reads register sets sees the vector forms. Newton
    bodies are left as they are. Only ``rhs`` and ``observe`` are fused.
    """
    code: list[Instr] = []
    for i in function.code:
        last = code[-1] if code else None
        if last is not None and i.op is Op.LOAD_Y and last.op in (Op.LOAD_Y, Op.VLOAD_Y):
            n = 1 if last.op is Op.LOAD_Y else last.c
            if i.a == last.a + n and i.dst == last.dst + n:
                code[-1] = Instr(Op.VLOAD_Y, last.dst, last.a, 0, n + 1)
                continue
        if last is not None and i.op is Op.STORE_OUT and last.op in (Op.STORE_OUT, Op.VSTORE_OUT):
            n = 1 if last.op is Op.STORE_OUT else last.c
            if i.a == last.a + n and i.b == last.b + n:
                code[-1] = Instr(Op.VSTORE_OUT, 0, last.a, last.b, n + 1)
                continue
        code.append(i)
    return replace(function, code=tuple(code))


# --------------------------------------------------------------------------------------
# Constant folding
# --------------------------------------------------------------------------------------

#: Instructions that write ``dst`` from their register operands and nothing else.
_PURE: Final = frozenset(
    {
        Op.MOVE, Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.POW,
        Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NE, Op.AND, Op.OR,
        Op.NEG, Op.NOT, Op.CALL1, Op.CALL2, Op.SELECT,
    }
)  # fmt: skip
#: Instructions that write ``dst`` and can go when nothing reads it.
_REMOVABLE: Final = _PURE | {Op.CONST, Op.TIME, Op.LOAD_Y, Op.LOAD_P, Op.LOAD_W}


def _operands(i: Instr) -> tuple[int, ...]:
    """The registers a pure instruction reads."""
    if i.op in (Op.MOVE, Op.NEG, Op.NOT, Op.CALL1):
        return (i.a,)
    if i.op is Op.SELECT:
        return (i.a, i.b, i.c)
    return (i.a, i.b)


def _apply(i: Instr, values: Mapping[int, float]) -> float:
    """What the pure instruction ``i`` computes from ``values``, as `evaluate` computes it."""
    with np.errstate(all="ignore"):
        if i.op is Op.MOVE:
            return values[i.a]
        if i.op is Op.SELECT:
            return values[i.b] if values[i.a] != 0.0 else values[i.c]
        if i.op is Op.CALL1:
            return float(_UNARY[i.c](np.float64(values[i.a])))
        if i.op is Op.CALL2:
            return float(_BINARY[i.c](np.float64(values[i.a]), np.float64(values[i.b])))
        b = values[i.b] if i.op not in (Op.NEG, Op.NOT) else 0.0
        return float(_arith(i.op, np.float64(values[i.a]), np.float64(b)))


#: The binary operators an `ArrayBlock` writes elementwise.
_VECTOR_OPS: Final = {"+": Op.VADD, "-": Op.VSUB, "*": Op.VMUL, "/": Op.VDIV}


def _reals(node: ir.Expr) -> ir.Expr:
    """``node`` with each integer literal outside a subscript written as a real.

    Causalization leaves the scalar equations that way; the register holds the same float.
    """
    match node:
        case ir.IntLiteral(value=value):
            return ir.RealLiteral(float(value))
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, _reals(lhs), _reals(rhs))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, _reals(operand))
        case ir.Call(name=name, args=args, named=named):
            return ir.Call(name, tuple(_reals(a) for a in args), named)
        case ir.IfExpr(condition=c, then=t, otherwise=o):
            return ir.IfExpr(_reals(c), _reals(t), _reals(o))
        case _:
            return node


def _reads_array(node: ir.Expr) -> bool:
    """Whether ``node`` holds a subscript, so its value differs from cell to cell."""
    match node:
        case ir.Subscript():
            return True
        case ir.Binary(lhs=lhs, rhs=rhs):
            return _reads_array(lhs) or _reads_array(rhs)
        case ir.Unary(operand=operand):
            return _reads_array(operand)
        case ir.Call(args=args):
            return any(_reads_array(a) for a in args)
        case ir.IfExpr(condition=c, then=t, otherwise=o):
            return _reads_array(c) or _reads_array(t) or _reads_array(o)
        case _:
            return False


def _vectorizable(node: ir.Expr, indices: frozenset[str]) -> bool:
    """Whether `_Lowering.array` can write ``node`` with array opcodes.

    A part with no subscript is computed once, so it may hold anything but a loop index.
    """
    node = _unwrap(node)
    if not _reads_array(node):
        return not (set(names(node)) & indices)
    match node:
        case ir.Subscript():
            return True
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return op in _VECTOR_OPS and _vectorizable(lhs, indices) and _vectorizable(rhs, indices)
        case ir.Unary(op="-", operand=operand):
            return _vectorizable(operand, indices)
        case ir.Call(name=name, args=(argument,)):
            return name in _UNARY_CALLS and _vectorizable(argument, indices)
        case _:
            return False


def _fold_code(  # noqa: C901, PLR0912 - one case per kind of instruction
    code: Sequence[Instr],
    known: dict[int, float],
    pool: dict[float, int],
    blocks: Sequence[NewtonBlock],
    bodies: list[tuple[Instr, ...]],
) -> tuple[Instr, ...]:
    """``code`` with every instruction whose operands are all constants made a constant.

    ``known`` maps a register to its constant value and grows as the code runs. A value that
    is NaN or -0.0 stays computed: the pool is keyed by value, and those two do not round-trip
    through a dict key.
    """
    values = {index: value for value, index in pool.items()}
    out: list[Instr] = []
    for i in code:
        if i.op is Op.CONST:
            known[i.dst] = values[i.a]
        elif i.op is Op.SELECT and i.a in known:
            chosen = i.b if known[i.a] != 0.0 else i.c
            i = Instr(Op.MOVE, i.dst, chosen)  # noqa: PLW2901 - the instruction it becomes
            if chosen in known:
                known[i.dst] = known[chosen]
            else:
                known.pop(i.dst, None)
        elif i.op in _PURE and all(r in known for r in _operands(i)):
            value = _apply(i, known)
            if value == value and not (value == 0.0 and np.signbit(value)):  # noqa: PLR0124 - NaN test
                if value not in pool:
                    pool[value] = len(pool)
                    values[pool[value]] = value
                i = Instr(Op.CONST, i.dst, pool[value])  # noqa: PLW2901
                known[i.dst] = value
            else:
                known.pop(i.dst, None)
        elif i.op is Op.SOLVE:
            for k in range(i.c):
                known.pop(i.dst + k, None)
        elif i.op is Op.NEWTON:
            block = blocks[i.a]
            n = len(block.unknowns)
            inner = dict(known)
            for register in (*block.unknowns, *range(block.step, block.step + n)):
                inner.pop(register, None)
                known.pop(register, None)
            bodies[i.a] = _fold_code(block.body, inner, pool, blocks, bodies)
        elif i.op in _REMOVABLE:
            known.pop(i.dst, None)
        out.append(i)
    return tuple(out)


def _fold_function(function: Function, pool: dict[float, int]) -> Function:
    bodies = [block.body for block in function.newton]
    code = _fold_code(function.code, {}, pool, function.newton, bodies)
    newton = tuple(
        replace(block, body=body) for block, body in zip(function.newton, bodies, strict=True)
    )
    return Function(code, newton, function.tables)


def _array_reads(i: Instr, tables: Sequence[int]) -> list[int]:
    """The registers an array opcode reads."""
    op, n = i.op, (i.b if i.op is Op.VCALL1 else i.c)
    if op is Op.VGATHER:
        return list(tables[i.a : i.a + n])
    if op is Op.VSCATTER:
        return list(range(i.b, i.b + n))
    if op is Op.VBROADCAST:
        return [i.a]
    if op in (Op.VNEG, Op.VCALL1):
        return list(range(i.a, i.a + n))
    return [*range(i.a, i.a + n), *range(i.b, i.b + n)]


def _array_runs(i: Instr) -> tuple[int, ...]:
    """The first registers of the runs an array opcode reads or writes."""
    if i.op is Op.VGATHER:
        return (i.dst,)
    if i.op is Op.VSCATTER:
        return (i.b,)
    if i.op is Op.VBROADCAST:
        return (i.dst,)
    if i.op in (Op.VNEG, Op.VCALL1):
        return (i.dst, i.a)
    return (i.dst, i.a, i.b)


def _array_writes(i: Instr, tables: Sequence[int]) -> list[int]:
    """The registers an array opcode writes."""
    if i.op is Op.VSCATTER:
        return list(tables[i.a : i.a + i.c])
    return list(range(i.dst, i.dst + (i.b if i.op is Op.VCALL1 else i.c)))


#: The array opcodes, and the scalar opcode each elementwise one applies.
_VECTOR_TO_SCALAR: Final = {Op.VADD: Op.ADD, Op.VSUB: Op.SUB, Op.VMUL: Op.MUL, Op.VDIV: Op.DIV}
_SCALAR_OF: Final = _VECTOR_TO_SCALAR
_ARRAY: Final = frozenset(
    {Op.VGATHER, Op.VSCATTER, Op.VBROADCAST, Op.VNEG, Op.VCALL1, *_VECTOR_TO_SCALAR}
)


def _reads(i: Instr, function: Function) -> list[int]:
    """The registers ``i`` reads, ranges included."""
    if i.op in _ARRAY:
        return _array_reads(i, function.tables)
    if i.op in _PURE:
        return list(_operands(i))
    if i.op in (Op.STORE_OUT, Op.STORE_W):
        return [i.b]
    if i.op is Op.SOLVE:
        return [*range(i.a, i.a + i.c * i.c), *range(i.b, i.b + i.c)]
    if i.op is Op.NEWTON:
        block = function.newton[i.a]
        n = len(block.unknowns)
        return [
            *block.unknowns,
            *range(block.matrix, block.matrix + n * n),
            *range(block.right, block.right + n),
        ]
    return []


def _prune(functions: Sequence[Function]) -> list[Function]:  # noqa: C901 - count, drop, rebuild
    """``functions`` without the instructions whose result nothing reads, transitively.

    One pass: count the reads of every register, drop a removable instruction whose register
    has none, and take its own reads off the count.
    """
    places: list[tuple[Sequence[Instr], Function]] = []
    for function in functions:
        places.append((function.code, function))
        places.extend((block.body, function) for block in function.newton)
    reads: dict[int, int] = {}
    writers: dict[int, list[tuple[int, int]]] = {}
    for where, (code, function) in enumerate(places):
        for position, i in enumerate(code):
            for r in _reads(i, function):
                reads[r] = reads.get(r, 0) + 1
            if i.op in _REMOVABLE:
                writers.setdefault(i.dst, []).append((where, position))
    dead: set[tuple[int, int]] = set()
    queue = [spot for r, spots in writers.items() if not reads.get(r) for spot in spots]
    while queue:
        spot = queue.pop()
        if spot in dead:
            continue
        dead.add(spot)
        code, function = places[spot[0]]
        for r in _reads(code[spot[1]], function):
            reads[r] -= 1
            if not reads[r]:
                queue.extend(writers.get(r, ()))
    kept = [
        tuple(i for position, i in enumerate(code) if (where, position) not in dead)
        for where, (code, _) in enumerate(places)
    ]
    out: list[Function] = []
    cursor = 0
    for function in functions:
        code = kept[cursor]
        bodies = kept[cursor + 1 : cursor + 1 + len(function.newton)]
        cursor += 1 + len(function.newton)
        newton = tuple(
            replace(block, body=body) for block, body in zip(function.newton, bodies, strict=True)
        )
        out.append(Function(code, newton, function.tables))
    return out


def _rewrite(i: Instr, alias: Mapping[int, int]) -> Instr:
    """``i`` reading ``alias[r]`` wherever it read a register ``r`` in ``alias``."""
    if i.op in (Op.MOVE, Op.NEG, Op.NOT, Op.CALL1):
        return Instr(i.op, i.dst, alias.get(i.a, i.a), i.b, i.c)
    if i.op is Op.SELECT:
        return Instr(i.op, i.dst, *(alias.get(r, r) for r in (i.a, i.b, i.c)))
    if i.op in _PURE:
        return Instr(i.op, i.dst, alias.get(i.a, i.a), alias.get(i.b, i.b), i.c)
    if i.op in (Op.STORE_OUT, Op.STORE_W):
        return Instr(i.op, i.dst, i.a, alias.get(i.b, i.b), i.c)
    return i


def _preload(program: Program, base: int) -> Program:  # noqa: C901 - one walk per kind of code
    """Constant ``k`` lives in register ``base + k``, set once when the program loads.

    Every ``CONST`` whose register is read only as a single operand goes, and its readers read
    the preloaded register. A ``CONST`` into a solve's matrix or right-hand side stays: those are
    read as a contiguous range.
    """
    functions = [
        *program.rhs, *program.jacobian, *program.observe, *program.indicators,
        *program.transition_prefix, *(f for branches in program.transition for f in branches),
        program.parameters, program.initial, program.latch,
    ]  # fmt: skip
    ranged: set[int] = set()
    for function in functions:
        for i in function.code:
            if i.op is Op.SOLVE:
                ranged.update(range(i.a, i.a + i.c * i.c))
                ranged.update(range(i.b, i.b + i.c))
            elif i.op in _ARRAY:
                # An array opcode reads its operands as runs or through a table, never through
                # a rewritten scalar operand, so a constant it reads keeps its register.
                ranged.update(_array_reads(i, function.tables))
        for block in function.newton:
            n = len(block.unknowns)
            ranged.update(range(block.matrix, block.matrix + n * n))
            ranged.update(range(block.right, block.right + n))
    alias: dict[int, int] = {}
    for function in functions:
        for code in (function.code, *(block.body for block in function.newton)):
            for i in code:
                if i.op is Op.CONST and i.dst not in ranged:
                    alias[i.dst] = base + i.a

    def move(function: Function) -> Function:
        def code(instructions: Sequence[Instr]) -> tuple[Instr, ...]:
            return tuple(
                _rewrite(i, alias)
                for i in instructions
                if not (i.op is Op.CONST and i.dst in alias)
            )

        newton = tuple(replace(block, body=code(block.body)) for block in function.newton)
        return Function(code(function.code), newton, function.tables)

    return replace(
        program,
        rhs=tuple(move(f) for f in program.rhs),
        jacobian=tuple(move(f) for f in program.jacobian),
        observe=tuple(move(f) for f in program.observe),
        indicators=tuple(move(f) for f in program.indicators),
        transition_prefix=tuple(move(f) for f in program.transition_prefix),
        transition=tuple(tuple(move(f) for f in branches) for branches in program.transition),
        parameters=move(program.parameters),
        initial=move(program.initial),
        latch=move(program.latch),
    )


def _writes(i: Instr, function: Function) -> list[int]:
    """The registers ``i`` writes, ranges included; a ``NEWTON``'s body counts as its own."""
    if i.op in _ARRAY:
        return _array_writes(i, function.tables)
    if i.op in _REMOVABLE:
        return [i.dst]
    if i.op is Op.SOLVE:
        return list(range(i.dst, i.dst + i.c))
    if i.op is Op.NEWTON:
        block = function.newton[i.a]
        inner = [r for j in block.body for r in _writes(j, function)]
        return [*block.unknowns, *range(block.step, block.step + len(block.unknowns)), *inner]
    return []


#: Fewer time-only instructions than this and a function is left as it is: the skip would save
#: less than the `HOIST` dispatch costs.
_HOIST_AT_LEAST: Final = 4


def _hoist(function: Function, base: int) -> Function:
    """``function`` with its time-only instructions first, then `Op.HOIST`, then the rest.

    An instruction moves when it is ``TIME``, ``CONST``, or pure with every operand a preloaded
    constant (a register at ``base`` or above) or the result of one that moved; when nothing
    else in the function writes its register; and when nothing reads that register before it.
    Then the order of what moved, and of what stayed, is unchanged, and so is every result. The
    engine skips the part before ``HOIST`` when ``t`` is the one of the function's last call.
    """
    code = function.code
    if not any(i.op is Op.TIME for i in code):
        return function
    written: dict[int, int] = {}
    for i in code:
        for r in _writes(i, function):
            written[r] = written.get(r, 0) + 1
    first_read: dict[int, int] = {}
    for position, i in enumerate(code):
        reads = _reads(i, function)
        if i.op is Op.NEWTON:
            reads += [r for j in function.newton[i.a].body for r in _reads(j, function)]
        for r in reads:
            first_read.setdefault(r, position)
    moved: set[int] = set()
    before: list[Instr] = []
    after: list[Instr] = []
    for position, i in enumerate(code):
        timely = i.op in (Op.TIME, Op.CONST) or (
            i.op in _PURE and all(r >= base or r in moved for r in _operands(i))
        )
        if timely and written[i.dst] == 1 and first_read.get(i.dst, position + 1) > position:
            moved.add(i.dst)
            before.append(i)
        else:
            after.append(i)
    if sum(i.op in _PURE for i in before) < _HOIST_AT_LEAST:
        return function
    return Function((*before, Instr(Op.HOIST, 0), *after), function.newton, function.tables)


#: The fields that name a register, by opcode; a range's first register too (`register_fields`).
_FIELDS: Final = {
    **dict.fromkeys((Op.CONST, Op.TIME, Op.LOAD_Y, Op.LOAD_P, Op.LOAD_W), ("dst",)),
    **dict.fromkeys((Op.STORE_OUT, Op.STORE_W), ("b",)),
    **dict.fromkeys(_PURE, ("dst", "a", "b")),
    **dict.fromkeys((Op.MOVE, Op.NEG, Op.NOT, Op.CALL1), ("dst", "a")),
    Op.SELECT: ("dst", "a", "b", "c"),
    Op.SOLVE: ("dst", "a", "b"),
    Op.VGATHER: ("dst",),
    Op.VSCATTER: ("b",),
    Op.VBROADCAST: ("dst", "a"),
    Op.VADD: ("dst", "a", "b"),
    Op.VSUB: ("dst", "a", "b"),
    Op.VMUL: ("dst", "a", "b"),
    Op.VDIV: ("dst", "a", "b"),
    Op.VNEG: ("dst", "a"),
    Op.VCALL1: ("dst", "a"),
}


def register_fields(i: Instr) -> tuple[str, ...]:
    """Which of ``dst``, ``a``, ``b``, ``c`` name a register in ``i``; a range's first one too."""
    return _FIELDS.get(i.op, ())


def _compact(program: Program, base: int) -> tuple[Program, int]:  # noqa: C901, PLR0912 - one walk
    """``program`` with its registers numbered in the order the functions first use them.

    `rhs` comes first, then the other functions the integrator calls, so what one `rhs` call
    touches sits together in memory instead of spread over every function's registers. A range
    a solve or a Newton block reads as one (and any range overlapping it) moves as one, so it
    stays contiguous. The constants stay last. Nothing is reused: every register keeps its own
    number, so no result changes. Returns the program and its first constant register.
    """
    functions = [
        *program.rhs, *program.jacobian, *program.indicators, *program.observe,
        *program.transition_prefix, *(f for branches in program.transition for f in branches),
        program.parameters, program.initial, program.latch,
    ]  # fmt: skip
    spans: list[tuple[int, int]] = []
    for function in functions:
        for code in (function.code, *(block.body for block in function.newton)):
            for i in code:
                if i.op is Op.SOLVE:
                    spans += [(i.a, i.c * i.c), (i.b, i.c), (i.dst, i.c)]
                elif i.op in _ARRAY:
                    n = i.b if i.op is Op.VCALL1 else i.c
                    spans += [(start, n) for start in _array_runs(i)]
        for block in function.newton:
            n = len(block.unknowns)
            spans += [(block.matrix, n * n), (block.right, n), (block.step, n)]
    merged: list[list[int]] = []
    for start, size in sorted((s, n) for s, n in spans if n > 0):
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], start + size)
        else:
            merged.append([start, start + size])
    group = {r: (first, last) for first, last in merged for r in range(first, last)}
    number: dict[int, int] = {}

    def touch(r: int) -> None:
        if r >= base or r in number:
            return
        if r in group:
            first, last = group[r]
            for k in range(first, last):
                number.setdefault(k, len(number))
        else:
            number[r] = len(number)

    for function in functions:
        for code in (function.code, *(block.body for block in function.newton)):
            for i in code:
                for name in register_fields(i):
                    touch(getattr(i, name))
                if i.op in (Op.VGATHER, Op.VSCATTER):
                    for r in function.tables[i.a : i.a + i.c]:
                        touch(r)
        for block in function.newton:
            for r in (*block.unknowns, block.matrix, block.right, block.step):
                touch(r)
        for r in function.tables:
            touch(r)
    size = len(number)

    def at(r: int) -> int:
        return size + (r - base) if r >= base else number[r]

    def instr(i: Instr) -> Instr:
        fields = _FIELDS.get(i.op)
        if not fields:
            return i
        return Instr(
            i.op,
            at(i.dst) if "dst" in fields else i.dst,
            at(i.a) if "a" in fields else i.a,
            at(i.b) if "b" in fields else i.b,
            at(i.c) if "c" in fields else i.c,
        )

    def move(function: Function) -> Function:
        newton = tuple(
            replace(
                block,
                unknowns=tuple(at(r) for r in block.unknowns),
                body=tuple(instr(i) for i in block.body),
                matrix=at(block.matrix),
                right=at(block.right),
                step=at(block.step),
            )
            for block in function.newton
        )
        return Function(
            tuple(instr(i) for i in function.code),
            newton,
            tuple(at(r) for r in function.tables),
        )

    compacted = replace(
        program,
        rhs=tuple(move(f) for f in program.rhs),
        jacobian=tuple(move(f) for f in program.jacobian),
        observe=tuple(move(f) for f in program.observe),
        indicators=tuple(move(f) for f in program.indicators),
        transition_prefix=tuple(move(f) for f in program.transition_prefix),
        transition=tuple(tuple(move(f) for f in branches) for branches in program.transition),
        parameters=move(program.parameters),
        initial=move(program.initial),
        latch=move(program.latch),
    )
    return compacted, size


def _unnegate(i: Instr, negated: Mapping[int, int]) -> Instr:
    """``i`` with an operand ``-s`` of ``ADD`` or the subtrahend ``-s`` of ``SUB`` read as ``s``."""
    if i.op is Op.ADD and i.b in negated:
        return Instr(Op.SUB, i.dst, i.a, negated[i.b])
    if i.op is Op.ADD and i.a in negated:
        return Instr(Op.SUB, i.dst, i.b, negated[i.a])
    if i.op is Op.SUB and i.b in negated:
        return Instr(Op.ADD, i.dst, i.a, negated[i.b])
    return i


def _peephole(function: Function) -> Function:
    """``function`` with copies read through and negations folded into ``ADD`` and ``SUB``.

    After ``MOVE d, s`` later instructions read ``s`` for ``d``; after ``NEG d, s``,
    ``x + d`` and ``d + x`` become ``x - s`` and ``x - d`` becomes ``x + s``. Each is exact in
    IEEE arithmetic. Only when ``d`` is written once and ``s`` is not written after the copy, so
    both hold one value for the rest of the call. `_prune` then drops the ``MOVE`` and ``NEG``
    nothing reads. A function with Newton blocks is left as it is: its bodies run many times.
    """
    if function.newton:
        return function
    written: dict[int, list[int]] = {}
    for position, i in enumerate(function.code):
        for r in _writes(i, function):
            written.setdefault(r, []).append(position)

    def settled(dst: int, source: int, position: int) -> bool:
        writes = written.get(source, [])
        return len(written[dst]) == 1 and len(writes) <= 1 and all(w < position for w in writes)

    alias: dict[int, int] = {}
    negated: dict[int, int] = {}
    code: list[Instr] = []
    for position, original in enumerate(function.code):
        i = _unnegate(_rewrite(original, alias), negated)
        if i.op is Op.MOVE and settled(i.dst, i.a, position):
            alias[i.dst] = i.a
        elif i.op is Op.NEG and settled(i.dst, i.a, position):
            negated[i.dst] = i.a
        code.append(i)
    return Function(tuple(code), function.newton, function.tables)


def _fold(program: Program, pool: dict[float, int]) -> Program:
    """Fold constant subexpressions, then drop instructions whose result nothing reads.

    Compiled Rust gets both from LLVM. On the tape they are dispatches: on vowel_i a third of
    ``rhs`` reads only constants. Registers stay where they are, so a `transition` branch still
    reads what its prefix wrote.
    """

    def every(fn: Callable[[Function], Function], p: Program) -> Program:
        return replace(
            p,
            rhs=tuple(fn(f) for f in p.rhs),
            jacobian=tuple(fn(f) for f in p.jacobian),
            observe=tuple(fn(f) for f in p.observe),
            indicators=tuple(fn(f) for f in p.indicators),
            transition_prefix=tuple(fn(f) for f in p.transition_prefix),
            transition=tuple(tuple(fn(f) for f in branches) for branches in p.transition),
            parameters=fn(p.parameters),
            initial=fn(p.initial),
            latch=fn(p.latch),
        )

    def functions(p: Program) -> list[Function]:
        return [
            *p.rhs, *p.jacobian, *p.observe, *p.indicators, *p.transition_prefix,
            *(f for branches in p.transition for f in branches),
            p.parameters, p.initial, p.latch,
        ]  # fmt: skip

    program = every(lambda f: _peephole(_fold_function(f, pool)), program)
    pruned = iter(_prune(functions(program)))
    return every(lambda _: next(pruned), program)


# --------------------------------------------------------------------------------------
# The reference evaluator
# --------------------------------------------------------------------------------------


def _sign(x: float) -> float:
    return float(int(x > 0) - int(x < 0))


def _mod(a: float, b: float) -> float:
    r = float(np.fmod(a, b))
    return r + b if r != 0.0 and (r < 0.0) != (b < 0.0) else r


def _max(a: float, b: float) -> float:
    return max(b, a)


def _min(a: float, b: float) -> float:
    return min(b, a)


_UNARY: Final[dict[int, Callable[[Any], Any]]] = {
    Builtin.ABS: np.abs,
    Builtin.ACOS: np.arccos,
    Builtin.ASIN: np.arcsin,
    Builtin.ATAN: np.arctan,
    Builtin.CEIL: np.ceil,
    Builtin.COS: np.cos,
    Builtin.COSH: np.cosh,
    Builtin.EXP: np.exp,
    Builtin.FLOOR: np.floor,
    Builtin.LOG: np.log,
    Builtin.LOG10: np.log10,
    Builtin.SIGN: _sign,
    Builtin.SIN: np.sin,
    Builtin.SINH: np.sinh,
    Builtin.SQRT: np.sqrt,
    Builtin.TAN: np.tan,
    Builtin.TANH: np.tanh,
    Builtin.TRUNC: np.trunc,
}
_BINARY: Final[dict[int, Callable[[Any, Any], Any]]] = {
    Builtin.ATAN2: np.arctan2,
    Builtin.MAX: _max,
    Builtin.MIN: _min,
    Builtin.MOD: _mod,
    Builtin.REM: np.fmod,
}


@dataclass
class _Machine:
    """The state one call of `evaluate` works on."""

    program: Program
    t: float
    y: Sequence[float]
    p: Sequence[float]
    w: list[float]
    out: list[float]
    r: list[float]

    def _array(self, i: Instr, tables: Sequence[int]) -> None:
        """One array opcode, element by element, as the engine runs it."""
        r, op = self.r, i.op
        n = i.b if op is Op.VCALL1 else i.c
        for k in range(n):
            if op is Op.VGATHER:
                r[i.dst + k] = r[tables[i.a + k]]
            elif op is Op.VSCATTER:
                r[tables[i.a + k]] = r[i.b + k]
            elif op is Op.VBROADCAST:
                r[i.dst + k] = r[i.a]
            elif op is Op.VNEG:
                r[i.dst + k] = -r[i.a + k]
            elif op is Op.VCALL1:
                r[i.dst + k] = float(_UNARY[i.c](np.float64(r[i.a + k])))
            else:
                scalar = _VECTOR_TO_SCALAR[op]
                r[i.dst + k] = float(_arith(scalar, np.float64(r[i.a + k]), np.float64(r[i.b + k])))

    def run(self, function: Function, code: Sequence[Instr]) -> None:
        # IEEE semantics, as the engine has them: a branch SELECT discards may compute
        # sqrt(-1) or 1/0, and that must give NaN or inf, not an exception.
        with np.errstate(all="ignore"):
            self._run(function, code)

    def _run(self, function: Function, code: Sequence[Instr]) -> None:  # noqa: C901, PLR0912 - one case per opcode
        r, w, out = self.r, self.w, self.out
        for i in code:
            op = i.op
            if op is Op.CONST:
                r[i.dst] = self.program.constants[i.a]
            elif op is Op.TIME:
                r[i.dst] = self.t
            elif op is Op.LOAD_Y:
                r[i.dst] = self.y[i.a]
            elif op is Op.LOAD_P:
                r[i.dst] = self.p[i.a]
            elif op is Op.LOAD_W:
                r[i.dst] = w[i.a]
            elif op is Op.STORE_OUT:
                out[i.a] = r[i.b]
            elif op is Op.STORE_W:
                w[i.a] = r[i.b]
            elif op is Op.VLOAD_Y:
                r[i.dst : i.dst + i.c] = self.y[i.a : i.a + i.c]
            elif op is Op.VSTORE_OUT:
                out[i.a : i.a + i.c] = r[i.b : i.b + i.c]
            elif op is Op.MOVE:
                r[i.dst] = r[i.a]
            elif op is Op.COPY_Y_OUT:
                out[: len(self.y)] = self.y
            elif op is Op.CLEAR_OUT:
                out[: i.a] = [0.0] * i.a
            elif op is Op.HOIST:
                pass
            elif op in _ARRAY:
                self._array(i, function.tables)
            elif op is Op.SELECT:
                r[i.dst] = r[i.b] if r[i.a] != 0.0 else r[i.c]
            elif op is Op.CALL1:
                r[i.dst] = float(_UNARY[i.c](np.float64(r[i.a])))
            elif op is Op.CALL2:
                r[i.dst] = float(_BINARY[i.c](np.float64(r[i.a]), np.float64(r[i.b])))
            elif op is Op.SOLVE:
                self.solve(i.c, i.a, i.b, i.dst)
            elif op is Op.NEWTON:
                self.newton(function, function.newton[i.a])
            else:
                r[i.dst] = float(_arith(op, np.float64(r[i.a]), np.float64(r[i.b])))

    def solve(self, n: int, matrix: int, right: int, step: int) -> None:
        """Dense LU with partial pivoting, the same steps as ``md_solve``."""
        r = self.r
        a = np.array(r[matrix : matrix + n * n], dtype=np.float64)
        b = np.array(r[right : right + n], dtype=np.float64)
        for k in range(n):
            pivot = max(range(k, n), key=lambda i: abs(a[i * n + k]))
            if pivot != k:
                for j in range(n):
                    a[k * n + j], a[pivot * n + j] = a[pivot * n + j], a[k * n + j]
                b[k], b[pivot] = b[pivot], b[k]
            for i in range(k + 1, n):
                factor = a[i * n + k] / a[k * n + k]
                a[i * n + k] = 0.0
                for j in range(k + 1, n):
                    a[i * n + j] -= factor * a[k * n + j]
                b[i] -= factor * b[k]
        for i in reversed(range(n)):
            total = b[i] - sum(a[i * n + j] * r[step + j] for j in range(i + 1, n))
            r[step + i] = float(total / a[i * n + i])

    def newton(self, function: Function, block: NewtonBlock) -> None:
        r = self.r
        n = len(block.unknowns)
        for _ in range(ITERATIONS):
            self.run(function, block.body)
            self.solve(n, block.matrix, block.right, block.step)
            for position, unknown in enumerate(block.unknowns):
                r[unknown] = float(np.float64(r[unknown]) + r[block.step + position])
            if max(abs(r[block.step + k]) for k in range(n)) < TOLERANCE:
                break


def _arith(op: Op, a: float, b: float) -> float:  # noqa: C901, PLR0911, PLR0912 - one case per opcode
    if op is Op.ADD:
        return a + b
    if op is Op.SUB:
        return a - b
    if op is Op.MUL:
        return a * b
    if op is Op.DIV:
        return a / b
    if op is Op.POW:
        return float(np.power(a, b))
    if op is Op.LT:
        return float(a < b)
    if op is Op.LE:
        return float(a <= b)
    if op is Op.GT:
        return float(a > b)
    if op is Op.GE:
        return float(a >= b)
    if op is Op.EQ:
        return float(a == b)
    if op is Op.NE:
        return float(a != b)
    if op is Op.AND:
        return float(a != 0.0 and b != 0.0)
    if op is Op.OR:
        return float(a != 0.0 or b != 0.0)
    if op is Op.NEG:
        return -a
    if op is Op.NOT:
        return float(a == 0.0)
    message = f"unknown opcode {op!r}"
    raise ValueError(message)


def evaluate(  # noqa: PLR0913, PLR0917 - the kernel signature
    program: Program,
    function: Function,
    t: float,
    y: Sequence[float],
    p: Sequence[float],
    w: list[float],
    out: list[float],
    *,
    then: Function | None = None,
) -> None:
    """Run ``function`` (and then ``then``, for a transition) in Python; writes ``out`` and ``w``."""
    registers = [0.0] * program.registers
    base = program.registers - len(program.constants)
    registers[base:] = program.constants
    machine = _Machine(program, t, y, p, w, out, registers)
    machine.run(function, function.code)
    if then is not None:
        machine.run(then, then.code)


def mode(program: Program, w: Sequence[float]) -> int:
    """Which mode ``w`` says is in use, as the emitted picker decides it."""
    if program.switching is None:
        return 0
    count = len(program.rhs)
    for k in range(count - 1):
        if w[program.switching] < k + 0.5:
            return k
    return count - 1


# --------------------------------------------------------------------------------------
# The flat form the engine reads
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoded:
    """A `Program` as the five flat arrays ``md_vm_new`` reads (``engine/src/lib.rs``).

    ``code`` holds every instruction as five ``int32`` (op, dst, a, b, c). ``functions`` holds
    four per function (first instruction, count, first Newton block, count), in the order
    `Encoded.index` gives. ``newton`` holds seven per block (first unknown, count, first body
    instruction, count, matrix, right, step), and ``unknowns`` the unknowns' registers.
    """

    registers: int
    constants: tuple[float, ...]
    code: tuple[int, ...]
    functions: tuple[int, ...]
    newton: tuple[int, ...]
    unknowns: tuple[int, ...]
    modes: int
    events: int

    def rhs(self, mode: int) -> int:
        """The function index of ``rhs`` in ``mode``."""
        return mode

    def observe(self, mode: int) -> int:
        """The function index of ``observe`` in ``mode``."""
        return self.modes + mode

    def indicators(self, mode: int) -> int:
        """The function index of ``indicators`` in ``mode``."""
        return 2 * self.modes + mode

    def transition_prefix(self, mode: int) -> int:
        """The function index of the transition prefix in ``mode``."""
        return 3 * self.modes + mode

    def jacobian(self, mode: int) -> int:
        """The function index of ``jacobian`` in ``mode``."""
        return 4 * self.modes + mode

    def transition(self, mode: int, event: int) -> int:
        """The function index of ``event``'s transition in ``mode``."""
        return 5 * self.modes + mode * self.events + event

    @property
    def parameters(self) -> int:
        """The function index of ``parameters``."""
        return 5 * self.modes + self.modes * self.events

    @property
    def initial(self) -> int:
        """The function index of ``initial``."""
        return self.parameters + 1

    @property
    def latch(self) -> int:
        """The function index of ``latch``."""
        return self.parameters + 2


def encode(program: Program) -> Encoded:
    """Pack ``program`` into flat integer and float arrays, in `Encoded`'s layout."""
    code: list[int] = []
    functions: list[int] = []
    newton: list[int] = []
    unknowns: list[int] = []

    def put(instrs: Sequence[Instr], tables: int = 0) -> tuple[int, int]:
        # A gather or scatter names its table by offset in the function's `tables`, which
        # sit in `unknowns` from ``tables`` on.
        start = len(code) // 5
        for i in instrs:
            a = i.a + tables if i.op in (Op.VGATHER, Op.VSCATTER) else i.a
            code.extend((int(i.op), i.dst, a, i.b, i.c))
        return start, len(instrs)

    def add(function: Function) -> None:
        tables = len(unknowns)
        unknowns.extend(function.tables)
        start, count = put(function.code, tables)
        first = len(newton) // 7
        for block in function.newton:
            body, length = put(block.body, tables)
            newton.extend(
                (
                    len(unknowns),
                    len(block.unknowns),
                    body,
                    length,
                    block.matrix,
                    block.right,
                    block.step,
                )
            )
            unknowns.extend(block.unknowns)
        functions.extend((start, count, first, len(function.newton)))

    groups = (
        program.rhs,
        program.observe,
        program.indicators,
        program.transition_prefix,
        program.jacobian,
    )
    for group in groups:
        for function in group:
            add(function)
    for branches in program.transition:
        for function in branches:
            add(function)
    for function in (program.parameters, program.initial, program.latch):
        add(function)
    return Encoded(
        registers=program.registers,
        constants=program.constants,
        code=tuple(code),
        functions=tuple(functions),
        newton=tuple(newton),
        unknowns=tuple(unknowns),
        modes=len(program.rhs),
        events=len(program.transition[0]) if program.transition else 0,
    )


_SYMBOLS: Final = {
    Op.ADD: "+", Op.SUB: "-", Op.MUL: "*", Op.DIV: "/", Op.POW: "^",
    Op.LT: "<", Op.LE: "<=", Op.GT: ">", Op.GE: ">=", Op.EQ: "==", Op.NE: "!=",
    Op.AND: "and", Op.OR: "or",
    Op.VADD: "+", Op.VSUB: "-", Op.VMUL: "*", Op.VDIV: "/",
}  # fmt: skip


def _line(i: Instr, program: Program, tables: Sequence[int]) -> str:  # noqa: C901, PLR0911, PLR0912 - one case per opcode
    """One instruction as text: registers ``r12``, a run ``r12[4]``, a constant its value."""
    op = i.op
    base = program.registers - len(program.constants)

    def reg(r: int) -> str:
        # A preloaded constant reads better as its value.
        return repr(program.constants[r - base]) if r >= base else f"r{r}"

    def run(start: int, n: int) -> str:
        return f"r{start}[{n}]"

    if op is Op.CONST:
        return f"r{i.dst} = {program.constants[i.a]!r}"
    if op is Op.TIME:
        return f"r{i.dst} = time"
    if op in (Op.LOAD_Y, Op.LOAD_P, Op.LOAD_W):
        return f"r{i.dst} = {op.name[-1].lower()}[{i.a}]"
    if op is Op.STORE_OUT:
        return f"out[{i.a}] = {reg(i.b)}"
    if op is Op.STORE_W:
        return f"w[{i.a}] = {reg(i.b)}"
    if op is Op.MOVE:
        return f"r{i.dst} = {reg(i.a)}"
    if op in (Op.NEG, Op.NOT):
        return f"r{i.dst} = {'-' if op is Op.NEG else 'not '}{reg(i.a)}"
    if op in _SYMBOLS and op not in _ARRAY:
        return f"r{i.dst} = {reg(i.a)} {_SYMBOLS[op]} {reg(i.b)}"
    if op is Op.CALL1:
        return f"r{i.dst} = {Builtin(i.c).name.lower()}({reg(i.a)})"
    if op is Op.CALL2:
        return f"r{i.dst} = {Builtin(i.c).name.lower()}({reg(i.a)}, {reg(i.b)})"
    if op is Op.SELECT:
        return f"r{i.dst} = {reg(i.b)} if {reg(i.a)} else {reg(i.c)}"
    if op is Op.SOLVE:
        return f"{run(i.dst, i.c)} = solve({run(i.a, i.c * i.c)}, {run(i.b, i.c)})"
    if op is Op.NEWTON:
        return f"newton block {i.a}"
    if op is Op.COPY_Y_OUT:
        return "out = y"
    if op is Op.CLEAR_OUT:
        return f"out[:{i.a}] = 0"
    if op is Op.HOIST:
        return "--- hoist: everything above reads only time and constants ---"
    if op is Op.VLOAD_Y:
        return f"{run(i.dst, i.c)} = y[{i.a}:{i.a + i.c}]"
    if op is Op.VSTORE_OUT:
        return f"out[{i.a}:{i.a + i.c}] = {run(i.b, i.c)}"
    if op is Op.VGATHER:
        return f"{run(i.dst, i.c)} = gather r{list(tables[i.a : i.a + i.c])}"
    if op is Op.VSCATTER:
        return f"scatter {run(i.b, i.c)} into r{list(tables[i.a : i.a + i.c])}"
    if op is Op.VBROADCAST:
        return f"{run(i.dst, i.c)} = {reg(i.a)}"
    if op is Op.VNEG:
        return f"{run(i.dst, i.c)} = -{run(i.a, i.c)}"
    if op is Op.VCALL1:
        return f"{run(i.dst, i.b)} = {Builtin(i.c).name.lower()}({run(i.a, i.b)})"
    return f"{run(i.dst, i.c)} = {run(i.a, i.c)} {_SYMBOLS[op]} {run(i.b, i.c)}"


def listing(program: Program, function: Function, *, width: int = 8) -> str:
    """``function`` as readable text, one instruction per line, numbered.

    A gather table longer than ``width`` entries is cut with ``...``. Meant for reading a tape,
    not for parsing it back.
    """
    lines: list[str] = []
    for number, i in enumerate(function.code):
        text = _line(i, program, function.tables)
        if "[" in text and text.count(",") > width:
            head, _, rest = text.partition("r[")
            entries = rest.split("]")[0].split(", ")
            text = f"{head}r[{', '.join(entries[:width])}, ...]" + rest.split("]", 1)[1]
        lines.append(f"{number:4d}  {text}")
    return "\n".join(lines)


def runtime() -> str:
    """The Radau runtime as ``engine/src/radau.rs`` holds it: `rust_radau`'s text, unchanged.

    The engine compiles the same integrator `rustsim` does. The file is generated so the
    engine crate builds without Python; ``tests/test_vm.py`` fails when it is out of date, and
    ``uv run python -c "from modelica.sim import vm; vm.write_runtime()"`` rewrites it.
    """
    from modelica.sim.rust_radau import (  # noqa: PLC0415 - only the generator needs the text
        CONSTANTS,
        FIRST_STEP,
        INTEGRATE,
        LINALG,
        NEWTON,
        STEP,
    )

    header = "// Generated from modelica.sim.rust_radau by vm.write_runtime(). Do not edit.\n"
    return header + "\n".join([*CONSTANTS, *LINALG, *STEP, *NEWTON, *FIRST_STEP, *INTEGRATE])


def write_runtime() -> None:
    """Rewrite ``engine/src/radau.rs`` from `runtime`."""
    from pathlib import Path  # noqa: PLC0415 - only the generator writes files

    target = Path(__file__).resolve().parents[3] / "engine" / "src" / "radau.rs"
    target.write_text(runtime(), encoding="utf-8")
