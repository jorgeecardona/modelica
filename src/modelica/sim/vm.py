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
from typing import TYPE_CHECKING, Final

from modelica.ir import nodes as ir
from modelica.ir.derive import simplify
from modelica.ir.rewrite import residual, substitute
from modelica.sim.emit import (
    ITERATIONS,
    TOLERANCE,
    EmitError,
    Emitter,
    _latched,  # pyright: ignore[reportPrivateUsage] - the same latching rule, not a copy
    _unwrap,  # pyright: ignore[reportPrivateUsage]
    block_jacobian,
    is_terminate,
    iterated,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from modelica.ir.system import Block, Mode, System

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
    observe: tuple[Function, ...]
    indicators: tuple[Function, ...]
    transition_prefix: tuple[Function, ...]
    transition: tuple[tuple[Function, ...], ...]
    switching: int | None
    """The workspace slot holding the mode, as `Code.switching`."""


@dataclass
class _Scope:
    """What one instruction stream is being written into, and what it already computed."""

    code: list[Instr]
    values: dict[ir.Expr, int]
    names: dict[str, int]
    newton: list[NewtonBlock]

    def child(self) -> _Scope:
        """A scope for a Newton body: it sees what is known, and leaks nothing back."""
        return _Scope([], dict(self.values), dict(self.names), self.newton)


@dataclass
class _Lowering:
    """Writes the tapes, reading every decision from an `Emitter` over the same system."""

    emitter: Emitter
    registers: int = 0
    constants: dict[float, int] = field(default_factory=dict[float, int])

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

    def blocks(self, scope: _Scope, mode: Mode) -> None:
        for block in mode.blocks:
            self.solve(scope, block)

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
        return Function(tuple(scope.code), tuple(scope.newton))

    def rhs(self, mode: Mode) -> Function:
        def body(scope: _Scope) -> None:
            self.unpack(scope, mode)
            self.blocks(scope, mode)
            for slot, state in enumerate(mode.states):
                source = self.ref(scope, mode.derivative[state])
                scope.code.append(Instr(Op.STORE_OUT, 0, slot, source))

        return self.function(body)

    def observe(self, mode: Mode) -> Function:
        emitter = self.emitter

        def body(scope: _Scope) -> None:
            self.unpack(scope, mode)
            self.blocks(scope, mode)
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


def lower(system: System) -> Program:
    """Write ``system``'s kernel functions as tapes.

    Raises `EmitError` for what the tape cannot express yet, with the emitter's message where
    the emitter would refuse too.
    """
    lowering = _Lowering(Emitter(system))
    modes = system.modes
    transitions = [lowering.transition(k, mode) for k, mode in enumerate(modes)]
    program = Program(
        registers=0,
        constants=(),
        rhs=tuple(lowering.rhs(mode) for mode in modes),
        observe=tuple(lowering.observe(mode) for mode in modes),
        indicators=tuple(lowering.indicators(k, mode) for k, mode in enumerate(modes)),
        transition_prefix=tuple(prefix for prefix, _ in transitions),
        transition=tuple(branches for _, branches in transitions),
        switching=lowering.emitter.switching,
    )
    pool = sorted(lowering.constants, key=lowering.constants.__getitem__)
    return replace(program, registers=lowering.registers, constants=tuple(pool))


# --------------------------------------------------------------------------------------
# The reference evaluator
# --------------------------------------------------------------------------------------


def _sign(x: float) -> float:
    return float((x > 0) - (x < 0))


def _mod(a: float, b: float) -> float:
    r = math.fmod(a, b)
    return r + b if r != 0.0 and (r < 0.0) != (b < 0.0) else r


def _max(a: float, b: float) -> float:
    return max(b, a)


def _min(a: float, b: float) -> float:
    return min(b, a)


_UNARY: Final[dict[int, Callable[[float], float]]] = {
    Builtin.ABS: abs,
    Builtin.ACOS: math.acos,
    Builtin.ASIN: math.asin,
    Builtin.ATAN: math.atan,
    Builtin.CEIL: math.ceil,
    Builtin.COS: math.cos,
    Builtin.COSH: math.cosh,
    Builtin.EXP: math.exp,
    Builtin.FLOOR: math.floor,
    Builtin.LOG: math.log,
    Builtin.LOG10: math.log10,
    Builtin.SIGN: _sign,
    Builtin.SIN: math.sin,
    Builtin.SINH: math.sinh,
    Builtin.SQRT: math.sqrt,
    Builtin.TAN: math.tan,
    Builtin.TANH: math.tanh,
    Builtin.TRUNC: math.trunc,
}
_BINARY: Final[dict[int, Callable[[float, float], float]]] = {
    Builtin.ATAN2: math.atan2,
    Builtin.MAX: _max,
    Builtin.MIN: _min,
    Builtin.MOD: _mod,
    Builtin.REM: math.fmod,
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

    def run(self, function: Function, code: Sequence[Instr]) -> None:  # noqa: C901, PLR0912 - one case per opcode
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
            elif op is Op.MOVE:
                r[i.dst] = r[i.a]
            elif op is Op.COPY_Y_OUT:
                out[: len(self.y)] = self.y
            elif op is Op.SELECT:
                r[i.dst] = r[i.b] if r[i.a] != 0.0 else r[i.c]
            elif op is Op.CALL1:
                r[i.dst] = float(_UNARY[i.c](r[i.a]))
            elif op is Op.CALL2:
                r[i.dst] = _BINARY[i.c](r[i.a], r[i.b])
            elif op is Op.SOLVE:
                self.solve(i.c, i.a, i.b, i.dst)
            elif op is Op.NEWTON:
                self.newton(function, function.newton[i.a])
            else:
                r[i.dst] = _arith(op, r[i.a], r[i.b])

    def solve(self, n: int, matrix: int, right: int, step: int) -> None:
        """Dense LU with partial pivoting, the same steps as ``md_solve``."""
        r = self.r
        a = r[matrix : matrix + n * n]
        b = r[right : right + n]
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
            r[step + i] = total / a[i * n + i]

    def newton(self, function: Function, block: NewtonBlock) -> None:
        r = self.r
        n = len(block.unknowns)
        for _ in range(ITERATIONS):
            self.run(function, block.body)
            self.solve(n, block.matrix, block.right, block.step)
            for position, unknown in enumerate(block.unknowns):
                r[unknown] += r[block.step + position]
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
        return float(a**b)
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
    machine = _Machine(program, t, y, p, w, out, [0.0] * program.registers)
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

    def transition(self, mode: int, event: int) -> int:
        """The function index of ``event``'s transition in ``mode``."""
        return 4 * self.modes + mode * self.events + event


def encode(program: Program) -> Encoded:
    """Pack ``program`` into flat integer and float arrays, in `Encoded`'s layout."""
    code: list[int] = []
    functions: list[int] = []
    newton: list[int] = []
    unknowns: list[int] = []

    def put(instrs: Sequence[Instr]) -> tuple[int, int]:
        start = len(code) // 5
        for i in instrs:
            code.extend((int(i.op), i.dst, i.a, i.b, i.c))
        return start, len(instrs)

    def add(function: Function) -> None:
        start, count = put(function.code)
        first = len(newton) // 7
        for block in function.newton:
            body, length = put(block.body)
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

    for group in (program.rhs, program.observe, program.indicators, program.transition_prefix):
        for function in group:
            add(function)
    for branches in program.transition:
        for function in branches:
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
