"""A C translation unit for a model, from the same emitter that writes the Python one.

What `docs/ideas/backends.md:42` asks for: "one translation unit -- FMU export, where a
shared library is the deliverable". This is the first half of it, and the half worth having
first, because it is the one that says whether the claim underneath the whole backends page
is true. That claim is that the emitters are *consequences* of the IR and adding one is
mechanical.

Measured against that claim, this module is 120 lines and it makes no decisions. It does not
choose what a block solves for, what order the blocks run in, which relations are latched, or
where a torn variable comes from -- `modelica.sim.emit` decides all of that once, and this
asks it for the same lines with :data:`modelica.sim.dialect.C` selected. What is left over
is what genuinely differs between the two languages:

* a statement ends in a semicolon;
* a local has to be declared, and its type named;
* the arrays arrive as pointers with a signature that says so.

The subset is stated rather than discovered. A model whose blocks are all explicit
assignments comes out; anything with a simultaneous block or more than one state set is
refused by name, because `np.linalg.solve` and the mode dispatcher have no C written for
them yet and emitting something that merely compiles would be worse than saying so.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from modelica.ir import nodes as ir
from modelica.ir.rewrite import residual
from modelica.sim.dialect import C
from modelica.sim.emit import (
    ITERATIONS,
    TOLERANCE,
    EmitError,
    Emitter,
    block_jacobian,
    iterated,
)

if TYPE_CHECKING:
    from modelica.ir.system import Block, Mode, System

__all__ = ["FUNCTIONS", "unit"]

_LOCAL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
"""A symbol that is a plain name, and therefore a local a C function has to declare.

The others are already expressions -- ``w[3]``, ``t`` -- and declaring one would shadow the
argument it reads from.
"""


FUNCTIONS: Final = ("rhs", "observe", "indicators", "transition")
"""The entry points a generated unit defines, in the order this module writes them."""


def _newton(emitter: Emitter, block: Block) -> list[str]:
    """The iteration, written here rather than shaped by the dialect, and here is why.

    Everything else C has needed is a *spelling* -- an operator, a library name, how a matrix
    entry is addressed -- and it fits in `Dialect` because the emitter's line structure is
    the same in both languages. An iteration is not a spelling. It is a loop, a mutable step
    and an early exit, and Python writes those with indentation where C writes braces, so a
    `Dialect` able to express it would have to describe statements rather than expressions:
    a second emitter wearing a table's clothes.

    So the six lines live here. What they are *made of* still comes from `emit` -- the same
    residuals, the same block Jacobian, the same iteration count and tolerance -- which is
    the part that must not be duplicated, because those are decisions and this is only shape.
    """
    unknowns = block.unknowns
    jacobian = block_jacobian(block)
    negated = [ir.Unary("-", residual(equation)) for equation in block.equations]
    guess = [f"{emitter.symbols[u]} = w[{emitter.slots[u]}]" for u in unknowns]
    keep = [f"w[{emitter.slots[u]}] = {emitter.symbols[u]}" for u in unknowns]

    if len(unknowns) == 1:
        name = emitter.symbols[unknowns[0]]
        body = [
            f"double _step = {emitter.expr(negated[0])} / ({emitter.expr(jacobian[0][0])})",
            f"{name} = {name} + _step",
            f"if (fabs(_step) < {TOLERANCE}) {{ break; }}",
        ]
    else:
        size = len(unknowns)
        body = [*emitter.system_lines(block, jacobian, negated)]
        body += [
            f"{emitter.symbols[u]} = {emitter.symbols[u]} + _x[{position}]"
            for position, u in enumerate(unknowns)
        ]
        body.append("double _worst = 0.0")
        body += [f"if (fabs(_x[{k}]) > _worst) {{ _worst = fabs(_x[{k}]); }}" for k in range(size)]
        body.append(f"if (_worst < {TOLERANCE}) {{ break; }}")

    return [
        *guess,
        f"for (int _i = 0; _i < {ITERATIONS}; ++_i) {{",
        *(f"    {line}" for line in body),
        "}",
        *keep,
    ]


def _blocks(emitter: Emitter, mode: Mode) -> list[str]:
    """Every block of a state set, with the iterated ones written the C way.

    `Emitter.solve` handles the rest unchanged -- an assignment, a linear solve -- because
    those are the same lines in both languages once the dialect has spelled the arithmetic.
    """
    lines: list[str] = []
    for block in mode.blocks:
        if block.assignment is None and iterated(block):
            lines += _newton(emitter, block)
            lines += [f"{emitter.symbols[n]} = {emitter.expr(v)}" for n, v in block.torn]
        else:
            lines += emitter.solve(block)
    return lines


def _used(name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def _live(body: list[str]) -> list[str]:
    """The lines whose result something later actually reads.

    Straight-line dead-code elimination, and it is not an optimisation -- it is what makes
    the unit compile. Every plain assignment here has no side effect, so one whose target is
    never read again can go; in Python that line costs a microsecond, and in C, under
    ``-Werror=unused-but-set-variable``, it costs the build.

    Anything that is not a plain ``name = expression`` stays: a store into ``dy``, ``out`` or
    ``w`` is the function's whole purpose, and so is every line of a Newton loop, whose
    targets are indented or declared or braces.

    Why there are dead lines at all: `emit` computes every block because `observe` needs
    every variable, while `rhs` needs only what the derivatives depend on. One body in
    Python serves both; C gets two, each pruned to what it is for.
    """
    keep: list[str] = []
    for line in reversed(body):
        target = line.split(" = ", 1)[0]
        if not _LOCAL.match(target) or _used(target, "\n".join(keep)):
            keep.insert(0, line)
    return keep


def _function(name: str, arguments: str, body: list[str], candidates: list[str]) -> list[str]:
    """One function, declaring exactly the locals its own body uses.

    Exactly, and not all of them: `-Werror=unused-variable` is on, so a declaration this
    function never touches fails the build. `rhs` and `observe` walk the same blocks and end
    differently, so the sets really do differ -- `observe` reads the aliases and `rhs` never
    mentions them.
    """
    written = "\n".join(body)
    used = [local for local in candidates if _used(local, written)]
    return [
        f"void {name}({arguments}) {{",
        *(f"    double {local};" for local in used),
        *(f"    {_ended(line)}" for line in body),
        "}",
        "",
    ]


def _ended(line: str) -> str:
    """A statement takes a semicolon; a brace does not.

    ``if (event == 0) {;`` and ``};`` are both legal C -- an empty statement is a statement --
    which is why the stray punctuation was invisible until somebody read the output.
    Generated code is read when something has gone wrong, and that is the worst moment to be
    explaining it.

    A closing brace is not enough to go on, though, and the first attempt at this rule broke
    the build: ``double _b[2] = {0}`` ends in a brace and is a declaration that very much
    needs its semicolon. So the test is what the line *is* -- an opening brace, a bare
    closing one, or one of the two braced forms this module writes -- and not what it ends
    with.
    """
    text = line.strip()
    block = (
        text.endswith("{")
        or text == "}"
        or (text.endswith("}") and text.startswith(("if (", "for (")))
    )
    return line if block or text.endswith(";") else f"{line};"


def _bodies(
    emitter: Emitter, system: System, mode: Mode, position: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Each entry point for one state set, pruned to what it actually reads."""
    rhs_rest = [
        *_blocks(emitter, mode),
        *(
            f"dy[{slot}] = {emitter.symbols[mode.derivative[state]]}"
            for slot, state in enumerate(mode.states)
        ),
    ]
    observe_rest = [
        *_blocks(emitter, mode),
        *(
            f"{emitter.symbols[name]} = {emitter.expr(node)}"
            for name, node in system.aliases.items()
        ),
        *(f"out[{slot}] = {emitter.symbols[name]}" for slot, name in enumerate(emitter.observed)),
    ]
    events = emitter.events()
    indicator_rest = [
        *(_blocks(emitter, mode) if events else []),
        *(
            f"z[{k}] = "
            + (
                emitter.expr(event.indicator)
                if event.origin in (None, position)
                # An event belonging to another state set is held negative so its root finder
                # has nothing to find, which is what `emit` does and for the same reason: the
                # solver watches one array whichever set is in use.
                else "-1.0"
            )
            for k, event in enumerate(events)
        ),
    ]
    slots = {name: slot for slot, name in enumerate(mode.states)}
    transition_rest: list[str] = [
        *(_blocks(emitter, mode) if events else []),
        # `out[:] = y` in Python. Written out because C has no slice assignment, and written
        # *before* the branches because an event that reinitialises one state must leave the
        # others exactly where they were.
        *(f"out[{slot}] = y[{slot}]" for slot in range(len(mode.states))),
    ]
    for k, event in enumerate(events):
        taken = [emitter.statement(statement, slots) for statement in event.body]
        if event.latch is not None:
            taken.append(f"    w[{event.latch}] = 1.0 - w[{event.latch}]")
        moving = event.trigger.switch
        if moving is not None and event.origin == position:
            taken.extend(emitter.rebase(system.modes[moving]))
            taken.append(f"    w[{emitter.switching}] = {float(moving)}")
        transition_rest.append(f"if (event == {k}) {{")
        transition_rest.extend(taken)
        transition_rest.append("}")
    return (
        _live([*emitter.unpack(mode), *rhs_rest]),
        _live([*emitter.unpack(mode), *observe_rest]),
        _live([*emitter.unpack(mode), *indicator_rest]),
        _live([*emitter.unpack(mode), *transition_rest]),
    )


def _dispatch(name: str, arguments: str, call: str, count: int, switching: int) -> list[str]:
    """The picker in front of one function per state set.

    The same shape the Python emitter writes, and the same comparison: ``w`` is an array of
    doubles, so the slot holding which set is in use is compared against ``k + 0.5`` rather
    than converted to an integer. A C ``switch`` would need the conversion and would say no
    more than this does.
    """
    body = [
        line
        for k in range(count - 1)
        for line in (f"if (w[{switching}] < {k + 0.5}) {{ {name}_{k}({call}); return; }}",)
    ]
    body.append(f"{name}_{count - 1}({call})")
    return [
        f"void {name}({arguments}) {{",
        *(f"    {line}" for line in body[:-1]),
        f"    {body[-1]};",
        "}",
        "",
    ]


def unit(system: System) -> str:
    """``rhs`` and ``observe`` for ``system``, as one C translation unit.

    The workspace layout is the Python one, unchanged and deliberately so: ``y`` the states,
    ``p`` the parameters in `System.parameters` order, ``w`` one slot per observed variable,
    then the latches, then the state set in use. A second backend that laid its arrays out
    differently would need its own reader, its own tests and its own bugs.
    """
    emitter = Emitter(system, dialect=C)
    declared = sorted(
        {symbol for symbol in emitter.symbols.values() if _LOCAL.match(symbol) and symbol != "t"}
    )
    signatures = {
        "rhs": "double t, const double *y, const double *p, double *w, double *dy",
        "observe": "double t, const double *y, const double *p, double *w, double *out",
        "indicators": "double t, const double *y, const double *p, double *w, double *z",
        "transition": (
            "int event, double t, const double *y, const double *p, double *w, double *out"
        ),
    }
    calls = {
        "rhs": "t, y, p, w, dy",
        "observe": "t, y, p, w, out",
        "indicators": "t, y, p, w, z",
        "transition": "event, t, y, p, w, out",
    }

    written: list[str] = []
    for position, name in enumerate(FUNCTIONS):
        bodies = [
            _bodies(emitter, system, mode, k)[position] for k, mode in enumerate(system.modes)
        ]
        if len(bodies) == 1:
            written += _function(name, signatures[name], bodies[0], declared)
            continue
        for k, body in enumerate(bodies):
            written += _function(f"{name}_{k}", signatures[name], body, declared)
        switching = emitter.switching
        if switching is None:  # pragma: no cover - several bodies means several state sets
            raise EmitError("C: several state sets and no slot saying which is in use")
        written += _dispatch(name, signatures[name], calls[name], len(bodies), switching)

    lines = [
        f"/* Generated from {system.name} by modelica.sim.cgen. Do not edit. */",
        *C.preamble,
        *written,
    ]
    return "\n".join(lines) + "\n"
