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

from modelica.sim.dialect import C
from modelica.sim.emit import EmitError, Emitter

if TYPE_CHECKING:
    from modelica.ir.system import Mode, System

__all__ = ["FUNCTIONS", "unit"]

_LOCAL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
"""A symbol that is a plain name, and therefore a local a C function has to declare.

The others are already expressions -- ``w[3]``, ``t`` -- and declaring one would shadow the
argument it reads from.
"""


FUNCTIONS: Final = ("rhs", "observe")
"""The entry points a generated unit defines, in the order this module writes them."""


def _refuse(system: System) -> Mode:
    """The one mode this can write, or a refusal that names what is missing.

    Both refusals are about code that does not exist yet rather than about the model. A
    simultaneous block needs a dense solve in C and a second state set needs the dispatcher
    as a switch; until those are written, a unit that quietly left them out would compile
    and give wrong answers, which is the one outcome worth ruling out.
    """
    if len(system.modes) != 1:
        raise EmitError(
            f"C: {len(system.modes)} state sets, and the mode dispatcher has no C yet"
        )
    (mode,) = system.modes
    implicit = [block for block in mode.blocks if block.assignment is None]
    if implicit:
        raise EmitError(
            f"C: {len(implicit)} simultaneous block(s), and the linear solve has no C yet"
        )
    return mode


def _used(name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def _live(body: list[str]) -> list[str]:
    """The lines whose result something later actually reads.

    Straight-line dead-code elimination, and it is not an optimisation -- it is what makes
    the unit compile. Every line here is ``target = expression`` with no side effect, so a
    line whose target is never read again can go; in Python that line costs a microsecond,
    and in C, under ``-Werror=unused-but-set-variable``, it costs the build.

    Two kinds of line always stay: a store into ``dy``, ``out`` or ``w`` is the function's
    whole purpose, and a target that is not a plain name is one of those.

    Why the sequence has dead lines at all is worth knowing. `emit` computes every block
    because `observe` needs every variable; `rhs` needs the ones the derivatives depend on
    and no others. The Python emitter writes one body and both callers use it. C gets two
    bodies, and each is pruned to what it is for.
    """
    keep: list[str] = []
    for line in reversed(body):
        target = line.split(" = ", 1)[0]
        later = "\n".join(keep)
        if not _LOCAL.match(target) or _used(target, later):
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
        *(f"    {line};" for line in body),
        "}",
        "",
    ]


def unit(system: System) -> str:
    """``rhs`` and ``observe`` for ``system``, as one C translation unit.

    The workspace layout is the Python one, unchanged and deliberately so: ``y`` the states,
    ``p`` the parameters in `System.parameters` order, ``w`` one slot per observed variable
    then the latches. A second backend that laid its arrays out differently would need its
    own reader, its own tests and its own bugs.
    """
    mode = _refuse(system)
    emitter = Emitter(system, dialect=C)
    declared = sorted(
        {
            symbol
            for symbol in emitter.symbols.values()
            if _LOCAL.match(symbol) and symbol != "t"
        }
    )

    rhs_rest = [
        *emitter.blocks(mode),
        *(
            f"dy[{slot}] = {emitter.symbols[mode.derivative[state]]}"
            for slot, state in enumerate(mode.states)
        ),
    ]
    rhs_body = _live([*emitter.unpack(mode), *rhs_rest])
    observe_rest = [
        *emitter.blocks(mode),
        *(
            f"{emitter.symbols[name]} = {emitter.expr(node)}"
            for name, node in system.aliases.items()
        ),
        *(
            f"out[{slot}] = {emitter.symbols[name]}"
            for slot, name in enumerate(emitter.observed)
        ),
    ]
    observe_body = _live([*emitter.unpack(mode), *observe_rest])

    lines = [
        f"/* Generated from {system.name} by modelica.sim.cgen. Do not edit. */",
        *C.preamble,
        *_function(
            "rhs",
            "double t, const double *y, const double *p, double *w, double *dy",
            rhs_body,
            declared,
        ),
        *_function(
            "observe",
            "double t, const double *y, const double *p, double *w, double *out",
            observe_body,
            declared,
        ),
    ]
    return "\n".join(lines) + "\n"
