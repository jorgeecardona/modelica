"""A Rust translation unit for a model, from the same emitter that writes the C one.

`docs/issues/03-09-2026-a-rust-backend-is-a-third-dialect-plus-a-port-of-radau.md` is the
plan this file starts on. Measured against the same bar as `modelica.sim.cgen`: this module
makes no modelling decisions. It asks `modelica.sim.emit` for the same lines with
:data:`modelica.sim.dialect.RUST` selected, and writes the Rust *structure* -- the part a
dialect table cannot hold -- around them.

It begins deliberately small. Only what a plain right-hand side needs is written: a single
state set, blocks that are assignments or a linear solve (the dialect's ``md_solve`` carries
the dense LU), and no events. A block that has to be iterated on -- Newton -- is refused,
because its loop is structure the emitter writes rather than spelling the dialect holds, and
that structure is a later increment. Everything refused raises an `EmitError` that names the
missing piece, exactly as `cgen` does, so a model moving off the refused set fails the test
and gets its check turned on rather than staying quietly untested. Newton, the mode
dispatcher, the row-at-a-time tangent and the event functions are what is left.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from modelica.sim.dialect import RUST
from modelica.sim.emit import EmitError, Emitter, iterated

if TYPE_CHECKING:
    from modelica.ir.system import Mode, System

__all__ = ["unit"]

_LOCAL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
"""A target that is a plain name, and therefore a local the Rust body has to bind with `let`.

The others are already places -- ``dy[0]``, ``w[3]`` -- and a `let` in front of one would not
be valid Rust.
"""

#: The one function written so far. Slices rather than raw pointers: this is a Rust-native
#: interface a `main` (and later the ported integrator) calls directly. A C ABI surface for
#: FMU export is a later increment, and a separate decision.
_SIGNATURES: Final = {
    "rhs": "_t: f64, y: &[f64], p: &[f64], w: &mut [f64], dy: &mut [f64]",
}


def _used(name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def _live(body: list[str]) -> list[str]:
    """The lines whose result something later actually reads.

    The same straight-line dead-code elimination `cgen._live` does, and for the same reason:
    a `let` binding nothing reads is an `unused_variables` warning, and the test compiles with
    ``-D warnings``. A store into ``dy`` or ``w`` is the function's purpose and always stays.
    """
    keep: list[str] = []
    for line in reversed(body):
        target = line.split(" = ", 1)[0]
        if not _LOCAL.match(target) or _used(target, "\n".join(keep)):
            keep.insert(0, line)
    return keep


def _bind(lines: list[str]) -> list[str]:
    """Turn the emitter's ``name = expr`` lines into Rust statements.

    A plain-name target assigned once becomes ``let name: f64 = expr;``; a store into an
    indexed place (``dy[0]``) keeps its assignment and takes a semicolon. A name assigned more
    than once is declared ``let mut`` up front so every later assignment is a plain one -- the
    straight-line case has none of these yet, but the rule is here so Newton does not surprise
    it later.
    """
    counts: dict[str, int] = {}
    for line in lines:
        target = line.split(" = ", 1)[0]
        if _LOCAL.match(target):
            counts[target] = counts.get(target, 0) + 1
    reassigned = {name for name, n in counts.items() if n > 1}

    out = [f"let mut {name}: f64;" for name in sorted(reassigned)]
    for line in lines:
        target, _, expr = line.partition(" = ")
        if _LOCAL.match(target) and target not in reassigned:
            out.append(f"let {target}: f64 = {expr};")
        else:
            out.append(f"{line};")
    return out


def _rhs(emitter: Emitter, mode: Mode) -> list[str]:
    """The derivative vector: unpack the workspace, run the blocks, store into ``dy``."""
    lines = [
        *emitter.unpack(mode),
        *emitter.blocks(mode),
        *(
            f"dy[{slot}] = {emitter.symbols[mode.derivative[state]]}"
            for slot, state in enumerate(mode.states)
        ),
    ]
    return _bind(_live(lines))


def _guard(system: System) -> tuple[Emitter, Mode]:
    """Refuse what this increment does not yet write, each with the piece it is missing."""
    if len(system.modes) != 1:
        raise EmitError("rust: several state sets are not emitted yet (no mode dispatcher)")
    emitter = Emitter(system, dialect=RUST)
    if emitter.events():
        raise EmitError("rust: events are not emitted yet (no indicators/transition)")
    mode = system.modes[0]
    for block in mode.blocks:
        if block.assignment is None and iterated(block):
            raise EmitError("rust: Newton iteration is not emitted yet (no iterated blocks)")
    return emitter, mode


def _function(name: str, body: list[str]) -> list[str]:
    """One Rust function, with the two lints a generated body trips deliberately allowed.

    ``non_snake_case`` because the locals are Modelica-derived names (``source.V`` becomes
    ``source_V``), so the casing is the model's and not ours to normalise -- the C side gets
    this for free, C has no such lint. ``unused_variables`` because the signature is fixed
    while a given model may touch neither ``t`` nor ``w``, and ``-Wall`` does not warn on an
    unused parameter either. Dead *locals* are still pruned by `_live`; this only forgives the
    fixed parameters and the inherited names.
    """
    return [
        "#[allow(non_snake_case, unused_variables)]",
        f"fn {name}({_SIGNATURES[name]}) {{",
        *(f"    {line}" for line in body),
        "}",
        "",
    ]


def unit(system: System) -> str:
    """``rhs`` for ``system``, as one Rust source file.

    The workspace layout is the Python and C one, unchanged: ``y`` the states, ``p`` the
    parameters in `System.parameters` order, ``w`` one slot per observed variable. A backend
    that laid its arrays out differently would need its own reader and its own bugs.
    """
    emitter, mode = _guard(system)
    lines = [
        f"// Generated from {system.name} by modelica.sim.rustgen. Do not edit.",
        *RUST.preamble,
        "",
        *_function("rhs", _rhs(emitter, mode)),
    ]
    return "\n".join(lines) + "\n"
