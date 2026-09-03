"""A Rust translation unit for a model, from the same emitter that writes the C one.

`docs/issues/03-09-2026-a-rust-backend-is-a-third-dialect-plus-a-port-of-radau.md` is the
plan this file starts on. Measured against the same bar as `modelica.sim.cgen`: this module
makes no modelling decisions. It asks `modelica.sim.emit` for the same lines with
:data:`modelica.sim.dialect.RUST` selected, and writes the Rust *structure* -- the part a
dialect table cannot hold -- around them.

It grows one kernel function at a time. Blocks that are assignments or a linear solve (the
dialect's ``md_solve`` carries the dense LU), and the five kernel functions ``rhs``, ``observe``,
``indicators``, ``transition`` and ``jacobian`` -- for a model with events too, so ``indicators``
reads real zero-crossing functions and ``transition`` writes the reset. A model with several
state sets gets one function per set behind a dispatcher that reads which set is in use, so a
mode switch is emitted like `cgen`. A block that has to be iterated on -- Newton -- is refused,
because its loop is structure the emitter writes rather than spelling the dialect holds. The
refusal raises an `EmitError` that names the missing piece, exactly as `cgen` does, so a model
moving off the refused set fails the test and gets its check turned on rather than staying
quietly untested. What is left: Newton (iterated blocks), which is the last thing between this
and every corpus model.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from modelica.ir.derive import ZERO
from modelica.ir.rewrite import residual
from modelica.sim.dialect import RUST
from modelica.sim.emit import EmitError, Emitter, block_jacobian, iterated

if TYPE_CHECKING:
    from collections.abc import Iterable

    from modelica.ir.system import Block, Mode, System

__all__ = ["unit"]

_LOCAL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
"""A target that is a plain name, and therefore a local the Rust body has to bind with `let`.

The others are already places -- ``dy[0]``, ``w[3]`` -- and a `let` in front of one would not
be valid Rust.
"""

#: The functions written so far, with their parameter lists. Slices rather than raw pointers:
#: this is a Rust-native interface a `main` (and later the ported integrator) calls directly.
#: A C ABI surface for FMU export is a later increment, and a separate decision. The last
#: parameter is the output the function fills; the others are the shared workspace layout.
_SIGNATURES: Final = {
    "rhs": "t: f64, y: &[f64], p: &[f64], w: &mut [f64], dy: &mut [f64]",
    "observe": "t: f64, y: &[f64], p: &[f64], w: &mut [f64], out: &mut [f64]",
    "indicators": "t: f64, y: &[f64], p: &[f64], w: &mut [f64], z: &mut [f64]",
    "transition": "event: usize, t: f64, y: &[f64], p: &[f64], w: &mut [f64], out: &mut [f64]",
    "jacobian": "t: f64, y: &[f64], p: &[f64], w: &mut [f64], out: &mut [f64]",
}

#: The order the functions are written in, and the argument names a dispatcher forwards to the
#: per-mode function it picks -- the parameter names of `_SIGNATURES` without their types.
_FUNCTIONS: Final = ("rhs", "observe", "indicators", "transition", "jacobian")
_CALLS: Final = {
    "rhs": "t, y, p, w, dy",
    "observe": "t, y, p, w, out",
    "indicators": "t, y, p, w, z",
    "transition": "event, t, y, p, w, out",
    "jacobian": "t, y, p, w, out",
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
    indexed place (``dy[0]``, ``row[_k]``) keeps its assignment and takes a semicolon. A name
    assigned more than once is declared ``let mut`` up front so every later assignment is a
    plain one -- the straight-line functions have none of these; the Jacobian's row loops do.

    The Jacobian body also carries structure, not only assignments: a ``for _k in 0..n {``
    header and its closing ``}`` are statements a semicolon would break, so a line that opens or
    closes a block is emitted verbatim. This is the Rust analog of `cgen._ended`.
    """
    counts: dict[str, int] = {}
    for line in lines:
        target = line.split(" = ", 1)[0]
        if _LOCAL.match(target):
            counts[target] = counts.get(target, 0) + 1
    reassigned = {name for name, n in counts.items() if n > 1}

    out = [f"let mut {name}: f64;" for name in sorted(reassigned)]
    for line in lines:
        stripped = line.strip()
        if stripped.endswith("{") or stripped == "}":
            out.append(line)
            continue
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


def _observe(emitter: Emitter, system: System, mode: Mode) -> list[str]:
    """Every observed variable: `rhs`'s blocks, then the aliases, then store into ``out``.

    The same body `rhs` runs, extended with the alias equations `rhs` did not need and the
    stores into ``out`` in `Emitter.observed` order. `_live` then prunes it to what an
    observation actually reads, which is a different set than the derivative did.
    """
    lines = [
        *emitter.unpack(mode),
        *emitter.blocks(mode),
        *(
            f"{emitter.symbols[name]} = {emitter.expr(node)}"
            for name, node in system.aliases.items()
        ),
        *(f"out[{slot}] = {emitter.symbols[name]}" for slot, name in enumerate(emitter.observed)),
    ]
    return _bind(_live(lines))


def _indicators(emitter: Emitter, position: int, mode: Mode) -> list[str]:
    """The zero-crossing functions, one ``z[k]`` per event.

    Unpack the workspace, run the blocks when there is an indicator to read, then store each
    event's function into ``z`` -- the same shape `cgen` writes. A model with no events makes an
    empty body (``_live`` prunes the unpack nothing reads); a model with events needs the unpack,
    because an indicator like BouncingBall's ``-h`` reads a state. An event owned by another state
    set is held negative so its root finder has nothing to find.
    """
    events = emitter.events()
    lines = [
        *emitter.unpack(mode),
        *(emitter.blocks(mode) if events else []),
        *(
            f"z[{k}] = "
            + (emitter.expr(event.indicator) if event.origin in (None, position) else "-1.0")
            for k, event in enumerate(events)
        ),
    ]
    return _bind(_live(lines))


def _transition(emitter: Emitter, position: int, system: System, mode: Mode) -> list[str]:
    """The post-event state: copy ``y`` through, then let each event's branch overwrite its slots.

    ``out[:] = y`` first (Rust has no slice assignment here either, so it is written out), and
    *before* the branches, so a crossing that reinitialises one state leaves the others exactly
    where they were. Then one ``if event == k`` per crossing runs that branch's `reinit` and
    discrete assignments, flips its latch, and -- for a crossing that switches state sets, and
    only from the set it leaves (``event.origin == position``) -- rebases the trajectory into the
    destination's variables and writes the new set into ``w``. The same body `cgen` writes.
    """
    events = emitter.events()
    slots = {name: slot for slot, name in enumerate(mode.states)}
    lines = [
        *emitter.unpack(mode),
        *(emitter.blocks(mode) if events else []),
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
        lines.append(f"if event == {k} {{")
        lines.extend(taken)
        lines.append("}")
    return _bind(_live(lines))


def _rows(text: str, tangents: Iterable[str]) -> str:
    """A row expression read one element at a time: subscript each row name by the loop index.

    `Emitter.combination` returns whole-row arithmetic (``a * _d_u + _d_v``); making every row
    name ``name[_k]`` turns it elementwise, the same rewrite `cgen._rows` does. Textual and safe
    to be, because a row name appears nowhere in an expression except as a row.
    """
    for row in tangents:
        text = re.sub(rf"\b{re.escape(row)}\b", f"{row}[_k]", text)
    return text


def _reached(groups: list[tuple[str | None, list[str]]]) -> list[str]:
    """The rows something downstream reads, and nothing else.

    A row is several lines -- a declaration, a loop, a store -- so it is pruned as a group and
    the group carries the row it defines. Not an optimisation: a row nothing reads is a `let`
    binding nothing reads, which ``-D warnings`` rejects.
    """
    keep: list[str] = []
    for name, lines in reversed(groups):
        if name is None or _used(name, "\n".join(keep)):
            keep[:0] = lines
    return keep


def _implicit(
    emitter: Emitter, block: Block, carried: dict[str, str], width: int, rows: list[str]
) -> list[tuple[str | None, list[str]]]:
    """The tangent of a solved block, without solving it again.

    ``g(u, v) = 0`` differentiates to ``(dg/du) u' = -(dg/dv) v'``: one solve against the matrix
    the block was solved with, evaluated at the answer. `Emitter.implicit` says this in numpy;
    here the right-hand side has a column per state and the dense solver takes one at a time.
    """
    residuals = [emitter.combination(residual(equation), carried) for equation in block.equations]
    if all(term is None for term in residuals):
        for unknown in block.solved:
            carried.pop(unknown, None)
        return []
    jacobian = block_jacobian(block)
    if len(block.unknowns) == 1:
        carried[block.unknowns[0]] = row = emitter.tangents[block.unknowns[0]]
        return [
            (
                row,
                [
                    f"let mut {row} = [0.0f64; {width}]",
                    f"for _k in 0..{width} {{",
                    (
                        f"    {row}[_k] = -({_rows(residuals[0] or '0.0', rows)})"
                        f" / ({emitter.expr(jacobian[0][0])})"
                    ),
                    "}",
                ],
            )
        ]
    size = len(block.unknowns)
    lines = [f"let mut {emitter.tangents[u]} = [0.0f64; {width}]" for u in block.unknowns]
    lines.append(f"for _k in 0..{width} {{")
    lines += [f"    let mut _M = [0.0f64; {size * size}]", f"    let mut _R = [0.0f64; {size}]"]
    for position, entries in enumerate(jacobian):
        lines += [
            f"    _M[{position * size + column}] = {emitter.expr(entry)}"
            for column, entry in enumerate(entries)
            if entry != ZERO
        ]
        term = residuals[position]
        if term is not None:
            lines.append(f"    _R[{position}] = -({_rows(term, rows)})")
    lines += [
        f"    let mut _D = [0.0f64; {size}]",
        f"    md_solve({size}, &mut _M, &mut _R, &mut _D)",
    ]
    for position, unknown in enumerate(block.unknowns):
        carried[unknown] = row = emitter.tangents[unknown]
        lines.append(f"    {row}[_k] = _D[{position}]")
    lines.append("}")
    return [(None, lines)]


def _tangent(emitter: Emitter, mode: Mode, width: int) -> list[tuple[str | None, list[str]]]:
    """Forward mode over the very blocks the right-hand side runs, a row at a time.

    `Emitter.tangent` in Rust, and the one place the two backends differ in data layout rather
    than spelling: a Jacobian row is a numpy vector in Python and a ``[f64; n]`` here, so every
    assignment to one becomes a loop. The decisions stay in `emit`: which variables have a
    tangent at all is `combination` returning ``None``, and the matrix a solved block
    differentiates against is `block_jacobian`. A structurally independent variable gets no row
    rather than a row of zeros, which is what keeps the emitted source proportional to the model.
    """
    rows = list(emitter.tangents.values())
    groups: list[tuple[str | None, list[str]]] = []
    carried: dict[str, str] = {}
    for slot, state in enumerate(mode.states):
        carried[state] = row = emitter.tangents[state]
        groups.append((row, [f"let mut {row} = [0.0f64; {width}]", f"{row}[{slot}] = 1.0"]))

    def elementwise(name: str, total: str | None) -> list[str]:
        """``d name = <combination>``, which in Rust is a loop over the row."""
        if total is None:
            carried.pop(name, None)
            return []
        carried[name] = row = emitter.tangents[name]
        return [
            f"let mut {row} = [0.0f64; {width}]",
            f"for _k in 0..{width} {{",
            f"    {row}[_k] = {_rows(total, rows)}",
            "}",
        ]

    for block in mode.blocks:
        groups.append((None, emitter.solve(block)))
        if block.assignment is not None:
            unknown = block.unknowns[0]
            groups.append(
                (
                    emitter.tangents[unknown],
                    elementwise(unknown, emitter.combination(block.assignment, carried)),
                )
            )
            continue
        groups.extend(_implicit(emitter, block, carried, width, rows))
        for name, value in block.torn:
            groups.append(
                (emitter.tangents[name], elementwise(name, emitter.combination(value, carried)))
            )

    for slot, state in enumerate(mode.states):
        row = carried.get(mode.derivative[state])
        groups.append(
            (
                None,
                [
                    f"for _k in 0..{width} {{",
                    f"    out[{slot} * {width} + _k] = " + (f"{row}[_k]" if row else "0.0"),
                    "}",
                ],
            )
        )
    return groups


def _jacobian(emitter: Emitter, mode: Mode) -> list[str]:
    """The ``n``-by-``n`` Jacobian in row-major order, forward mode, a row at a time.

    Two prunings, because there are two kinds of dead line: a row nothing reads, which
    `_reached` drops as a group, and then the block values only that row read, which `_live`
    drops once the row is gone.
    """
    width = len(mode.states)
    return _bind(_live([*emitter.unpack(mode), *_reached(_tangent(emitter, mode, width))]))


def _guard(system: System) -> Emitter:
    """Refuse what this increment does not yet write, each with the piece it is missing."""
    emitter = Emitter(system, dialect=RUST)
    for mode in system.modes:
        for block in mode.blocks:
            if block.assignment is None and iterated(block):
                raise EmitError("rust: Newton iteration is not emitted yet (no iterated blocks)")
    return emitter


def _function(name: str, signature: str, body: list[str]) -> list[str]:
    """One Rust function, with the lints a generated entry point trips deliberately allowed.

    ``name`` is the emitted name, which for a multi-mode model is ``rhs_0`` and the like, so it
    is passed rather than reused as the `_SIGNATURES` key -- ``signature`` carries that.

    ``non_snake_case`` because the locals are Modelica-derived names (``source.V`` becomes
    ``source_V``), so the casing is the model's and not ours to normalise -- the C side gets
    this for free, C has no such lint. ``unused_variables`` because the signature is fixed
    while a given model may touch neither ``t`` nor ``w``, and ``-Wall`` does not warn on an
    unused parameter either. ``dead_code`` because a unit defines every entry point and a given
    caller -- a test's `main`, later the integrator -- may use only some, exactly as the C
    shared object exports all five whether or not a consumer calls them. Dead *locals* are
    still pruned by `_live`; this only forgives the fixed parameters and the whole functions.
    """
    return [
        "#[allow(dead_code, non_snake_case, unused_variables)]",
        f"fn {name}({signature}) {{",
        *(f"    {line}" for line in body),
        "}",
        "",
    ]


def _dispatch(name: str, switching: int, count: int) -> list[str]:
    """The picker in front of one function per state set, `cgen._dispatch` in Rust.

    ``w`` holds which set is in use as a float, so the slot is compared against ``k + 0.5``
    rather than converted to an integer -- the same comparison the Python and C emitters write.
    Each arm forwards this function's own arguments to the mode's, and the last set is the
    fall-through with no test.
    """
    call = _CALLS[name]
    body = [
        f"if w[{switching}] < {k + 0.5} {{ {name}_{k}({call}); return; }}" for k in range(count - 1)
    ]
    body.append(f"{name}_{count - 1}({call});")
    return _function(name, _SIGNATURES[name], body)


def _body(name: str, emitter: Emitter, system: System, mode: Mode, position: int) -> list[str]:
    """The body of one kernel function for one state set."""
    if name == "rhs":
        return _rhs(emitter, mode)
    if name == "observe":
        return _observe(emitter, system, mode)
    if name == "indicators":
        return _indicators(emitter, position, mode)
    if name == "transition":
        return _transition(emitter, position, system, mode)
    return _jacobian(emitter, mode)


def unit(system: System) -> str:
    """``rhs``, ``observe``, ``indicators``, ``transition`` and ``jacobian``, as one Rust file.

    The workspace layout is the Python and C one, unchanged: ``y`` the states, ``p`` the
    parameters in `System.parameters` order, ``w`` one slot per observed variable, then the
    latches, then the state set in use. A model with several state sets gets one function per set
    named ``rhs_0``, ``rhs_1``, ... behind a dispatcher that reads that last slot, exactly as
    `cgen` does; a single-set model gets the bare names and no dispatcher. A backend that laid
    its arrays out differently would need its own reader and its own bugs.
    """
    emitter = _guard(system)
    lines = [
        f"// Generated from {system.name} by modelica.sim.rustgen. Do not edit.",
        *RUST.preamble,
        "",
    ]
    for name in _FUNCTIONS:
        bodies = [_body(name, emitter, system, mode, k) for k, mode in enumerate(system.modes)]
        if len(bodies) == 1:
            lines += _function(name, _SIGNATURES[name], bodies[0])
            continue
        for k, body in enumerate(bodies):
            lines += _function(f"{name}_{k}", _SIGNATURES[name], body)
        switching = emitter.switching
        if switching is None:  # pragma: no cover - several sets always carry a switching slot
            raise EmitError("rust: several state sets without a switching slot")
        lines += _dispatch(name, switching, len(bodies))
    return "\n".join(lines) + "\n"
