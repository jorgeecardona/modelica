"""Two tiers of the same model, at one state, stopped at the first block they disagree on.

`docs/ideas/backends.md:27` sells this as the reason for owning the compiler: "When a
compiled run disagrees with an interpreted run, step both and find the first equation where
they diverge. You cannot even ask that question of a generated-C pipeline, because there is
nothing to compare against." This module is that question, asked.

The comparison is *not* between two runs. It is between two evaluations at one state, and
that is the whole design rather than a simplification: a trajectory that has already
separated tells you where the two implementations ended up, and where they *started* to
disagree is a different instant, usually much earlier, and the only one worth reading. Give
this the state at which they first differed and it names the block.

**How the two tiers are read.** Not by instrumenting either one -- neither has a hook, and a
tier that had to grow one would no longer be the tier that shipped. Both are asked for
``observe``, which writes *every* variable the model has (`Code.observed`), and the block
schedule attributes each variable to the block that computes it. Measured over the corpus:
every name in every `Block.solved`, in every mode, is in `Code.observed`. So the block-level
picture is already there in a value both tiers agree to produce.

**What counts as a tier**: anything with the emitted `observe` signature. A `Kernel` passes
itself; a compiled C unit passes a ctypes wrapper; an FMU will pass whatever wraps an FMU.
That is why the arguments are two observers rather than two kernels -- the interpreter is not
privileged here, and the pairs that will matter (C against interpreter, fused-step against
per-block, FMU against the model it came from) do not include it on both sides.

A divergence names the equations it found, in Modelica, and where they were written when the
span survived flattening. It survives for the equations a modeller actually wrote; a
connection equation and an equation created by index reduction have no line, because nobody
wrote one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from modelica.lang.unparse import unparse_equation
from modelica.sim.kernel import Kernel, build

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from modelica.ir.nodes import Span
    from modelica.ir.system import Block, Mode, System

__all__ = ["Divergence", "Observer", "diverges"]

TOLERANCE: float = 1e-12
"""Default agreement, absolute and relative both.

Not zero: the engine and the Python path call different ``libm`` builds, which can round
``exp`` differently in the last place.
"""


class Observer(Protocol):
    """The emitted ``observe``: every variable of the model, written into ``out``."""

    def __call__(
        self,
        t: float,
        y: NDArray[np.float64],
        p: NDArray[np.float64],
        w: NDArray[np.float64],
        out: NDArray[np.float64],
        /,
    ) -> object: ...


@dataclass(frozen=True)
class Divergence:
    """The first block whose two evaluations disagree, and by how much."""

    position: int | None
    """Index into the mode's blocks, or ``None`` for a variable no block computes.

    ``None`` is the alias case and it is reported rather than swallowed: an alias is
    recovered *after* the schedule has run, so two tiers can agree on every block and still
    disagree on a variable a user reads.
    """

    variables: tuple[str, ...]
    """The variables of that block that differ. Not all of them -- the ones that do."""

    left: tuple[float, ...]
    right: tuple[float, ...]
    equations: tuple[str, ...]
    """The block's equations, unparsed back to Modelica with flat names."""

    span: Span | None
    """Where the first of those equations was written, when that survived flattening."""

    @property
    def worst(self) -> float:
        """The largest absolute disagreement among :attr:`variables`."""
        return max(abs(a - b) for a, b in zip(self.left, self.right, strict=True))

    def __str__(self) -> str:
        where = f"block {self.position}" if self.position is not None else "an alias"
        head = f"{where} disagrees by {self.worst:.3g}"
        if self.span is not None:
            head += f", written at {self.span}"
        rows = [
            f"  {name}: {a!r} vs {b!r}"
            for name, a, b in zip(self.variables, self.left, self.right, strict=True)
        ]
        return "\n".join([head, *rows, *(f"  {equation}" for equation in self.equations)])


def diverges(  # noqa: PLR0913 - a state, two tiers and two tolerances
    system: System,
    left: Kernel | Observer,
    right: Kernel | Observer,
    *,
    t: float = 0.0,
    y: NDArray[np.float64] | None = None,
    p: NDArray[np.float64] | None = None,
    w: NDArray[np.float64] | None = None,
    absolute: float = TOLERANCE,
    relative: float = TOLERANCE,
) -> Divergence | None:
    """The first block ``left`` and ``right`` disagree on at ``(t, y)``, or ``None``.

    First in *schedule* order, which is what makes the answer useful. A wrong value in an
    early block reaches every later block that reads it, so a report listing every
    disagreement would be a list of one cause and its consequences with no way to tell them
    apart. The block returned is the one to look at.

    ``y``, ``p`` and ``w`` default to the model's own start point at ``t``, taken from a
    kernel built for the purpose -- the one place a state is guaranteed to be one the model
    was written for. Pass them to ask about somewhere else, which is the usual case: a
    divergence is found by a run, and the state that run was in is the interesting one.

    Both tiers are read through ``observe`` at the same arrays, and neither is mutated: each
    gets its own copy of ``w``, because ``observe`` writes the workspace as it goes.
    """
    reference = build(system)
    code = reference.code
    if y is None or p is None or w is None:
        start_p, start_y, start_w = reference.start(t)
        p = start_p if p is None else p
        y = start_y if y is None else y
        w = start_w if w is None else w

    here = _seen(left, t, y, p, w, len(code.observed))
    there = _seen(right, t, y, p, w, len(code.observed))
    slot = {name: index for index, name in enumerate(code.observed)}
    apart = {
        name
        for name, index in slot.items()
        if abs(here[index] - there[index]) > absolute + relative * abs(there[index])
    }
    if not apart:
        return None

    mode = _mode(system, w, code.switching)
    for position, block in enumerate(mode.blocks):
        names = tuple(name for name in block.solved if name in apart)
        if names:
            return _found(position, names, here, there, slot, block)
    # Every block agreed and something still differs: an alias, recovered after the schedule.
    return _found(None, tuple(name for name in code.observed if name in apart), here, there, slot)


def _seen(  # noqa: PLR0913, PLR0917 - the generated calling convention, plus the width
    tier: Kernel | Observer,
    t: float,
    y: NDArray[np.float64],
    p: NDArray[np.float64],
    w: NDArray[np.float64],
    width: int,
) -> NDArray[np.float64]:
    """One tier's view of every variable, on its own copy of the workspace."""
    observe = tier.observe if isinstance(tier, Kernel) else tier
    out = np.zeros(width)
    observe(t, y, p, w.copy(), out)
    return out


def _mode(system: System, w: NDArray[np.float64], switching: int | None) -> Mode:
    """Which state set the workspace says is in use, which is what ``observe`` dispatched on."""
    return system.modes[0] if switching is None else system.modes[int(w[switching])]


def _found(  # noqa: PLR0913, PLR0917 - assembling a record from the pieces that made it
    position: int | None,
    names: tuple[str, ...],
    here: NDArray[np.float64],
    there: NDArray[np.float64],
    slot: dict[str, int],
    block: Block | None = None,
) -> Divergence:
    equations = () if block is None else tuple(block.equations)
    return Divergence(
        position=position,
        variables=names,
        left=tuple(float(here[slot[name]]) for name in names),
        right=tuple(float(there[slot[name]]) for name in names),
        equations=tuple(
            " ".join(line.strip() for line in unparse_equation(equation)) for equation in equations
        ),
        span=next((equation.span for equation in equations if equation.span is not None), None),
    )
