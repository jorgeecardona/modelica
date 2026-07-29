"""What a model looks like once it has been given a direction of computation.

A :class:`~modelica.ir.flat.FlatModel` is a pile of equations that happen to be true. A
:class:`System` is the same content arranged so a solver can walk it: states first, then
blocks in an order where every block's inputs were computed by an earlier one.

Like :mod:`modelica.ir.flat` this module holds data and nothing else. The pass that
produces it lives in :mod:`modelica.ir.causalize`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from modelica.ir import nodes as ir
    from modelica.ir.flat import FlatVariable

__all__ = ["Block", "System"]


def _no_aliases() -> dict[str, ir.Expr]:
    return {}


@dataclass(frozen=True)
class Block:
    """One diagonal entry of the block-lower-triangular form.

    ``unknowns`` and ``equations`` always have the same length: a block is square by
    construction, being one strongly connected component of the dependency graph. Most
    blocks are 1x1 and carry an ``assignment`` -- the unknown written explicitly in terms
    of things already known, which is the whole point of sorting. A block without an
    assignment has to be handed to a numerical solver, and its size says how bad that is.
    """

    unknowns: tuple[str, ...]
    equations: tuple[ir.Equal, ...]
    assignment: ir.Expr | None = None

    @property
    def is_linear_scalar(self) -> bool:
        """True when this block reduces to a single explicit assignment."""
        return self.assignment is not None

    def __len__(self) -> int:
        return len(self.unknowns)


@dataclass(frozen=True)
class System:
    """A causalized model: states, sorted blocks, and the bookkeeping to read it back.

    ``derivative`` maps a state to the variable holding its time derivative. That is not
    always ``der_x``: alias elimination routinely discovers that a variable the modeller
    already named *is* the derivative -- ``mass.v`` is ``der(mass.s)`` -- and keeping the
    modeller's name is what makes the result readable.

    ``aliases`` maps every variable that was eliminated to the expression that replaces
    it, resolved so that a single substitution recovers it. Nothing downstream needs it to
    integrate, but everything downstream needs it to report results: a user who asks for
    ``r.p.v`` should get a number, not a shrug.

    ``candidates`` lists the state sets that *would* also have worked, :attr:`states` first.
    An index-reduced model usually has more than one admissible selection, and which one is
    numerically usable changes along the trajectory -- the pendulum's ``(y, vy)`` goes
    singular at the bottom of its swing where ``(x, vx)`` is fine. Nothing switches between
    them yet; they are recorded because the alternatives are a property of the model, and a
    runtime that will eventually have to choose should be handed the choice rather than
    rediscover it.

    Admissible means two things: the remaining equations determine everything the set does
    not, and the set is closed under integration -- if ``vy`` is a state and the model knows
    ``vy`` is ``der(y)``, then ``y`` is a state too. The second is not pedantry. Integrating
    ``vy`` while reading ``y`` off a constraint leaves nothing tying the two together, since
    ``der(y) = vy`` was spent making them one variable, and the trajectory drifts while
    satisfying every equation. Sets that fail either test never reach this list.
    """

    name: str
    variables: tuple[FlatVariable, ...]
    states: tuple[str, ...]
    derivative: Mapping[str, str]
    blocks: tuple[Block, ...]
    index: int = 1
    aliases: Mapping[str, ir.Expr] = field(default_factory=_no_aliases)
    whens: tuple[ir.When, ...] = ()
    initial_equations: tuple[ir.Equal, ...] = ()
    candidates: tuple[tuple[str, ...], ...] = ()

    @cached_property
    def _by_name(self) -> dict[str, FlatVariable]:
        return {v.name: v for v in self.variables}

    def __getitem__(self, name: str) -> FlatVariable:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    @cached_property
    def derivatives(self) -> tuple[str, ...]:
        """The derivative variable of each state, in the same order as :attr:`states`."""
        return tuple(self.derivative[state] for state in self.states)

    @cached_property
    def parameters(self) -> tuple[FlatVariable, ...]:
        return tuple(v for v in self.variables if v.is_constant)

    @cached_property
    def largest_block(self) -> int:
        """The size of the biggest algebraic loop -- 1 when the model sorts completely."""
        return max((len(block) for block in self.blocks), default=0)

    @cached_property
    def is_explicit(self) -> bool:
        """True when every block is a plain assignment, so no iteration is needed."""
        return all(block.is_linear_scalar for block in self.blocks)
