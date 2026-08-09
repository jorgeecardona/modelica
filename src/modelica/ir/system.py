"""What a model looks like once it has been given a direction of computation.

A :class:`~modelica.ir.flat.FlatModel` is a pile of equations that happen to be true. A
:class:`System` is the same content arranged so a solver can walk it: states first, then
blocks in an order where every block's inputs were computed by an earlier one.

Like :mod:`modelica.ir.flat` this module holds data and nothing else. The pass that
produces it lives in :mod:`modelica.ir.causalize`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from modelica.ir import nodes as ir
    from modelica.ir.flat import FlatVariable

__all__ = ["Block", "FrozenMap", "Mode", "System"]

_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenMap(Mapping[_K, _V]):
    """A mapping that is a value: immutable, comparable, and hashable.

    Two fields of this module are mappings -- :attr:`Mode.derivative` and
    :attr:`System.aliases` -- and a ``dict`` in a frozen dataclass makes the whole dataclass
    unhashable. That is not a formality. `docs/ideas/backends.md:70` promises "a ``FlatModel``
    has a content hash and needs no extra machinery to get one. That hash is the cache key for
    the compiled kernel", and `System` sits between `FlatModel` and `Code`, both of which
    hashed while it did not -- so the key was available at both ends of the pipeline and not
    in the middle.

    A tuple of pairs is what `ir.Call.named` does for the same reason and it was the obvious
    move. It is not the one taken: ten places read these two fields with ``[]``, ``.items()``
    and ``in``, and turning a mapping into a sequence to win a hash would make every one of
    them say ``dict(...)`` first. This keeps the reading and fixes the hashing.

    Equality is inherited and untouched: `collections.abc.Mapping` already compares two
    mappings by content, so ``system.aliases == {...}`` in a test goes on reading the way it
    reads. Only ``__hash__`` is added -- ``Mapping`` sets it to ``None``, and that single
    ``None`` is what made a whole `System` unhashable.

    The hash is over a ``frozenset`` of the pairs, so it agrees with that inherited equality:
    two maps with the same pairs in a different order are equal and therefore must hash the
    same. The cost is stated rather than hidden. Iteration order *is* visible downstream --
    the emitter writes aliases out in it -- so two systems differing only in the order of
    otherwise identical aliases would compare equal and share a cache entry while generating
    different source. Causalization is deterministic, so that pair cannot arise from the
    pipeline; it is a hole in the argument and not one in the behaviour.
    """

    __slots__ = ("_hash", "_lookup")

    def __init__(self, items: Mapping[_K, _V] | None = None) -> None:
        self._lookup: dict[_K, _V] = {} if items is None else dict(items)
        self._hash: int = hash(frozenset(self._lookup.items()))

    def __getitem__(self, key: _K) -> _V:
        return self._lookup[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self._lookup)

    def __len__(self) -> int:
        return len(self._lookup)

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"FrozenMap({self._lookup!r})"


def _no_aliases() -> FrozenMap[str, ir.Expr]:
    return FrozenMap()


@dataclass(frozen=True)
class Block:
    """One diagonal entry of the block-lower-triangular form.

    ``unknowns`` and ``equations`` always have the same length: a block is square by
    construction, being one strongly connected component of the dependency graph. Most
    blocks are 1x1 and carry an ``assignment`` -- the unknown written explicitly in terms
    of things already known, which is the whole point of sorting. A block without an
    assignment has to be handed to a numerical solver, and its size says how bad that is.

    ``pivot`` is ``d residual / d unknown`` for a 1x1 block: the number the solve divides
    by, kept as an expression so it can be evaluated along the trajectory. Where it passes
    through zero the block is unsolvable no matter how the arithmetic is arranged, and that
    is not a defect of the block but a fact about the state selection that produced it --
    which is why :class:`Mode` collects them.
    """

    unknowns: tuple[str, ...]
    equations: tuple[ir.Equal, ...]
    assignment: ir.Expr | None = None
    pivot: ir.Expr | None = None
    torn: tuple[tuple[str, ir.Expr], ...] = ()
    """Variables this block computes by substitution once its ``unknowns`` are known.

    Tearing splits a simultaneous block in two. What stays in ``unknowns`` and ``equations``
    is the part a solver still has to iterate on; what moves here is the part that follows
    from it by forward substitution, in the order given, each expression written over the
    torn unknowns and whatever was already known. ``len(unknowns)`` is therefore the size of
    the *numerical* problem, and ``len(self)`` is the size of the block as causalization
    found it.

    Empty for a block that was not torn, which is every 1x1 block and any block where no
    variable came out.
    """

    @property
    def solved(self) -> tuple[str, ...]:
        """Every variable this block computes: the ones it iterates on, then the torn ones."""
        return (*self.unknowns, *(name for name, _ in self.torn))

    @property
    def is_linear_scalar(self) -> bool:
        """True when this block reduces to a single explicit assignment."""
        return self.assignment is not None

    def __len__(self) -> int:
        """How many variables the block computes, torn or not -- its size before tearing."""
        return len(self.unknowns) + len(self.torn)


@dataclass(frozen=True)
class Mode:
    """One admissible state selection, and the computation that follows from it.

    An index-reduced model usually has more than one set of variables it could integrate,
    and the sets are not interchangeable: each one causalizes to its own blocks, and each
    one breaks somewhere different. The pendulum has two, ``(y, vy)`` and ``(x, vx)``, and
    a full swing needs both -- the first cannot get through the bottom, the second cannot
    get through the sides. A mode is that whole package, not just the names.

    ``pivot`` is the product of the 1x1 blocks' pivots: the mode is usable where that is far
    from zero, and two modes compare by comparing theirs. For the pendulum it comes out
    ``4x²`` against ``4y²``, so the switch is at ``|x| = |y|``, which is 45 degrees, which is
    the answer the literature gives. It is ``None`` when the model has only one mode, since
    a conditioning number with nothing to compare against has no reader.
    """

    states: tuple[str, ...]
    derivative: Mapping[str, str]
    blocks: tuple[Block, ...]
    pivot: ir.Expr | None = None

    def __post_init__(self) -> None:
        """Whatever mapping was handed in becomes a `FrozenMap`, so the mode is hashable.

        Coerced rather than demanded, because the type stays `Mapping` and every caller goes
        on passing a dict. A hole that has to be remembered is a hole.
        """
        object.__setattr__(self, "derivative", FrozenMap(self.derivative))


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

    ``modes`` holds every admissible state selection, the chosen one first, each already
    causalized into its own blocks. Most models have exactly one and the plural never comes
    up. Index-reduced ones have several, and which of them is numerically usable changes
    along the trajectory, so the list is the menu a runtime switches between rather than a
    curiosity.

    Admissible means two things: the remaining equations determine everything the set does
    not, and the set is closed under integration -- if ``vy`` is a state and the model knows
    ``vy`` is ``der(y)``, then ``y`` is a state too. The second is not pedantry. Integrating
    ``vy`` while reading ``y`` off a constraint leaves nothing tying the two together, since
    ``der(y) = vy`` was spent making them one variable, and the trajectory drifts while
    satisfying every equation. Sets that fail either test never reach this list.

    :attr:`states`, :attr:`derivative` and :attr:`blocks` read the first mode. A caller that
    does not care about switching -- which is most of them, since most models have one mode
    -- can go on treating a system as a flat list of blocks and never learn the word.
    """

    name: str
    variables: tuple[FlatVariable, ...]
    modes: tuple[Mode, ...]
    index: int = 1
    aliases: Mapping[str, ir.Expr] = field(default_factory=_no_aliases)
    whens: tuple[ir.When, ...] = ()
    initial_equations: tuple[ir.Equal, ...] = ()

    def __post_init__(self) -> None:
        """See `Mode.__post_init__`: the one dict left between `FlatModel` and `Code`."""
        object.__setattr__(self, "aliases", FrozenMap(self.aliases))

    @property
    def mode(self) -> Mode:
        """The chosen selection. Every model has one; only some have alternatives."""
        return self.modes[0]

    @property
    def states(self) -> tuple[str, ...]:
        return self.modes[0].states

    @property
    def derivative(self) -> Mapping[str, str]:
        return self.modes[0].derivative

    @property
    def blocks(self) -> tuple[Block, ...]:
        return self.modes[0].blocks

    @cached_property
    def candidates(self) -> tuple[tuple[str, ...], ...]:
        """The state sets, :attr:`states` first -- :attr:`modes` with the detail dropped."""
        return tuple(mode.states for mode in self.modes)

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
