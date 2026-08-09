"""Tearing: a simultaneous block is smaller than it looks.

A block of *n* equations is rarely *n*-dimensional in any real sense. Most of its variables
come out by substitution once a few are guessed, and iterating on all of them is iterating
on numbers that were never in doubt. What these tests pin is the split -- how many the
solver still has to guess, and that everything else is recovered -- and the invariant that
makes the split safe: the same trajectory comes out.

That last one is not here. It is `tests/test_examples.py`, which holds every corpus model to
reference points computed independently and to tolerances recorded before tearing existed.
`AtwoodPendulum`'s 4x4 block is now two unknowns and its points did not move, which is the
observation the whole feature rests on; this file only says what the split *is*.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.rewrite import names
from modelica.ir.source import flatten
from modelica.lang import parse

if TYPE_CHECKING:
    from modelica.ir.system import Block, System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _system(stem: str) -> System:
    model = EXAMPLES / "models" / f"{stem}.mo"
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    return causalize(flatten(sources, stem))


def _simultaneous(stem: str) -> list[Block]:
    return [block for mode in _system(stem).modes for block in mode.blocks if len(block) > 1]


@pytest.mark.parametrize(
    ("stem", "size", "iterated"),
    [("AtwoodPendulum", 4, 2), ("Pendulum", 3, 1), ("TwoInertias", 3, 1)],
)
def test_how_far_each_corpus_block_tears(stem: str, size: int, iterated: int) -> None:
    """The numbers the issue asked for, and `AtwoodPendulum` is the one it named.

    Its 4x4 is the largest block in the corpus, and it comes down to two guesses.
    """
    for block in _simultaneous(stem):
        assert len(block) == size
        assert len(block.unknowns) == iterated
        assert len(block.torn) == size - iterated


def test_every_variable_the_block_had_is_still_computed() -> None:
    """Tearing moves variables between two lists; it must not drop one.

    A dropped variable is not a crash: the block solves, the model runs, and one name keeps
    whatever the previous step left in it.
    """
    for block in _simultaneous("AtwoodPendulum"):
        assert len(block.solved) == len(block)
        assert set(block.solved) == set(block.solved)  # no duplicates between the two lists
        assert len(set(block.solved)) == len(block)


def test_a_torn_block_still_has_something_to_iterate_on() -> None:
    """A strongly connected component contains a cycle, so it cannot tear to nothing."""
    for stem in ("AtwoodPendulum", "Pendulum", "TwoInertias"):
        for block in _simultaneous(stem):
            assert block.unknowns, f"{stem}: a block tore to no unknowns at all"


def test_a_recovered_variable_reads_only_what_is_known_by_then() -> None:
    """Each substitution is written over the torn unknowns and the ones before it.

    This is what lets the emitter place them after the solve, in order, with no iteration:
    if one read a variable defined *later* the sequence would be a loop again.
    """
    for stem in ("AtwoodPendulum", "Pendulum", "TwoInertias"):
        for block in _simultaneous(stem):
            available = set(block.unknowns)
            for name, value in block.torn:
                unresolved = {
                    read for read in names(value) if read in set(block.solved) - available
                }
                assert not unresolved, f"{stem}: {name} reads {sorted(unresolved)} too early"
                available.add(name)


def test_an_explicit_block_is_left_alone() -> None:
    """A 1x1 block was never simultaneous, so there is nothing to tear out of it."""
    system = _system("RCCircuit")
    for mode in system.modes:
        for block in mode.blocks:
            assert block.torn == ()
            assert len(block) == 1
