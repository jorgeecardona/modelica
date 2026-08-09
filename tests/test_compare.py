"""A debugging tool whose failure mode is untested is one that gets distrusted when it matters.

So the divergences here are *planted*, one block at a time, and the assertion is not "it found
something" but "it found that block and no other". The hard half is the second one: a wrong
value in an early block reaches every later block that reads it, so a tool that reported every
disagreement would be right and useless. Each test below plants at a block with something
downstream of it and checks that the downstream blocks are not what comes back.

Planting is done on the `System`, not on the generated text: one block's assignment is
multiplied by a constant and the model is rebuilt, so both sides are real emitted kernels for
two models that differ in one equation. Patching a string in the source would test the patch.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelica.ir import nodes as ir
from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim.compare import diverges

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _system(stem: str) -> System:
    model = EXAMPLES / "models" / f"{stem}.mo"
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    flatten = pytest.importorskip("modelica.sim").flatten
    sources = [
        parse(
            (EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"),
            str(EXAMPLES / "library" / f"{name}.mo"),
        )
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8"), str(model)))
    return causalize(flatten(sources, stem))


def _bend(system: System, variable: str, shift: float = 1.0) -> tuple[System, int]:
    """Add ``shift`` to the block that computes ``variable``, and say which block that was.

    Added and not scaled. Scaling was the first attempt and it plants nothing where the value
    is zero, which at a model's start point is most of them -- `DCMotor` starts at rest, so
    ``motor.winding.v * 1.001`` is ``0.0`` on both sides and the test passed by finding no
    divergence at all.
    """
    blocks = list(system.mode.blocks)
    for position, block in enumerate(blocks):
        if block.assignment is not None and block.unknowns[0] == variable:
            blocks[position] = dataclasses.replace(
                block,
                assignment=ir.Binary("+", block.assignment, ir.RealLiteral(shift)),
            )
            mode = dataclasses.replace(system.mode, blocks=tuple(blocks))
            return dataclasses.replace(system, modes=(mode, *system.modes[1:])), position
    message = f"no block assigns {variable!r}"
    raise AssertionError(message)


def _spread(system: System, bent: System) -> int:
    """How many blocks the planted value actually reached.

    Without this the "and no other" assertions could be passing because nothing propagated,
    which would make them a test of an easy case dressed as a test of the hard one.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    kernel = build(system)
    p, y, w = kernel.start(0.0)
    here = numpy.zeros(len(kernel.code.observed))
    there = numpy.zeros_like(here)
    kernel.observe(0.0, y, p, w.copy(), here)
    build(bent).observe(0.0, y, p, w.copy(), there)
    slot = {name: index for index, name in enumerate(kernel.code.observed)}
    return sum(
        any(here[slot[name]] != there[slot[name]] for name in block.solved)
        for block in system.mode.blocks
    )


def test_a_tier_does_not_diverge_from_itself() -> None:
    """The negative case, and it comes first.

    A tool that always finds something is not reporting anything. Every corpus model, the
    same kernel on both sides.
    """
    build = pytest.importorskip("modelica.sim.kernel").build
    for model in sorted((EXAMPLES / "models").glob("*.mo")):
        system = _system(model.stem)
        kernel = build(system)
        assert diverges(system, kernel, kernel) is None, model.stem


@pytest.mark.parametrize(
    ("stem", "variable"),
    [
        # The first two have blocks after them that read them, which is the point. The
        # third has none, and is here for the opposite reason: an independent block must not
        # drag its neighbours into the report either.
        ("RoastingMachine", "C_bean"),
        ("DCMotor", "motor.winding.v"),
        ("HeatBar", "der_T[4]"),
    ],
)
def test_the_planted_block_is_the_one_named(stem: str, variable: str) -> None:
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(stem)
    bent, planted = _bend(system, variable)

    found = diverges(system, build(system), build(bent))

    assert found is not None
    assert found.position == planted
    assert variable in found.variables


def test_a_downstream_block_is_not_what_comes_back() -> None:
    """The consequence is not the cause.

    `DCMotor`'s ``motor.winding.v`` is block 0 and later blocks read it, so bending it makes
    several of them disagree. `_spread` counts how many, so this is known to be the hard case
    rather than an easy one that looks like it.
    """
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system("DCMotor")
    bent, planted = _bend(system, "motor.winding.v")

    assert _spread(system, bent) > 1, "nothing propagated, so naming one block proves nothing"

    found = diverges(system, build(system), build(bent))

    assert found is not None
    assert found.position == planted
    assert found.variables == ("motor.winding.v",)


def test_the_report_names_the_equation_and_where_it_was_written() -> None:
    """A block number is not an answer. The equation, in Modelica, and a line to open."""
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system("RoastingMachine")
    bent, _ = _bend(system, "C_bean")

    found = diverges(system, build(system), build(bent))

    assert found is not None
    assert found.equations == ("C_bean = m_dry * cp_dry + m_w * cp_water;",)
    assert found.span is not None
    assert found.span.file.endswith("RoastingMachine.mo")
    assert (
        (EXAMPLES / "models" / "RoastingMachine.mo")
        .read_text(encoding="utf-8")
        .splitlines()[found.span.line - 1]
        .strip()
        .startswith("C_bean =")
    )
    assert "C_bean" in str(found)


def test_a_disagreement_smaller_than_the_tolerance_is_not_one() -> None:
    """The tolerance is the caller's, because what counts as agreement differs per pair.

    Two Python tiers are bit-for-bit; a C tier is not, and a default that only the first pair
    could meet would make the tool useless for the second.
    """
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system("RoastingMachine")
    bent, _ = _bend(system, "C_bean", shift=1e-6)
    left, right = build(system), build(bent)

    assert diverges(system, left, right) is not None
    assert diverges(system, left, right, relative=1e-3, absolute=1e-3) is None


def test_any_callable_with_the_emitted_signature_is_a_tier() -> None:
    """Not only a `Kernel`. The C unit and the FMU exporter will both arrive as wrappers."""
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system("RoastingMachine")
    kernel = build(system)
    slot = kernel.code.observed.index("C_bean")

    def wrong(t: float, y: object, p: object, w: object, out: object) -> object:
        kernel.observe(t, y, p, w, out)  # pyright: ignore[reportArgumentType]
        numpy.asarray(out)[slot] += 1.0
        return out

    found = diverges(system, kernel, wrong)

    assert found is not None
    assert found.variables == ("C_bean",)
    assert found.worst == pytest.approx(1.0)


def test_an_alias_that_differs_is_reported_with_no_block() -> None:
    """A variable no block computes still has to be findable.

    Aliases are recovered after the schedule has run, so two tiers can agree on every block
    and disagree on something a user reads. Reported as a divergence with `position` of
    ``None`` rather than passed over, which is what "no block diverged" would have meant.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system("RCCircuit")
    kernel = build(system)
    aliases = [
        name
        for name in kernel.code.observed
        if name in system.aliases and name not in kernel.code.states
    ]
    assert aliases, "RCCircuit has no aliases, so this test is checking nothing"
    slot = kernel.code.observed.index(aliases[0])

    def wrong(t: float, y: object, p: object, w: object, out: object) -> object:
        kernel.observe(t, y, p, w, out)  # pyright: ignore[reportArgumentType]
        numpy.asarray(out)[slot] += 1.0
        return out

    found = diverges(system, kernel, wrong)

    assert found is not None
    assert found.position is None
    assert found.variables == (aliases[0],)
