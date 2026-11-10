"""`modelica.sim.fv`: heat conduction on a mesh, boundary rules applied by opcodes (step 11)."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from modelica.lang import parse
from modelica.sim import causalize, flatten, fv, mesh, vm

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _program(
    box: mesh.Mesh, rules: Mapping[str, fv.Rule]
) -> tuple[vm.Program, vm.Function, list[float]]:
    """A program whose ``rhs`` is `fv.heat` on ``box``, and the ``p`` it reads.

    Built on Lorenz's program for its other tapes. The new registers go past the old file, and
    the constant pool is padded down over them so every constant stays where it was read.
    """
    source = (EXAMPLES / "models" / "Lorenz.mo").read_text(encoding="utf-8")
    base = vm.lower(causalize(flatten(parse(source), "Lorenz")))
    fresh = [base.registers]

    def register(size: int) -> int:
        first = fresh[0]
        fresh[0] += size
        return first

    tape = fv.heat(box, 2.5, rules, register)
    function = vm.Function(tape.code, (), tape.tables)
    pool = base.constants
    program = dataclasses.replace(
        base,
        rhs=(function,),
        registers=fresh[0] + len(pool),
        constants=(*pool, *(0.0,) * (fresh[0] - base.registers), *pool),
    )
    return program, function, list(tape.parameters)


def _steady(box: mesh.Mesh, rules: Mapping[str, fv.Rule]) -> NDArray[np.float64]:
    """The steady state of the tape, ``A T + b = 0``, with ``A`` and ``b`` read from it."""
    program, function, p = _program(box, rules)
    n = box.cells

    def rhs(t: NDArray[np.float64]) -> NDArray[np.float64]:
        out = [0.0] * n
        vm.evaluate(program, function, 0.0, list(t), p, [], out)
        return np.array(out)

    b = rhs(np.zeros(n))
    a = np.column_stack([rhs(np.eye(n)[j]) - b for j in range(n)])
    return np.linalg.solve(a, -b)


BOX = mesh.box((10, 4, 1), (1.0, 0.4, 0.1))
WALLS: dict[str, fv.Rule] = {side: fv.ZeroGradient() for side in ("ymin", "ymax", "zmin", "zmax")}


def test_two_fixed_walls_give_the_linear_profile() -> None:
    """100 at x = 0 and 20 at x = 1, insulated elsewhere: ``T = 100 - 80 x`` at cell centres."""
    rules = {"xmin": fv.FixedValue(100.0), "xmax": fv.FixedValue(20.0), **WALLS}
    steady = _steady(BOX, rules)
    np.testing.assert_allclose(steady, 100.0 - 80.0 * BOX.cell_centres[:, 0], rtol=0, atol=1e-8)


def test_a_fixed_gradient_wall_gives_the_same_profile() -> None:
    """A gradient of -80 out of the x = 1 wall in place of its fixed value: the same line."""
    rules = {"xmin": fv.FixedValue(100.0), "xmax": fv.FixedGradient(-80.0), **WALLS}
    steady = _steady(BOX, rules)
    np.testing.assert_allclose(steady, 100.0 - 80.0 * BOX.cell_centres[:, 0], rtol=0, atol=1e-8)


def test_every_patch_needs_a_rule() -> None:
    """A patch with no rule is an error, never a default."""
    with pytest.raises(ValueError, match="xmax"):
        fv.heat(BOX, 1.0, {"xmin": fv.ZeroGradient(), **WALLS}, lambda _: 0)


def test_the_engine_runs_the_tape_as_python_does() -> None:
    """The fixed-value and fixed-gradient tape gives the same bits in the engine."""
    from modelica.sim.engine import installed, load  # noqa: PLC0415 - only this test

    if installed() is None and shutil.which("rustc") is None:
        pytest.skip("no engine in this install and no rustc to build one")
    rules = {"xmin": fv.FixedValue(100.0), "xmax": fv.FixedGradient(-80.0), **WALLS}
    program, function, p = _program(BOX, rules)
    t = np.linspace(20.0, 100.0, BOX.cells)
    expected = [0.0] * BOX.cells
    vm.evaluate(program, function, 0.0, list(t), p, [], expected)
    loaded = load(program)
    out = np.zeros(BOX.cells)
    loaded.call(loaded.encoded.rhs(0), 0.0, t, np.array(p), np.zeros(0), out)
    loaded.close()
    np.testing.assert_array_equal(out, expected)


def test_the_jacobian_of_a_mesh_tape_is_its_matrix() -> None:
    """`vm._Tangents` differentiates `fv.heat` through its scatter-adds: ``J = V^-1 A``."""
    source = (EXAMPLES / "models" / "Lorenz.mo").read_text(encoding="utf-8")
    base = vm.lower(causalize(flatten(parse(source), "Lorenz")))
    lowering = vm._Lowering(None)  # type: ignore[arg-type] # noqa: SLF001 - no emitter is read
    lowering.registers = base.registers
    for index, value in enumerate(base.constants):
        lowering.constants.setdefault(value, index)
    box = mesh.box((6, 3, 1), (1.0, 0.5, 0.1))
    rules = {"xmin": fv.FixedValue(100.0), "xmax": fv.FixedGradient(-80.0), **WALLS}
    tape = fv.heat(box, 2.5, rules, lowering.register)
    rhs = vm.Function(tape.code, (), tape.tables)
    n = box.cells
    tangents = vm._Tangents(lowering, n)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    jacobian = tangents.function(rhs)
    pool = tuple(sorted(lowering.constants, key=lowering.constants.__getitem__))
    program = dataclasses.replace(
        base,
        rhs=(rhs,),
        jacobian=(jacobian,),
        registers=lowering.registers + len(pool),
        constants=pool,
    )
    p = list(tape.parameters)

    def evaluate(function: vm.Function, t: NDArray[np.float64], size: int) -> NDArray[np.float64]:
        out = [0.0] * size
        vm.evaluate(program, function, 0.0, list(t), p, [], out)
        return np.array(out)

    t = np.linspace(20.0, 100.0, n)
    b = evaluate(rhs, np.zeros(n), n)
    matrix = np.column_stack([evaluate(rhs, np.eye(n)[j], n) - b for j in range(n)])
    got = evaluate(jacobian, t, n * n).reshape(n, n)
    np.testing.assert_allclose(got, matrix, rtol=1e-13, atol=1e-12)
