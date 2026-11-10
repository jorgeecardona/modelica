"""`modelica.sim.fem`: P1 triangles as a tape, with `vm.spmv` (plan step 14)."""

from __future__ import annotations

import dataclasses
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from modelica.lang import parse
from modelica.sim import causalize, fem, flatten, vm

if TYPE_CHECKING:
    from numpy.typing import NDArray

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _exact(x: float, y: float) -> float:
    return math.sin(math.pi * x) * math.sin(math.pi * y)


def _source(x: float, y: float) -> float:
    """``-laplacian(u)`` for `_exact`: the steady state of ``u' = laplacian(u) + s``."""
    return 2.0 * math.pi**2 * _exact(x, y)


def _program(cells: int) -> tuple[vm.Program, vm.Function, fem.Tape]:
    """Lorenz's program with ``rhs`` replaced by `fem.heat` on a ``cells x cells`` square."""
    source = (EXAMPLES / "models" / "Lorenz.mo").read_text(encoding="utf-8")
    base = vm.lower(causalize(flatten(parse(source), "Lorenz")))
    fresh = [base.registers]

    def register(size: int) -> int:
        first = fresh[0]
        fresh[0] += size
        return first

    tape = fem.heat(fem.square(cells), _source, register)
    function = vm.Function(tape.code, (), tape.tables)
    pool = base.constants
    program = dataclasses.replace(
        base,
        rhs=(function,),
        registers=fresh[0] + len(pool),
        constants=(*pool, *(0.0,) * (fresh[0] - base.registers), *pool),
    )
    return program, function, tape


def _steady_error(cells: int) -> float:
    """The largest nodal error of the steady state ``K u = M s`` read from the tape."""
    program, function, tape = _program(cells)
    p, n = list(tape.parameters), len(tape.unknowns)

    def rhs(u: NDArray[np.float64]) -> NDArray[np.float64]:
        out = [0.0] * n
        vm.evaluate(program, function, 0.0, list(u), p, [], out)
        return np.array(out)

    b = rhs(np.zeros(n))
    a = np.column_stack([rhs(np.eye(n)[j]) - b for j in range(n)])
    steady = np.linalg.solve(a, -b)
    mesh = fem.square(cells)
    exact = np.array([_exact(*mesh.points[point]) for point in tape.unknowns])
    return float(np.max(np.abs(steady - exact)))


def test_the_stiffness_matrix_is_symmetric_and_its_rows_sum_to_zero() -> None:
    """P1 stiffness: symmetric, every row sums to 0, and the lumped mass is the area."""
    assembled = fem.assemble(fem.square(4))
    entries = dict(
        zip(zip(assembled.rows, assembled.columns, strict=True), assembled.values, strict=True)
    )
    for (a, b), value in entries.items():
        assert entries[b, a] == pytest.approx(value, abs=1e-14)
    sums: dict[int, float] = {}
    for (a, _), value in entries.items():
        sums[a] = sums.get(a, 0.0) + value
    assert max(abs(total) for total in sums.values()) < 1e-13
    assert sum(assembled.mass) == pytest.approx(1.0, rel=1e-14)


def test_poisson_on_triangles_converges_at_second_order() -> None:
    """Halving the mesh size divides the nodal error by about 4, against the exact solution."""
    errors = [_steady_error(cells) for cells in (4, 8, 16)]
    rates = [math.log2(errors[k] / errors[k + 1]) for k in range(2)]
    assert errors[-1] < 1e-2
    assert all(rate > 1.9 for rate in rates), (errors, rates)


def test_the_engine_runs_the_fem_tape_as_python_does() -> None:
    """The same bits from the engine as from `vm.evaluate`."""
    from modelica.sim.engine import installed, load  # noqa: PLC0415 - only this test

    if installed() is None and shutil.which("rustc") is None:
        pytest.skip("no engine in this install and no rustc to build one")
    program, function, tape = _program(6)
    n = len(tape.unknowns)
    u = np.linspace(0.0, 1.0, n)
    expected = [0.0] * n
    vm.evaluate(program, function, 0.0, list(u), list(tape.parameters), [], expected)
    loaded = load(program)
    out = np.zeros(n)
    loaded.call(loaded.encoded.rhs(0), 0.0, u, np.array(tape.parameters), np.zeros(0), out)
    loaded.close()
    np.testing.assert_array_equal(out, expected)
