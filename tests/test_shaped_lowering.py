"""Array loops with each construct the flat array path refused now lower to shaped opcodes.

Each model's loop is written with views, and every run is bit for bit the scalar tape's.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from modelica.lang import parse
from modelica.sim import causalize, flatten, simulate, vm
from modelica.sim.solve import SimulationError

if TYPE_CHECKING:
    from modelica.ir.system import System

TEMPLATE = """
model Loop
  parameter Integer n = 6;
  parameter Real k = 0.7;
  Real x[n](start = {S});
equation
  for i in {RANGE} loop
    der(x[i]) = {BODY};
  end for;
{REST}end Loop;
"""

#: One loop body per construct, with what the shaped tape must hold.
CASES: dict[str, tuple[str, vm.Op]] = {
    "if": ("if x[i] > 0.2 then -k * x[i] else -2 * k * x[i]", vm.Op.TSELECT),
    "comparison": ("-x[i] * (1 + (x[i] >= 0.5))", vm.Op.TBINARY),
    "power": ("-k * x[i] ^ 3", vm.Op.TBINARY),
    "max": ("-max(x[i], 0.3) * k", vm.Op.TCALL2),
    "mod": ("-x[i] + mod(i, 3)", vm.Op.TCALL2),
    "index": ("-k * i * x[i]", vm.Op.TINDEX),
    "not": ("if not (x[i] < 0.4) then -x[i] else -0.5 * x[i]", vm.Op.TUNARY),
}


def _system(body: str) -> System:
    source = TEMPLATE.format(S="{1.0, 0.8, 0.6, 0.4, 0.2, 0.1}", RANGE="1:n", BODY=body, REST="")
    return causalize(flatten([parse(source, "Loop.mo")], "Loop"))


def _rhs(program: vm.Program, y: list[float]) -> list[float]:
    p = [0.0] * len(program.layout.parameters)
    vm.evaluate(program, program.parameters, 0.0, [], [], [], p)
    out = [0.0] * len(y)
    vm.evaluate(program, program.rhs[0], 0.0, y, p, [0.0] * program.layout.workspace, out)
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_loop_is_shaped_and_has_the_scalar_bits(case: str) -> None:
    body, expected = CASES[case]
    system = _system(body)
    shaped, scalar = vm.lower(system), vm.lower(system, arrays=False)
    ops = {i.op for i in shaped.rhs[0].code}
    assert expected in ops, vm.listing(shaped, shaped.rhs[0])
    y = [0.9, -0.3, 0.45, 0.6, 0.15, 0.25]
    assert _rhs(shaped, y) == _rhs(scalar, y)


#: A relation inside arithmetic has no derivative, so neither backend's jacobian takes it.
_NOT_DIFFERENTIABLE: Final = frozenset({"comparison"})


@pytest.mark.parametrize("case", sorted(set(CASES) - _NOT_DIFFERENTIABLE))
def test_the_engine_and_scipy_agree_on_the_loop(case: str) -> None:
    from modelica.sim import engine  # noqa: PLC0415 - only this test needs the engine

    if engine.installed() is None and shutil.which("rustc") is None:
        pytest.skip("no engine in this install and no rustc to build one")
    system = _system(CASES[case][0])
    try:
        on_engine = simulate(system, stop=1.0, points=11, tolerance=1e-9, backend="engine")
    except SimulationError as error:  # pragma: no cover - a refusal is a failure here
        pytest.fail(str(error))
    on_scipy = simulate(system, stop=1.0, points=11, tolerance=1e-9, backend="python")
    np.testing.assert_allclose(on_engine.values, on_scipy.values, rtol=1e-6, atol=1e-8)


def test_an_index_from_two_moves_to_its_start() -> None:
    """``for i in 2:n`` reads the index as ``i``, not its position along the loop."""
    source = TEMPLATE.format(
        S="{1.0, 0.8, 0.6, 0.4, 0.2, 0.1}",
        RANGE="2:n",
        BODY="-i * x[i]",
        REST="  der(x[1]) = -x[1];\n",
    )
    system = causalize(flatten([parse(source, "Loop.mo")], "Loop"))
    shaped, scalar = vm.lower(system), vm.lower(system, arrays=False)
    y = [0.9, -0.3, 0.45, 0.6, 0.15, 0.25]
    assert _rhs(shaped, y) == _rhs(scalar, y)


PLATE = """
model Plate "Heat on a square plate: the edges cool on their own, the interior diffuses"
  parameter Integer n = 7;
  parameter Real a = 1.5;
  Real T[n, n];
initial equation
  for i in 1:n loop
    for j in 1:n loop
      T[i, j] = sin(3.14159 * (i - 1) / (n - 1)) * sin(3.14159 * (j - 1) / (n - 1));
    end for;
  end for;
equation
  for i in 2:n-1 loop
    for j in 2:n-1 loop
      der(T[i, j]) = a * (T[i - 1, j] + T[i + 1, j] + T[i, j - 1] + T[i, j + 1] - 4 * T[i, j]);
    end for;
  end for;
  for j in 1:n loop
    der(T[1, j]) = -T[1, j];
    der(T[n, j]) = -T[n, j];
  end for;
  for i in 2:n-1 loop
    der(T[i, 1]) = -T[i, 1];
    der(T[i, n]) = -T[i, n];
  end for;
end Plate;
"""


def _plate() -> System:
    return causalize(flatten([parse(PLATE, "Plate.mo")], "Plate"))


def test_an_interior_stencil_reads_slices_with_no_table() -> None:
    """``T[i - 1, j]`` over ``i in 2:n-1`` is a view with a moved base: no gather, no roll."""
    system = _plate()
    shaped, scalar = vm.lower(system), vm.lower(system, arrays=False)
    function = shaped.rhs[0]
    ops = [i.op for i in function.code]
    assert vm.Op.TBINARY in ops, vm.listing(shaped, function)
    assert vm.Op.ROLL not in ops
    inner = [v for v in function.views if v.shape == (5, 5) and v.strides != (0, 0)]
    assert len({v.base for v in inner}) >= 5  # the centre and four neighbours, one block
    y = np.random.default_rng(3).normal(size=len(system.states)).tolist()
    assert _rhs(shaped, y) == _rhs(scalar, y)


def test_the_plate_runs_the_same_on_the_engine_and_on_scipy() -> None:
    from modelica.sim import engine  # noqa: PLC0415 - only this test needs the engine

    if engine.installed() is None and shutil.which("rustc") is None:
        pytest.skip("no engine in this install and no rustc to build one")
    system = _plate()
    on_engine = simulate(system, stop=0.2, points=5, tolerance=1e-9, backend="engine")
    on_scipy = simulate(system, stop=0.2, points=5, tolerance=1e-9, backend="python")
    np.testing.assert_allclose(on_engine.values, on_scipy.values, rtol=1e-6, atol=1e-9)


def _dense_jacobian(program: vm.Program, y: list[float]) -> np.ndarray:
    n = len(y)
    p = [0.0] * len(program.layout.parameters)
    vm.evaluate(program, program.parameters, 0.0, [], [], [], p)
    out = [0.0] * len(program.pattern)
    vm.evaluate(program, program.jacobian[0], 0.0, y, p, [0.0] * program.layout.workspace, out)
    matrix = np.zeros((n, n))
    for k, (row, column) in enumerate(program.pattern):
        matrix[row, column] = out[k]
    return matrix


def test_the_jacobian_stores_each_stencil_offset_in_one_instruction() -> None:
    """TorusHeat's five offsets, 64 values each, sit side by side: one ``VSTORE_OUT`` of 320."""
    torus = Path(__file__).resolve().parent.parent / "examples" / "models" / "TorusHeat.mo"
    system = causalize(flatten([parse(torus.read_text(), str(torus))], "TorusHeat"))
    shaped, scalar = vm.lower(system), vm.lower(system, arrays=False)
    code = shaped.jacobian[0].code
    stores = [i for i in code if i.op is vm.Op.VSTORE_OUT]
    assert [i.c for i in stores] == [5 * 64]
    assert not any(i.op is vm.Op.STORE_OUT for i in code)
    assert len(code) < len(scalar.jacobian[0].code) // 20
    y = np.random.default_rng(4).normal(size=len(system.states)).tolist()
    np.testing.assert_array_equal(_dense_jacobian(shaped, y), _dense_jacobian(scalar, y))


def test_a_product_of_two_runs_of_one_state_adds_both_partials() -> None:
    """``r u (1 - u)`` in FisherKPP: d/du is ``r (1 - u) - r u``, both terms, in every cell."""
    model = Path(__file__).resolve().parent.parent / "examples" / "models" / "FisherKPP.mo"
    system = causalize(flatten([parse(model.read_text(), str(model))], "FisherKPP"))
    shaped, scalar = vm.lower(system), vm.lower(system, arrays=False)
    y = np.random.default_rng(7).random(len(system.states)).tolist()
    np.testing.assert_array_equal(_dense_jacobian(shaped, y), _dense_jacobian(scalar, y))


BOX3D = """
model Box "Heat in an insulated cube: a neighbour past a face is the cell itself"
  parameter Integer n = 13;
  parameter Real a = 2.5 * n ^ 2;
  Real T[n, n, n];
initial equation
  for i in 1:n loop
    for j in 1:n loop
      for k in 1:n loop
        T[i, j, k] = i + 0.5 * j - 0.25 * k;
      end for;
    end for;
  end for;
equation
  for i in 1:n loop
    for j in 1:n loop
      for k in 1:n loop
        der(T[i, j, k]) = a * (T[max(i - 1, 1), j, k] + T[min(i + 1, n), j, k]
          + T[i, max(j - 1, 1), k] + T[i, min(j + 1, n), k]
          + T[i, j, max(k - 1, 1)] + T[i, j, min(k + 1, n)] - 6 * T[i, j, k]);
      end for;
    end for;
  end for;
end Box;
"""


def test_a_3d_box_matches_the_finite_volume_tape_and_solves_iteratively() -> None:
    """``Real T[n, n, n]`` with zero-gradient faces is `fv.heat` on `mesh.box`, cell for cell.

    Cell ``(i, j, k)`` of the box is number ``i + n (j + n k)``, which is ``T[k, j, i]`` in
    row-major order; the cube is symmetric, so the two orders are one.
    """
    from modelica.sim import fv, mesh  # noqa: PLC0415
    from test_fv import (  # noqa: PLC0415 - one test reuses its program builder
        _program,  # pyright: ignore[reportUnknownVariableType, reportPrivateUsage]
    )

    system = causalize(flatten([parse(BOX3D, "Box.mo")], "Box"))
    program = vm.lower(system)
    assert program.iterative
    cube = mesh.box((13, 13, 13))
    rules = {patch.name: fv.ZeroGradient() for patch in cube.patches}
    reference, function, p = _program(cube, rules)  # pyright: ignore[reportUnknownVariableType]
    y = np.random.default_rng(5).normal(size=13**3).tolist()
    expected = [0.0] * len(y)
    vm.evaluate(reference, function, 0.0, y, p, [], expected)  # pyright: ignore[reportUnknownArgumentType]
    np.testing.assert_allclose(_rhs(program, y), expected, rtol=1e-12, atol=1e-9)
    # The clamped reads name the centre state at a face; the jacobian adds both partials there.
    scalar = vm.lower(system, arrays=False)
    assert len(program.jacobian[0].code) < len(scalar.jacobian[0].code) // 10
    np.testing.assert_array_equal(_dense_jacobian(program, y), _dense_jacobian(scalar, y))
