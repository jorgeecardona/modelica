"""Array loops with each construct the flat array path refused now lower to shaped opcodes.

Each model's loop is written with views, and every run is bit for bit the scalar tape's.
"""

from __future__ import annotations

import shutil
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
