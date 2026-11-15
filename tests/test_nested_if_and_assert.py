"""A run-time if-equation may hold an if-equation in a branch; ``assert`` assigns nothing.

``Electrical.PowerConverters.DCDC.Examples.DemoPowerSupply`` needs both: ``DCPowerSupply`` nests
an if-equation in a run-time one, and its algorithm checks a value with ``assert``.
"""

from __future__ import annotations

import math
import shutil
from typing import Literal

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

NESTED = """\
model M
  parameter Boolean fast = true;
  Real x(start = -1, fixed = true);
  Real y;
equation
  der(x) = 1;
  if x < 0 then
    y = 0;
  else
    if fast then
      y = 2 * x;
    else
      y = x;
    end if;
  end if;
end M;
"""

ASSERTED = """\
model A
  Real x(start = 1, fixed = true);
  Real y;
equation
  der(x) = -x;
algorithm
  assert(x >= 0, "x went negative");
  y := 2 * x;
end A;
"""


def _skip_without_engine(backend: str) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")


@pytest.mark.parametrize("backend", ["python", "engine"])
@pytest.mark.parametrize(("fast", "last"), [("true", 2.0), ("false", 1.0)])
def test_a_branch_may_hold_a_structural_if(
    backend: Literal["python", "engine"], fast: str, last: float
) -> None:
    _skip_without_engine(backend)
    source = NESTED.replace("fast = true", f"fast = {fast}")
    result = simulate(causalize(flatten(parse(source), "M")), stop=2.0, backend=backend)
    assert float(result["y"][0]) == pytest.approx(0.0, abs=1e-9)
    assert float(result["y"][-1]) == pytest.approx(last, abs=1e-6)


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_a_branch_may_hold_a_run_time_if(backend: Literal["python", "engine"]) -> None:
    _skip_without_engine(backend)
    source = NESTED.replace("if fast then", "if x > 0.5 then")
    result = simulate(causalize(flatten(parse(source), "M")), stop=2.0, backend=backend)
    assert float(result.at(1.25)["y"]) == pytest.approx(0.25, abs=1e-6)
    assert float(result["y"][-1]) == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_an_assert_statement_assigns_nothing(backend: Literal["python", "engine"]) -> None:
    _skip_without_engine(backend)
    result = simulate(causalize(flatten(parse(ASSERTED), "A")), stop=1.0, backend=backend)
    assert float(result["y"][-1]) == pytest.approx(2 * math.exp(-1), rel=1e-4)
