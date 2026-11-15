"""An initial equation that is not an explicit value is solved for its one free state.

``Blocks.Continuous.PI`` with ``initType = InitialOutput`` writes ``y = y_start``, where ``y``
is ``k * (x + u)``: the state ``x`` comes out of it. That was refused: "this initial equation
is not an explicit value for a state".
"""

from __future__ import annotations

import shutil
from typing import Literal

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate
from modelica.sim.emit import EmitError

SOURCE = """\
model PI
  parameter Real k = 2;
  parameter Real y_start = 3;
  Real u;
  Real x;
  Real y;
initial equation
  y = y_start;
equation
  u = 0.5;
  der(x) = u;
  y = k * (x + u);
end PI;
"""


def _skip_without_engine(backend: str) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_an_implicit_initial_equation_solves_for_its_free_state(
    backend: Literal["python", "engine"],
) -> None:
    _skip_without_engine(backend)
    result = simulate(causalize(flatten(parse(SOURCE), "PI")), stop=1.0, backend=backend)
    # y = 2 * (x + 0.5) = 3 gives x(0) = 1.
    assert float(result["x"][0]) == pytest.approx(1.0, abs=1e-9)
    assert float(result["y"][0]) == pytest.approx(3.0, abs=1e-9)
    assert float(result["x"][-1]) == pytest.approx(1.5, abs=1e-6)


def test_a_fixed_state_is_not_the_one_solved_for() -> None:
    source = SOURCE.replace("Real x;", "Real x(start = 0, fixed = true);")
    with pytest.raises(EmitError, match="exactly one free state"):
        simulate(causalize(flatten(parse(source), "PI")), stop=1.0, backend="python")
