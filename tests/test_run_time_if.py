"""An if-equation whose condition changes during the run is an if-expression per equation.

``FluidHeatFlow`` pipes write ``if volumeFlow > 0 then dp = a; else dp = b; end if;``. The
condition reads a variable, so it cannot pick the equations at flatten time. Each branch holds
the same number of equations, so they pair by position.
"""

from __future__ import annotations

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import parse
from modelica.sim import simulate

SAME_SIDE = """\
model M
  Real x(start = -1, fixed = true);
  Real y;
equation
  der(x) = 1;
  if x > 0 then
    y = 2 * x;
  elseif x > -0.5 then
    y = 0;
  else
    y = -x;
  end if;
end M;
"""


def test_the_same_left_side_becomes_an_if_expression() -> None:
    flat = flatten(parse(SAME_SIDE), "M")
    result = simulate(causalize(flat), stop=2.0)
    assert float(result["y"][0]) == pytest.approx(1.0)
    assert float(result["y"][-1]) == pytest.approx(2.0, abs=1e-6)


def test_different_left_sides_become_a_residual() -> None:
    source = SAME_SIDE.replace("y = 0;", "0 = y;")
    result = simulate(causalize(flatten(parse(source), "M")), stop=2.0)
    assert float(result["y"][-1]) == pytest.approx(2.0, abs=1e-6)


def test_branches_of_unequal_size_are_still_refused() -> None:
    source = SAME_SIDE.replace("y = 0;", "y = 0;\n    x = 0;")
    with pytest.raises(FlattenError, match="must be known when the model is flattened"):
        flatten(parse(source), "M")
