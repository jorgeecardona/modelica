"""A variable a ``when`` assigns is discrete, written ``discrete`` or not.

``Modelica.Blocks.Sources.Pulse`` declares ``Integer count`` and ``Real T_start`` and sets
them in a ``when``. Read as continuous, nothing determined them between events and 21 MSL
examples stopped as structurally singular.
"""

from __future__ import annotations

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

COUNTER = """\
model W
  {kind} count(start = 0);
  Real y;
  Real x(start = 1);
equation
  der(x) = -x;
  when time >= 0.5 then
    count = pre(count) + 1;
    reinit(x, 1);
  end when;
  y = count * 2.0;
end W;
"""


@pytest.mark.parametrize("kind", ["Integer", "Real"])
def test_a_when_assigned_variable_steps_at_the_event(kind: str) -> None:
    flat = flatten(parse(COUNTER.format(kind=kind)), "W")
    variability = {v.name: v.variability for v in flat.variables}
    assert variability["count"] == "discrete"
    assert variability["x"] == "continuous"  # reinit is not an assignment
    result = simulate(causalize(flat), stop=1.0)
    assert float(result["y"][0]) == 0.0
    assert float(result["y"][-1]) == 2.0


def test_a_relation_outside_a_when_stays_solved() -> None:
    source = "model M\n  Boolean off;\n  Real u = time - 0.5;\nequation\n  off = u < 0;\nend M;\n"
    flat = flatten(parse(source), "M")
    assert "off" not in {v.name for v in flat.variables if v.variability == "discrete"}
    causalize(flat)
