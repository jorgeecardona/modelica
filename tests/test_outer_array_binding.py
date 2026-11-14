"""An array binding written by the enclosing class reaches each element as its own share.

``VoltageBridge2Pulse`` sets ``final vStart = fill(vStart, 1)`` on a block that declares
``vStart[m] = zeros(m)``. Each element carried the whole ``fill(...)`` call, which no backend
translates.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
model A
  parameter Integer m = {m};
  parameter Real v[m] = zeros(m);
  Real y;
equation
  y = v[m];
end A;
model M
  parameter Real v0 = 3;
  A a(final v = fill(v0, {m}));
  A b(v = {{v0 * i for i in 1:{m}}});
end M;
"""


@pytest.mark.parametrize("m", [1, 2])
def test_each_element_gets_its_share(m: int) -> None:
    flat = flatten(parse(SOURCE.format(m=m)), "M")
    bindings = {v.name: v.binding for v in flat.variables}
    assert bindings[f"a.v[{m}]"] == ir.Ref(("v0",))
    result = simulate(causalize(flat), stop=1.0)
    assert float(result["a.y"][-1]) == pytest.approx(3.0)
    assert float(result["b.y"][-1]) == pytest.approx(3.0 * m)
