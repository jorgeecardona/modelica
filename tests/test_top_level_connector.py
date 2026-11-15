"""A flow of a connector declared in the simulated model itself is zero.

Nothing outside the model feeds it. ``OpAmps.DifferentialAmplifier`` and ``CauerLowPassOPV``
declare their own pins, and without this the model had one equation too few.
"""

from __future__ import annotations

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
connector Pin
  Real v;
  flow Real i;
end Pin;
model Resistor
  Pin p;
  Pin n;
  parameter Real R = 2;
equation
  p.v - n.v = R * p.i;
  p.i + n.i = 0;
end Resistor;
model Source
  Pin p;
  Pin n;
equation
  p.v - n.v = 4;
  p.i + n.i = 0;
end Source;
model Ground
  Pin p;
equation
  p.v = 0;
end Ground;
model Circuit
  Pin probe;
  Source source;
  Resistor r;
  Ground ground;
equation
  connect(source.p, r.p);
  connect(r.n, source.n);
  connect(source.n, ground.p);
  connect(probe, r.p);
end Circuit;
"""


def test_a_pin_of_the_model_itself_carries_no_current() -> None:
    result = simulate(causalize(flatten(parse(SOURCE), "Circuit")), stop=1.0, backend="python")
    assert float(result["probe.v"][-1]) == pytest.approx(4.0)
    assert float(result["r.p.i"][-1]) == pytest.approx(2.0)
