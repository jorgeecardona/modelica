"""``b = sample(...); when b then`` fires at the sample instants.

``Blocks.Interfaces.DiscreteBlock`` writes ``sampleTrigger = sample(startTime, samplePeriod)``
and ``when sampleTrigger``, which was refused: "a when-condition has to be a relation".
"""

from __future__ import annotations

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
model Z
  Boolean trigger;
  discrete Real held(start = 0);
  Real x(start = 0, fixed = true);
equation
  der(x) = 1;
  trigger = sample(0.1, 0.25);
  when trigger then
    held = x;
  end when;
end Z;
"""


def test_a_boolean_set_by_sample_triggers_the_when() -> None:
    result = simulate(causalize(flatten(parse(SOURCE), "Z")), stop=1.0, backend="python")
    # Samples at 0.1, 0.35, 0.6, 0.85: held is x at the last one before t.
    assert float(result.at(0.5)["held"]) == pytest.approx(0.35, abs=1e-6)
    assert float(result.at(1.0)["held"]) == pytest.approx(0.85, abs=1e-6)
