"""A component's own ``input`` connected inside it is not a second source.

``Analog.Basic.Potentiometer`` writes ``connect(r, rInt)`` between two of its own inputs, and
the model outside drives ``r``. That set was refused: "which all write the signal".
"""

from __future__ import annotations

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
connector RealInput = input Real;
connector RealOutput = output Real;
model Gain
  RealInput r;
  RealOutput y;
protected
  RealInput rInt;
equation
  connect(r, rInt);
  y = 2 * rInt;
end Gain;
model Source
  RealOutput y;
equation
  y = 3;
end Source;
model Top
  Source source;
  Gain gain;
equation
  connect(source.y, gain.r);
end Top;
"""


def test_a_component_input_connected_inside_it_reads_the_outer_source() -> None:
    result = simulate(causalize(flatten(parse(SOURCE), "Top")), stop=1.0, backend="python")
    assert float(result["gain.y"][-1]) == pytest.approx(6.0)


def test_two_outputs_on_one_signal_are_still_refused() -> None:
    source = SOURCE.replace(
        "connect(source.y, gain.r);", "connect(source.y, gain.r);\n  connect(gain.y, gain.r);"
    )
    with pytest.raises(FlattenError, match="one source"):
        flatten(parse(source), "Top")
