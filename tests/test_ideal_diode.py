"""An ideal diode's ``off = s < 0`` in an algebraic loop: it builds, and it switches.

``Analog.Examples.Rectifier`` writes the diode this way. Two things stopped it. Tearing could
guess ``off``, which leaves a Newton step on a relation (no slope). And with ``Ron = 1e-5`` the
on-mode is stiff, so Radau stepped over the whole off interval and saw no sign change in
``s``. Now ``off`` is never the guess, and a watched crossing caps the step at the output
spacing.
"""

from __future__ import annotations

import numpy as np
import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
model D
  parameter Real Ron = 1e-5, Goff = 1e-5, R = 1, C = 0.01;
  Real v, i, s, vr(start = 0, fixed = true);
  Boolean off;
equation
  v + vr = sin(6.283185307179586 * time + 0.3);
  i = C * der(vr) + vr / R;
  off = s < 0;
  v = s * (if off then 1 else Ron);
  i = s * (if off then Goff else 1);
end D;
"""


def test_the_diode_switches_off_and_on_each_period() -> None:
    system = causalize(flatten(parse(SOURCE), "D"))
    result = simulate(system, stop=2.0, points=81, backend="python")
    times = [event.time for event in result.events]
    assert len(times) == 4  # off and on, twice
    # The first switch is where C*der(src) + src/R = 0 after the peak: t = 0.4423.
    assert times[0] == pytest.approx(0.4423, abs=1e-3)
    assert float(np.min(result["vr"])) > -1e-3
    assert bool(result.at(0.7)["off"])
