"""Crossings that share one root settle once, and stay settled.

At a 50 Hz source zero the thyristor's ``s < 0`` and its timer's ``u`` cross together. Each
settle at that instant kept only its own nudges, so the next found the others again where
they had already acted and flipped them back: "202 events at t = 0.01 without advancing"
(MSL ``PowerConverters.Examples.ACDC.Rectifier1Pulse.Thyristor1Pulse_R``).
"""

from __future__ import annotations

import math

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
model SharedRoot
  Real v = sin(2 * 3.141592653589793 * 50 * time);
  Boolean positive = v > 0;
  Boolean negative = v < 0;
  Boolean off;
  discrete Real entry(start = 0);
  Real x(start = 0, fixed = true);
equation
  off = v < 0 or pre(off) and not positive;
  when negative then
    entry = time;
  end when;
  der(x) = if off then 0 else v;
end SharedRoot;
"""


def test_crossings_at_one_root_do_not_chatter() -> None:
    result = simulate(causalize(flatten(parse(SOURCE), "SharedRoot")), stop=0.05, backend="python")
    # Conducting on each of the three positive half periods: 2 / (100 pi) each.
    assert float(result.at(0.05)["x"]) == pytest.approx(3 * 2 / (100 * math.pi), rel=1e-4)
