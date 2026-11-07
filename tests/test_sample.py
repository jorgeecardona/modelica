"""``when sample(start, interval)`` fires at ``start + k * interval`` (#1)."""

from __future__ import annotations

import pytest

from modelica.lang import parse
from modelica.sim import causalize, flatten, simulate

np = pytest.importorskip("numpy")

PULSE = """\
model Pulse
  parameter Real f0 = 120;
  discrete Real tStart(start = 0, fixed = true);
equation
  when sample(0, 1 / f0) then
    tStart = time;
  end when;
end Pulse;
"""


def test_sample_lowers_to_a_time_relation() -> None:
    """Flattening replaces the ``sample`` condition with ``time >= _sample1``."""
    model = flatten(parse(PULSE), "Pulse")
    assert "_sample1" in model
    assert model["_sample1"].variability == "discrete"
    condition = model.whens[0].branches[0].condition
    assert getattr(condition, "op", None) == ">="


def test_sample_fires_once_per_interval() -> None:
    """Over 0.05 s at 120 Hz the clause fires at 0, 1/120, ..., 6/120."""
    result = simulate(causalize(flatten(parse(PULSE), "Pulse")), stop=0.05)
    fired = np.unique(np.round(result["tStart"], 9))
    np.testing.assert_allclose(fired, [k / 120 for k in range(7)], atol=1e-9)
