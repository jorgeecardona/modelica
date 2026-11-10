"""Clocked partitions in flatten (plan step 6b, commit 1): data and inference, no tick yet."""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.causalize import CausalizationError
from modelica.ir.clocks import ClockError
from modelica.lang import parse
from modelica.sim import causalize, flatten

PI = """\
model PI
  Real x(start = 1, fixed = true);
  Real u;
  Real e;
  Real i(start = 0);
equation
  der(x) = -x + hold(u);
  e = sample(-x, Clock(0.1));
  i = previous(i) + 0.1 * e;
  u = 2 * e + i;
end PI;
"""


def test_a_sampled_controller_is_one_partition() -> None:
    """``sample``, ``previous`` and what reads them form one partition on ``Clock(0.1)``."""
    model = flatten(parse(PI), "PI")
    (partition,) = model.partitions
    assert partition.interval == ir.RealLiteral(0.1)
    assert partition.variables == ("e", "i", "u")
    assert len(model.equations) == 1  # der(x) reads u only through hold(u)
    assert model["u"].variability == "discrete"
    assert model["x"].variability == "continuous"


def test_two_clocks_in_one_partition_are_refused() -> None:
    """Two ``Clock`` intervals meeting through a shared variable is an error, not a guess."""
    source = PI.replace("u = 2 * e + i;", "u = 2 * e + i + sample(x, Clock(0.2));")
    with pytest.raises(ClockError, match="exactly one Clock"):
        flatten(parse(source), "PI")


def test_sub_sample_is_refused() -> None:
    """Clock forms past the minimal scope are refused by name."""
    source = PI.replace("e = sample(-x, Clock(0.1));", "e = subSample(sample(-x, Clock(0.1)), 2);")
    with pytest.raises(ClockError, match="subSample"):
        flatten(parse(source), "PI")


def test_a_clocked_model_is_not_simulated_yet() -> None:
    """Causalize refuses partitions until a tick runs them, rather than hold them at start."""
    with pytest.raises(CausalizationError, match="not simulated yet"):
        causalize(flatten(parse(PI), "PI"))


def test_a_time_sample_is_not_a_clock() -> None:
    """``when sample(start, interval)`` stays a time event: no partition."""
    source = """\
model Pulse
  discrete Real t0(start = 0, fixed = true);
equation
  when sample(0, 0.1) then
    t0 = time;
  end when;
end Pulse;
"""
    assert flatten(parse(source), "Pulse").partitions == ()
