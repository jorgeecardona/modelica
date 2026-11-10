"""Clocked partitions (plan step 6b): inference in flatten, and the tick as a time event."""

from __future__ import annotations

import math
import shutil
from typing import Literal

import pytest

from modelica import ir
from modelica.ir.clocks import ClockError
from modelica.lang import parse
from modelica.sim import causalize, flatten, simulate

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


def _recurrence(stop: float) -> float:
    """The plant ``x' = -x + u`` between ticks, solved exactly, with the controller at each tick."""
    x, i, t = 1.0, 0.0, 0.0
    while t <= stop:
        e = -x
        i += 0.1 * e
        u = 2 * e + i
        step = min(0.1, stop - t)
        x = u + (x - u) * math.exp(-step)
        t = round(t + 0.1, 12)
    return x


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_a_sampled_controller_runs_tick_by_tick(backend: Literal["python", "engine"]) -> None:
    """The PI controller ticks at 0, 0.1, ... and the plant follows the exact recurrence."""
    if backend == "engine":
        from modelica.sim.engine import installed  # noqa: PLC0415 - only this case needs it

        if installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    result = simulate(
        causalize(flatten(parse(PI), "PI")), stop=0.95, tolerance=1e-10, backend=backend
    )
    assert [round(event.time, 9) for event in result.events] == [k / 10 for k in range(10)]
    assert result["x"][-1] == pytest.approx(_recurrence(0.95), abs=1e-12)


def test_an_algebraic_loop_inside_a_tick_is_refused() -> None:
    """Two clocked equations that read each other at the same tick are refused."""
    source = PI.replace("i = previous(i) + 0.1 * e;", "i = u + 0.1 * e;")
    with pytest.raises(ClockError, match="cycle"):
        flatten(parse(source), "PI")


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
