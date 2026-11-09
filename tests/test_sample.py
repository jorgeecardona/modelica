"""``when sample(start, interval)`` fires at ``start + k * interval`` (#1)."""

from __future__ import annotations

import shutil
from typing import Literal

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


DECAY = """\
model Decay
  Real x(start = 1, fixed = true);
  discrete Real tA(start = 0, fixed = true);
  discrete Real tB(start = 0, fixed = true);
equation
  der(x) = -x;
  when sample(0.05, 0.1) then
    tA = time;
  end when;
  when sample(0.05, 0.2) then
    tB = time;
  end when;
end Decay;
"""


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_a_sample_is_a_time_event_at_its_exact_instant(
    backend: Literal["python", "engine"],
) -> None:
    """Each sample fires at ``pre(next) + interval`` exactly, and two that meet both fire."""
    system = causalize(flatten(parse(DECAY), "Decay"))
    if backend == "engine":
        from modelica.sim.engine import installed  # noqa: PLC0415 - only this case needs it

        if installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    result = simulate(system, stop=1.0, backend=backend)
    expected: list[float] = []
    for interval in (0.1, 0.2):
        instant = 0.05
        while instant <= 1.0:
            expected.append(instant)
            instant += interval
    assert sorted(event.time for event in result.events) == sorted(expected)
    assert result["x"][-1] == pytest.approx(np.exp(-1.0), rel=1e-5)


STEP = """\
model Step
  discrete Real a(start = 0, fixed = true);
equation
  when time >= 0.01 then
    a = 1;
  end when;
end Step;
"""


@pytest.mark.parametrize(("source", "name"), [(PULSE, "Pulse"), (STEP, "Step")])
def test_a_model_with_no_states_runs_on_the_engine(source: str, name: str) -> None:
    """A model with only discrete unknowns gives the same events on the engine as on SciPy."""
    from modelica.sim.engine import installed  # noqa: PLC0415 - only this test needs it

    if installed() is None and shutil.which("rustc") is None:
        pytest.skip("no engine in this install and no rustc to build one")
    system = causalize(flatten(parse(source), name))
    runs = [simulate(system, stop=0.05, backend=backend) for backend in ("python", "engine")]
    times = [[event.time for event in run.events] for run in runs]
    np.testing.assert_allclose(times[1], times[0], rtol=0, atol=1e-12)
    assert times[0]
