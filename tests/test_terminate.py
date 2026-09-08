"""``terminate(message)`` -- the standard operator that ends the run when its ``when`` fires.

`terminate` is a Modelica built-in: a run reaches an instant past which the model is no longer
defined -- a tank that empties, a decay past a threshold, a bouncing ball's Zeno accumulation --
and says so, and the analysis stops there rather than integrating meaningless numbers to the
horizon. It is a state event like any other: the ``when`` condition is a zero crossing located
the ordinary way, and the only new thing is the action, which is "stop and report" instead of
``reinit``. So the event *location* is the machinery that already exists; these tests hold that
the run ends at the located instant, keeps the trajectory up to it and nothing past it, and
reports the message as the cause -- in the interpreter and in the compiled Rust backend alike.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten, rustsim, simulate

if TYPE_CHECKING:
    from pathlib import Path

# A first-order decay that stops when it has fallen below a tenth: x(t) = e^{-t} crosses 0.1 at
# t = ln(10), a terminate time known in closed form, so the located instant can be checked exactly.
DECAY = """
model Decay
  Real x(start = 1, fixed = true);
equation
  der(x) = -x;
  when x <= 0.1 then
    terminate("x fell below a tenth");
  end when;
end Decay;
"""

_TERMINATE_TIME = math.log(10.0)


def _system(source: str, name: str):  # noqa: ANN202 - the causalized System
    return causalize(flatten([parse(source)], name))


def test_the_run_stops_at_the_located_instant() -> None:
    """The interpreter ends the run exactly where the condition became true, not at the stop."""
    result = simulate(_system(DECAY, "Decay"), stop=10.0, tolerance=1e-9)
    assert result.time[-1] == pytest.approx(_TERMINATE_TIME, abs=1e-6)
    # The whole point: the horizon was 10 and the run is a fifth of it, because past the crossing
    # the model says there is nothing worth integrating.
    assert result.time[-1] < 3.0
    assert not (result.time > _TERMINATE_TIME + 1e-6).any()
    assert result["x"][-1] == pytest.approx(0.1, abs=1e-6)


def test_the_terminate_is_the_last_event_and_carries_its_message() -> None:
    """One event, at the terminate instant, whose cause is the message the model wrote."""
    result = simulate(_system(DECAY, "Decay"), stop=10.0, tolerance=1e-9)
    assert len(result.events) == 1
    event = result.events[-1]
    assert event.time == pytest.approx(_TERMINATE_TIME, abs=1e-6)
    assert event.cause == "terminate: x fell below a tenth"


def test_a_bare_terminate_still_stops_without_a_message() -> None:
    """``terminate()`` with no message ends the run; the reason is just left empty."""
    source = """
    model Bare
      Real x(start = 1, fixed = true);
    equation
      der(x) = -x;
      when x <= 0.5 then
        terminate();
      end when;
    end Bare;
    """
    result = simulate(_system(source, "Bare"), stop=10.0, tolerance=1e-9)
    assert result.time[-1] == pytest.approx(math.log(2.0), abs=1e-6)
    assert result.events[-1].cause == "terminate"


def test_the_trigger_records_that_it_terminates() -> None:
    """The discrete structure is on the public record: emit marks the crossing as terminating."""
    build = pytest.importorskip("modelica.sim.kernel").build
    kernel = build(_system(DECAY, "Decay"))
    terminating = [trigger for trigger in kernel.code.events if trigger.terminates]
    assert len(terminating) == 1
    assert terminating[0].message == "x fell below a tenth"
    assert not any(trigger.reinitialises for trigger in kernel.code.events)


@pytest.mark.skipif(rustsim.compiler() is None, reason="no rustc on this machine")
def test_the_rust_backend_terminates_at_the_same_instant(tmp_path: Path) -> None:
    """The compiled backend stops where the interpreter does, and reports the same cause.

    The Rust run fills a fixed sample grid and truncates it at the terminate instant, so its last
    grid point sits at or just before the crossing; the reported end time and the event are the
    located instant itself.
    """
    system = _system(DECAY, "Decay")
    run = rustsim.simulate(system, stop=10.0, tolerance=1e-9, cache=tmp_path)
    assert run.end == pytest.approx(_TERMINATE_TIME, abs=1e-6)
    assert not (run.result.time > _TERMINATE_TIME + 1e-6).any()
    assert run.events == 1
    assert run.event_list[-1].cause == "terminate: x fell below a tenth"
    assert run.event_list[-1].time == pytest.approx(_TERMINATE_TIME, abs=1e-6)
