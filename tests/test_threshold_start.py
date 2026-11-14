"""A latched relation that starts exactly on its zero is not a crossing.

The MSL sources write ``time < startTime`` with ``startTime = 0``. At t = 0 each one reads 0.0,
was counted as a crossing, and toggled its latch. Three of them toggled each other forever:
"201 events at t = 0 without advancing".
"""

from __future__ import annotations

import shutil

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
model M
  parameter Real t0 = 0;
  parameter Real t1 = 0;
  parameter Real t2 = 0;
  Real x0(start = 0, fixed = true);
  Real x1(start = 0, fixed = true);
  Real x2(start = 0, fixed = true);
equation
  der(x0) = if time < t0 then 0 else 1;
  der(x1) = if time < t1 then 0 else 2;
  der(x2) = if time < t2 then 0 else 3;
end M;
"""


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_three_relations_on_their_threshold_at_the_start_settle(backend: str) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    result = simulate(causalize(flatten(parse(SOURCE), "M")), stop=1.0, backend=backend)
    assert not [e for e in result.events if e.time == 0.0]
    end = result.at(1.0)
    assert float(end["x0"]) == pytest.approx(1.0)
    assert float(end["x1"]) == pytest.approx(2.0)
    assert float(end["x2"]) == pytest.approx(3.0)
