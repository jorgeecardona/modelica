"""A ``fixed = false`` parameter that an initial equation sets is fixed at the initial point.

``Blocks.Math.Mean`` writes ``parameter SI.Time t0(fixed = false); initial equation t0 = time;``
and samples from ``t0 + 1/f``. It lives in the workspace as a discrete variable, so the
parameter vector the user sets is unchanged.
"""

from __future__ import annotations

import shutil
from typing import Literal

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
model Mn
  parameter Real f = 4;
  parameter Real t0(fixed = false);
  Real x(start = 0);
  discrete Real y_last;
initial equation
  t0 = time;
  x = 0;
  y_last = 0;
equation
  der(x) = 1;
  when sample(t0 + 1/f, 1/f) then
    y_last = f * pre(x);
    reinit(x, 0);
  end when;
end Mn;
"""


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_t0_is_the_start_time_and_the_samples_follow_it(
    backend: Literal["python", "engine"],
) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    system = causalize(flatten(parse(SOURCE), "Mn"))
    result = simulate(system, start=0.1, stop=1.0, points=19, backend=backend)
    assert float(result["t0"].min()) == pytest.approx(0.1)
    assert float(result["t0"].max()) == pytest.approx(0.1)
    # The first sample is at t0 + 1/f = 0.35; x has integrated 0.25 by then, so y_last = 1.
    assert float(result.at(0.3)["y_last"]) == 0.0
    assert float(result.at(0.4)["y_last"]) == pytest.approx(1.0)
