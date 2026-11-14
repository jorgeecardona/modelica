"""An initial equation may set a discrete variable a ``when`` assigns.

``Blocks.Sources.Pulse`` writes ``initial equation count = integer((time - startTime) / period)``
and then counts periods in a ``when``. The initial equation was refused because ``count`` is
not a state.
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
model P
  parameter Real period = 0.25;
  Integer count;
  Real y;
  Real x(start = 0, fixed = true);
initial equation
  count = integer(time / period) + 3;
equation
  der(x) = 1;
  when time >= (pre(count) - 2) * period then
    count = pre(count) + 1;
  end when;
  y = count;
end P;
"""


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_the_initial_equation_sets_the_discrete_variable(
    backend: Literal["python", "engine"],
) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    system = causalize(flatten(parse(SOURCE), "P"))
    result = simulate(system, stop=0.3, points=4, backend=backend)
    assert float(result["y"][0]) == 3.0
    assert float(result["y"][-1]) == 4.0
    # From t = 0.5 the initial equation reads time: count = 2 + 3.
    later = simulate(system, start=0.5, stop=0.6, points=3, backend=backend)
    assert float(later["y"][0]) == 5.0


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_an_initial_equation_on_a_state_may_read_time(
    backend: Literal["python", "engine"],
) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    source = (
        "model S\n  Real x;\ninitial equation\n  x = time + 1;\nequation\n  der(x) = 0;\nend S;\n"
    )
    result = simulate(causalize(flatten(parse(source), "S")), start=0.5, stop=1.0, backend=backend)
    assert float(result["x"][0]) == 1.5
