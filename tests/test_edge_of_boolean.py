"""``when edge(off)`` fires when the relation behind ``off`` becomes true.

``Analog.Interfaces.IdealSwitchWithArc`` writes ``when edge(off) then tSwitch = time;`` and
``OpenerWithArc`` writes ``off = control``, so the relation is two names away. That was refused:
"a when-condition has to be a relation, not 'edge(switch2.off)'".
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
model Switch
  Boolean control;
  Boolean off;
  Boolean on;
  discrete Real tSwitch(start = -1);
  discrete Real tClose(start = -1);
  Real x(start = 0, fixed = true);
equation
  der(x) = 1;
  control = x >= 0.3;
  off = control;
  on = not control;
  when edge(off) then
    tSwitch = time;
  end when;
  when edge(on) then
    tClose = time;
  end when;
end Switch;
"""


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_edge_of_an_aliased_boolean_fires_at_its_crossing(
    backend: Literal["python", "engine"],
) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    result = simulate(causalize(flatten(parse(SOURCE), "Switch")), stop=1.0, backend=backend)
    assert float(result["tSwitch"][-1]) == pytest.approx(0.3, abs=1e-6)
    # `not control` never becomes true after the start, so that when never fires.
    assert float(result["tClose"][-1]) == pytest.approx(-1.0)
