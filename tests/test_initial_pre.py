"""``initial equation pre(x) = v`` gives a discrete variable its value before the first event.

``Blocks.Logical.Timer`` writes ``pre(entryTime) = 0`` and sets ``entryTime = time`` in a
``when``. The initial equation was refused because its left side is not a name.
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
model T
  discrete Real entryTime;
  Real x(start = 0, fixed = true);
initial equation
  pre(entryTime) = -1;
equation
  der(x) = 1;
  when x > 0.5 then
    entryTime = time;
  end when;
end T;
"""


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_pre_sets_the_value_before_the_first_event(backend: Literal["python", "engine"]) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    result = simulate(causalize(flatten(parse(SOURCE), "T")), stop=1.0, points=11, backend=backend)
    assert float(result["entryTime"][0]) == -1.0
    assert float(result["entryTime"][-1]) == pytest.approx(0.5, abs=1e-6)
