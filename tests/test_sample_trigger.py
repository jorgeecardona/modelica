"""``b = sample(...); when b then`` fires at the sample instants.

``Blocks.Interfaces.DiscreteBlock`` writes ``sampleTrigger = sample(startTime, samplePeriod)``
and ``when sampleTrigger``, which was refused: "a when-condition has to be a relation".
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
model Z
  Boolean trigger;
  discrete Real held(start = 0);
  Real x(start = 0, fixed = true);
equation
  der(x) = 1;
  trigger = sample(0.1, 0.25);
  when trigger then
    held = x;
  end when;
end Z;
"""


def test_a_boolean_set_by_sample_triggers_the_when() -> None:
    result = simulate(causalize(flatten(parse(SOURCE), "Z")), stop=1.0, backend="python")
    # Samples at 0.1, 0.35, 0.6, 0.85: held is x at the last one before t.
    assert float(result.at(0.5)["held"]) == pytest.approx(0.35, abs=1e-6)
    assert float(result.at(1.0)["held"]) == pytest.approx(0.85, abs=1e-6)


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_an_array_condition_with_initial_fires_at_the_start_too(
    backend: Literal["python", "engine"],
) -> None:
    """``when {sampleTrigger, initial()}`` (the PWM's zero-order hold) fires on either."""
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    source = SOURCE.replace("when trigger then", "when {trigger, initial()} then").replace(
        "x(start = 0, fixed = true)", "x(start = 0.5, fixed = true)"
    )
    result = simulate(causalize(flatten(parse(source), "Z")), stop=1.0, backend=backend)
    assert float(result.at(0.05)["held"]) == pytest.approx(0.5, abs=1e-9)
    assert float(result.at(0.5)["held"]) == pytest.approx(0.85, abs=1e-6)
