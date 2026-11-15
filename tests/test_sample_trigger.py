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


HOLD = """\
model Hold
  parameter Real startTime = 0;
  Real x(start = 1, fixed = true);
  Real u;
  Real ySample(start = 0, fixed = true);
  Real y;
equation
  der(x) = -x;
  u = if x > 0.5 then 2 * x else -7;
  when {sample(startTime, 0.25), initial()} then
    ySample = if time >= startTime then u else pre(ySample);
  end when;
  y = pre(ySample);
end Hold;
"""


@pytest.mark.parametrize("backend", ["python", "engine"])
@pytest.mark.parametrize(("start_time", "first"), [(0.0, 2.0), (0.1, 0.0)])
def test_a_hold_started_by_initial_reads_the_model_equations(
    backend: Literal["python", "engine"], start_time: float, first: float
) -> None:
    """``Blocks.Discrete.ZeroOrderHold``: the initial value reads ``u`` and ``pre(ySample)``."""
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    source = HOLD.replace("startTime = 0;", f"startTime = {start_time};")
    result = simulate(causalize(flatten(parse(source), "Hold")), stop=1.0, backend=backend)
    held = result["ySample"]
    assert float(held[0]) == pytest.approx(first, abs=1e-9)
    # x crosses 0.5 at t = ln 2, so the sample at 0.75 takes the else branch.
    assert float(held[-1]) == pytest.approx(-7.0, abs=1e-9)


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_an_initial_value_read_from_an_algebraic_loop_is_refused_by_name(
    backend: Literal["python", "engine"],
) -> None:
    from modelica.sim.emit import EmitError  # noqa: PLC0415 - only this case needs it

    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    source = (
        HOLD.replace(
            "u = if x > 0.5 then 2 * x else -7;", "u = x - v * v * v; v = u + sin(v);\n  Real v;"
        )
        .replace("  Real y;\nequation", "  Real y;\n  Real v;\nequation")
        .replace("\n  Real v;\n  when", "\n  when")
    )
    with pytest.raises(EmitError, match="'u'"):
        simulate(causalize(flatten(parse(source), "Hold")), stop=1.0, backend=backend)
