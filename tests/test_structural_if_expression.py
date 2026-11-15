"""An if-expression on a constant or an ``Evaluate = true`` parameter is its chosen branch.

``Analog.Basic.InductorDC`` writes ``v = if quasiStatic then 0 else L*der(i)``. Kept, the if
was solved for ``der(i)`` and divided by zero in ``DCPM_QuasiStatic``.
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
model Inductor
  parameter Real L = 1;
  parameter Boolean quasiStatic = false annotation(Evaluate = true);
  Real v;
  Real i(start = 0);
equation
  v = if quasiStatic then 0 else L * der(i);
end Inductor;
model Machine
  Inductor la(final quasiStatic = quasiStatic);
  constant Boolean quasiStatic = true;
  parameter Real R = 2;
  Real x(start = 0, fixed = true);
equation
  la.v = 1 - R * la.i;
  der(x) = la.i;
end Machine;
"""

PLAIN = SOURCE.replace(" annotation(Evaluate = true)", "").replace(
    "constant Boolean quasiStatic", "parameter Boolean quasiStatic"
)


def _has_if(source: str) -> bool:
    flat = flatten(parse(source), "Machine")
    return any("IfExpr" in repr(equation) for equation in flat.equations)


def test_an_if_expression_on_an_evaluate_parameter_is_its_chosen_branch() -> None:
    assert not _has_if(SOURCE)


def test_a_plain_parameter_condition_is_not_folded() -> None:
    assert _has_if(PLAIN)


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_the_folded_model_simulates(backend: Literal["python", "engine"]) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    result = simulate(causalize(flatten(parse(SOURCE), "Machine")), stop=1.0, backend=backend)
    assert float(result["la.i"][-1]) == pytest.approx(0.5, abs=1e-9)
    assert float(result["x"][-1]) == pytest.approx(0.5, abs=1e-6)
