"""``homotopy(actual, simplified)`` -- the standard operator, taken at its actual value.

`homotopy` helps an initialisation converge by walking a path from a simplified model to the
actual one (specification 3.7.2.4). A simulator that does not run that continuation takes the
*actual* value, which is what the expression is worth once initialisation is done -- so here
homotopy is transparent to its **first** argument, the opposite end from ``noEvent``/``smooth``,
which are transparent to their last. It evaluates, emits (in the interpreter and the compiled
Rust backend) and differentiates as that first argument, and the simplified branch is discarded --
including any relations in it, which must not become events nothing reads.
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from modelica.ir import nodes as ir
from modelica.ir.causalize import causalize
from modelica.ir.derive import d_dt
from modelica.ir.evaluate import evaluate
from modelica.lang import parse
from modelica.sim import engine, flatten, simulate

_X = ir.Ref(("x",))


def _system(source: str, name: str):  # noqa: ANN202 - the causalized System
    return causalize(flatten([parse(source)], name))


def test_homotopy_has_the_value_of_its_actual_argument() -> None:
    """The value is the first argument; the simplified second is not read."""
    expr = ir.Call("homotopy", (ir.Binary("*", ir.RealLiteral(3.0), _X), _X))
    assert evaluate(expr, {"x": 4.0}) == pytest.approx(12.0)


def test_homotopy_differentiates_as_its_actual_argument() -> None:
    """The derivative is the derivative of the actual, with no wrapper left behind.

    ``homotopy`` is not a run-time operator -- emit strips it -- so unlike ``smooth`` its
    derivative keeps no wrapper: d/dt homotopy(x*x, x) is just d/dt (x*x).
    """
    actual = ir.Binary("*", _X, _X)
    expr = ir.Call("homotopy", (actual, _X))
    assert d_dt(expr) == d_dt(actual)
    assert evaluate(d_dt(expr), {"x": 4.0, "der(x)": 1.0}) == pytest.approx(8.0)


def test_a_linear_actual_simulates_as_that_actual() -> None:
    """``der(x) = -homotopy(a*x, x)`` is ``der(x) = -a*x``: x decays as e^{-a t}."""
    result = simulate(_system(LINEAR, "Homo"), stop=1.0, tolerance=1e-11)
    exact = np.exp(-2.0 * result.time)
    assert np.max(np.abs(result["x"] - exact)) < 1e-12


def test_a_named_homotopy_reads_as_the_positional_one() -> None:
    """The library writes ``homotopy(actual = ..., simplified = ...)``; it is the positional form.

    A named call reached the differentiator with no positional arguments and crashed with an
    IndexError (MSL's `IdealizedOpAmpLimited`, so `OpAmps.Multivibrator` and its neighbours). The
    names are resolved to ``homotopy(actual, simplified)`` at flatten, so it flattens, causalizes,
    differentiates and simulates the same as if it had been written positionally.
    """
    result = simulate(_system(NAMED, "HomoNamed"), stop=1.0, tolerance=1e-11)
    exact = np.exp(-2.0 * result.time)
    assert np.max(np.abs(result["x"] - exact)) < 1e-12


def test_a_nonlinear_actual_exercises_the_jacobian_rule() -> None:
    """``der(x) = -homotopy(x^3, x)`` needs d/dx of the actual (3x^2) to be right.

    If homotopy differentiated to its simplified argument (or to zero) the implicit solver's
    Jacobian would be wrong and the stiff-ish decay would drift; the closed form ``x(t) =
    1/sqrt(1/x0^2 + 2 t)`` catches that.
    """
    result = simulate(_system(CUBIC, "HomoCubic"), stop=1.0, tolerance=1e-10)
    exact = 1.0 / np.sqrt(1.0 / 4.0 + 2.0 * result.time)  # x0 = 2
    assert np.max(np.abs(result["x"] - exact)) < 1e-8
    assert result["x"][-1] == pytest.approx(1.0 / math.sqrt(1.0 / 4.0 + 2.0), abs=1e-8)


@pytest.mark.slow
@pytest.mark.skipif(
    engine.installed() is None and shutil.which("rustc") is None,
    reason="no engine in this install and no rustc to build one",
)
def test_the_engine_takes_the_actual_too() -> None:
    """The engine strips homotopy as well: same trajectory."""
    system = _system(CUBIC, "HomoCubic")
    grid = np.linspace(0.0, 1.0, 41)
    reference = simulate(system, stop=1.0, times=grid, tolerance=1e-9)
    run = engine.simulate(system, stop=1.0, times=grid, tolerance=1e-9)
    on_grid = np.interp(run.result.time, reference.time, reference["x"])
    assert np.max(np.abs(on_grid - run.result["x"])) < 1e-9


LINEAR = """
model Homo
  Real x(start = 1, fixed = true);
  parameter Real a = 2;
equation
  der(x) = -homotopy(a * x, x);
end Homo;
"""

CUBIC = """
model HomoCubic
  Real x(start = 2, fixed = true);
equation
  der(x) = -homotopy(x ^ 3, x);
end HomoCubic;
"""

NAMED = """
model HomoNamed
  Real x(start = 1, fixed = true);
  parameter Real a = 2;
equation
  der(x) = -homotopy(actual = a * x, simplified = x);
end HomoNamed;
"""
