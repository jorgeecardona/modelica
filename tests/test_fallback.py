"""What happens when the engine cannot run: the ``auto`` warning, and the pure-wheel error."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from modelica.lang import parse
from modelica.sim import (
    EngineFallbackWarning,
    SimulationError,
    causalize,
    engine,
    flatten,
    simulate,
    sweep,
)
from modelica.sim.emit import EmitError

DECAY = """
model Decay
  parameter Real k = 1;
  Real x(start = 1, fixed = true);
equation
  der(x) = -k * x;
end Decay;
"""


def _system() -> Any:
    return causalize(flatten([parse(DECAY, "Decay.mo")], "Decay"))


def _refuse(*_: object, **__: object) -> None:
    message = "an opcode the engine does not have"
    raise EmitError(message)


@pytest.fixture
def refusing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed engine that refuses every model."""
    monkeypatch.setattr(engine, "installed", lambda: Path("engine.so"))
    monkeypatch.setattr(engine, "result", _refuse)
    monkeypatch.setattr(engine, "results", _refuse)


@pytest.mark.usefixtures("refusing")
def test_simulate_warns_and_runs_on_scipy() -> None:
    with pytest.warns(EngineFallbackWarning, match="Decay: the engine cannot run this model"):
        result = simulate(_system(), stop=1.0, points=3)
    assert result["x"][-1] == pytest.approx(0.36788, rel=1e-3)


@pytest.mark.usefixtures("refusing")
def test_sweep_warns_and_runs_on_scipy() -> None:
    with pytest.warns(EngineFallbackWarning):
        results = sweep(_system(), [{"k": 1.0}, {"k": 2.0}], stop=1.0, points=3)
    assert len(results) == 2


@pytest.mark.usefixtures("refusing")
def test_backend_engine_raises_instead_of_warning() -> None:
    with pytest.raises(EmitError):
        simulate(_system(), stop=1.0, points=3, backend="engine")


def test_an_explicit_method_falls_back_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", EngineFallbackWarning)
        simulate(_system(), stop=1.0, points=3, method="LSODA")


def test_a_pure_wheel_never_calls_rustc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No bundled engine and no ``engine/src``: a clear error, and no compiler is started."""
    monkeypatch.setattr(engine, "installed", lambda: None)
    monkeypatch.setattr(engine, "SOURCE", tmp_path / "missing")

    def no_compiler(*_: object, **__: object) -> None:
        pytest.fail("rustc was called on a pure wheel")

    monkeypatch.setattr(engine.subprocess, "run", no_compiler)
    with pytest.raises(SimulationError, match="no engine in this install"):
        engine.library()
