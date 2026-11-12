"""`modelica.sim.run` reads files and gives the result the four stages give."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modelica.ir.source import AmbiguousModelError
from modelica.lang import parse
from modelica.sim import SimulationError, causalize, flatten, run, simulate
from modelica.sim.emit import EmitError

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "examples" / "models" / "RLCCircuit.mo"
LIBRARY = ROOT / "examples" / "library" / "Electrical.mo"


def test_run_gives_the_values_of_the_four_stages() -> None:
    """Same files, same model, same ``times``: the same trajectory."""
    times = np.linspace(0.0, 0.01, 11)
    result = run(MODEL, LIBRARY, stop=0.01, times=times)
    programs = [parse(path.read_text(encoding="utf-8"), str(path)) for path in (MODEL, LIBRARY)]
    expected = simulate(causalize(flatten(programs, "RLCCircuit")), stop=0.01, times=times)
    for name in expected.names:
        np.testing.assert_array_equal(result[name], expected[name])


def test_run_asks_for_a_model_name_when_the_files_hold_several(tmp_path: Path) -> None:
    """Two models and no ``model``: the error lists both."""
    source = tmp_path / "two.mo"
    source.write_text(
        "model A Real x; equation x = 1; end A;\nmodel B Real y; equation y = 2; end B;\n"
    )
    with pytest.raises(AmbiguousModelError, match=r"\['A', 'B'\]"):
        run(source, stop=1.0)
    assert run(source, model="B", stop=1.0, times=[0.0, 1.0])["y"][-1] == 2.0


def test_stop_start_and_tolerance_come_from_the_experiment_annotation(tmp_path: Path) -> None:
    """``experiment(StartTime = 1, StopTime = 3)`` sets the span when ``run`` is not told."""
    source = tmp_path / "decay.mo"
    source.write_text(
        "model Decay\n  Real x(start = 1);\nequation\n  der(x) = -x;\n"
        "  annotation(experiment(StartTime = 1, StopTime = 3, Tolerance = 1e-8));\nend Decay;\n"
    )
    result = run(source)
    assert result.time[0] == 1.0
    assert result.time[-1] == 3.0
    assert run(source, stop=2.0).time[-1] == 2.0


def test_run_without_stop_or_annotation_says_what_is_missing(tmp_path: Path) -> None:
    """No ``stop`` and no ``StopTime``: the error names both."""
    source = tmp_path / "plain.mo"
    source.write_text("model Plain\n  Real x(start = 1);\nequation\n  der(x) = -x;\nend Plain;\n")
    with pytest.raises(SimulationError, match="StopTime"):
        run(source)


DECAY = "model Decay\n  parameter Real k = 1;\n  Real x(start = 1);\nequation\n  der(x) = -k * x;\nend Decay;\n"


def test_a_parameter_set_by_name_changes_the_trajectory(tmp_path: Path) -> None:
    """``k = 2`` halves the time constant: ``x(1) = exp(-2)``."""
    source = tmp_path / "decay.mo"
    source.write_text(DECAY)
    result = run(source, stop=1.0, times=[0.0, 1.0], parameters={"k": 2.0}, tolerance=1e-10)
    np.testing.assert_allclose(result["x"][-1], np.exp(-2.0), rtol=1e-6)


def test_a_misspelled_parameter_is_refused_with_the_close_name(tmp_path: Path) -> None:
    """``kk`` is not a parameter; the error suggests ``k``."""
    source = tmp_path / "decay.mo"
    source.write_text(DECAY)
    with pytest.raises(EmitError, match="did you mean k"):
        run(source, stop=1.0, parameters={"kk": 2.0})
