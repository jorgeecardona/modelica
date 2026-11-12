"""`modelica.sim.run` reads files and gives the result the four stages give."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from modelica.ir.source import AmbiguousModelError
from modelica.lang import parse
from modelica.lang.library import modelicapath
from modelica.sim import SimulationError, causalize, flatten, read, run, simulate
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


def test_the_command_line_writes_a_mat_file_read_back_equal(tmp_path: Path) -> None:
    """``python -m modelica.sim ... -o out.mat`` writes what `read` loads, with ``--set`` applied."""
    source = tmp_path / "decay.mo"
    source.write_text(DECAY)
    out = tmp_path / "decay.mat"
    command = [sys.executable, "-m", "modelica.sim", str(source), "--stop", "1"]
    subprocess.run([*command, "--set", "k=2", "--points", "11", "-o", str(out)], check=True)  # noqa: S603
    loaded = read(out)
    expected = run(source, stop=1.0, points=11, parameters={"k": 2.0})
    np.testing.assert_allclose(loaded["x"], expected["x"], rtol=1e-12)


def test_the_command_line_prints_csv_without_an_output_file(tmp_path: Path) -> None:
    """No ``-o``: a ``time`` column, then the variables, one row per point."""
    source = tmp_path / "decay.mo"
    source.write_text(DECAY)
    command = [sys.executable, "-m", "modelica.sim", str(source), "--stop", "1", "--points", "3"]
    lines = subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()  # noqa: S603
    assert lines[0].split(",")[0] == "time"
    assert len(lines) == 4


def test_import_modelica_loads_no_numpy_and_run_is_there() -> None:
    """``modelica.run`` is loaded on first use; the bare import stays NumPy-free."""
    code = (
        "import sys, modelica\n"
        "assert 'numpy' not in sys.modules\n"
        "assert modelica.run.__module__ == 'modelica.sim.files'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_a_model_on_a_directory_library_equals_its_single_file_form(tmp_path: Path) -> None:
    """``Lib/package.mo`` + ``Lib/Decay.mo`` as a directory gives the run a one-file ``Lib`` gives."""
    (tmp_path / "Lib").mkdir()
    (tmp_path / "Lib" / "package.mo").write_text("package Lib\nend Lib;\n")
    (tmp_path / "Lib" / "Decay.mo").write_text("within Lib;\n" + DECAY)
    (tmp_path / "Lib" / "Broken.mo").write_text("never read")
    single = tmp_path / "single.mo"
    single.write_text("package Lib\n" + DECAY + "end Lib;\n")
    use = tmp_path / "use.mo"
    use.write_text("model Use\n  Lib.Decay d(k = 2);\nend Use;\n")
    times = [0.0, 0.5, 1.0]
    from_directory = run(use, libraries=[tmp_path / "Lib"], stop=1.0, times=times)
    from_file = run(use, single, model="Use", stop=1.0, times=times)
    np.testing.assert_array_equal(from_directory["d.x"], from_file["d.x"])
    assert run(use, tmp_path / "Lib", stop=1.0, times=times)["d.x"][-1] == from_directory["d.x"][-1]


def _directory_library(tmp_path: Path) -> Path:
    (tmp_path / "libs" / "Lib").mkdir(parents=True)
    (tmp_path / "libs" / "Lib" / "package.mo").write_text("package Lib\nend Lib;\n")
    (tmp_path / "libs" / "Lib" / "Decay.mo").write_text("within Lib;\n" + DECAY)
    use = tmp_path / "use.mo"
    use.write_text("model Use\n  Lib.Decay d(k = 2);\nend Use;\n")
    return use


def test_the_command_line_takes_a_library_directory(tmp_path: Path) -> None:
    """``--lib DIR`` and ``--modelicapath`` with ``MODELICAPATH`` set both find ``Lib``."""
    use = _directory_library(tmp_path)
    base = [sys.executable, "-m", "modelica.sim", str(use), "--stop", "1", "--points", "2"]
    by_lib = subprocess.run(  # noqa: S603
        [*base, "--lib", str(tmp_path / "libs" / "Lib")], check=True, capture_output=True, text=True
    ).stdout
    environment = {**os.environ, "MODELICAPATH": str(tmp_path / "libs")}
    by_path = subprocess.run(  # noqa: S603
        [*base, "--modelicapath"], check=True, capture_output=True, text=True, env=environment
    ).stdout
    assert by_lib == by_path
    assert "d.x" in by_lib.splitlines()[0]


def test_modelicapath_lists_package_directories(tmp_path: Path) -> None:
    """An entry is a package itself, or a directory of packages; other entries add nothing."""
    _directory_library(tmp_path)
    value = os.pathsep.join([str(tmp_path / "libs"), str(tmp_path / "libs" / "Lib"), "/no/such"])
    assert modelicapath(value) == [tmp_path / "libs" / "Lib", tmp_path / "libs" / "Lib"]
