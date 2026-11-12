"""`modelica.sim.run` reads files and gives the result the four stages give."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modelica.ir.source import AmbiguousModelError
from modelica.lang import parse
from modelica.sim import causalize, flatten, run, simulate

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
