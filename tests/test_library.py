"""`library.closure` reads only the files of a directory library that a model can reach."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelica.lang import parse
from modelica.lang.library import closure, load
from modelica.sim import flatten

FILES = {
    "Lib/package.mo": "package Lib\nend Lib;\n",
    "Lib/Basic/package.mo": "within Lib;\npackage Basic\nend Basic;\n",
    "Lib/Basic/Resistor.mo": (
        "within Lib.Basic;\nmodel Resistor\n  extends Interfaces.OnePort;\n"
        "  parameter Real R = 1;\nequation\n  v = R * i;\nend Resistor;\n"
    ),
    "Lib/Interfaces.mo": (
        "within Lib;\npackage Interfaces\n  partial model OnePort\n    Real v;\n    Real i;\n"
        "  end OnePort;\nend Interfaces;\n"
    ),
    "Lib/Examples/package.mo": "within Lib;\npackage Examples\nend Examples;\n",
    "Lib/Examples/Resistor.mo": "this file does not parse",
    "Lib/Units.mo": "within Lib;\npackage Units\n  type Voltage = Real;\nend Units;\n",
}

MODEL = """
model Use
  import V = Lib.Units.Voltage;
  Lib.Basic.Resistor r(R = 2);
  V u;
equation
  r.i = 1;
  u = r.v;
end Use;
"""


@pytest.fixture
def library(tmp_path: Path) -> Path:
    for name, text in FILES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path / "Lib"


def test_a_model_reads_what_it_names_and_not_the_rest(library: Path) -> None:
    """``Lib.Basic.Resistor`` reads Basic, its ``extends`` target and the import, never Examples.

    ``Interfaces.OnePort`` is found from inside ``Lib.Basic`` by looking outward to ``Lib``;
    the broken ``Examples/Resistor.mo`` shares a name with what the model uses and is not read.
    """
    program = closure([library], [MODEL])
    flat = flatten([program, parse(MODEL)], "Use")
    assert {v.name for v in flat.variables} >= {"r.v", "r.i", "r.R", "u"}


def test_a_broken_file_the_model_needs_says_what_named_it(library: Path) -> None:
    """A model on ``Lib.Examples.Resistor`` reaches the broken file; the error names the chain."""
    model = "model Bad\n  Lib.Examples.Resistor r;\nend Bad;\n"
    with pytest.raises(SyntaxError, match=r"Lib\.Examples\.Resistor.*named in the model"):
        closure([library], [model])


def test_load_reads_everything_and_records_what_does_not_parse(library: Path) -> None:
    """`load` is the whole tree; with ``skipped`` a broken member is left out and recorded."""
    skipped: dict[Path, str] = {}
    lib = load(library, skipped)
    assert {c.name for c in lib.classes} == {"Basic", "Examples", "Interfaces", "Units"}
    assert [path.name for path in skipped] == ["Resistor.mo"]


def test_an_example_beside_a_used_class_pulls_in_nothing(tmp_path: Path) -> None:
    """A model on ``Lib.Good`` never reads a broken file that only an inline example extends.

    Names are followed per class, so ``Examples.Bad`` beside ``Good`` pulls in nothing.
    """
    (tmp_path / "Lib" / "Far").mkdir(parents=True)
    (tmp_path / "Lib" / "package.mo").write_text(
        "package Lib\n  model Good\n    Real x(start = 1);\n  equation\n    der(x) = -x;\n"
        "  end Good;\n  package Examples\n    model Bad\n      extends Lib.Far.Broken;\n"
        "    end Bad;\n  end Examples;\nend Lib;\n"
    )
    (tmp_path / "Lib" / "Far" / "package.mo").write_text("within Lib;\npackage Far\nend Far;\n")
    (tmp_path / "Lib" / "Far" / "Broken.mo").write_text("this file does not parse")
    model = "model Use\n  Lib.Good g;\nend Use;\n"
    flat = flatten([closure([tmp_path / "Lib"], [model]), parse(model)], "Use")
    assert "g.x" in {v.name for v in flat.variables}


MSL = Path(__file__).resolve().parents[1] / ".cache" / "msl" / "v4.1.0"


@pytest.mark.skipif(not (MSL / "Modelica" / "package.mo").exists(), reason="no MSL 4.1.0 cache")
def test_an_rc_model_runs_on_the_real_msl() -> None:
    """Resistor, Capacitor and Ground from MSL 4.1.0: ``exp(-t / RC)``, nothing from Machines."""
    import numpy as np  # noqa: PLC0415

    from modelica.sim import run  # noqa: PLC0415

    model = Path(__file__).resolve().parent / "data" / "msl_rc.mo"
    result = run(
        model, libraries=[MSL / "Modelica", MSL / "ModelicaServices"], stop=1.0, times=[0, 1]
    )
    np.testing.assert_allclose(result["c.v"], [1.0, np.exp(-1.0)], rtol=1e-4)
