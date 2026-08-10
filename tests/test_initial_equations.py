"""`initial equation`, which used to be parsed, carried and silently dropped.

The entry this closes was found by writing `HeatBar`'s initial condition as an equation
instead of a `start` attribute. Every cell began at zero, `T[1]` at t = 2 came out 42.77
against a reference of 54.21, and nothing said a word. A wrong number from an ignored
equation is the worst shape a simulator can fail in: the run completes, the plot looks
plausible, and the only clue is a comparison against something else.

So there are two kinds of test here and both matter. One says the explicit case is *honoured*
-- the two spellings of one model agree. The other says everything else is *refused by name*,
which is the standing rule for every unimplemented construct in this project and was the one
thing this construct did not do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim.emit import EmitError, emit

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
HEATBAR = (EXAMPLES / "models" / "HeatBar.mo").read_text(encoding="utf-8")

#: `HeatBar` with its initial condition moved out of the `start` attribute and into an
#: `initial equation`. The same model, said the other way, which is exactly the comparison
#: the entry asked for.
AS_EQUATION = HEATBAR.replace(
    'Real T[n](start = T_0, fixed = true, unit = "degC")', 'Real T[n](unit = "degC")'
).replace(
    "equation", "initial equation\n  for i in 1:n loop\n    T[i] = T_0;\n  end for;\nequation", 1
)


def _system(source: str, name: str = "HeatBar"):  # noqa: ANN202 - a `System`
    flatten = pytest.importorskip("modelica.sim").flatten
    return causalize(flatten([parse(source)], name))


def test_the_two_spellings_of_one_model_agree() -> None:
    """The observation the entry named as what closes it."""
    numpy = pytest.importorskip("numpy")
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    grid = numpy.linspace(0.0, 2.0, 9)

    attribute = simulate(_system(HEATBAR), stop=2.0, times=grid, tolerance=1e-10)
    equation = simulate(_system(AS_EQUATION), stop=2.0, times=grid, tolerance=1e-10)

    assert attribute.names == equation.names
    numpy.testing.assert_array_equal(attribute.values, equation.values)
    # The number the bug produced, so a regression is recognisable rather than merely red.
    assert attribute["T[1]"][-1] == pytest.approx(54.21324087, rel=1e-6)


def test_the_initial_value_is_what_the_first_sample_reports() -> None:
    """`y` and the workspace slot both, or the run disagrees with its own first point.

    The state is what the solver integrates from; the workspace slot is what `observe`
    reports at t = 0. Setting only the first leaves the sample at t = 0 holding the `start`
    attribute -- zero here -- while the trajectory begins at 20.
    """
    pytest.importorskip("numpy")
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    result = simulate(_system(AS_EQUATION), stop=1.0, points=3)
    assert result["T[4]"][0] == pytest.approx(20.0)


def test_one_initial_equation_may_read_another() -> None:
    """Ordered by what each reads, so ``y = x + 1`` runs after ``x = 1``."""
    pytest.importorskip("numpy")
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    source = """
    model Chain
      Real x;
      Real y;
    initial equation
      y = x + 1;
      x = 3;
    equation
      der(x) = 0;
      der(y) = 0;
    end Chain;
    """
    result = simulate(_system(source, "Chain"), stop=1.0, points=3)
    assert result["x"][0] == pytest.approx(3.0)
    assert result["y"][0] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        # Not an explicit value for a state: this is the case that needs a real solve.
        ("x + y = 3;\n  x - y = 1;", "not an explicit value"),
        # Explicit, and circular.
        ("x = y + 1;\n  y = x - 1;", "read each other in a cycle"),
    ],
)
def test_what_cannot_be_honoured_is_refused_by_name(body: str, message: str) -> None:
    """The half of this that had to land whichever way the other half went.

    Modelica's initialization is a system -- the initial equations, the model's own equations
    and the `fixed = true` start values solved together for a consistent point. That is not
    implemented. What is not implemented says so.
    """
    source = f"""
    model Hard
      Real x;
      Real y;
    initial equation
      {body}
    equation
      der(x) = 0;
      der(y) = 0;
    end Hard;
    """
    with pytest.raises(EmitError, match=message):
        emit(_system(source, "Hard"))


def test_a_model_with_no_initial_equations_is_unchanged() -> None:
    """Every corpus model takes this path, so it is the one that must not move."""
    source = emit(_system(HEATBAR)).source
    assert "initial equation" not in source
    assert source == emit(_system(HEATBAR)).source
