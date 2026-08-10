"""`parameter Real t[4] = {0, 1, 2, 3}` -- an array given its values, not just its shape.

Arrays existed before this: `HeatBar` declares `Real T[n]` and runs. What did not exist was a
way to say what is *in* one. Every array in the corpus takes a scalar attribute that
broadcasts -- `Real T[n](start = T_0)` gives every cell the same start -- and the moment a
component needs a different value per element the model ended at "array equations are not
supported yet".

The order is the load-bearing test. A binding that reversed itself, or that gave every element
the first value, would flatten, causalize, compile and simulate; only the numbers would say
otherwise. And the broadcast has its own test, because indexing into a scalar is the obvious
way to break `HeatBar` while making this work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import FlattenError
from modelica.lang import parse

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _flat(source: str, name: str = "T"):  # noqa: ANN202 - a `FlatModel`
    flatten = pytest.importorskip("modelica.sim").flatten
    return flatten([parse(source)], name)


def _bindings(source: str, name: str = "T") -> dict[str, object]:
    """Every parameter's value, evaluated in dependency order the way the emitter does."""
    settings = pytest.importorskip("modelica.sim.emit").settings
    return dict(settings(causalize(_flat(source, name))))


def test_each_element_gets_its_own_value_in_order() -> None:
    """The whole feature, and the failure that passes every structural check.

    Four names and four values, paired the way they were written. A binding that reversed
    itself has the same number of parameters, the same shapes and the same equations.
    """
    source = """
    model T
      parameter Real ts[4] = {0.0, 1.0, 2.5, 3.0};
      Real x(start = 0, fixed = true);
    equation
      der(x) = ts[3];
    end T;
    """
    assert _bindings(source) == {
        "ts[1]": 0.0,
        "ts[2]": 1.0,
        "ts[3]": 2.5,
        "ts[4]": 3.0,
    }


def test_the_values_reach_the_run() -> None:
    """Flattened is not enough: the emitted parameter vector has to carry them too."""
    pytest.importorskip("numpy")
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    source = """
    model T
      parameter Real gain[3] = {2.0, 4.0, 8.0};
      parameter Real ts[4] = {0.0, 1.0, 2.5, 3.0};
      Real x(start = 0, fixed = true);
    equation
      der(x) = gain[2] * ts[3];
    end T;
    """
    result = simulate(causalize(_flat(source)), stop=1.0, points=3, tolerance=1e-10)
    assert result["x"][-1] == pytest.approx(10.0, rel=1e-9)


def test_a_scalar_binding_still_broadcasts() -> None:
    """The case that must not move. Every array in the corpus relies on it.

    `Real T[n](start = T_0)` gives every cell the same start. Indexing into a scalar is the
    obvious way to make this feature work and `HeatBar` stop.
    """
    pytest.importorskip("numpy")
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    heatbar = (EXAMPLES / "models" / "HeatBar.mo").read_text(encoding="utf-8")
    result = simulate(causalize(_flat(heatbar, "HeatBar")), stop=2.0, points=3, tolerance=1e-10)
    assert result["T[1]"][-1] == pytest.approx(54.21324087, rel=1e-6)


def test_a_binding_shorter_than_the_declaration_says_both_numbers() -> None:
    """Silently short is the other way this goes wrong, and it reads as a zero."""
    source = """
    model T
      parameter Real ts[4] = {0.0, 1.0};
      Real x(start = 0, fixed = true);
    equation
      der(x) = ts[1];
    end T;
    """
    with pytest.raises(FlattenError, match="lists 2"):
        _flat(source)


def test_a_value_may_be_an_expression_and_not_only_a_literal() -> None:
    """Each element is bound to whatever was written in that slot, evaluated as usual."""
    source = """
    model T
      parameter Real k = 3.0;
      parameter Real ts[3] = {k, 2 * k, k + 1};
      Real x(start = 0, fixed = true);
    equation
      der(x) = ts[2];
    end T;
    """
    assert _bindings(source) == {"k": 3.0, "ts[1]": 3.0, "ts[2]": 6.0, "ts[3]": 4.0}


def test_the_python_front_end_says_it_too() -> None:
    """Every corpus feature is written twice, and an array literal is one expression."""
    build = pytest.importorskip("modelica.build")
    unparse_expr = pytest.importorskip("modelica.lang.unparse").unparse_expr
    written = build.array(0.0, 1.0, 2.5)
    assert unparse_expr(written.node) == "{0.0, 1.0, 2.5}"
