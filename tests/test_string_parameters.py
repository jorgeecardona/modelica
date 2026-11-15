"""A String parameter is resolved when the model is flattened, and comparisons on it fold.

The MSL transformers compare ``C1 == "D"`` in a ``Real`` binding, where ``C1`` comes from the
``VectorGroup`` String parameter. A String reached the parameter vector, which holds floats,
and the run stopped with "could not convert string to float".
"""

from __future__ import annotations

import math

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
model S
  parameter String g = "Dy";
  constant String c = g;
  parameter Real k = if c == "Dy" then 2 else 3;
  Real x(start = 1, fixed = true);
equation
  der(x) = -k * x;
end S;
model M
  S s(g = "{group}");
end M;
"""


@pytest.mark.parametrize(("group", "rate"), [("Dy", 2.0), ("Yy", 3.0)])
def test_a_string_parameter_chooses_the_branch(group: str, rate: float) -> None:
    flat = flatten(parse(SOURCE.format(group=group)), "M")
    assert all(v.base_type != "String" for v in flat.variables)
    result = simulate(causalize(flat), stop=1.0, backend="python")
    assert float(result["s.x"][-1]) == pytest.approx(math.exp(-rate), rel=1e-4)


SUBSTRING = """\
package Modelica
  package Utilities
    package Strings
      function substring
        input String string;
        input Integer startIndex;
        input Integer endIndex;
        output String result;
        external "C" result = ModelicaStrings_substring(string, startIndex, endIndex);
      end substring;
    end Strings;
  end Utilities;
end Modelica;
model S
  parameter String group = "Dy05";
  constant String c1 = Modelica.Utilities.Strings.substring(group, 1, 1);
  parameter Real k = if c1 == "D" then 2 else 3;
  Real x(start = 1, fixed = true);
equation
  der(x) = -k * x;
end S;
model M
  S s(group = "{group}");
end M;
"""


@pytest.mark.parametrize(("group", "rate"), [("Dy05", 2.0), ("Yy00", 3.0)])
def test_substring_of_a_string_parameter_chooses_the_branch(group: str, rate: float) -> None:
    """``substring(VectorGroup, 1, 1)`` as the transformers write it: an external C function."""
    flat = flatten(parse(SUBSTRING.format(group=group)), "M")
    result = simulate(causalize(flat), stop=1.0, backend="python")
    assert float(result["s.x"][-1]) == pytest.approx(math.exp(-rate), rel=1e-4)
