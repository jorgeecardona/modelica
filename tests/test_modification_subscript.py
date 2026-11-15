"""A subscript in a binding of a nested component names the element its index picks.

MultiBody's ``World`` writes ``groundAxis_u = if abs(n[1]) >= 0.99 then {0,1,0} else {1,0,0}``.
In a nested component the binding was prefixed without the parameter values, and the
subscript was refused: "subscripted references are not expanded here" (30 MSL examples).
"""

from __future__ import annotations

from modelica import ir
from modelica.ir.evaluate import evaluate
from modelica.ir.flatten import flatten
from modelica.lang import parse

SOURCE = """\
model Inner
  parameter Real n[3] = {0, -1, 0};
  parameter Real u[3] = if abs(n[1]) >= 0.99 then {0, 1, 0} else {1, 0, 0};
  parameter Integer k = 2;
  Real x(start = n[k]);
equation
  der(x) = u[1];
end Inner;
model Top
  Inner w;
  Inner v(n = {1, 0, 0});
end Top;
"""


def test_the_subscript_reads_the_element_of_the_component() -> None:
    flat = flatten(parse(SOURCE), "Top")
    variables = {v.name: v for v in flat.variables}
    values = {
        name: float(evaluate(v.binding, {}))
        for name, v in variables.items()
        if v.name.endswith(("n[1]", "n[2]", "n[3]")) and v.binding is not None
    }

    def value(name: str) -> float:
        binding = variables[name].binding
        assert binding is not None
        return float(evaluate(binding, values))

    assert [value(f"w.u[{i}]") for i in (1, 2, 3)] == [1.0, 0.0, 0.0]
    assert [value(f"v.u[{i}]") for i in (1, 2, 3)] == [0.0, 1.0, 0.0]
    assert variables["w.x"].start == ir.Ref(("w", "n[2]"))
