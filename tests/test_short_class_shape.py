"""``type Axis = Real[3]``: a short class definition with a shape.

The MultiBody library declares ``Axis``, ``Color`` and ``TransformationMatrix`` this way, and
the parser stopped at the ``[``. An instance takes the type's shape after its own.
"""

from __future__ import annotations

from modelica.ir.flatten import flatten
from modelica.lang import parse, unparse

SOURCE = """\
type V = Real[3](each unit = "m");
model M
  V x[2];
  V y = {1, 2, 3};
end M;
"""


def test_the_shape_round_trips() -> None:
    program = parse(SOURCE)
    v = program.find("V")
    assert v is not None
    assert len(v.dimensions) == 1
    assert parse(unparse(program)) == program


def test_an_instance_takes_the_type_shape_after_its_own() -> None:
    flat = flatten(parse(SOURCE), "M")
    names = {v.name: v for v in flat.variables}
    expected = {f"x[{i}, {j}]" for i in (1, 2) for j in (1, 2, 3)} | {"y[1]", "y[2]", "y[3]"}
    assert expected <= set(names)
    assert names["x[2, 3]"].unit == "m"


def test_a_two_dimensional_type() -> None:
    source = "type T = Real[2, 2];\nmodel M\n  T a;\nend M;\n"
    names = {v.name for v in flatten(parse(source), "M").variables}
    assert {"a[1, 1]", "a[2, 2]"} <= names
