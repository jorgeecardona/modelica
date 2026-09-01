"""Conditional components: ``Sensor s if useSensor;`` and the removal that follows.

The MSL measurement (docs/ideas/coverage.md) counts 53 example models whose first refusal
was this construct. Unlike ``expandable connector`` this one is implemented, not merely
accepted: the condition is evaluated during flattening -- it is structural the way a
dimension is, deciding which variables and equations exist -- and a component whose
condition is false disappears together with every ``connect()`` that touches it, which is
the one reference the specification allows to survive the removal.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import parse, unparse

LIB = """\
connector Pin
  Real v;
  flow Real i;
end Pin;
model Source
  Pin p;
equation
  p.v = 1;
end Source;
model Load
  Pin p;
equation
  p.i = p.v;
end Load;
"""


def model(use: str) -> ir.Program:
    return parse(
        LIB
        + f"""\
model M
  parameter Boolean useLoad = {use};
  Source s;
  Load l if useLoad;
equation
  connect(s.p, l.p);
end M;
""",
    )


def test_the_condition_is_read_and_round_trips() -> None:
    program = model("false")
    m = program.find("M")
    assert m is not None
    load = next(c for c in m.components if c.name == "l")
    assert load.condition == ir.Ref(("useLoad",))
    assert parse(unparse(program)) == program


def test_a_false_condition_removes_the_component_and_its_connects() -> None:
    flat = flatten(model("false"), "M")
    names = [v.name for v in flat.variables]
    assert not any(name.startswith("l.") for name in names)
    # The severed connect leaves the source's pin alone in its connection set: its flow
    # sums to zero on its own, and no equation mentions the load.
    assert not any("l." in str(equation) for equation in flat.equations)


def test_a_true_condition_keeps_the_component() -> None:
    flat = flatten(model("true"), "M")
    names = [v.name for v in flat.variables]
    assert "l.p.v" in names
    assert "l.p.i" in names


def test_a_condition_nobody_can_evaluate_is_refused() -> None:
    source = LIB + "model M\n  Real x;\n  Load l if x > 0;\nequation\n  x = 1;\nend M;\n"
    with pytest.raises(FlattenError, match="must be known when the model is flattened"):
        flatten(parse(source), "M")


def test_a_condition_that_is_not_a_boolean_is_refused() -> None:
    source = LIB + "model M\n  Load l if 3;\nend M;\n"
    with pytest.raises(FlattenError, match="must be a Boolean"):
        flatten(parse(source), "M")
