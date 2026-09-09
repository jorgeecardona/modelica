"""Type attributes: the descriptive ones are accepted and ignored, an unknown one is refused.

A Modelica type carries attributes that reach the numeric model -- `start`, `fixed`, `min`,
`max`, `nominal` -- and attributes that are purely descriptive: `unit`, `displayUnit`, and
`quantity`, which names the physical quantity a number measures. The whole `Modelica.Units.SI`
type set is written `Real(quantity="ElectricPotential", unit="V")`, so refusing `quantity`
blocked the electrical and machine libraries wholesale in the coverage run. It is accepted and
ignored now, the way `unit` and `displayUnit` already were.

The refusal is kept for a genuinely unknown attribute, so a typo does not pass silently.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from modelica import ir
from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse

if TYPE_CHECKING:
    from modelica.ir.flat import FlatVariable

QUANTITY = """\
model M
  type Voltage = Real(quantity="ElectricPotential", unit="V");
  Voltage v(start=2.0, nominal=100.0);
equation
  der(v) = -v;
end M;
"""

UNKNOWN = """\
model M
  Real x(qauntity="typo");
equation
  der(x) = -x;
end M;
"""


def _variable(source: str, name: str) -> FlatVariable:
    flat = flatten(parse(source, "<test>"), "M")
    return next(variable for variable in flat.variables if variable.name == name)


def test_quantity_is_accepted_and_the_useful_attributes_survive() -> None:
    """The `quantity` is dropped; `unit`, `start` and `nominal` on the same type reach the model."""
    v = _variable(QUANTITY, "v")
    assert v.unit == "V"
    assert v.start is not None
    assert v.nominal is not None


def test_an_unknown_attribute_is_still_refused_by_name() -> None:
    with pytest.raises(FlattenError, match=re.escape("unknown attribute(s) ['qauntity']")):
        flatten(parse(UNKNOWN, "<test>"), "M")


STATE_SELECT = """\
model M
  parameter StateSelect ss = StateSelect.default;
  Real x(stateSelect = StateSelect.prefer, start = 3.0);
equation
  der(x) = -x;
end M;
"""


def test_state_select_attribute_and_type_resolve() -> None:
    """`StateSelect` is provided as a predefined enumeration; the hint is ignored, `start` kept."""
    ss = _variable(STATE_SELECT, "ss")
    x = _variable(STATE_SELECT, "x")
    assert isinstance(ss.binding, ir.IntLiteral)
    assert ss.binding.value == 3  # never, avoid, default -> default is the third literal
    assert x.start is not None  # the stateSelect hint is dropped, start is kept


def test_every_state_select_member_maps_to_its_ordinal() -> None:
    for ordinal, member in enumerate(["never", "avoid", "default", "prefer", "always"], start=1):
        source = (
            f"model M parameter StateSelect s = StateSelect.{member};"
            " Real y(start=1); equation der(y) = -y; end M;"
        )
        s = next(v for v in flatten(parse(source, "<test>"), "M").variables if v.name == "s")
        assert isinstance(s.binding, ir.IntLiteral)
        assert s.binding.value == ordinal
