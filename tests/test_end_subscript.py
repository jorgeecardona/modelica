"""``end`` in a subscript is the size of the dimension it indexes."""

from __future__ import annotations

import pytest

from modelica.ir import nodes as ir
from modelica.lang import parse
from modelica.sim import flatten

MODEL = """
model M
  parameter Integer n = 4;
  parameter Integer idx[2] = {1, 3};
  Real x[n];
  Real last;
  Real before;
  Real s;
  Real nested;
equation
  for i in 1:n loop
    x[i] = i;
  end for;
  last = x[end];
  before = x[end - 1];
  s = sum(x[2:end]);
  nested = x[idx[end]];
end M;
"""


def _right_side(name: str) -> ir.Expr:
    flat = flatten([parse(MODEL)], "M")
    return next(e.rhs for e in flat.equations if e.lhs == ir.Ref((name,)))


@pytest.mark.parametrize(
    ("name", "element"), [("last", "x[4]"), ("before", "x[3]"), ("nested", "x[3]")]
)
def test_end_reads_the_size_of_its_own_dimension(name: str, element: str) -> None:
    """In ``x[idx[end]]`` the ``end`` is the size of ``idx`` (2), so the element is ``x[3]``."""
    assert _right_side(name) == ir.Ref((element,))


def test_end_closes_a_range() -> None:
    """``x[2:end]`` is ``x[2]``, ``x[3]``, ``x[4]``."""
    assert set(_names(_right_side("s"))) == {"x[2]", "x[3]", "x[4]"}


def test_end_outside_a_subscript_still_closes_a_block() -> None:
    """The keyword keeps its other meaning: ``end M;`` and ``end for;`` still parse."""
    assert parse(MODEL).classes[0].name == "M"


def _names(node: ir.Expr) -> list[str]:
    match node:
        case ir.Ref(parts=parts):
            return [".".join(parts)]
        case ir.Binary(lhs=lhs, rhs=rhs):
            return [*_names(lhs), *_names(rhs)]
        case _:
            return []
