"""A user function called in a parameter binding is inlined, as in an equation (#2)."""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.lang import parse
from modelica.sim import causalize, flatten, simulate

SOURCE = """\
package P
  function sq
    input Real x;
    output Real y;
  algorithm
    y := x * x;
  end sq;
  model M
    parameter Real a = 3;
    final parameter Real b = sq(a);
    Real v(start = 0, fixed = true);
  equation
    der(v) = b;
  end M;
end P;
"""


def test_binding_call_is_inlined() -> None:
    """The binding of ``b`` holds ``a * a``, not a call to ``sq``."""
    binding = flatten(parse(SOURCE), "P.M")["b"].binding
    assert binding == ir.Binary("*", ir.Ref(("a",)), ir.Ref(("a",)))


def test_binding_call_simulates() -> None:
    """``der(v) = sq(3)`` integrates to ``v = 9 t``."""
    pytest.importorskip("numpy")
    result = simulate(causalize(flatten(parse(SOURCE), "P.M")), stop=1.0)
    assert result["v"][-1] == pytest.approx(9.0, rel=1e-6)
