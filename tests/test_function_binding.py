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


LOOPS = """\
package Q
  function powsum
    input Real x;
    input Integer n = 4;
    output Real y;
  algorithm
    y := 0;
    for i in 1:n loop
      y := y + x ^ i;
    end for;
  end powsum;
  function halve3
    input Real x;
    output Real y;
  protected
    Integer i;
  algorithm
    y := x;
    i := 0;
    while i < 3 loop
      y := y / 2;
      i := i + 1;
    end while;
  end halve3;
  model M
    parameter Real a = powsum(2);
    parameter Real b = halve3(8);
    Real z = powsum(time, 2);
  end M;
end Q;
"""


def test_for_and_while_loops_unroll() -> None:
    """A ``for`` over a known range and a ``while`` on a constant counter inline (#2)."""
    from modelica.ir.evaluate import evaluate  # noqa: PLC0415 - only this test needs it

    model = flatten(parse(LOOPS), "Q.M")
    a, b = model["a"].binding, model["b"].binding
    assert a is not None
    assert b is not None
    assert evaluate(a, {}) == 2 + 4 + 8 + 16
    assert evaluate(b, {}) == 1


def test_loop_with_a_variable_bound_is_refused() -> None:
    """``for i in 1:n`` with ``n`` a model variable cannot be unrolled at flatten time."""
    from modelica.ir.flatten import FlattenError  # noqa: PLC0415 - only this test needs it

    source = LOOPS.replace("powsum(time, 2)", "powsum(time, k)").replace(
        "model M\n", "model M\n    Integer k = 2;\n"
    )
    with pytest.raises(FlattenError, match="range"):
        flatten(parse(source), "Q.M")


ARRAY_OUT = """\
package R
  function areas
    input Real t;
    input Integer n;
    output Real A[n];
  algorithm
    for k in 1:n loop
      A[k] := k * (1 + t);
    end for;
  end areas;
  model M
    parameter Integer N = 3;
    parameter Real B[N] = areas(1, N);
    Real A[N];
  equation
    A = areas(time, N);
  end M;
end R;
"""


def test_array_output_from_a_loop_with_a_parameter_bound() -> None:
    """``A[k] := ...`` in a loop over ``1:n``, with ``n`` a model parameter, builds the array (#2)."""
    from modelica.ir.evaluate import evaluate  # noqa: PLC0415 - only this test needs it

    model = flatten(parse(ARRAY_OUT), "R.M")
    for k in (1, 2, 3):
        binding = model[f"B[{k}]"].binding
        assert binding is not None
        assert evaluate(binding, {}) == 2 * k


def test_array_output_simulates() -> None:
    """``A[k] = k * (1 + time)`` at every output point."""
    pytest.importorskip("numpy")
    result = simulate(causalize(flatten(parse(ARRAY_OUT), "R.M")), stop=1.0)
    assert result["A[3]"][-1] == pytest.approx(6.0)
