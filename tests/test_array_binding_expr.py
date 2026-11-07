"""A binding that is an expression over whole arrays is read out per element (#3)."""

from __future__ import annotations

import pytest

from modelica.ir.evaluate import evaluate
from modelica.lang import parse
from modelica.sim import causalize, flatten, simulate

SOURCE = """\
model Tube
  parameter Integer N = 3;
  parameter Real dx = 0.5;
  parameter Real A[N] = {1, 2, 3};
  parameter Real C[N] = A * dx / 10;
  parameter Real D[N] = A .* A;
  Real p[N](each start = 0, each fixed = true);
equation
  for i in 1:N loop
    C[i] * der(p[i]) = 1;
  end for;
end Tube;
"""


def test_each_element_gets_its_own_binding() -> None:
    """``C[i] = A[i] * dx / 10`` and ``D[i] = A[i] * A[i]``, not the whole-array expression."""
    model = flatten(parse(SOURCE), "Tube")
    values = {"dx": 0.5, "A[1]": 1.0, "A[2]": 2.0, "A[3]": 3.0}
    for i, a in enumerate((1.0, 2.0, 3.0), start=1):
        c, d = model[f"C[{i}]"].binding, model[f"D[{i}]"].binding
        assert c is not None
        assert d is not None
        assert evaluate(c, values) == pytest.approx(a * 0.5 / 10)
        assert evaluate(d, values) == pytest.approx(a * a)


def test_array_binding_simulates() -> None:
    """``der(p[i]) = 1 / C[i]``, so ``p[i](1) = 20 / A[i]``."""
    pytest.importorskip("numpy")
    result = simulate(causalize(flatten(parse(SOURCE), "Tube")), stop=1.0)
    for i, a in enumerate((1.0, 2.0, 3.0), start=1):
        assert result[f"p[{i}]"][-1] == pytest.approx(20 / a, rel=1e-6)


def test_matrix_times_vector_in_an_equation() -> None:
    """``der(x) = -K * x`` with a diagonal ``K`` decays each element at its own rate."""
    np = pytest.importorskip("numpy")
    source = """\
model M
  parameter Real K[2, 2] = [1, 0; 0, 2];
  Real x[2](each start = 1, each fixed = true);
equation
  der(x) = -K * x;
end M;
"""
    result = simulate(causalize(flatten(parse(source), "M")), stop=1.0)
    assert result["x[1]"][-1] == pytest.approx(np.exp(-1.0), rel=1e-5)
    assert result["x[2]"][-1] == pytest.approx(np.exp(-2.0), rel=1e-5)


def test_when_assigns_a_whole_array() -> None:
    """``h = A`` in a ``when`` sets each element of ``h`` at the event."""
    pytest.importorskip("numpy")
    source = """\
model W
  parameter Real A[2] = {3, 4};
  discrete Real h[2](each start = 0, each fixed = true);
equation
  when time > 0.5 then
    h = A;
  end when;
end W;
"""
    result = simulate(causalize(flatten(parse(source), "W")), stop=1.0)
    assert result["h[1]"][0] == 0
    assert result["h[1]"][-1] == pytest.approx(3.0)
    assert result["h[2]"][-1] == pytest.approx(4.0)


SIZE_CAT = """\
model S
  parameter Real A[3] = {1, 2, 3};
  parameter Integer M = size(A, 1);
  parameter Real B[M + 1] = cat(1, A, {4});
  Real x[M](each start = 0, each fixed = true);
equation
  der(x) = A;
end S;
"""


def test_size_sets_a_dimension_and_cat_joins_arrays() -> None:
    """``M = size(A, 1)`` is 3, and ``cat(1, A, {4})`` gives ``B = {1, 2, 3, 4}`` (#3)."""
    model = flatten(parse(SIZE_CAT), "S")
    assert "x[3]" in model
    assert "x[4]" not in model
    values = {"A[1]": 1.0, "A[2]": 2.0, "A[3]": 3.0}
    for i, expected in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        binding = model[f"B[{i}]"].binding
        assert binding is not None
        assert evaluate(binding, values) == expected
