"""An equation written between whole arrays, scalarised element by element.

``Real x[2]; equation der(x) = -x;`` is two equations, ``der(x[1]) = -x[1]`` and
``der(x[2]) = -x[2]``, the same split the declaration already does for the variables. Before
this the array form reached ``emit`` intact and stopped at "array equations are not supported
yet", so a vector field had to be written as an explicit ``for`` loop over its elements (as
``TorusHeat`` still does). The loop form keeps working; this adds the whole-array form beside
it, including the per-element share of an array-valued ``start``.

What is *not* elementwise is refused rather than mis-scalarised: ``x * y`` between two arrays
is a product with no element-by-element reading, and it says so and names the loop as the way
to write the thing it cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import FlattenError
from modelica.lang import parse
from modelica.sim import flatten, simulate


def _system(source: str, name: str):  # noqa: ANN202 - the causalized System
    return causalize(flatten([parse(source)], name))


def test_a_vector_decay_scalarises_into_its_elements() -> None:
    """``der(x) = -x`` for ``x[2]`` becomes two scalar states that decay independently."""
    system = _system(
        """
        model Vec
          Real x[2](start = {1, 2}, fixed = true);
        equation
          der(x) = -x;
        end Vec;
        """,
        "Vec",
    )
    assert sorted(str(s) for s in system.states) == ["x[1]", "x[2]"]

    result = simulate(system, stop=2.0, tolerance=1e-10)
    time = result.time
    # The per-element start comes from the array literal: x[1] starts at 1, x[2] at 2, and
    # each is its own e^{-t}. If the start had not been split too, both would start at the
    # whole literal and the model would not have built at all.
    assert np.max(np.abs(result["x[1]"] - np.exp(-time))) < 1e-9
    assert np.max(np.abs(result["x[2]"] - 2 * np.exp(-time))) < 1e-9


def test_a_scalar_broadcasts_across_the_array() -> None:
    """``der(x) = -a * x + 1`` mixes a scalar parameter and a scalar literal into a vector.

    Each element settles to ``1 / a`` along ``x(t) = 1/a + (x0 - 1/a) e^{-a t}``; the point is
    that ``a`` and ``1`` are left whole while ``x`` is indexed, which is the broadcast an
    equation between an array and a scalar has to do.
    """
    system = _system(
        """
        model Scaled
          parameter Real a = 2;
          Real x[2](start = {1, 1}, fixed = true);
        equation
          der(x) = -a * x + 1;
        end Scaled;
        """,
        "Scaled",
    )
    result = simulate(system, stop=3.0, tolerance=1e-10)
    time = result.time
    exact = 0.5 + 0.5 * np.exp(-2.0 * time)
    assert np.max(np.abs(result["x[1]"] - exact)) < 1e-9
    assert np.max(np.abs(result["x[2]"] - exact)) < 1e-9


def test_a_matrix_equation_scalarises_with_per_element_starts() -> None:
    """``der(T) = -T`` for ``T[2, 2]`` with a nested-literal start decays cell by cell."""
    system = _system(
        """
        model Grid2
          Real T[2, 2](start = {{1, 2}, {3, 4}}, fixed = true);
        equation
          der(T) = -T;
        end Grid2;
        """,
        "Grid2",
    )
    assert sorted(str(s) for s in system.states) == [
        "T[1, 1]",
        "T[1, 2]",
        "T[2, 1]",
        "T[2, 2]",
    ]
    result = simulate(system, stop=1.0, tolerance=1e-10)
    time = result.time
    for name, start in (("T[1, 1]", 1), ("T[1, 2]", 2), ("T[2, 1]", 3), ("T[2, 2]", 4)):
        assert np.max(np.abs(result[name] - start * np.exp(-time))) < 1e-9


def test_an_elementwise_function_call_scalarises() -> None:
    """``-sin(x)`` over a vector is ``-sin(x[i])`` per element: a call broadcasts."""
    system = _system(
        """
        model Func
          Real x[2](start = {0, 1}, fixed = true);
        equation
          der(x) = -sin(x);
        end Func;
        """,
        "Func",
    )
    assert sorted(str(s) for s in system.states) == ["x[1]", "x[2]"]


def test_an_array_product_is_refused_and_names_the_loop() -> None:
    """``x * y`` between two arrays is a contraction, not elementwise: refused by name."""
    with pytest.raises(FlattenError, match="array product is not supported"):
        _system(
            """
            model Prod
              Real x[2](start = {1, 1}, fixed = true);
              Real y[2](start = {1, 1}, fixed = true);
            equation
              der(x) = x * y;
              der(y) = -y;
            end Prod;
            """,
            "Prod",
        )


def test_a_scalar_start_still_broadcasts_over_a_sized_array() -> None:
    """A regression guard: ``Real T[n](start = T0)`` gives every element the same start.

    The declaration path relies on a scalar start broadcasting, the way ``HeatBar`` and
    ``TorusHeat`` write it; indexing an array-valued start must not have started indexing a
    scalar one too.
    """
    system = _system(
        """
        model Broadcast
          parameter Integer n = 3;
          parameter Real x0 = 5;
          Real x[n](start = x0, fixed = true);
        equation
          der(x) = -x;
        end Broadcast;
        """,
        "Broadcast",
    )
    result = simulate(system, stop=1.0, tolerance=1e-10)
    time = result.time
    for name in ("x[1]", "x[2]", "x[3]"):
        assert np.max(np.abs(result[name] - 5.0 * np.exp(-time))) < 1e-9
