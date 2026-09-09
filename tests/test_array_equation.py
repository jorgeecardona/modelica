"""An equation written between whole arrays, scalarised element by element.

``Real x[2]; equation der(x) = -x;`` is two equations, ``der(x[1]) = -x[1]`` and
``der(x[2]) = -x[2]``, the same split the declaration already does for the variables. Before
this the array form reached ``emit`` intact and stopped at "array equations are not supported
yet", so a vector field had to be written as an explicit ``for`` loop over its elements (as
``TorusHeat`` still does). The loop form keeps working; this adds the whole-array form beside
it, including the per-element share of an array-valued ``start``.

A ``*`` between two arrays is a contraction, not elementwise, and is scalarised as such: the
dot product ``a * x``, the matrix-vector ``A * x`` (``der(x) = A * x``, the linear system), and
the matrix-matrix ``A * B`` all become the sums they stand for. Array ``/`` and ``^`` have no
such reading and are refused, naming the loop as the way to write them.

A reduction folds a whole array into one scalar: ``sum(x)`` is ``x[1] + ... + x[n]``,
``product(x)`` the same with ``*``, and ``min(x)`` / ``max(x)`` fold through the pairwise
``min(a, b)`` / ``max(a, b)``. The fold happens at flatten, so causalization, the Jacobian and
every emitter see ordinary arithmetic and need no reduction rule of their own; the two-argument
``min(a, b)`` stays the pairwise operator it was. And the fold reaches the scalar contexts an
equation split never touches -- a ``when`` condition (``when sum(x) < 1``), an assert
(``assert(sum(x) < top, ...)``) -- through the flattener's expression rewrite, so a reduction
there means what it says instead of reaching the emitter as an untranslated ``sum``.
"""

from __future__ import annotations

import numpy as np
import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import FlattenError
from modelica.lang import parse
from modelica.sim import flatten, simulate
from modelica.sim.solve import SimulationError


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


def test_a_matrix_vector_product_is_the_linear_system() -> None:
    """``der(x) = A * x`` with a rotation matrix turns ``x`` in the plane: cos and sin.

    The case the whole feature is named for. ``(A * x)[i]`` is the sum over ``j`` of
    ``A[i, j] * x[j]``, and the implicit solver's Jacobian of it has to be ``A`` for the run to
    hold its shape -- the closed form catches a wrong contraction and a wrong derivative alike.
    """
    system = _system(
        """
        model Rotation
          parameter Real A[2, 2] = {{0, -1}, {1, 0}};
          Real x[2](start = {1, 0}, fixed = true);
        equation
          der(x) = A * x;
        end Rotation;
        """,
        "Rotation",
    )
    result = simulate(system, stop=3.0, tolerance=1e-10)
    time = result.time
    assert np.max(np.abs(result["x[1]"] - np.cos(time))) < 1e-9
    assert np.max(np.abs(result["x[2]"] - np.sin(time))) < 1e-9


def test_a_dot_product_scalarises_to_a_sum() -> None:
    """``s = a * x`` between two vectors is the scalar sum over the elements."""
    system = _system(
        """
        model Dot
          parameter Real a[3] = {1, 2, 3};
          Real x[3](start = {1, 1, 1}, fixed = true);
          Real s;
        equation
          der(x) = -x;
          s = a * x;
        end Dot;
        """,
        "Dot",
    )
    result = simulate(system, stop=1.0, tolerance=1e-10)
    # a . x = (1 + 2 + 3) * e^{-t} = 6 e^{-t}.
    assert np.max(np.abs(result["s"] - 6.0 * np.exp(-result.time))) < 1e-9


def test_a_matrix_matrix_product_scalarises() -> None:
    """``C = A * B`` is the matrix product, each entry a sum over the shared dimension."""
    system = _system(
        """
        model MatMul
          parameter Real A[2, 2] = {{2, 0}, {0, 3}};
          parameter Real B[2, 2] = {{1, 4}, {5, 1}};
          Real C[2, 2];
          Real k(start = 0, fixed = true);
        equation
          C = A * B;
          der(k) = 0;
        end MatMul;
        """,
        "MatMul",
    )
    result = simulate(system, stop=0.1, tolerance=1e-10)
    # [[2,0],[0,3]] * [[1,4],[5,1]] = [[2,8],[15,3]].
    assert result["C[1, 1]"][0] == pytest.approx(2.0)
    assert result["C[1, 2]"][0] == pytest.approx(8.0)
    assert result["C[2, 1]"][0] == pytest.approx(15.0)
    assert result["C[2, 2]"][0] == pytest.approx(3.0)


def test_sum_reduces_an_array_to_a_scalar() -> None:
    """``sum(x)`` folds the whole array: ``x[1] + x[2] + x[3]``, not ``sum(x[1])`` per element."""
    system = _system(
        """
        model Sum
          Real x[3](each start = 1, each fixed = true);
          Real total;
        equation
          der(x) = -x;
          total = sum(x);
        end Sum;
        """,
        "Sum",
    )
    result = simulate(system, stop=1.0, tolerance=1e-10)
    # sum of three copies of e^{-t} is 3 e^{-t}.
    assert np.max(np.abs(result["total"] - 3.0 * np.exp(-result.time))) < 1e-9


def test_product_reduces_an_array_to_a_scalar() -> None:
    """``product(x)`` folds with ``*``: three copies of ``2 e^{-t}`` multiply to ``8 e^{-3t}``."""
    system = _system(
        """
        model Prod
          Real x[3](each start = 2, each fixed = true);
          Real p;
        equation
          der(x) = -x;
          p = product(x);
        end Prod;
        """,
        "Prod",
    )
    result = simulate(system, stop=1.0, tolerance=1e-10)
    assert np.max(np.abs(result["p"] - 8.0 * np.exp(-3.0 * result.time))) < 1e-8


def test_min_and_max_reduce_a_single_array() -> None:
    """``min(x)`` / ``max(x)`` fold through the pairwise operators; the extremes track the ends."""
    system = _system(
        """
        model Extremes
          Real x[3](start = {1, 2, 3}, fixed = true);
          Real lo;
          Real hi;
        equation
          der(x) = -x;
          lo = min(x);
          hi = max(x);
        end Extremes;
        """,
        "Extremes",
    )
    result = simulate(system, stop=0.5, tolerance=1e-10)
    # Each x[i] decays keeping its order, so min is x[1] = e^{-t} and max is x[3] = 3 e^{-t}.
    assert np.max(np.abs(result["lo"] - np.exp(-result.time))) < 1e-9
    assert np.max(np.abs(result["hi"] - 3.0 * np.exp(-result.time))) < 1e-9


def test_a_two_argument_min_is_still_the_pairwise_operator() -> None:
    """The reduction is only for a single array; ``min(a, b)`` of two scalars is untouched."""
    system = _system(
        """
        model Pair
          Real x(start = 1, fixed = true);
          Real y(start = 3, fixed = true);
          Real lo;
        equation
          der(x) = -x;
          der(y) = 0;
          lo = min(x, y);
        end Pair;
        """,
        "Pair",
    )
    result = simulate(system, stop=1.0, tolerance=1e-10)
    # x = e^{-t} < y = 3 throughout, so min(x, y) = x.
    assert np.max(np.abs(result["lo"] - np.exp(-result.time))) < 1e-9


def test_a_reduction_inside_a_derivative_builds_its_jacobian() -> None:
    """``der(x[2]) = -x[2] + sum(x)`` is a coupled system whose Jacobian needs the folded sum."""
    system = _system(
        """
        model Coupled
          Real x[2](start = {1, 2}, fixed = true);
        equation
          der(x[1]) = -x[1];
          der(x[2]) = -x[2] + sum(x);
        end Coupled;
        """,
        "Coupled",
    )
    result = simulate(system, stop=1.0, tolerance=1e-10)
    # sum(x) = x[1] + x[2], so der(x[2]) = x[1] = e^{-t}; from x[2](0) = 2 that is 3 - e^{-t}.
    assert np.max(np.abs(result["x[1]"] - np.exp(-result.time))) < 1e-9
    assert np.max(np.abs(result["x[2]"] - (3.0 - np.exp(-result.time)))) < 1e-8


def test_a_two_dimensional_sum_folds_every_element() -> None:
    """``sum(T)`` over a 2-D array folds all of its elements, not one row."""
    system = _system(
        """
        model Grid
          Real T[2, 2](each start = 1, each fixed = true);
          Real s;
        equation
          der(T) = -T;
          s = sum(T);
        end Grid;
        """,
        "Grid",
    )
    result = simulate(system, stop=1.0, tolerance=1e-10)
    assert np.max(np.abs(result["s"] - 4.0 * np.exp(-result.time))) < 1e-9


def test_a_reduction_in_a_when_condition_fires_at_the_right_instant() -> None:
    """``when sum(x) < 1`` is a state event on the folded total, not an untranslated ``sum``.

    A ``when`` condition never passes through the equation split, so before the fold reached the
    flattener's expression rewrite this stopped at the emitter with "no translation for 'sum'".
    Now the total is folded there and the event is an ordinary crossing of ``x[1] + x[2]``.
    """
    system = _system(
        """
        model WhenSum
          Real x[2](start = {1, 1}, each fixed = true);
          discrete Real hit(start = 0, fixed = true);
        equation
          der(x[1]) = -x[1];
          der(x[2]) = -x[2];
          when sum(x) < 1.0 then
            hit = 1.0;
          end when;
        end WhenSum;
        """,
        "WhenSum",
    )
    result = simulate(system, stop=2.0, tolerance=1e-10)
    # sum(x) = 2 e^{-t} crosses 1 at t = ln 2; hit is 0 before and 1 after.
    time = result.time
    hit = np.asarray(result["hit"])
    assert hit[0] == 0.0
    assert hit[-1] == 1.0
    crossing = time[np.argmax(hit > 0.5)]
    assert crossing == pytest.approx(np.log(2.0), abs=1e-2)


def test_a_reduction_in_an_assert_is_evaluated_not_skipped() -> None:
    """``assert(sum(x) < top, ...)`` guards the total, and a violated guard stops the run.

    An assert whose condition named the array ``x`` (rather than its elements) was quietly
    skipped, because the run reports ``x[1]``/``x[2]`` and not ``x``: a conservation guard that
    did nothing. With the reduction folded the condition names the reported elements, so it is
    evaluated -- it raises when violated and passes when satisfied.
    """
    source = """
    model AssertSum
      Real x[2](start = {1, 1}, each fixed = true);
    equation
      der(x[1]) = -x[1];
      der(x[2]) = -x[2];
      assert(sum(x) < BOUND, "sum too large");
    end AssertSum;
    """
    # sum(x) starts at 2: a bound of 1.5 is violated at t = 0, a bound of 3 never is.
    with pytest.raises(SimulationError, match="sum too large"):
        simulate(_system(source.replace("BOUND", "1.5"), "AssertSum"), stop=1.0, tolerance=1e-10)
    result = simulate(
        _system(source.replace("BOUND", "3.0"), "AssertSum"), stop=1.0, tolerance=1e-10
    )
    assert result["x[1]"][-1] == pytest.approx(np.exp(-1.0), rel=1e-7)


def test_array_divide_and_power_are_refused_and_name_the_loop() -> None:
    """Only ``*`` contracts; array ``/`` and ``^`` have no reading and say so."""
    for operator in ("/", "^"):
        with pytest.raises(FlattenError, match="is not supported"):
            _system(
                f"""
                model Bad
                  Real x[2](start = {{1, 1}}, fixed = true);
                  Real y[2](start = {{1, 1}}, fixed = true);
                equation
                  der(x) = x {operator} y;
                  der(y) = -y;
                end Bad;
                """,
                "Bad",
            )


def test_a_nonconforming_product_is_refused() -> None:
    """A matrix and a vector whose inner dimensions differ cannot be multiplied."""
    with pytest.raises(FlattenError, match="do not conform for a product"):
        _system(
            """
            model Mismatch
              parameter Real M[2, 3] = {{1, 2, 3}, {4, 5, 6}};
              parameter Real w[2] = {1, 1};
              Real y[2];
              Real k(start = 0, fixed = true);
            equation
              y = M * w;
              der(k) = 0;
            end Mismatch;
            """,
            "Mismatch",
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
