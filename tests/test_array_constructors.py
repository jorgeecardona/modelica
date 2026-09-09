"""The array-constructor functions in a declaration.

``zeros``, ``ones``, ``fill``, ``identity`` and ``linspace`` are the ordinary way to give an array
declaration a start or a value -- ``Real x[n](start =
zeros(n))``, ``parameter Real A[n, n] = identity(n)``, ``parameter Real g[n] = linspace(0, 1, n)``.
Each one is array-valued and structural, so it is expanded at flatten into the literal array of the
declared shape, exactly as a written ``{...}`` would be; before that the whole call reached the
emitter as an untranslated function and the model would not build.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import FlattenError
from modelica.lang import parse
from modelica.sim import flatten, simulate


def _run(source: str, name: str, stop: float = 1.0):  # noqa: ANN202 - a simulated Result
    return simulate(causalize(flatten([parse(source)], name)), stop=stop, tolerance=1e-11)


def test_ones_and_zeros_fill_a_start() -> None:
    """``start = ones(n)`` gives every element 1, ``start = zeros(n)`` gives every element 0."""
    ones = _run(
        "model M Real x[3](start = ones(3), each fixed = true); equation der(x) = -x; end M;", "M"
    )
    for i in range(1, 4):
        assert np.max(np.abs(ones[f"x[{i}]"] - np.exp(-ones.time))) < 1e-9
    # zeros from der(x) = -x - 1 settles to -1 along -1 + e^{-t}.
    zeros = _run(
        "model M Real x[2](start = zeros(2), each fixed = true); equation der(x) = -x - 1; end M;",
        "M",
    )
    assert np.max(np.abs(zeros["x[1]"] - (-1 + np.exp(-zeros.time)))) < 1e-9


def test_fill_repeats_its_value() -> None:
    """``fill(v, n)`` fills the array with ``v``: three copies of ``2`` decay as ``2 e^{-t}``."""
    result = _run(
        "model M Real x[3](start = fill(2.0, 3), each fixed = true); equation der(x) = -x; end M;",
        "M",
    )
    assert np.max(np.abs(result["x[2]"] - 2.0 * np.exp(-result.time))) < 1e-9


def test_identity_is_the_unit_matrix_not_all_ones() -> None:
    """``identity(n)`` is 1 on the diagonal and 0 off it: ``der(x) = identity * x`` does not mix.

    A wrong expansion (all ones, say) would couple the two states, so each growing exponential is
    what pins the off-diagonal zeros: ``der(x) = A x`` with ``A = I`` leaves ``x_i = x_i(0) e^{t}``.
    """
    result = _run(
        "model M parameter Real A[2, 2] = identity(2); Real x[2](start = {1, 2}, each fixed = true);"
        " equation der(x) = A * x; end M;",
        "M",
        stop=0.5,
    )
    assert np.max(np.abs(result["x[1]"] - np.exp(result.time))) < 1e-8
    assert np.max(np.abs(result["x[2]"] - 2.0 * np.exp(result.time))) < 1e-8


def test_linspace_steps_evenly_with_exact_endpoints() -> None:
    """``linspace(0, 2, 3)`` is ``{0, 1, 2}``: evenly spaced, and the endpoints are exact."""
    result = _run(
        "model M parameter Real g[3] = linspace(0, 2, 3); Real y(start = 0, fixed = true);"
        " Real z(start = 0, fixed = true); equation der(y) = g[2]; der(z) = g[3]; end M;",
        "M",
    )
    # g[2] = 1 so y = t, g[3] = 2 (the exact endpoint) so z = 2 t.
    assert result["y"][-1] == pytest.approx(1.0, abs=1e-9)
    assert result["z"][-1] == pytest.approx(2.0, abs=1e-9)


def test_identity_of_a_non_square_shape_is_refused() -> None:
    """``identity`` on a shape that is not square is a modelling error, named as such."""
    with pytest.raises(FlattenError, match="identity"):
        causalize(
            flatten(
                [
                    parse(
                        "model M parameter Real A[2, 3] = identity(2);"
                        " Real y(start = 0, fixed = true); equation der(y) = A[1, 1]; end M;"
                    )
                ],
                "M",
            )
        )


def test_a_two_dimensional_fill_covers_every_cell() -> None:
    """``fill(v, n, m)`` fills a matrix: every cell of the 2x2 starts at 3."""
    result = _run(
        "model M Real T[2, 2](start = fill(3.0, 2, 2), each fixed = true); equation"
        " der(T) = -T; end M;",
        "M",
    )
    for cell in ("T[1, 1]", "T[1, 2]", "T[2, 1]", "T[2, 2]"):
        assert result[cell][-1] == pytest.approx(3.0 * math.exp(-1.0), rel=1e-7)
