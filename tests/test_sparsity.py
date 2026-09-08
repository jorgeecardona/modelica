"""The Jacobian's pattern, and the sparse output that fills it.

`docs/architecture.md:120` and `docs/ideas/benchmarks.md:63` both claim an analytic *and
sparse* Jacobian comes free from the incidence matching. Free means read off the schedule --
not probed, not coloured, not sampled -- and `Mode.sparsity` is that read.

A sparsity pattern is allowed to be wrong in exactly one direction. Claiming an entry can be
non-zero when it is always zero costs arithmetic; claiming an entry is always zero when it is
not loses a term, and the solver then converges to the wrong thing or not at all. Both tests
below are written for that asymmetry: the pattern must *cover* every non-zero the real
Jacobian has, and it is allowed to be looser.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MODELS = sorted((EXAMPLES / "models").glob("*.mo"))


def _system(model: Path) -> System:
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    flatten = pytest.importorskip("modelica.sim").flatten
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    return causalize(flatten(sources, model.stem))


@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_the_pattern_covers_every_entry_the_jacobian_actually_has(model: Path) -> None:
    """The direction that matters. A missing entry is a lost term, not a slow one."""
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)
    width = len(states)
    computed = numpy.zeros((width, width))
    kernel.jacobian(0.0, states, parameters, workspace, computed)

    pattern = system.sparsity

    assert len(pattern) == width
    outside = [
        (row, column)
        for row in range(width)
        for column in range(width)
        if computed[row, column] != 0.0 and column not in pattern[row]
    ]
    assert outside == [], f"{model.stem}: non-zero outside the pattern at {outside[:3]}"


@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_the_sparse_jacobian_rebuilds_the_dense_one_exactly(model: Path) -> None:
    """``jacobian_sparse`` writes the same numbers, in the pattern's own order.

    Exactly, not to a tolerance: it is the same expressions in the same order, gathered
    rather than stored whole, so any difference at all would be a difference in what was
    computed.
    """
    numpy = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    kernel = build(system)
    parameters, states, workspace = kernel.start(0.0)
    width = len(states)
    pattern = kernel.code.sparsity

    dense = numpy.zeros((width, width))
    kernel.jacobian(0.0, states, parameters, workspace, dense)
    indices = numpy.array([j for row in pattern for j in row], dtype=numpy.int32)
    indptr = numpy.cumsum([0, *(len(row) for row in pattern)], dtype=numpy.int32)
    data = numpy.zeros(len(indices))
    kernel.jacobian_sparse(0.0, states, parameters, workspace, data)

    rebuilt = sparse.csr_array((data, indices, indptr), shape=(width, width)).toarray()
    numpy.testing.assert_array_equal(rebuilt, dense)


def test_a_stencil_is_tridiagonal_and_the_pattern_says_so() -> None:
    """The shape the whole entry is about, at a size where dense storage is the problem.

    3n - 2 non-zeros out of n squared. At n = 1000 that is 0.3 per cent, and the pattern is
    computed from the schedule in a few milliseconds -- no Jacobian is evaluated to find it.
    """
    flatten = pytest.importorskip("modelica.sim").flatten
    size = 1000
    source = f"""
    model Bar
      parameter Integer n = {size};
      parameter Real a = 0.4;
      Real T[n](start = 20, fixed = true);
    equation
      der(T[1]) = a * (100 - 2 * T[1] + T[2]);
      for i in 2:n - 1 loop
        der(T[i]) = a * (T[i - 1] - 2 * T[i] + T[i + 1]);
      end for;
      der(T[n]) = a * (T[n - 1] - T[n]);
    end Bar;
    """
    pattern = causalize(flatten([parse(source)], "Bar")).sparsity

    assert sum(len(row) for row in pattern) == 3 * size - 2
    assert all(set(row) <= {index - 1, index, index + 1} for index, row in enumerate(pattern))


def test_only_the_grid_model_takes_the_sparse_path() -> None:
    """Exactly one model crosses the 16-state threshold onto the sparse path.

    Every hand-sized model is under it and stays dense, so its reference points come through the
    same linear algebra they always did; ``TorusHeat``, the 64-state 2-D stencil, is the one that
    crosses over. Asserted in both directions: a small model quietly switching to sparse would
    change the numerics under an existing reference point, and the grid model failing to switch
    would mean the corpus is not exercising the sparse path at all -- its references were taken
    on it.
    """
    pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    run = pytest.importorskip("modelica.sim.solve")
    for model in MODELS:
        system = _system(model)
        kernel = build(system)
        parameters, _, workspace = kernel.start(0.0)
        plan = run._Run(kernel, parameters, workspace, 1e-6, 1e-6, "Radau")  # noqa: SLF001
        expected = model.stem == "TorusHeat"
        assert plan.sparse == expected, (
            f"{model.stem}: sparse path is {plan.sparse}, expected {expected}"
        )
