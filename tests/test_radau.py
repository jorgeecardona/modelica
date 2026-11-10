"""The fused Radau loop, checked against what the corpus already knows.

`modelica.sim.radau` is a second integrator, so the first question is not whether it is fast
but whether it is right. The evidence used here is the evidence the corpus carries: every
``.toml`` with a ``[simulate]`` section lists reference points with a source -- ``analytic``
for a closed form, ``reference`` for a trusted solver -- and `tests/test_examples.py` already
holds the SciPy path to them. Holding a second integrator to the *same* numbers is a stronger
statement than holding it to the first one's output, because neither is grading the other.

"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten
from modelica.sim.kernel import build
from modelica.sim.radau import (
    FusionError,
    _factor,  # pyright: ignore[reportPrivateUsage]
    _solve,  # pyright: ignore[reportPrivateUsage]
    integrate,
    simulate,
)

if TYPE_CHECKING:
    from modelica.ir.system import System

# Whole file is slow: it integrates every corpus model with the fused Radau loop. `make test`
# skips it; CI and `make test-all` run it.
pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MODELS = sorted((EXAMPLES / "models").glob("*.mo"))


def _system(model: Path) -> System:
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec.get("requires", [])
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    return causalize(flatten(sources, model.stem))


def _spec(model: Path) -> dict[str, Any]:
    return tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_the_fused_loop_reaches_the_reference_points(model: Path) -> None:
    """Every point the corpus publishes, from this integrator, at the stated tolerance."""
    spec: dict[str, Any] | None = _spec(model).get("simulate")
    if spec is None:
        pytest.skip("no simulate section")
    points: list[dict[str, Any]] = spec.get("point", [])
    if not points:
        pytest.skip("no reference points")
    system = _system(model)
    if build(system).events:
        pytest.skip("the fused loop has no event handling")

    result, stats = simulate(
        system,
        stop=spec["stop"],
        tolerance=spec.get("tolerance", 1e-6),
        absolute=spec.get("absolute"),
        times=sorted({point["t"] for point in points}),
    )
    assert stats.status == 0
    for point in points:
        sampled = result.at(point["t"])
        for name, expected in point["values"].items():
            got = sampled[name]
            assert np.isclose(
                got, expected, rtol=point.get("rtol", 0.0), atol=point.get("atol", 0.0)
            ), f"{name} at t={point['t']}: got {got!r}, want {expected!r} ({point['source']})"


def test_a_model_with_events_is_refused_by_name() -> None:
    """And refused with the alternative in the message, not merely with a type."""
    system = _system(EXAMPLES / "models" / "BouncingBall.mo")
    with pytest.raises(FusionError, match="event"):
        simulate(system, stop=1.0)


def test_folding_the_parameters_does_not_change_the_answer() -> None:
    """The two compile modes are the emitter's, and the loop is indifferent to which ran.

    ``fold=True`` burns the parameters into the generated source as literals; ``fold=False``
    keeps them as an argument so one artefact serves a sweep. Same trajectory either way, or
    the folding is not a specialisation but a second model.
    """
    system = _system(EXAMPLES / "models" / "Robertson.mo")
    grid = np.linspace(0.0, 400.0, 17)
    loose, _ = simulate(system, stop=400.0, times=grid, tolerance=1e-9, absolute=1e-9)
    tight, _ = simulate(system, stop=400.0, times=grid, tolerance=1e-9, absolute=1e-9, fold=True)
    for name in system.states:
        assert np.allclose(loose[name], tight[name], rtol=0.0, atol=0.0)


def test_a_requested_time_is_interpolated_and_does_not_move_the_steps() -> None:
    """Dense output, and the property that makes it worth having.

    The state at t = 3 is the same whether or not t = 1, 2 and 4 were also asked for. If it
    were not, the grid would be steering the integrator, and a coarse output request would be
    quietly buying a different answer from a fine one.
    """
    system = _system(EXAMPLES / "models" / "VanDerPol.mo")
    dense, _ = simulate(system, stop=5.0, points=101, tolerance=1e-8, absolute=1e-8)
    sparse, _ = simulate(system, stop=5.0, times=[5.0], tolerance=1e-8, absolute=1e-8)
    for name in system.states:
        assert dense[name][-1] == pytest.approx(sparse[name][-1], rel=0.0, abs=0.0)


def test_the_counters_add_up() -> None:
    """A step costs at least one Newton iteration, and an iteration is three stages."""
    system = _system(EXAMPLES / "models" / "Robertson.mo")
    _, stats = simulate(system, stop=1e4, points=11, tolerance=1e-8, absolute=1e-8)
    assert stats.steps > 0
    assert stats.evaluations >= 3 * stats.steps
    assert stats.jacobians >= 1
    assert stats.factorisations >= 1
    assert stats.time == pytest.approx(1e4)


def test_the_loop_reports_where_it_stopped_rather_than_pretending() -> None:
    """A right-hand side that blows up ends as a status, not as a wrong trajectory."""

    def rhs(t: float, y: object, p: object, w: object, out: object) -> object:
        del t, p, w
        out[0] = 1.0 / (1.0 - y[0])  # type: ignore[index]
        return out

    def jac(t: float, y: object, p: object, w: object, out: object) -> object:
        del t, p, w
        out[0, 0] = 1.0 / (1.0 - y[0]) ** 2  # type: ignore[index]
        return out

    grid = np.linspace(0.0, 10.0, 11)
    out = np.zeros((11, 1))
    status, *_ = integrate(
        rhs, jac, 0.0, 10.0, np.zeros(1), np.zeros(0), np.zeros(0), 1e-8, 1e-8, grid, out
    )
    assert status != 0


@pytest.mark.parametrize("size", [1, 2, 5, 9])
def test_the_factorisation_solves_what_numpy_solves(size: int) -> None:
    """Both element types, against the library this deliberately does not call."""
    rng = np.random.default_rng(size)
    for matrix in (
        rng.standard_normal((size, size)),
        rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size)),
    ):
        b = rng.standard_normal(size).astype(matrix.dtype)
        wanted = np.linalg.solve(matrix, b)
        work = matrix.copy()
        piv = np.empty(size, dtype=np.int64)
        assert _factor(work, piv)
        got = b.copy()
        _solve(work, piv, got)
        assert np.allclose(got, wanted, rtol=1e-9, atol=1e-12)


def test_a_singular_matrix_is_reported_and_not_divided_by() -> None:
    """The pivot search returning zero is the only thing standing between here and a NaN."""
    matrix = np.array([[1.0, 2.0], [2.0, 4.0]])
    piv = np.empty(2, dtype=np.int64)
    assert not _factor(matrix, piv)


def test_pivoting_actually_happens() -> None:
    """A zero on the diagonal is not a singular matrix, and the row swap is what says so."""
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    piv = np.empty(2, dtype=np.int64)
    assert _factor(matrix, piv)
    b = np.array([3.0, 5.0])
    _solve(matrix, piv, b)
    assert np.allclose(b, [5.0, 3.0])
