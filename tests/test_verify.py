"""The discrete half of every example, enumerated rather than sampled.

:mod:`modelica.verify` claims that the event iteration settles, and settles in one place,
from *every* latch configuration. That claim is worth having as a test for two reasons: it
is the property :mod:`modelica.sim.solve` relies on when it applies the first stale event it
finds rather than choosing an order, and it is a property no simulation can establish,
because a run only ever visits the configurations its own trajectory reaches.

The interesting instants are the ones a run walks past, so the corpus is simulated first and
each event's instant checked afterwards.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from modelica.verify import (
    Instant,
    check_iteration,
    check_run,
    check_selection,
    iteration,
    settled,
)
from test_examples import UNSIMULATED

if TYPE_CHECKING:
    from collections.abc import Callable

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LIBRARY = EXAMPLES / "library"
MODELS = EXAMPLES / "models"

MODEL_FILES = sorted(MODELS.glob("*.mo"))


def _ids(paths: list[Path]) -> list[str]:
    return [path.stem for path in paths]


def _require(module: str, what: str) -> Callable[..., Any]:
    pytest.importorskip(module)
    imported = pytest.importorskip(module)
    if not hasattr(imported, what):
        pytest.skip(f"{module}.{what} is not written yet")
    return getattr(imported, what)


def _system(model: Path) -> Any:
    parse = _require("modelica.lang", "parse")
    flatten = _require("modelica.sim", "flatten")
    causalize = _require("modelica.sim", "causalize")
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    files = [LIBRARY / f"{name}.mo" for name in spec["requires"]] + [model]
    return causalize(flatten([parse(f.read_text(encoding="utf-8")) for f in files], model.stem))


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_event_iteration_settles(model: Path) -> None:
    """At the initial instant, from every configuration the latches could be in."""
    report = check_iteration(_system(model))
    assert report.ok, str(report)


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_event_iteration_settles_at_every_event(model: Path) -> None:
    """And at every instant a real run stopped at, which is where events actually are.

    A model that cannot be simulated yet has no instants to offer; the reason it cannot is
    recorded once, in ``test_examples.UNSIMULATED``, and read from there so the two suites
    cannot disagree about what is still missing.
    """
    if model.stem in UNSIMULATED:
        pytest.skip(UNSIMULATED[model.stem])
    simulate = _require("modelica.sim", "simulate")
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    run = spec.get("simulate")
    if run is None:
        pytest.skip("no simulate section")
    system = _system(model)
    result = simulate(system, stop=run["stop"], tolerance=run.get("tolerance", 1e-6))
    for report in check_run(system, result):
        assert report.ok, str(report)


def test_a_settled_state_has_nothing_to_do() -> None:
    """The terminal predicate and the actions have to agree, or a fixed point is a deadlock."""
    system = _system(MODELS / "DrainingTank.mo")
    model = iteration(system)
    for start in model.initial_states():
        for action in model.actions:
            assert not (settled(system, start) and action.enabled(start)), (
                f"{action.name} is enabled in a state called settled: {start}"
            )


def test_every_configuration_is_a_starting_point() -> None:
    """A model with one watched relation is checked from both of its configurations.

    Not a detail: the configuration a run happens to start in is one of two, and the bug
    this module exists to find is the one hiding in the other.
    """
    starts = iteration(_system(MODELS / "DrainingTank.mo")).initial_states()
    assert {start.latches for start in starts} == {(False,), (True,)}
    assert all(start.fired == frozenset() for start in starts)


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_state_selection_always_recovers(model: Path) -> None:
    """Switching state sets never runs out of sets to switch to, and never chatters."""
    report = check_selection(_system(model))
    assert report.ok, str(report)


def test_the_pendulum_has_the_second_state_set_the_literature_names() -> None:
    """``(x, vx)`` is the whole point: it is regular exactly where ``(y, vy)`` is not.

    It is two exchanges away from the chosen set, which is why the candidates are
    enumerated rather than found by perturbing the selection one variable at a time.

    And there are exactly two. The four unknowns admit five structurally valid selections,
    but the three mixed ones -- ``(x, vy)``, ``(y, vx)``, ``(vx, vy)`` -- take a velocity
    without its position, which integrates the velocity forward while reading the position
    off the constraint with nothing left to say the two agree. A runtime switching into one
    of those would drift, so they are inadmissible rather than merely unhelpful.
    """
    candidates = _system(MODELS / "Pendulum.mo").candidates
    assert candidates == (("y", "vy"), ("x", "vx"))


def test_a_moving_jacobian_during_a_search_breaks_liveness() -> None:
    """The assumption `numerics` records is load-bearing, and this is the counterexample.

    Let the numerics change while the engine is part-way through trying candidates and a
    set can recover *after* being ruled out -- so the engine exhausts its list at a point
    where one of them was usable, and never accepts another step. Safety survives (nothing
    illegal happened, and the environment can always still move, so it is not a deadlock);
    liveness is what dies, which is why the check asks for both.
    """
    report = check_selection(_system(MODELS / "Pendulum.mo"), quiescent=False)
    assert report.safety.ok, "no invariant is broken -- the failure is not a safety one"
    assert not report.liveness.ok
    assert report.liveness.kind == "p-unreachable"


def test_instant_reads_as_its_latches() -> None:
    assert str(Instant((True, False), (1.0,), frozenset({2}))) == "latches=10 fired=[2]"
