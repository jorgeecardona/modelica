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


@pytest.mark.slow
@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_event_iteration_settles(model: Path) -> None:
    """At the initial instant, from every configuration the latches could be in."""
    report = check_iteration(_system(model))
    assert report.ok, str(report)


@pytest.mark.slow
@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_event_iteration_settles_at_every_event(model: Path) -> None:
    """And at every instant a real run stopped at, which is where events actually are.

    A model that cannot be simulated yet has no instants to offer; the reason it cannot is
    recorded once, in ``test_examples.UNSIMULATED``, and read from there so the two suites
    cannot disagree about what is still missing.

    A model with no watched relations has none either, and skipping it is not a weakening.
    Read `check_run`: it returns ``[check_iteration(kernel)]`` and then loops over
    ``result.events``. With no events the loop body never runs, so the whole trajectory is
    computed and thrown away and what is left is exactly what
    `test_event_iteration_settles` already asserts, above. It was a duplicate that paid for a
    full integration -- **31.7 s on `AtwoodPendulum` alone**, of a 144.6 s file, for an
    assertion made two functions up.
    """
    if model.stem in UNSIMULATED:
        pytest.skip(UNSIMULATED[model.stem])
    build = _require("modelica.sim", "build")
    simulate = _require("modelica.sim", "simulate")
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    run = spec.get("simulate")
    if run is None:
        pytest.skip("no simulate section")
    system = _system(model)
    if not build(system).events:
        pytest.skip("no watched relations, so `check_run` would only repeat the check at t = 0")
    result = simulate(
        system,
        stop=run["stop"],
        tolerance=run.get("tolerance", 1e-6),
        absolute=run.get("absolute"),
        method=run.get("method", "Radau"),
    )
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


CONTESTED = """model Contested "Two `when` bodies at one instant, and one reads what the other wrote"
  Real x(start = 3, fixed = true);
  discrete Real a(start = 0, fixed = true);
  discrete Real b(start = 0, fixed = true);
equation
  der(x) = -1;
  when x <= 2 then
    a = pre(a) + 1;
  end when;
  when x <= 2.5 then
    b = a;
  end when;
end Contested;
"""
"""A model whose fixed point depends on which `when` acts first.

Below ``x = 2`` both conditions hold, and ``b = a`` reads a variable the other branch
writes: act on the first and ``b`` ends at 1, act on the second and it ends at 0. Legal
Modelica -- each variable is assigned in exactly one ``when`` -- and exactly the shape
`docs/ideas/verification.md:118` said would arrive with discrete variables.

Written here rather than added to ``examples/models`` on purpose. The corpus is models worth
simulating; this is a model worth *checking*, and it is order-dependent by construction,
which is a property no example should quietly carry.
"""


def _contested() -> Any:
    parse = _require("modelica.lang", "parse")
    flatten = _require("modelica.sim", "flatten")
    causalize = _require("modelica.sim", "causalize")
    return causalize(flatten([parse(CONTESTED)], "Contested"))


def test_a_when_body_writing_memory_is_a_move_the_checker_sees() -> None:
    """The move `Thermostat` makes at its threshold, which used to be invisible.

    Firing the heat-on branch sets ``heating`` and increments ``cycles``, and neither is a
    latch or a continuous state -- so before the memory was part of `Instant`, the state
    after the event compared *equal* to the state before it, and the checker reported
    "settles in one place" about a model that had moved.
    """
    build = _require("modelica.sim", "build")
    numpy = pytest.importorskip("numpy")
    kernel = build(_system(MODELS / "Thermostat.mo"))
    assert kernel.code.remembered == ("heating", "cycles")

    report = check_iteration(kernel, y=numpy.array([18.0]))  # below T_low, so the branch fires
    assert report.ok, str(report)
    ((start, (end,)),) = report.outcomes.items()
    assert start.memory == (0.0, 0.0)
    assert end.memory == (1.0, 1.0), "the heater came on and the counter moved"
    assert end.state == start.state, "no continuous state was reinitialised"


def test_two_when_bodies_racing_over_memory_is_reported() -> None:
    """One starting configuration, two fixed points, and the memory is the whole difference.

    This is the finding the entry asked for: with only the latches in the state, both orders
    end at the same `Instant` and the report says the iteration settles in one place. It is
    also why the memory is *not* dropped from the outcomes the way ``fired`` is.
    """
    numpy = pytest.importorskip("numpy")
    report = check_iteration(_contested(), y=numpy.array([1.0]))
    assert not report.ok
    ((_, ends),) = report.divergent
    assert {end.memory for end in ends} == {(1.0, 0.0), (1.0, 1.0)}
    assert len({end.latches for end in ends}) == 1, "the latches agree; only the memory differs"


def test_instant_reads_as_its_memory_too() -> None:
    instant = Instant((True,), (1.0,), frozenset({0}), (0.0, 3.0))
    assert str(instant) == "latches=1 memory=0,3 fired=[0]"
