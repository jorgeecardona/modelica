"""One compiled model over a matrix of parameters.

Three things have to be true and each is checked differently.

*It agrees.* Every point of a sweep equals what `simulate` returns for that parameter value
on its own. That is the only claim that matters and it is differential, not a reference.

*It compiles once.* Counted rather than timed. `modelica.sim.kernel.cache_info` reports
misses, so "one compile for a hundred points" is an integer, and an integer does not flake on
a loaded machine the way a wall-clock comparison does.

*It changes the right things.* Overriding a parameter has to reach everything computed from
it -- the derived parameters *and* the initial conditions. That is the failure that produces a
smooth, plausible, wrong curve, so it gets the most direct test here.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim.emit import EmitError, settings
from modelica.sim.kernel import cache_info, clear_cache
from modelica.sim.sweep import grid, sweep, vector

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _system(stem: str) -> System:
    model = EXAMPLES / "models" / f"{stem}.mo"
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    flatten = pytest.importorskip("modelica.sim").flatten
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    return causalize(flatten(sources, stem))


def test_every_point_matches_the_same_run_on_its_own() -> None:
    """The whole claim, differentially. A sweep is not allowed to be its own reference."""
    numpy = pytest.importorskip("numpy")
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    system = _system("RCCircuit")
    points = grid(**{"r.R": [0.5, 1.0, 2.0], "c.C": [0.5, 1.5]})
    times = numpy.linspace(0.0, 3.0, 11)

    swept = sweep(system, points, stop=3.0, times=times)

    assert len(swept) == len(points)
    for point, mine in zip(points, swept, strict=True):
        alone = simulate(system, stop=3.0, times=times, parameters=vector(system, point))
        for name in mine.names:
            numpy.testing.assert_array_equal(mine[name], alone[name], err_msg=f"{point} {name}")


def test_a_hundred_points_are_one_compile() -> None:
    """Counted, not timed. An integer does not flake on a loaded machine."""
    pytest.importorskip("numpy")
    system = _system("RCCircuit")
    clear_cache()
    sweep(system, [{"r.R": 1.0}], stop=0.5)
    after_one = cache_info().misses

    sweep(system, [{"r.R": 1.0 + k / 100} for k in range(100)], stop=0.5)

    assert after_one == 1
    assert cache_info().misses == 1, "a hundred points compiled more than once"


def test_the_sweep_actually_changes_the_answer() -> None:
    """A driver that silently ignored the overrides would pass everything above.

    Every test before this one compares a sweep against `simulate` given *the same vector*,
    so a `sweep` that dropped its points on the floor and a `vector` that ignored its values
    would agree with each other perfectly.
    """
    numpy = pytest.importorskip("numpy")
    system = _system("BouncingBall")
    # The bounce, not the RC: a swept resistance changes only how fast the capacitor gets to
    # the same place, and by any time worth sampling both have arrived. A swept restitution
    # changes where the ball *is*.
    # 0.5 and not 0.3: a very inelastic ball reaches its Zeno point inside one second and
    # the run raises rather than returning, which would be testing the chatter guard.
    grid_times = numpy.linspace(0.0, 1.0, 9)
    dull, lively = sweep(system, [{"e": 0.5}, {"e": 0.9}], stop=1.0, times=grid_times)
    at = numpy.searchsorted(dull.time, grid_times), numpy.searchsorted(lively.time, grid_times)
    # A fixed grid, because the two runs bounce at different instants and so take different
    # steps: the solver's own samples are not the same length and cannot be compared at all.
    assert lively["h"][at[1]][-1] > dull["h"][at[0]][-1] * 2
    assert not numpy.allclose(lively["h"][at[1]], dull["h"][at[0]])


def test_an_override_reaches_the_parameters_computed_from_it() -> None:
    """`RoastingMachine` writes ``m_dry = m_green * (1 - moisture)``.

    Poking the parameter vector after the fact leaves `m_dry` derived from the old moisture:
    a model that is *nearly* the one asked for, which is worse than one that refuses.
    """
    system = _system("RoastingMachine")
    computed = settings(system, {"moisture": 0.20})
    assert computed["moisture"] == pytest.approx(0.20)
    assert computed["m_dry"] == pytest.approx(13.8 * 0.80)


def test_an_override_reaches_the_initial_conditions() -> None:
    """`m_w` starts at ``m_green * moisture``, so a swept moisture has to move the start.

    The other half of the same failure, and the one a sweep notices first: change a parameter
    the initial state is written in terms of, and the run has to *begin* somewhere else.
    """
    numpy = pytest.importorskip("numpy")
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    system = _system("RoastingMachine")

    swept = sweep(system, [{"moisture": 0.20}], stop=1.0, points=2)

    assert swept[0]["m_w"][0] == pytest.approx(13.8 * 0.20)
    default = simulate(system, stop=1.0, points=2)
    assert default["m_w"][0] == pytest.approx(13.8 * 0.13)
    assert not numpy.allclose(swept[0]["m_w"][0], default["m_w"][0])


def test_a_parameter_the_model_does_not_have_is_refused_by_name() -> None:
    """A typo in a sweep axis is a silent no-op otherwise, over every point."""
    system = _system("RCCircuit")
    with pytest.raises(EmitError, match=re.escape("no parameter 'r.Rr'")):
        vector(system, {"r.Rr": 1.0})


def test_the_default_vector_is_what_the_kernel_computes_for_itself() -> None:
    """Two implementations of one thing, so they have to agree.

    `vector` recomputes in Python what the emitted ``parameters`` function computes inside the
    generated module.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    for stem in ("RCCircuit", "RoastingMachine", "PhaseChange", "DCMotor"):
        system = _system(stem)
        emitted, _, _ = build(system).start(0.0)
        numpy.testing.assert_allclose(vector(system), emitted, rtol=1e-15, err_msg=stem)


def test_the_grid_is_the_cross_product_with_the_last_axis_fastest() -> None:
    assert grid(a=[1, 2], b=[10, 20, 30]) == (
        {"a": 1, "b": 10},
        {"a": 1, "b": 20},
        {"a": 1, "b": 30},
        {"a": 2, "b": 10},
        {"a": 2, "b": 20},
        {"a": 2, "b": 30},
    )


def test_points_do_not_share_arrays() -> None:
    """Nothing is carried from one point to the next.

    The property a process pool or a device queue would need, asserted while the loop is
    still a `for`: a sweep run backwards gives the same answers as one run forwards.
    """
    numpy = pytest.importorskip("numpy")
    system = _system("BouncingBall")
    points = [{"e": 0.5}, {"e": 0.7}, {"e": 0.9}]
    # Stopped at 1 s. Past about 2 s a lively ball reaches its Zeno point and the run raises
    # rather than returning, which would be testing the chatter guard and not this.
    forwards = sweep(system, points, stop=1.0, points=9)
    backwards = sweep(system, list(reversed(points)), stop=1.0, points=9)
    for mine, theirs in zip(forwards, reversed(backwards), strict=True):
        numpy.testing.assert_array_equal(mine["h"], theirs["h"])
