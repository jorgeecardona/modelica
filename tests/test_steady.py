"""Where a model settles, and the two analyses agreeing about it.

`simulate` answers what a model does; `steady` answers where it ends up. The claim worth
testing is not that Newton converges -- it is that **the two analyses agree**, because that is
the "no model drift" argument made checkable instead of asserted: one set of equations, two
workflows, one answer.

So every model with a `[steady]` section is checked twice. Once against a closed form, which
is why each of those sections says `source = "analytic"`; and once against its own trajectory,
integrated far enough that the transient is gone.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim.steady import SteadyError, steady

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MODELS = sorted((EXAMPLES / "models").glob("*.mo"))

#: How long each is integrated for the agreement test. Long enough that the slowest mode has
#: died; no longer, because a `PIControl` run is not free.
SETTLES: dict[str, float] = {
    "NewtonCooling": 60.0,
    "RCCircuit": 40.0,
    "RLCCircuit": 60.0,
    "PIControl": 60.0,
}


def _spec(model: Path) -> Any:
    return tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))


def _system(model: Path) -> System:
    flatten = pytest.importorskip("modelica.sim").flatten
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in _spec(model)["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    return causalize(flatten(sources, model.stem))


WITH_STEADY = [m for m in MODELS if "steady" in _spec(m)]


def test_the_corpus_declares_some_equilibria() -> None:
    """A guard on the list below: an empty parametrize passes and proves nothing."""
    assert len(WITH_STEADY) >= 2, [m.stem for m in WITH_STEADY]


@pytest.mark.parametrize("model", WITH_STEADY, ids=[m.stem for m in WITH_STEADY])
def test_the_equilibrium_is_what_the_algebra_says(model: Path) -> None:
    pytest.importorskip("numpy")
    spec = _spec(model)["steady"]
    found = steady(_system(model))
    for name, expected in spec["values"].items():  # pyright: ignore[reportIndexIssue]
        assert found[name] == pytest.approx(
            expected,
            rel=spec.get("rtol", 0.0),
            abs=spec.get("atol", 0.0),  # pyright: ignore[reportAttributeAccessIssue]
        ), f"{name}: {found[name]!r} against {expected!r} ({spec['source']})"  # pyright: ignore[reportIndexIssue]


@pytest.mark.parametrize("model", WITH_STEADY, ids=[m.stem for m in WITH_STEADY])
def test_integrating_far_enough_arrives_where_the_solve_says(model: Path) -> None:
    """The two analyses, tied together. This is the claim the entry was filed for.

    Held to 1e-6 and not tighter: a trajectory approaches its equilibrium and never reaches
    it, so this measures how long `SETTLES` waits as much as it measures the solve. Tightening
    it would be tuning the horizon.
    """
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    stop = SETTLES[model.stem]
    found = steady(_system(model))
    run = simulate(_system(model), stop=stop, tolerance=1e-10)
    for name, value in zip(found.states, found.values, strict=True):
        assert run[name][-1] == pytest.approx(float(value), abs=1e-6), f"{name} after {stop}s"


def test_an_unstable_fixed_point_is_reported_and_not_recommended() -> None:
    """`VanDerPol` has a limit cycle around the origin. Newton lands on the origin.

    Reported, and nothing here claims the model settles there -- which is the difference
    between finding a root and finding an equilibrium a system actually reaches.
    """
    pytest.importorskip("numpy")
    found = steady(_system(EXAMPLES / "models" / "VanDerPol.mo"))
    assert found["x"] == pytest.approx(0.0, abs=1e-12)
    assert found["y"] == pytest.approx(0.0, abs=1e-12)


def test_a_conserved_quantity_leaves_a_family_and_that_is_refused() -> None:
    """`TwoBodyHeat` has no equilibrium that `rhs = 0` determines, and this is a correction.

    The entry expected it to settle at the heat-capacity-weighted mean of its two initial
    temperatures. That is true of the *trajectory* and it is not a root of the residual: two
    masses joined by a conductor stop moving whenever their temperatures are *equal*, at any
    common value, so the residual has a one-parameter family of zeros and its Jacobian is
    singular everywhere on it. Picking the right one needs the conserved energy as an extra
    equation, which a steady solve on `rhs` alone does not have.

    Refused, with that said in the message, rather than returning whichever member of the
    family the arithmetic drifted to.
    """
    pytest.importorskip("numpy")
    with pytest.raises(SteadyError, match="singular"):
        steady(_system(EXAMPLES / "models" / "TwoBodyHeat.mo"))


def test_a_residual_that_never_shrinks_is_refused_by_name() -> None:
    """`DrainingTank`'s square root goes imaginary under Newton and the residual is NaN."""
    pytest.importorskip("numpy")
    with pytest.raises(SteadyError, match="Newton steps"):
        steady(_system(EXAMPLES / "models" / "DrainingTank.mo"), iterations=8)


def test_the_answer_says_which_state_set_it_belongs_to() -> None:
    """An equilibrium belongs to a mode, and a model with two has one for each."""
    pytest.importorskip("numpy")
    found = steady(_system(EXAMPLES / "models" / "RLCCircuit.mo"))
    assert found.mode == found.states == ("l.i", "c.v")


def test_a_parameter_can_be_overridden_for_the_solve() -> None:
    """Sizing asks where it settles *at another operating point*, without editing the model."""
    pytest.importorskip("numpy")
    model = EXAMPLES / "models" / "NewtonCooling.mo"
    assert steady(_system(model))["T"] == pytest.approx(25.0)
    assert steady(_system(model), values={"T_amb": 4.0})["T"] == pytest.approx(4.0)
