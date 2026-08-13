"""``dy/dp`` integrated, against ``dy/dp`` differenced.

`modelica.sim.sensitivity` builds the forward-sensitivity ODE out of two partials the emitter
already writes exactly -- ``df/dy`` and ``df/dp`` -- so nothing in it is a difference quotient.
The way to check that is to difference the thing it claims to compute: run the model twice with
one parameter nudged either way, subtract, and compare.

The two disagree for reasons that belong to the *quotient*, not to the integration. A central
difference at ``h`` carries an ``O(h^2)`` truncation error and an ``O(eps/h)`` cancellation
error, and both runs are only as good as the solver's tolerance. Measured at ``h = 1e-5`` and
``rtol = 1e-9`` on the four models below, the worst agreement is 8.9e-7 -- which is about where
a quotient of two 1e-9 trajectories should land, and is the reason the assertion is 1e-5 rather
than machine precision. Tightening the tolerance moves the number; tightening ``h`` past this
makes it worse, which is the classic shape and the reason exact sensitivities are worth having.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from modelica.ir.causalize import causalize
from modelica.ir.source import flatten
from modelica.lang import parse

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LIBRARY = EXAMPLES / "library"
MODELS = EXAMPLES / "models"

STEP = 1e-5
"""Relative nudge for the central difference. See the module docstring for why not smaller."""

TOLERANCE = 1e-9
"""What both the integration and the two differenced runs are asked for."""

AGREEMENT = 1e-5
"""How close the exact and differenced sensitivities have to be, relative to the larger."""

CASES = [
    ("NewtonCooling", "k"),
    ("RCCircuit", "r.R"),
    ("RCCircuit", "c.C"),
    ("MassSpringDamper", "spring.c"),
]
"""Model and parameter, chosen so that the four cover the shapes that matter.

`NewtonCooling` is one state and a closed form. `RCCircuit` is differentiated twice, against a
resistance the derivative depends on directly and a capacitance that scales the whole
right-hand side. `MassSpringDamper` is two states, so ``S`` is a matrix rather than a column
and the ``(df/dy) S`` product is doing something.

None of them has an event, because `sensitivity` refuses those -- which is its own test below.
"""


def _system(stem: str) -> System:
    spec = tomllib.loads((MODELS / f"{stem}.toml").read_text(encoding="utf-8"))
    files = [LIBRARY / f"{name}.mo" for name in spec["requires"]] + [MODELS / f"{stem}.mo"]
    return causalize(flatten([parse(f.read_text(encoding="utf-8")) for f in files], stem))


def _spec(stem: str) -> dict[str, Any]:
    return tomllib.loads((MODELS / f"{stem}.toml").read_text(encoding="utf-8"))["simulate"]


@pytest.mark.parametrize(("stem", "parameter"), CASES, ids=[f"{s}/{p}" for s, p in CASES])
def test_the_integrated_sensitivity_matches_a_central_difference(stem: str, parameter: str) -> None:
    """The whole claim, on every observed variable at eight points of the run."""
    build = pytest.importorskip("modelica.sim.kernel").build
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    forward = pytest.importorskip("modelica.sim.sensitivity").sensitivity
    vector = pytest.importorskip("modelica.sim.sweep").vector

    system = _system(stem)
    stop = float(_spec(stem)["stop"])
    grid = np.linspace(0.0, stop, 9)[1:]

    exact = forward(system, [parameter], stop=stop, times=grid, tolerance=TOLERANCE)
    kernel = build(system, sensitivity=(parameter,))
    names = [variable.name for variable in system.parameters]
    value = float(kernel.parameters(np.zeros(len(names)))[names.index(parameter)])

    step = STEP * max(abs(value), 1.0)
    runs = [
        simulate(
            system,
            stop=stop,
            times=grid,
            tolerance=TOLERANCE,
            parameters=vector(system, {parameter: value + delta}),
        )
        for delta in (step, -step)
    ]
    columns = [np.searchsorted(run.time, grid) for run in runs]
    crude = np.array(
        [
            (runs[0][name][columns[0]] - runs[1][name][columns[1]]) / (2 * step)
            for name in kernel.code.observed
        ]
    )
    taken = np.searchsorted(exact.time, grid)
    emitted = exact.output[:, 0, :][:, taken]

    scale = np.maximum(1.0, np.abs(crude).max(axis=1, keepdims=True))
    relative = np.abs(emitted - crude) / scale
    worst = int(np.argmax(relative.max(axis=1)))
    assert relative.max() <= AGREEMENT, (
        f"{stem} d/d{parameter}: {relative.max():.2e} on {kernel.code.observed[worst]}"
    )


def test_a_state_and_an_observed_variable_agree_where_they_are_the_same_thing() -> None:
    """`Sensitivity.state` and `Sensitivity.output` are two views of one integration.

    A state is also an observed variable -- ``observe`` writes every variable the model has --
    so the row for ``T`` has to be the same number by either route. It is not a tautology:
    `state` comes straight out of the augmented integration and `output` comes from the chain
    rule over ``d(obs)/dy`` and ``d(obs)/dp``, which is a different pair of emitted functions.
    """
    forward = pytest.importorskip("modelica.sim.sensitivity").sensitivity
    system = _system("NewtonCooling")
    exact = forward(system, ["k"], stop=float(_spec("NewtonCooling")["stop"]), tolerance=TOLERANCE)
    state = exact.state[exact.states.index("T"), 0]
    output = exact.output[exact.observed.index("T"), 0]
    np.testing.assert_allclose(output, state, rtol=1e-12, atol=1e-14)


def test_a_model_with_events_is_refused_rather_than_approximated() -> None:
    """`RoastingMachine` has nine watched relations, and this would be quietly wrong for it.

    Measured: integrated through the augmented system with no event handling, its *own
    trajectory* -- before any sensitivity is read -- comes out 8.8e-2 from `simulate`'s,
    because the latches are set once and never refreshed. A number that wrong that still looks
    like a number is the outcome the refusal exists to prevent.
    """
    forward = pytest.importorskip("modelica.sim.sensitivity").sensitivity
    error = pytest.importorskip("modelica.sim.solve").SimulationError
    with pytest.raises(error, match="does not handle events"):
        forward(_system("RoastingMachine"), ["m_green"], stop=10.0)


def test_asking_for_a_parameter_the_kernel_was_not_built_for_is_refused() -> None:
    """A kernel built without `sensitivity=` has no columns, and would return zeros.

    Zeros are a plausible answer -- a parameter nothing depends on has exactly that
    sensitivity -- so the failure has to be loud rather than empty.
    """
    build = pytest.importorskip("modelica.sim.kernel").build
    forward = pytest.importorskip("modelica.sim.sensitivity").sensitivity
    error = pytest.importorskip("modelica.sim.solve").SimulationError
    system = _system("NewtonCooling")
    with pytest.raises(error, match="build it with `sensitivity=`"):
        forward(build(system), ["k"], stop=1.0)


# --------------------------------------------------------------------------------------
# Which parameters the measurements can tell apart
# --------------------------------------------------------------------------------------


def test_a_resistance_and_a_capacitance_cannot_be_separated_from_the_voltage_alone() -> None:
    """The failure the fit has to be protected from, on a corpus model, with the exact answer.

    ``c.v`` decays as ``exp(-t / (R*C))``, so the voltage sees the *product* and nothing else.
    Double ``R`` and halve ``C`` and every sample is unchanged, which means the direction
    ``log C - log R`` is invisible and a least-squares fit over both would converge to
    whichever pair the optimiser wandered into, report a small residual, and be wrong about
    the current.

    The report says so from the model alone: condition **1.95e14** at ``rtol = 1e-9``, and the
    weakest direction
    comes out ``[-0.707, +0.707]`` -- the analytic answer, not an approximation of it, since
    the columns are scaled to ``d/d log p`` and the null direction of ``log R + log C`` is
    exactly the difference at 45 degrees.
    """
    report = pytest.importorskip("modelica.sim.sensitivity").identifiability
    found = report(
        _system("RCCircuit"),
        ["r.R", "c.C"],
        ["c.v"],
        stop=float(_spec("RCCircuit")["stop"]),
        tolerance=TOLERANCE,
    )
    assert found.condition > 1e12
    weak = dict(found.inseparable())
    assert set(weak) == {"r.R", "c.C"}
    assert weak["r.R"] * weak["c.C"] < 0, "the invisible move is a trade, not a scaling"
    np.testing.assert_allclose([abs(v) for v in weak.values()], np.sqrt(0.5), rtol=1e-3)


def test_measuring_the_current_as_well_separates_them() -> None:
    """And this is the advice the entry's source gives, arrived at from the model.

    His procedure is two staged fits ordered by sensitivity, because "temperatures close to
    the weld line are influenced by both the thermal efficiency and the heat source shape".
    The same shape here: ``r.i`` is ``C * d(c.v)/dt``, so it carries ``C`` on its own and
    breaks the tie the voltage cannot.

    Condition 1.95e14 with the voltage alone, **3.32** with the current beside it. Nothing was
    measured to find that out -- it is a statement about the model and the sensor set.

    Both are read at ``rtol = 1e-9`` rather than the default, and that matters for the first
    number only: an invisible direction's singular value is round-off, so it is bounded by how
    accurately the sensitivity was integrated. The same call at the default 1e-6 reads 1.7e10
    instead of 1.9e14. The *separable* number does not move.
    """
    report = pytest.importorskip("modelica.sim.sensitivity").identifiability
    stop = float(_spec("RCCircuit")["stop"])
    alone = report(_system("RCCircuit"), ["r.R", "c.C"], ["c.v"], stop=stop, tolerance=TOLERANCE)
    both = report(
        _system("RCCircuit"), ["r.R", "c.C"], ["c.v", "r.i"], stop=stop, tolerance=TOLERANCE
    )
    assert both.condition < 10
    assert alone.condition / both.condition > 1e12
    assert both.inseparable() == (), str(both)


@pytest.mark.parametrize(
    ("stem", "parameters", "measured"),
    [
        ("NewtonCooling", ["k", "T_amb"], ["T"]),
        ("MassSpringDamper", ["spring.c", "damper.d"], ["mass.s"]),
    ],
    ids=["cooling", "oscillator"],
)
def test_parameters_that_are_separable_are_reported_as_separable(
    stem: str, parameters: list[str], measured: list[str]
) -> None:
    """The negative case, which is what stops the report from always finding something.

    `NewtonCooling` reads 2.92 and `MassSpringDamper` 5.06 -- a rate and an ambient move the
    trajectory in different shapes, and so do a stiffness and a damping. Both are fourteen
    orders away from the RC reading above, which is the gap `Identifiability.inseparable`'s
    default threshold sits in the middle of.
    """
    report = pytest.importorskip("modelica.sim.sensitivity").identifiability
    found = report(_system(stem), parameters, measured, stop=float(_spec(stem)["stop"]))
    assert found.condition < 100
    assert found.inseparable() == (), str(found)


def test_measuring_something_the_model_does_not_have_is_refused() -> None:
    report = pytest.importorskip("modelica.sim.sensitivity").identifiability
    error = pytest.importorskip("modelica.sim.solve").SimulationError
    with pytest.raises(error, match="no variable 'thermocouple'"):
        report(_system("NewtonCooling"), ["k"], ["thermocouple"], stop=1.0)
