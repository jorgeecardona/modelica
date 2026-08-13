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
