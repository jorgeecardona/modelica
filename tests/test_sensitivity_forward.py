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


def _system_from(source: str, name: str) -> System:
    return causalize(flatten([parse(source)], name))


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


CROSSING = """model Crossing "A latch flip whose time moves with a parameter, in closed form"
  parameter Real k = 2;
  Real x(start = 1, fixed = true);
equation
  der(x) = if x > 0 then -k else -1;
end Crossing;
"""
"""``x`` falls at ``k`` until it reaches zero, then at one.

So ``t* = 1/k`` and ``dt*/dk = -1/k^2``, and after the crossing ``x = -(t - 1/k)``, giving
``dx/dk = -1/k^2 = -0.25`` for every later ``t``. Before it, ``dx/dk = -t``.

The model earns its place by discriminating. Integrating ``S`` through the crossing with **no
jump** leaves it at its pre-crossing value of ``-0.5`` and looks perfectly reasonable; getting
the jump's sign backwards gives ``-0.75``. Both are wrong and neither is obviously wrong.
"""


def test_the_sensitivity_jumps_across_a_crossing_by_the_closed_form_amount() -> None:
    """The whole of event sensitivity, on a model whose answer is a fraction.

    ``S+ = S- + (f- - f+) dt*/dp``. Here ``S- = -0.5``, ``f- = -2``, ``f+ = -1`` and
    ``dt*/dp = -0.25``, so the jump is ``+0.25`` and the answer is ``-0.25`` -- which is what
    differentiating ``-(t - 1/k)`` gives, and what a central difference of two runs gives.
    """
    forward = pytest.importorskip("modelica.sim.sensitivity").sensitivity
    system = _system_from(CROSSING, "Crossing")
    grid = np.array([0.2, 0.4, 0.6, 1.0])
    exact = forward(system, ["k"], stop=1.0, times=grid, tolerance=1e-11)
    taken = np.searchsorted(exact.time, grid)
    got = exact.state[exact.states.index("x"), 0][taken]
    np.testing.assert_allclose(got, [-0.2, -0.4, -0.25, -0.25], atol=1e-7)


def test_roasting_machine_integrates_through_all_nine_of_its_crossings() -> None:
    """The model the fitting entry's probe needs, and the reason this work was done.

    Nine watched relations, eight of them schedules whose ``g_y`` is zero and which therefore
    move ``S`` not at all, and one -- ``T_bean > T_dry`` -- whose crossing time genuinely
    depends on the parameters. Checked against a central difference over every observed
    variable at eight points: **1.6e-8**.

    It used to be refused. Integrated with no event handling its own trajectory came out
    8.8e-2 from `simulate`'s, before any sensitivity was read.
    """
    forward = pytest.importorskip("modelica.sim.sensitivity").sensitivity
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    build = pytest.importorskip("modelica.sim.kernel").build
    vector = pytest.importorskip("modelica.sim.sweep").vector

    system = _system("RoastingMachine")
    stop = float(_spec("RoastingMachine")["stop"])
    grid = np.linspace(0.0, stop, 9)[1:]
    exact = forward(system, ["UA"], stop=stop, times=grid, tolerance=1e-10)

    kernel = build(system, sensitivity=("UA",))
    declared = list(kernel.code.parameters)
    value = float(kernel.parameters(np.zeros(len(declared)))[declared.index("UA")])
    step = 1e-6 * abs(value)
    runs = [
        simulate(
            system,
            stop=stop,
            times=grid,
            tolerance=1e-10,
            parameters=vector(system, {"UA": value + delta}),
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
    scale = np.maximum(1.0, np.abs(crude).max(axis=1, keepdims=True))
    assert (np.abs(exact.output[:, 0, :][:, taken] - crude) / scale).max() <= 1e-5


def test_a_when_that_only_sets_a_discrete_variable_goes_through() -> None:
    """What is refused is a crossing that moves ``y``, which is narrower than "has a `when`".

    ``S+ = S- + (f- - f+) dt*/dp`` is the whole story exactly when ``h`` is the identity.
    `Thermostat`'s two `when` clauses assign ``heating`` and ``cycles`` and touch no state, so
    they change what ``f`` *is* without moving where ``y`` is -- and the ``f- - f+`` term is
    precisely that change. Measured against a central difference: **1.2e-8**.

    The first version of the refusal turned away every `when`, which was broader than the
    mathematics needs. `Trigger.reinitialises` is what tells the two apart, and it had to be
    put on the record because `latch` cannot: both kinds of `when` have none.
    """
    forward = pytest.importorskip("modelica.sim.sensitivity").sensitivity
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    build = pytest.importorskip("modelica.sim.kernel").build
    vector = pytest.importorskip("modelica.sim.sweep").vector

    system = _system("Thermostat")
    stop = float(_spec("Thermostat")["stop"])
    grid = np.linspace(0.0, stop, 9)[1:]
    exact = forward(system, ["k"], stop=stop, times=grid, tolerance=1e-10)

    kernel = build(system, sensitivity=("k",))
    declared = list(kernel.code.parameters)
    value = float(kernel.parameters(np.zeros(len(declared)))[declared.index("k")])
    step = 1e-6 * abs(value)
    runs = [
        simulate(
            system,
            stop=stop,
            times=grid,
            tolerance=1e-10,
            parameters=vector(system, {"k": value + delta}),
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
    scale = np.maximum(1.0, np.abs(crude).max(axis=1, keepdims=True))
    assert (np.abs(exact.output[:, 0, :][:, taken] - crude) / scale).max() <= 1e-5


def test_a_when_that_reinitialises_a_state_is_still_refused() -> None:
    """`BouncingBall` turns its velocity round, so ``h`` is not the identity.

    ``dh/dy`` is `Kernel.transition` differentiated and nothing emits it. Refused with the
    reason rather than approximated -- and the reason now names what is actually wrong,
    which is the reinitialisation and not the `when`.
    """
    forward = pytest.importorskip("modelica.sim.sensitivity").sensitivity
    error = pytest.importorskip("modelica.sim.solve").SimulationError
    with pytest.raises(error, match="reinitialises a state"):
        forward(_system("BouncingBall"), ["e"], stop=1.0)


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


# --------------------------------------------------------------------------------------
# The fit, and what it refuses
# --------------------------------------------------------------------------------------


def _synthetic(
    stem: str, moved: dict[str, float], watched: list[str], points: int = 20
) -> tuple[Any, float, dict[str, tuple[Any, Any]]]:
    """A run at known parameters, sampled, as if it were a measurement."""
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    vector = pytest.importorskip("modelica.sim.sweep").vector
    system = _system(stem)
    stop = float(_spec(stem)["stop"])
    grid = np.linspace(0.0, stop, points + 1)[1:]
    truth = simulate(
        system, stop=stop, times=grid, tolerance=1e-10, parameters=vector(system, moved)
    )
    columns = np.searchsorted(truth.time, grid)
    return system, stop, {name: (grid, truth[name][columns]) for name in watched}


def test_a_known_parameter_is_recovered_from_its_own_trajectory() -> None:
    """The modest claim, and the entry that asked for this says it is the modest one.

    Data made by the model at ``k = 0.3``, fit started from the ``0.5`` the model declares.
    Six Gauss-Newton steps, residual 1e-10, ``k`` back to nine figures. What this shows is
    that the optimiser converges on an identifiable problem with exact gradients -- not that
    the procedure recovers parameters from real-shaped data, which is what
    `docs/issues/09-08-2026-nothing-fits-a-parameter-to-data-and-the-reason-that-is-hard.md`
    reserves its `verify: probe` for.
    """
    fitter = pytest.importorskip("modelica.sim.sensitivity").fit
    system, stop, data = _synthetic("NewtonCooling", {"k": 0.3}, ["T"])
    found = fitter(system, ["k"], data, stop=stop)
    np.testing.assert_allclose(found.values, [0.3], rtol=1e-7)
    assert found.residual < 1e-8
    assert found.conditioning.condition < 10


def test_the_fit_refuses_the_pair_the_measurement_cannot_separate() -> None:
    """The whole reason this is beside `identifiability` and not in a caller's script.

    ``R`` and ``C`` from the capacitor voltage would converge -- to *a* pair on the invisible
    direction, with a residual near zero, and wrong about the current. It is refused with the
    combination named, so the failure is a sentence rather than a plausible number.
    """
    module = pytest.importorskip("modelica.sim.sensitivity")
    system, stop, data = _synthetic("RCCircuit", {}, ["c.v"])
    with pytest.raises(module.FitError, match="cannot separate") as caught:
        module.fit(system, ["r.R", "c.C"], data, stop=stop)
    message = str(caught.value)
    assert "invisible combination" in message
    assert "r.R" in message
    assert "c.C" in message


def test_the_same_pair_fits_once_the_current_is_measured_too() -> None:
    """And the refusal lifts for the reason it should: the data separates them now.

    Same two parameters, same model, one more sensor. Started from a doubled resistance and a
    halved capacitance -- which is *exactly* the invisible direction of the previous test, so
    a fit that could not see it would have no reason to come back.
    """
    module = pytest.importorskip("modelica.sim.sensitivity")
    build = pytest.importorskip("modelica.sim.kernel").build

    # The measurement is made at *doubled* R and *halved* C, which is exactly the direction
    # the previous test showed the voltage cannot see. Fitting from the model's declared
    # values has to walk along it, so a fit that could not see it would have no reason to move.
    system, stop, data = _synthetic("RCCircuit", {"r.R": 200.0, "c.C": 0.005}, ["c.v", "r.i"])
    found = module.fit(system, ["r.R", "c.C"], data, stop=stop)

    kernel = build(system, sensitivity=("r.R", "c.C"))
    declared = list(kernel.code.parameters)
    started = kernel.parameters(np.zeros(len(declared)))
    assert started[declared.index("r.R")] != 200.0, "the fit would have had nothing to do"
    np.testing.assert_allclose(found.values, [200.0, 0.005], rtol=1e-5)
    assert found.conditioning.condition < 10, str(found)


def test_fitting_anyway_is_possible_and_says_what_it_did() -> None:
    """``separable=0`` is a decision a caller may take, and `Fit` carries the consequence.

    Somebody fitting ``R`` and ``C`` to the voltage may genuinely want the product. The
    refusal is a default, not a prohibition -- and what comes back has the conditioning on it,
    so the residual can never be read alone.
    """
    module = pytest.importorskip("modelica.sim.sensitivity")
    system, stop, data = _synthetic("RCCircuit", {}, ["c.v"])
    found = module.fit(system, ["r.R", "c.C"], data, stop=stop, separable=0.0)
    assert found.residual < 1e-6
    assert found.conditioning.condition > 1e6, "a fit nobody should trust, and it says so"
    assert found.conditioning.inseparable(), str(found)


def test_measuring_something_the_model_does_not_have_is_refused_by_the_fit_too() -> None:
    module = pytest.importorskip("modelica.sim.sensitivity")
    with pytest.raises(module.FitError, match="no variable 'thermocouple'"):
        module.fit(_system("NewtonCooling"), ["k"], {"thermocouple": ([1.0], [2.0])}, stop=5.0)


def test_a_measurement_after_the_horizon_is_refused() -> None:
    """Silently ignoring it would fit to fewer points than the caller thinks they gave."""
    module = pytest.importorskip("modelica.sim.sensitivity")
    with pytest.raises(module.FitError, match=r"leave \[0, 5\]"):
        module.fit(_system("NewtonCooling"), ["k"], {"T": ([1.0, 9.0], [2.0, 3.0])}, stop=5.0)


# --------------------------------------------------------------------------------------
# The two terms an event-crossing sensitivity needs
# --------------------------------------------------------------------------------------


EVENTFUL = ["BouncingBall", "BumpStop", "DrainingTank", "Pendulum", "Thermostat"]
"""Corpus models with a crossing whose indicator depends on the state.

`RoastingMachine`, `PhaseChange`, `Setback` and `TwoInertias` have events too and are left out
here only to keep this file quick -- `tests/test_examples.py` runs the ``g_y`` check over all
nine. What this file adds is the *slope*, and these five are the ones where it is not a
constant.
"""


@pytest.mark.parametrize("stem", EVENTFUL)
def test_the_indicator_slope_is_the_total_time_derivative(stem: str) -> None:
    """``indicator_slope`` against a difference quotient taken *along the trajectory*.

    ``dt*/dp = -(g_y S) / (g_t + g_y f)``, and the denominator is ``dg/dt`` as one quantity.
    The quotient has to move the state too -- ``g(t + h, y + h f)`` against ``g(t - h, y - h
    f)`` -- because a partial in ``t`` alone is the wrong thing and would agree with a wrong
    implementation on every model whose crossings are schedules.

    Measured when it landed: 1.9e-8 worst over all nine event models, and 1.3e-10 or better on
    the five here. The two loose ones are `RoastingMachine` and `PhaseChange`, whose slopes
    are constants, so the number is the quotient's own error rather than the emitter's.
    """
    build = pytest.importorskip("modelica.sim.kernel").build
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    run = _spec(stem)
    system = _system(stem)
    kernel = build(system)
    result = simulate(system, stop=float(run["stop"]), tolerance=1e-9)
    rows = [result.names.index(name) for name in kernel.code.states]
    p, _, w = kernel.start()

    worst = 0.0
    for column in np.unique(np.linspace(1, len(result.time) - 2, 6).astype(int)):
        t = float(result.time[column])
        y = np.ascontiguousarray([result.values[row, column] for row in rows])
        kernel.latch(t, y, p, w)
        exact = kernel.indicator_slope(t, y, p, w.copy(), np.zeros(kernel.events)).copy()
        f = kernel.rhs(t, y, p, w.copy(), np.empty(len(y))).copy()
        step = 1e-6 * max(abs(t), 1.0)
        ahead = kernel.indicators(
            t + step, y + step * f, p, w.copy(), np.empty(kernel.events)
        ).copy()
        behind = kernel.indicators(
            t - step, y - step * f, p, w.copy(), np.empty(kernel.events)
        ).copy()
        crude = (ahead - behind) / (2 * step)
        worst = max(worst, float(np.abs(exact - crude).max() / max(1.0, np.abs(crude).max())))
    assert worst <= 1e-6, f"{stem}: {worst:.2e}"


def test_a_schedule_has_no_state_gradient_but_a_perfectly_good_slope() -> None:
    """The two functions answer different questions, and `RoastingMachine` shows both.

    Eight of its nine crossings are ``time < c``. Their ``g_y`` is identically zero -- the
    crossing does not move when a parameter does -- while their ``dg/dt`` is ``-1``, because
    time reaches them at one second per second. A caller that read one for the other would get
    a division by zero on every schedule in the corpus.
    """
    build = pytest.importorskip("modelica.sim.kernel").build
    kernel = build(_system("RoastingMachine"))
    p, y, w = kernel.start()
    gradient = kernel.indicator_jacobian(0.0, y, p, w.copy(), np.zeros((kernel.events, len(y))))
    slope = kernel.indicator_slope(0.0, y, p, w.copy(), np.zeros(kernel.events))

    flat = [k for k in range(kernel.events) if not gradient[k].any()]
    assert len(flat) == 8, f"expected the eight schedules, got {flat}"
    assert all(slope[k] != 0.0 for k in flat), "a schedule is still reached at one per second"
    np.testing.assert_allclose([slope[k] for k in flat], -1.0)
