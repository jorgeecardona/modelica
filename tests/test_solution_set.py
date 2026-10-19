"""The schedule still has to satisfy the equations the modeller wrote.

Every other test of causalization in this repo is structural or end-to-end. The structural
ones (`tests/test_causalize_properties.py`) ask about incidence -- which block reads what,
whether the matching is a bijection -- and they pass on a `d/dt` that is arithmetically wrong,
because a differentiated equation has the same incidence whatever the arithmetic did to it.
The end-to-end ones (`tests/test_examples.py`) check numbers at the far side against a
reference, which catches a wrong `d/dt` but names the model rather than the equation.

This file asks the question in between, and it is the one `docs/ideas/verification.md:255`
lists third: **differentiating preserves the solution set**. Take the flat equations *before*
causalization, take the trajectory the causalized model produced, and evaluate one against the
other. Nothing about blocks, nothing about a reference implementation: the equations as
written, at points the run actually reached.

Two halves, and they check different things.

For an equation the schedule solves directly, the residual is a check on the schedule -- it
should be zero to arithmetic precision, and a value that is not says a block computed
something its own equation does not agree with.

For an equation index reduction **differentiated**, it is the real question. Pantelides
replaces a constraint by its derivative to get a solvable system; whether the original still
holds afterwards is a statement about the reduction, and the answer is allowed to be "no, it
drifts" -- that is what a dummy derivative exists to prevent. Measured on the corpus, it does
not drift at all, because the reduction keeps the original constraint in the schedule and
integrates one fewer state. `Pendulum` is index 3 and its ``x^2 + y^2 = L^2`` holds to
5.6e-17 relative; the day that becomes 1e-6 instead, this file is what says so.

## Does it have teeth? Five mutations, measured

A residual that is zero everywhere is only worth having if a wrong `d/dt` makes it non-zero.
Each rule in `modelica.ir.derive._binary` was broken in turn -- a real edit to the file, not a
monkeypatch -- and the property re-run:

============================== ============= ========== =============
mutation of ``d/dt``           `Constrained` `Pendulum` `TwoInertias`
============================== ============= ========== =============
product rule drops a term      caught        caught     caught
power rule forgets the exponent passed       **0.39**   passed
sum rule drops the right term  passed        caught     **0.71**
difference becomes a sum       **1.2**       passed     **2.0**
unary minus loses its sign     **1.3**       passed     **1.4**
============================== ============= ========== =============

**Five of five caught**, none by every model. Bold is a residual; "caught" without a number is
a model that stopped being causalizable or divided by zero, which is also an answer. No single
model is enough, which is the argument for checking the whole corpus rather than the two
index-reduced examples.

Stated plainly because it is the limit of what was shown: on *these five* mutations the
generated family catches nothing the corpus does not. Its value is that it is generated --
hundreds of coefficient combinations rather than three fixed models -- and that it holds the
index-2 shape the corpus has only by accident of two mechanical examples. It is not evidence
of extra coverage, and the table above is the reason to say so rather than assume.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from modelica.ir.causalize import causalize
from modelica.ir.evaluate import evaluate
from modelica.ir.rewrite import residual
from modelica.ir.source import flatten
from modelica.lang import parse

if TYPE_CHECKING:
    from modelica.ir.flat import FlatModel
    from modelica.ir.system import System
    from modelica.sim.result import Result

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LIBRARY = EXAMPLES / "library"
MODELS = EXAMPLES / "models"
MODEL_FILES = sorted(MODELS.glob("*.mo"))

SAMPLES = 20
"""How many points of each trajectory to check. Every sample would be no stronger.

A residual that is wrong is wrong along a whole stretch of the run, not at one isolated
column: it comes from an equation the schedule got wrong or a constraint that is drifting, and
neither is a spike. Twenty evenly spaced points cost milliseconds on the whole corpus and
would have to be very unlucky to step over a defect that a full sweep would find.
"""

TOLERANCE = 1e-11
"""Relative residual above which an equation is not satisfied.

Not a round number picked for comfort. Measured across the corpus on 2026-08-13, the worst
relative residual of any original equation at any sampled point is **6.2e-15**
(`AtwoodPendulum`; `DrainingTank` 3.1e-13, everything else at or below 1e-15). So this sits
about thirty times above the worst real reading and five orders of magnitude below the
solvers' own ``rtol``, which is 1e-6 for most of the corpus.

That gap is not slack, it is the point. For a *continuous* equation the residual is not
limited by the integration tolerance: an algebraic equation is solved at each point rather
than integrated, so it holds to arithmetic precision, and a constraint the reduction kept in
the schedule holds for the same reason. A residual that climbs to anywhere near ``rtol`` means
an equation stopped being solved and started being merely approximated -- exactly the failure
this file is for -- so the threshold belongs down here with the arithmetic and not up there
with the solver.

**An event instant is the one place the sampling has to be careful**, and it is a question
about timestamps rather than about tolerance. An event emits two rows sharing one ``t``, and a
strict relation on ``time`` cannot tell the left limit from the right. `_worst` skips those
columns and says what it measured; everything either side of them is sampled as usual.

Relative: ``|lhs - rhs| / (1 + max(|lhs|, |rhs|))``. Index reduction on a model whose terms are
in the thousands has a scale, and an absolute number would be a different test on every model.
"""


def _environment(flat: FlatModel, system: System, result: Result, column: int) -> dict[str, Any]:
    """Every name the original equations can read, at one column of the trajectory.

    `Result` carries a row per *observed* variable, which is every variable the model has, so
    the values are already there; what is missing is the spelling. The original equations say
    ``der(x)`` -- `modelica.ir.evaluate` reads that as a lookup of the literal key ``der(x)``
    -- while the trajectory calls it whatever causalization named it, which is ``der_x`` for a
    derivative this stage invented and the modeller's own name when alias elimination found
    one. Both routes are laid in, and `Mode.derivative` is what makes the second work.
    """
    env: dict[str, Any] = {
        name: float(result.values[row, column]) for row, name in enumerate(result.names)
    }
    env["time"] = float(result.time[column])
    for name in list(env):
        if name.startswith("der_"):
            env.setdefault(f"der({name[4:]})", env[name])
    for state, derivative in system.modes[0].derivative.items():
        if derivative in env:
            env[f"der({state})"] = env[derivative]
    for variable in flat.variables:
        env.setdefault(variable.name, variable.start if variable.start is not None else 0.0)
    return env


def _worst(flat: FlatModel, system: System, result: Result) -> tuple[float, str]:
    """The largest relative residual over the sampled columns, and the equation it came from.

    Columns at an event instant are not sampled, and that is a statement about timestamps
    rather than about events. An event emits **two** rows sharing one ``t`` -- the left limit
    and the right -- and a relation on ``time`` cannot tell them apart: ``time < 520`` is
    false at exactly 520, while the left-limit row was computed with it true. Evaluating there
    compares one side's values against the other side's branch.

    Measured 2026-08-13, at a 1e-6 tolerance floor: `PhaseChange` reads 1.735 at t = 520,
    `RoastingMachine` 0.674 at t = 93, `Setback` 0.333 at t = 4, `BumpStop` 1.0e-7. Dropping
    the event columns takes all four to machine precision -- 1.3e-16, 1.3e-16, 0, 0 -- on
    18 to 20 columns instead of 20 to 24.

    The instant *before* and the instant *after* are both still sampled, which is where the
    interesting question is. What is skipped is the one timestamp that names two states.
    """
    worst, culprit = 0.0, ""
    instants = {event.time for event in result.events}
    step = max(1, len(result.time) // SAMPLES)
    for column in range(0, len(result.time), step):
        if float(result.time[column]) in instants:
            continue
        env = _environment(flat, system, result, column)
        for equation in flat.equations:
            left = float(evaluate(equation.lhs, env))  # pyright: ignore[reportArgumentType]
            right = float(evaluate(equation.rhs, env))  # pyright: ignore[reportArgumentType]
            relative = abs(float(evaluate(residual(equation), env))) / (  # pyright: ignore[reportArgumentType]
                1.0 + max(abs(left), abs(right))
            )
            if relative > worst:
                worst, culprit = relative, f"{equation!r} at t = {env['time']:.6g}"
    return worst, culprit


# --------------------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------------------


def _ids(paths: list[Path]) -> list[str]:
    return [path.stem for path in paths]


@pytest.mark.slow
@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_the_original_equations_hold_along_the_trajectory(model: Path) -> None:
    """Every equation the modeller wrote, at twenty points of the run, on every corpus model.

    This is the only test here that touches the two index-reduced models in the corpus with
    the equations they were written with. `tests/test_examples.py` checks their numbers
    against a reference and `tests/test_causalize.py` checks the shape of what came out;
    neither reads `x^2 + y^2 = L^2` back at the trajectory and asks whether it is still true.
    """
    simulate = pytest.importorskip("modelica.sim").simulate
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    run = spec.get("simulate")
    if run is None:
        pytest.skip("no simulate section")
    files = [LIBRARY / f"{name}.mo" for name in spec["requires"]] + [model]
    flat = flatten([parse(path.read_text(encoding="utf-8")) for path in files], model.stem)
    system = causalize(flat)
    result = simulate(
        system,
        stop=run["stop"],
        # Floored at 1e-6, the same rule `tests/test_fold.py` and the difference-quotient
        # tests use. This file needs *a* trajectory, not an accurate one: an algebraic
        # equation is solved at each point rather than integrated, so how well it holds does
        # not follow the integration tolerance. `_worst` carries the one place where that was
        # not true, and it was the timestamps rather than the tolerance.
        tolerance=max(float(run.get("tolerance", 1e-6)), 1e-6),
        absolute=run.get("absolute"),
        method=run.get("method", "Radau"),
    )
    worst, culprit = _worst(flat, system, result)
    assert worst <= TOLERANCE, f"{model.stem}: relative residual {worst:.3e} from {culprit}"


# --------------------------------------------------------------------------------------
# Generated models that need index reduction
# --------------------------------------------------------------------------------------


@st.composite
def constrained(draw: st.DrawFn) -> tuple[str, float]:
    """Two positions moved by one shared flow and tied by a constraint: the smallest index-2.

    ``der(x) = c1*u + f1``, ``der(y) = c2*u + f2``, ``a*x + b*y = k``. Structurally singular as
    written -- three equations, and no equation for ``u`` until the constraint is
    differentiated -- so Pantelides has to act, which is what makes these worth generating.
    `tests/test_causalize_properties.py`'s `solvable` cannot: it builds triangles, and a
    triangle is index 1 by construction.

    The consistent point is built in rather than solved for. ``k`` is computed from the chosen
    starts, so ``a*x0 + b*y0 = k`` holds exactly at ``t = 0``; only ``x`` is `fixed`, because
    a constraint between two states leaves one of them to the constraint and pinning both
    would be an over-determined initialisation rather than an index-2 model.

    The returned float is ``u``, which the differentiated constraint fixes at
    ``-(a*f1 + b*f2) / (a*c1 + b*c2)`` for the whole run: a closed form to check the reduction
    against, and the reason ``a*c1 + b*c2`` is drawn away from zero.
    """
    a, b = draw(st.integers(1, 4)), draw(st.integers(2, 5))
    # Not both unit coefficients, or the constraint is an *alias* -- `a = +/-b + k` is the
    # shape `_pair` substitutes away, so `x + y = 0` never reaches index reduction at all and
    # the model comes out index 1. Hypothesis found that on the second draw.
    if a == b:
        b += 1
    c1, c2 = draw(st.integers(1, 3)), draw(st.integers(-3, -1))
    f1, f2 = draw(st.integers(-3, 3)), draw(st.integers(-3, 3))
    x0, y0 = draw(st.integers(-4, 4)), draw(st.integers(-4, 4))
    if a * c1 + b * c2 == 0:  # pragma: no cover - the draw ranges make this rare
        c1 += 1
    source = f"""model Constrained
  Real x(start = {x0}, fixed = true);
  Real y(start = {y0});
  Real u;
equation
  der(x) = {c1} * u + {f1};
  der(y) = {c2} * u + {f2};
  {a} * x + {b} * y = {a * x0 + b * y0};
end Constrained;
"""
    return source, -(a * f1 + b * f2) / (a * c1 + b * c2)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(constrained())
def test_index_reduction_leaves_the_original_constraint_true(drawn: tuple[str, float]) -> None:
    """The constraint that was differentiated away still holds where the run went.

    This is the assertion the structural properties cannot make. Pantelides replaced
    ``a*x + b*y = k`` by ``a*der(x) + b*der(y) = 0`` in order to have an equation for ``u``;
    whether the *original* is still true along the trajectory is a fact about the arithmetic
    of `_differentiate`, and a sign error in it would leave every structural property passing.

    ``u`` is checked against its closed form for the same reason. A reduction that satisfied
    the constraint by solving for the wrong variable would keep the residual at zero and still
    be wrong.
    """
    simulate = pytest.importorskip("modelica.sim").simulate
    source, u = drawn
    flat = flatten([parse(source)], "Constrained")
    system = causalize(flat)
    assert system.index >= 2, f"not an index-2 model\n{source}"

    result = simulate(system, stop=2.0, tolerance=1e-8)
    worst, culprit = _worst(flat, system, result)
    assert worst <= TOLERANCE, f"relative residual {worst:.3e} from {culprit}\n{source}"
    assert np.allclose(result.values[result.names.index("u")], u, rtol=1e-8, atol=1e-8), (
        f"u should be the constant {u:.6g}\n{source}"
    )
