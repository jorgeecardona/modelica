"""``d(der)/dp`` and ``d(observed)/dp``, against central differences on every corpus model.

The word *sensitivity* has an owner in this field: it is the derivative with respect to a
**parameter**, which is what identification and uncertainty propagation ask for. Both want it
exact — differencing it costs one evaluation per parameter and puts truncation error straight
into the quantity being fitted.

It is the same forward mode `jacobian` uses with the seed moved from the states to the
parameters, so it is checked the same way `jacobian` is: against a difference quotient, on
every model, at the tolerance `tests/test_examples.py` already holds that comparison to.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim.emit import EmitError, emit
from modelica.sim.sweep import vector

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


def _chosen(system: System) -> tuple[str, ...]:
    """Up to three parameters the model actually reads, which is the realistic ask.

    A parameter nothing downstream depends on has a column of zeros, and a test made only of
    those would pass against an emitter that returned zeros for everything.
    """
    from modelica.ir.rewrite import names  # noqa: PLC0415 - one helper needs it

    read: set[str] = set()
    for block in system.mode.blocks:
        for equation in block.equations:
            read |= set(names(equation.lhs)) | set(names(equation.rhs))
        if block.assignment is not None:
            read |= set(names(block.assignment))
    return tuple(v.name for v in system.parameters if v.name in read)[:3]


@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_the_parameter_derivative_agrees_with_a_difference_quotient(model: Path) -> None:
    """The exact derivative against a differenced one, parameter by parameter.

    The quotient is taken by rebuilding the parameter *vector* through
    `modelica.sim.sweep.vector`, not by poking one slot: a parameter written in terms of
    another has to move with it, and a difference taken without that measures a model nobody
    wrote. `RoastingMachine`'s `m_dry` is that case.

    And it is taken at a **fixed state**, which is the second thing to get right. This is a
    partial derivative at a point; several corpus models start a state *from* a parameter, so
    a quotient that let the start point move too would be differencing something else.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(model)
    chosen = _chosen(system)
    if not chosen:
        pytest.skip(f"{model.stem} has no parameter its equations read")

    kernel = build(system, sensitivity=chosen)
    settings = pytest.importorskip("modelica.sim.emit").settings
    defaults = settings(system)
    parameters, states, workspace = kernel.start(0.0)
    width = len(states)

    exact = numpy.zeros((width, len(chosen)))
    kernel.jacobian_parameters(0.0, states, parameters, workspace, exact)

    plain = build(system)
    for column, name in enumerate(chosen):
        base = float(defaults[name])
        step = 1e-6 * max(abs(base), 1.0)
        moved: list[Any] = []
        for delta in (step, -step):
            p = vector(system, {name: base + delta})
            # The *same* state on both sides. `jacobian_parameters` is the partial at a fixed
            # point, and several corpus models start a state from a parameter -- `m_w` begins
            # at `m_green * moisture` -- so letting `start` move it as well would difference
            # a different quantity and blame the emitter for the mismatch. It did, once.
            _, _, w = plain.start(0.0, p)
            out = numpy.zeros(width)
            plain.rhs(0.0, states, p, w, out)
            moved.append(out)
        quotient = (moved[0] - moved[1]) / (2 * step)
        numpy.testing.assert_allclose(
            exact[:, column], quotient, rtol=1e-5, atol=1e-5, err_msg=f"{model.stem}/{name}"
        )


def test_a_parameter_the_model_does_not_have_is_refused_by_name() -> None:
    system = _system(EXAMPLES / "models" / "NewtonCooling.mo")
    with pytest.raises(EmitError, match="no parameter 'kk'"):
        emit(system, sensitivity=["kk"])


def test_the_columns_are_in_the_order_asked_for() -> None:
    """`Code.sensitivity` is how a caller tells which column is which."""
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    system = _system(EXAMPLES / "models" / "NewtonCooling.mo")
    kernel = build(system, sensitivity=("k", "T_amb"))
    assert kernel.code.sensitivity == ("k", "T_amb")

    parameters, states, workspace = kernel.start(0.0)
    out = numpy.zeros((1, 2))
    kernel.jacobian_parameters(0.0, states, parameters, workspace, out)

    # der(T) = -k (T - T_amb), so d/dk is -(T - T_amb) and d/dT_amb is k. T starts at 90.
    assert out[0, 0] == pytest.approx(-(90.0 - 25.0))
    assert out[0, 1] == pytest.approx(0.5)


def test_asking_for_nothing_costs_nothing() -> None:
    """The default. A model nobody is fitting should not carry rows it will never read."""
    system = _system(EXAMPLES / "models" / "RoastingMachine.mo")
    assert emit(system).sensitivity == ()
    assert "_pseed = np.eye(0)" in emit(system).source
