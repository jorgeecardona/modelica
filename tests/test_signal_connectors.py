"""``connector RealInput = input Real``: the causal signal connector the MSL Blocks use.

A signal connector is a connector that *is* a scalar rather than a record of fields: it aliases
a predefined type and carries a causality, but holds no members. It lays down as one leaf
variable, and ``connect(out, in)`` equates the two -- an output driving an input, the whole point
of a signal wire. This is the front-end gate that kept ``Modelica.Blocks`` out; these tests hold
the gate open in the shapes the domain relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from modelica import ir
from modelica.ir.causalize import causalize
from modelica.ir.flatten import FlattenError
from modelica.lang import parse
from modelica.sim import flatten, simulate

CONNECTORS = """
connector RealInput = input Real;
connector RealOutput = output Real;
"""


def _flat(body: str, name: str):  # noqa: ANN202 - a FlatModel
    return flatten([parse(CONNECTORS + body)], name)


def test_a_signal_connector_is_one_scalar_leaf() -> None:
    """`RealOutput y` is the variable `y` itself, not a record with a field."""
    flat = _flat(
        """
        block Constant
          parameter Real k = 1;
          RealOutput y;
        equation
          y = k;
        end Constant;
        model M
          Constant c(k = 3);
        end M;
        """,
        "M",
    )
    names = {v.name for v in flat.variables}
    assert "c.y" in names
    assert "c.y.signal" not in names


def test_connect_equates_an_output_to_an_input() -> None:
    """`connect(c.y, g.u)` is the equation `c.y = g.u`; the wire carries the value across."""
    flat = _flat(
        """
        block Constant
          parameter Real k = 1;
          RealOutput y;
        equation
          y = k;
        end Constant;
        block Gain
          parameter Real g = 1;
          RealInput u;
          RealOutput y;
        equation
          y = g * u;
        end Gain;
        model M
          Constant c(k = 3);
          Gain gain(g = 4);
        equation
          connect(c.y, gain.u);
        end M;
        """,
        "M",
    )
    assert flat.is_balanced()
    equated = {
        frozenset((eq.lhs.parts, eq.rhs.parts))
        for eq in flat.equations
        if isinstance(eq.lhs, ir.Ref) and isinstance(eq.rhs, ir.Ref)
    }
    assert frozenset((("c", "y"), ("gain", "u"))) in equated


def test_a_state_flows_through_a_signal_connector() -> None:
    """An integrator fed a constant ramps: `der(y) = u`, `u` arriving over a wire, `y = k*t`."""
    system = causalize(
        _flat(
            """
            block Constant
              parameter Real k = 1;
              RealOutput y;
            equation
              y = k;
            end Constant;
            block Integrator
              RealInput u;
              RealOutput y(start = 0, fixed = true);
            equation
              der(y) = u;
            end Integrator;
            model M
              Constant c(k = 2);
              Integrator i;
            equation
              connect(c.y, i.u);
            end M;
            """,
            "M",
        )
    )
    assert sorted(system.states) == ["i.y"]
    result = simulate(system, stop=5.0, times=np.linspace(0.0, 5.0, 200))
    assert abs(float(np.asarray(result["i.y"])[-1]) - 10.0) < 1e-6


def test_two_outputs_on_one_wire_are_refused() -> None:
    """A signal has one source; joining two outputs is a modelling error, named as one."""
    with pytest.raises(FlattenError, match="one source"):
        _flat(
            """
            block Src
              parameter Real k = 1;
              RealOutput y;
            equation
              y = k;
            end Src;
            model M
              Src a(k = 1);
              Src b(k = 2);
            equation
              connect(a.y, b.y);
            end M;
            """,
            "M",
        )
