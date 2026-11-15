"""A connect to a connector nested in the class's own connector joins the parent's set.

``Machines.BasicMachines.PartialBasicMachine`` writes ``connect(thermalAmbient.thermalPort,
internalThermalPort)`` and ``connect(ra.heatPort, internalThermalPort.heatPortArmature)``. The
second end was taken as a component's connector, which gave the loss a wrong sign and an extra
``Q_flow = 0`` that over-determined the model.
"""

from __future__ import annotations

import shutil
from typing import Literal

import pytest

from modelica.ir import nodes as ir
from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.ir.rewrite import names
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
connector HeatPort
  Real T;
  flow Real Q_flow;
end HeatPort;
connector ThermalPort
  HeatPort armature;
  HeatPort core;
end ThermalPort;
model Ambient
  ThermalPort port;
equation
  port.armature.T = 300;
  port.core.T = 310;
end Ambient;
model Loss
  HeatPort heatPort;
  parameter Real power = 5;
equation
  heatPort.Q_flow = -power;
end Loss;
model Machine
  Loss ra(power = 5);
  Loss rc(power = 7);
  Ambient ambient;
protected
  ThermalPort internal;
equation
  connect(ambient.port, internal);
  connect(ra.heatPort, internal.armature);
  connect(rc.heatPort, internal.core);
end Machine;
model T
  Machine m;
  Real x(start = 0, fixed = true);
equation
  der(x) = m.ambient.port.armature.Q_flow + m.rc.heatPort.T;
end T;
"""


def test_one_flow_equation_joins_the_loss_and_the_ambient() -> None:
    flat = flatten(parse(SOURCE), "T")
    joined = [
        equation
        for equation in flat.equations
        if {"m.ra.heatPort.Q_flow", "m.ambient.port.armature.Q_flow"}
        <= set(names(ir.Binary("-", equation.lhs, equation.rhs)))
    ]
    assert len(joined) == 1


@pytest.mark.parametrize("backend", ["python", "engine"])
def test_connect_to_a_nested_connector_joins_the_parent_set(
    backend: Literal["python", "engine"],
) -> None:
    if backend == "engine":
        from modelica.sim import engine  # noqa: PLC0415 - only this case needs it

        if engine.installed() is None and shutil.which("rustc") is None:
            pytest.skip("no engine in this install and no rustc to build one")
    result = simulate(causalize(flatten(parse(SOURCE), "T")), stop=1.0, backend=backend)
    # The ambient takes the 5 W loss, and the core is at 310 K.
    assert float(result["x"][-1]) == pytest.approx(315.0, abs=1e-6)
    assert float(result["m.ra.heatPort.T"][-1]) == pytest.approx(300.0, abs=1e-9)
