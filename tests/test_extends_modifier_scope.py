"""A modification on an ``extends`` clause resolves where the clause was written.

``Machines.BasicMachines.DCMachines.DC_PermanentMagnet`` writes ``extends
PartialBasicDCMachine(final ViNominal = VaNominal - Machines.Thermal.convertResistance(...),
wNominal(start = 1410*2*pi/60))``. The call reached emit as a call and ``pi`` as an unbound name.
"""

from __future__ import annotations

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
package P
  constant Real pi = 3.14159;
  function conv
    input Real r;
    input Real k;
    output Real y;
  algorithm
    y := r * k;
  end conv;
  partial model Base
    parameter Real Vi = 0;
    parameter Real w = 0;
    Real x(start = 0, fixed = true);
  equation
    der(x) = Vi + w;
  end Base;
  model Machine
    parameter Real Ra = 0.25;
    extends Base(final Vi = 1 - P.conv(Ra, 2), w(start = 2 * pi) = 2 * pi);
  end Machine;
end P;
model T
  P.Machine m;
end T;
"""


def test_an_extends_modifier_inlines_a_call_and_reads_a_package_constant() -> None:
    result = simulate(causalize(flatten(parse(SOURCE), "T")), stop=1.0, backend="python")
    # Vi = 1 - 0.25 * 2 = 0.5, w = 2 * pi.
    assert float(result["m.x"][-1]) == pytest.approx(0.5 + 2 * 3.14159, abs=1e-6)
