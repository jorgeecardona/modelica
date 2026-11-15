"""An inherited equation resolves names in the class that wrote it.

``Blocks.Interfaces.Adaptors`` writes ``u = Functions.state1({f, u1}, time)`` in a base class.
From a derived class in another package, ``Functions`` did not resolve.
"""

from __future__ import annotations

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.flatten import flatten
from modelica.lang import parse
from modelica.sim import simulate

SOURCE = """\
package P
  constant Real k = 3;
  package Functions
    function second
      input Real u[2];
      output Real y;
    algorithm
      y := u[2];
    end second;
  end Functions;
  partial model Base
    Real x(start = 0, fixed = true);
    parameter Real a = 1;
  equation
    der(x) = k * Functions.second({a, 2});
  end Base;
end P;
package Q
  model Derived
    extends P.Base;
  end Derived;
end Q;
model T
  Q.Derived d;
end T;
"""


def test_an_inherited_equation_reads_names_of_the_writing_package() -> None:
    result = simulate(causalize(flatten(parse(SOURCE), "T")), stop=1.0, backend="python")
    # der(x) = 3 * 2.
    assert float(result["d.x"][-1]) == pytest.approx(6.0, abs=1e-6)
