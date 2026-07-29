"""The Python twin of ``RCCircuit.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Electrical

from modelica.build import Model, connect, mod, to_modelica


class RCCircuit(Model):
    """A capacitor charging through a resistor from a 1 V source"""

    source = Electrical.ConstantVoltage(V=1)
    r = Electrical.Resistor(R=100)
    c = Electrical.Capacitor(C=0.01, v=mod(start=0, fixed=True))
    ground = Electrical.Ground()

    def equations(self):
        yield connect(self.source.p, self.r.p)
        yield connect(self.r.n, self.c.p)
        yield connect(self.c.n, self.source.n)
        yield connect(self.source.n, self.ground.p)


if __name__ == "__main__":
    print(to_modelica(RCCircuit), end="")
