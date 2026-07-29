"""The Python twin of ``TwoBodyHeat.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Thermal

from modelica.build import Model, connect, mod, to_modelica


class TwoBodyHeat(Model):
    """Two thermal masses equilibrating through a conductor -- energy is conserved"""

    hot = Thermal.HeatCapacitor(C=100, T=mod(start=400, fixed=True))
    cold = Thermal.HeatCapacitor(C=200, T=mod(start=300, fixed=True))
    conductor = Thermal.ThermalConductor(G=5)

    def equations(self):
        yield connect(self.hot.port, self.conductor.port_a)
        yield connect(self.conductor.port_b, self.cold.port)


if __name__ == "__main__":
    print(to_modelica(TwoBodyHeat), end="")
