"""The Python twin of ``DCMotorThermal.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Electrical
import Machines
import Rotational
import Thermal

from modelica.build import Model, connect, mod, to_modelica


class DCMotorThermal(Model):
    """A loaded DC motor that warms as it runs: the winding's heat raises its own resistance"""

    supply = Electrical.ConstantVoltage(V=12)
    ground = Electrical.Ground()
    winding = Electrical.HeatingResistor(R_ref=0.5, alpha=0.0039, T_ref=293.15)
    coil = Electrical.Inductor(L=0.05, i=mod(start=0, fixed=True))
    converter = Machines.EMF(k=0.5)
    rotor = Rotational.Inertia(J=0.02, phi=mod(start=0, fixed=True), w=mod(start=0, fixed=True))
    friction = Rotational.Damper(d=0.05)
    winding_mass = Thermal.HeatCapacitor(C=20, T=mod(start=293.15, fixed=True))
    to_ambient = Thermal.ThermalConductor(G=0.5)
    ambient = Thermal.FixedTemperature(T=293.15)

    def equations(self):
        yield connect(self.supply.p, self.winding.p)
        yield connect(self.winding.n, self.coil.p)
        yield connect(self.coil.n, self.converter.p)
        yield connect(self.converter.n, self.supply.n)
        yield connect(self.supply.n, self.ground.p)
        yield connect(self.converter.flange, self.rotor.flange_a)
        yield connect(self.rotor.flange_b, self.friction.flange)
        yield connect(self.winding.heat_port, self.winding_mass.port)
        yield connect(self.winding_mass.port, self.to_ambient.port_a)
        yield connect(self.to_ambient.port_b, self.ambient.port)


if __name__ == "__main__":
    print(to_modelica(DCMotorThermal), end="")
