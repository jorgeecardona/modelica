"""The Python twin of ``DCMotor.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Electrical
import Machines
import Rotational

from modelica.build import Model, connect, mod, to_modelica


class DCMotor(Model):
    """A DC motor started from rest on 12 V, running up against viscous friction"""

    supply = Electrical.ConstantVoltage(V=12)
    ground = Electrical.Ground()
    motor = Machines.Motor(
        Rw=0.5,
        Lw=0.05,
        k=0.5,
        J=0.02,
        coil=mod(i=mod(start=0, fixed=True)),
        rotor=mod(phi=mod(start=0, fixed=True), w=mod(start=0, fixed=True)),
    )
    friction = Rotational.Damper(d=0.002)

    def equations(self):
        yield connect(self.supply.p, self.motor.p)
        yield connect(self.motor.n, self.supply.n)
        yield connect(self.supply.n, self.ground.p)
        yield connect(self.motor.flange, self.friction.flange)


if __name__ == "__main__":
    print(to_modelica(DCMotor), end="")
