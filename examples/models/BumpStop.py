"""The Python twin of ``BumpStop.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Translational

from modelica.build import Model, Real, call, connect, mod, to_modelica


class BumpStop(Model):
    """A mass on a spring meeting an elastic stop -- the contact law is a function"""

    wall = Translational.Fixed(s0=0)
    spring = Translational.Spring(c=4, s_rel0=0)
    stop = Translational.Stop(k=100, gap=0.5)
    mass = Translational.Mass(m=1, s=mod(start=-1, fixed=True), v=mod(start=0, fixed=True))
    E = Real("Total energy -- constant, because nothing here dissipates", unit="J")

    def equations(self):
        yield connect(self.wall.flange, self.spring.flange_a)
        yield connect(self.wall.flange, self.stop.flange_a)
        yield connect(self.spring.flange_b, self.mass.flange_a)
        yield connect(self.stop.flange_b, self.mass.flange_a)
        kinetic = 0.5 * self.mass.m * self.mass.v**2
        strain = 0.5 * self.spring.c * self.spring.s_rel**2
        contact = call(
            "Translational.contactEnergy", self.stop.s_rel, self.stop.gap, self.stop.k
        )
        yield self.E == kinetic + strain + contact


if __name__ == "__main__":
    print(to_modelica(BumpStop), end="")
