"""The Python twin of ``MassSpringDamper.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Translational

from modelica.build import Model, connect, mod, to_modelica


class MassSpringDamper(Model):
    """A mass on a spring and damper, released from 1 m -- the RLC circuit's twin"""

    wall = Translational.Fixed(s0=0)
    spring = Translational.Spring(c=4, s_rel0=0)
    damper = Translational.Damper(d=0.4)
    mass = Translational.Mass(m=1, s=mod(start=1, fixed=True), v=mod(start=0, fixed=True))

    def equations(self):
        yield connect(self.wall.flange, self.spring.flange_a)
        yield connect(self.wall.flange, self.damper.flange_a)
        yield connect(self.spring.flange_b, self.mass.flange_a)
        yield connect(self.damper.flange_b, self.mass.flange_a)


if __name__ == "__main__":
    print(to_modelica(MassSpringDamper), end="")
