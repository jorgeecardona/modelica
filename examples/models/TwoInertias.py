"""The Python twin of ``TwoInertias.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Rotational

from modelica.build import Model, connect, mod, to_modelica


class TwoInertias(Model):
    """Two inertias on one rigid shaft, braked: a constraint between states, so index 3"""

    big = Rotational.Inertia(J=2, phi=mod(start=0, fixed=True), w=mod(start=10, fixed=True))
    small = Rotational.Inertia(J=0.5)
    brake = Rotational.Damper(d=0.25)

    def equations(self):
        yield connect(self.big.flange_b, self.small.flange_a)
        yield connect(self.small.flange_b, self.brake.flange)


if __name__ == "__main__":
    print(to_modelica(TwoInertias), end="")
