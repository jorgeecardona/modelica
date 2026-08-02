"""The Python twin of ``PIControl.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Blocks

from modelica.build import Model, connect, mod, to_modelica


class PIControl(Model):
    """A first-order plant under PI control, brought from 0 to a setpoint of 1"""

    setpoint = Blocks.Constant(k=1)
    error = Blocks.Feedback()
    controller = Blocks.PI(k=2, T=0.5, x=mod(start=0, fixed=True))
    plant = Blocks.FirstOrder(k=1, T=1, y=mod(signal=mod(start=0, fixed=True)))

    def equations(self):
        yield connect(self.setpoint.y, self.error.u1)
        yield connect(self.plant.y, self.error.u2)
        yield connect(self.error.y, self.controller.u)
        yield connect(self.controller.y, self.plant.u)


if __name__ == "__main__":
    print(to_modelica(PIControl), end="")
