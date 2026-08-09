"""The Python twin of ``TwoFalls.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Translational

from modelica.build import Model, Real, inner, mod, to_modelica


class TwoFalls(Model):
    """Two bodies dropped from different heights, sharing one gravity"""

    environment = inner(Translational.Environment(g=9.81))
    high = Translational.FallingBody(s=mod(start=100, fixed=True), v=mod(start=0, fixed=True))
    low = Translational.FallingBody(s=mod(start=20, fixed=True), v=mod(start=0, fixed=True))
    gap = Real("How far apart they are -- constant, because gravity ignores mass", unit="m")

    def equations(self):
        yield self.gap == self.high.s - self.low.s


if __name__ == "__main__":
    print(to_modelica(TwoFalls), end="")
