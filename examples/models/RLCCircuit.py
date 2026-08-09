"""The Python twin of ``RLCCircuit.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Electrical

from modelica.build import Model, connect, import_, mod, place, to_modelica


class RLCCircuit(Model):
    """A series RLC circuit stepped with 1 V -- damped oscillation, two states"""

    IMPORTS = (import_("Electrical", "E"),)

    source = Electrical.ConstantVoltage(V=1, annotation=place([(-92, -18), (-56, 18)], -90))
    r = Electrical.Resistor(R=1, annotation=place([(-38, 42), (-2, 78)]))
    l = Electrical.Inductor(L=1, i=mod(start=0, fixed=True), annotation=place([(16, 42), (52, 78)]))
    c = Electrical.Capacitor(
        C=1, v=mod(start=0, fixed=True), annotation=place([(56, -18), (92, 18)], -90)
    )
    ground = Electrical.Ground(annotation=place([(-18, -96), (18, -60)]))

    def equations(self):
        yield connect(self.source.p, self.r.p)
        yield connect(self.r.n, self.l.p)
        yield connect(self.l.n, self.c.p)
        yield connect(self.c.n, self.source.n)
        yield connect(self.source.n, self.ground.p)


if __name__ == "__main__":
    print(to_modelica(RLCCircuit), end="")
