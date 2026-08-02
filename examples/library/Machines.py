"""The Python twin of ``Machines.mo``."""

import sys
from pathlib import Path

# This library is built on two others, which sit beside it.
sys.path.append(str(Path(__file__).resolve().parent))

import Electrical
import Rotational

from modelica.build import Model, Real, connect, der, package, parameter, ref, to_modelica


class EMF(Model):
    """Ideal converter between current and torque, and between speed and voltage"""

    k = parameter(Real, 1, unit="N.m/A", doc="Torque constant, equal to the back-EMF constant")
    p = Electrical.Pin("Positive terminal")
    n = Electrical.Pin("Negative terminal")
    flange = Rotational.Flange("Shaft it drives")
    v = Electrical.Voltage("Voltage across the terminals")
    i = Electrical.Current("Current through the machine")
    w = Rotational.AngularVelocity("Speed of the shaft")

    def equations(self):
        yield self.v == self.p.v - self.n.v
        yield self.i == self.p.i
        yield self.p.i + self.n.i == 0
        yield self.w == der(self.flange.phi)
        yield self.v == self.k * self.w
        yield self.flange.tau == -self.k * self.i


class Motor(Model):
    """A DC motor as one component: winding, converter and rotor inside"""

    Rw = parameter(Real, 0.5, unit="Ohm", doc="Winding resistance")
    Lw = parameter(Real, 0.05, unit="H", doc="Winding inductance")
    k = parameter(Real, 0.5, unit="N.m/A", doc="Torque constant")
    J = parameter(Real, 0.02, unit="kg.m2", doc="Moment of inertia of the rotor")
    p = Electrical.Pin("Positive terminal")
    n = Electrical.Pin("Negative terminal")
    flange = Rotational.Flange("Output shaft")
    # `ref` and not the descriptor itself: inside a class body a declaration has no name
    # yet -- Python assigns it through `__set_name__` after the body has run.
    winding = Electrical.Resistor(R=ref("Rw"))
    coil = Electrical.Inductor(L=ref("Lw"))
    converter = EMF(k=ref("k"))
    rotor = Rotational.Inertia(J=ref("J"))

    def equations(self):
        yield connect(self.p, self.winding.p)
        yield connect(self.winding.n, self.coil.p)
        yield connect(self.coil.n, self.converter.p)
        yield connect(self.converter.n, self.n)
        yield connect(self.converter.flange, self.rotor.flange_a)
        yield connect(self.rotor.flange_b, self.flange)


PACKAGE = package("Machines", __name__, "Where the electrical domain meets the rotational one")

if __name__ == "__main__":
    print(to_modelica(PACKAGE), end="")
