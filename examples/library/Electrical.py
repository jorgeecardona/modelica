"""The Python twin of ``Electrical.mo``.

Same package, written with :mod:`modelica.build` instead of Modelica syntax. Import it and
the module *is* the package -- ``Electrical.Resistor(R=100)`` reads exactly like the
Modelica it stands for.
"""

from modelica.build import Connector, Model, Real, alias, der, flow, package, parameter, to_modelica

Voltage = alias("Voltage", Real, unit="V")
Current = alias("Current", Real, unit="A")
Resistance = alias("Resistance", Real, unit="Ohm")
Capacitance = alias("Capacitance", Real, unit="F")
Inductance = alias("Inductance", Real, unit="H")


class Pin(Connector):
    """Electrical pin: voltage is shared, current sums to zero"""

    v = Voltage("Potential at the pin")
    i = flow(Current("Current flowing into the pin"))


class OnePort(Model, partial=True):
    """A component with two pins carrying the same current"""

    p = Pin("Positive pin")
    n = Pin("Negative pin")
    v = Voltage("Voltage drop from p to n")
    i = Current("Current flowing from p to n")

    def equations(self):
        yield self.v == self.p.v - self.n.v
        yield self.i == self.p.i
        yield self.p.i + self.n.i == 0


class Resistor(OnePort):
    """Ohm's law: v = R*i"""

    R = parameter(Resistance, 1, doc="Resistance")

    def equations(self):
        yield self.v == self.R * self.i


class Capacitor(OnePort):
    """i = C*der(v) -- introduces one state, v"""

    C = parameter(Capacitance, 1, doc="Capacitance")

    def equations(self):
        yield self.i == self.C * der(self.v)


class Inductor(OnePort):
    """v = L*der(i) -- introduces one state, i"""

    L = parameter(Inductance, 1, doc="Inductance")

    def equations(self):
        yield self.v == self.L * der(self.i)


class ConstantVoltage(OnePort):
    """Ideal source holding a fixed voltage"""

    V = parameter(Voltage, 1, doc="Source voltage")

    def equations(self):
        yield self.v == self.V


class Ground(Model):
    """Reference potential; every circuit needs exactly one"""

    p = Pin()

    def equations(self):
        yield self.p.v == 0


PACKAGE = package(
    "Electrical", __name__, "Minimal analog electrical components, after Modelica.Electrical.Analog"
)

if __name__ == "__main__":
    print(to_modelica(PACKAGE), end="")
