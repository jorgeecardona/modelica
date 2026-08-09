"""The Python twin of ``Electrical.mo``.

Same package, written with :mod:`modelica.build` instead of Modelica syntax. Import it and
the module *is* the package -- ``Electrical.Resistor(R=100)`` reads exactly like the
Modelica it stands for.
"""

from modelica.build import (
    Connector,
    Model,
    Real,
    alias,
    connect,
    der,
    flow,
    icon,
    mod,
    package,
    parameter,
    place,
    replaceable,
    shape,
    to_modelica,
)

PIN = [(-94.0, 4.0), (-86.0, -4.0)]
SOLID = "FillPattern.Solid"
NAME = [(-100.0, 62.0), (100.0, 40.0)]
PIN_LEFT = place([(-94.0, -4.0), (-86.0, 4.0)])
PIN_RIGHT = place([(86.0, -4.0), (94.0, 4.0)])
PIN_TOP = place([(-4.0, 86.0), (4.0, 94.0)])


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

    p = Pin("Positive pin", annotation=PIN_LEFT)
    n = Pin("Negative pin", annotation=PIN_RIGHT)
    v = Voltage("Voltage drop from p to n")
    i = Current("Current flowing from p to n")

    def equations(self):
        yield self.v == self.p.v - self.n.v
        yield self.i == self.p.i
        yield self.p.i + self.n.i == 0


class Resistor(OnePort):
    """Ohm's law: v = R*i"""

    R = parameter(Resistance, 1, doc="Resistance")

    ICON = icon(
        shape("Line", points=[(-90, 0), (-52, 0)]),
        shape(
            "Line",
            points=[
                (-52, 0),
                (-42, 22),
                (-22, -22),
                (-2, 22),
                (18, -22),
                (38, 22),
                (48, 0),
                (52, 0),
            ],
        ),
        shape("Line", points=[(52, 0), (90, 0)]),
        shape("Text", extent=NAME, textString="%name"),
    )

    def equations(self):
        yield self.v == self.R * self.i


class Capacitor(OnePort):
    """i = C*der(v) -- introduces one state, v"""

    C = parameter(Capacitance, 1, doc="Capacitance")

    ICON = icon(
        shape("Line", points=[(-90, 0), (-12, 0)]),
        shape("Line", points=[(12, 0), (90, 0)]),
        shape("Line", points=[(-12, 34), (-12, -34)]),
        shape("Line", points=[(12, 34), (12, -34)]),
        shape("Text", extent=NAME, textString="%name"),
    )

    def equations(self):
        yield self.i == self.C * der(self.v)


class Inductor(OnePort):
    """v = L*der(i) -- introduces one state, i"""

    L = parameter(Inductance, 1, doc="Inductance")

    ICON = icon(
        shape("Line", points=[(-90, 0), (-60, 0)]),
        shape("Line", points=[(-60, 0), (-60, 14), (-45, 22), (-30, 14), (-30, 0)]),
        shape("Line", points=[(-30, 0), (-30, 14), (-15, 22), (0, 14), (0, 0)]),
        shape("Line", points=[(0, 0), (0, 14), (15, 22), (30, 14), (30, 0)]),
        shape("Line", points=[(30, 0), (30, 14), (45, 22), (60, 14), (60, 0)]),
        shape("Line", points=[(60, 0), (90, 0)]),
        shape("Text", extent=NAME, textString="%name"),
    )

    def equations(self):
        yield self.v == self.L * der(self.i)


class ConstantVoltage(OnePort):
    """Ideal source holding a fixed voltage"""

    V = parameter(Voltage, 1, doc="Source voltage")

    ICON = icon(
        shape("Line", points=[(-90, 0), (-30, 0)]),
        shape("Line", points=[(30, 0), (90, 0)]),
        shape("Ellipse", extent=[(-30, 30), (30, -30)]),
        shape("Line", points=[(-16, 0), (-4, 0)]),
        shape("Line", points=[(-10, -6), (-10, 6)]),
        shape("Line", points=[(4, 0), (16, 0)]),
        shape("Text", extent=NAME, textString="%name"),
    )

    def equations(self):
        yield self.v == self.V


class Ground(Model):
    """Reference potential; every circuit needs exactly one"""

    p = Pin(annotation=PIN_TOP)

    ICON = icon(
        shape("Line", points=[(0, 90), (0, 20)]),
        shape("Line", points=[(-46, 20), (46, 20)]),
        shape("Line", points=[(-28, 2), (28, 2)]),
        shape("Line", points=[(-11, -16), (11, -16)]),
        shape("Text", extent=[(-100, -30), (100, -52)], textString="%name"),
    )

    def equations(self):
        yield self.p.v == 0


class Probe(Model):
    """A 1 V source charging a capacitor through one swappable element"""

    element = replaceable(Resistor(R=100, doc="The component under test"), constrainedby=OnePort)
    source = ConstantVoltage(V=1)
    c = Capacitor(C=0.01, v=mod(start=0, fixed=True))
    ground = Ground()

    def equations(self):
        yield connect(self.source.p, self.element.p)
        yield connect(self.element.n, self.c.p)
        yield connect(self.c.n, self.source.n)
        yield connect(self.source.n, self.ground.p)


PACKAGE = package(
    "Electrical", __name__, "Minimal analog electrical components, after Modelica.Electrical.Analog"
)

if __name__ == "__main__":
    print(to_modelica(PACKAGE), end="")
