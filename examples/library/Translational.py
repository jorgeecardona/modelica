"""The Python twin of ``Translational.mo``."""

from modelica.build import Connector, Model, Real, alias, der, flow, package, parameter, to_modelica

Position = alias("Position", Real, unit="m")
Velocity = alias("Velocity", Real, unit="m/s")
Acceleration = alias("Acceleration", Real, unit="m/s2")
Force = alias("Force", Real, unit="N")


class Flange(Connector):
    """Mechanical flange: position is shared, force sums to zero"""

    s = Position("Absolute position")
    f = flow(Force("Force exerted on the flange"))


class Compliant(Model, partial=True):
    """Two flanges with a force between them (spring, damper, ...)"""

    flange_a = Flange()
    flange_b = Flange()
    s_rel = Position("flange_b.s - flange_a.s")
    f = Force("Force from flange_a to flange_b")

    def equations(self):
        yield self.s_rel == self.flange_b.s - self.flange_a.s
        yield self.flange_b.f == self.f
        yield self.flange_a.f == -self.f


class Mass(Model):
    """Newton's second law -- introduces two states, s and v"""

    m = parameter(Real, 1, unit="kg", doc="Mass")
    flange_a = Flange()
    flange_b = Flange()
    s = Position("Absolute position of the centre")
    v = Velocity()
    a = Acceleration()

    def equations(self):
        yield self.flange_a.s == self.s
        yield self.flange_b.s == self.s
        yield self.v == der(self.s)
        yield self.a == der(self.v)
        yield self.m * self.a == self.flange_a.f + self.flange_b.f


class Spring(Compliant):
    """Linear spring: f = c*(s_rel - s_rel0)"""

    c = parameter(Real, 1, unit="N/m", doc="Stiffness")
    s_rel0 = parameter(Position, 0, doc="Unstretched length")

    def equations(self):
        yield self.f == self.c * (self.s_rel - self.s_rel0)


class Damper(Compliant):
    """Linear viscous damper: f = d*der(s_rel)"""

    d = parameter(Real, 1, unit="N.s/m", doc="Damping constant")

    def equations(self):
        yield self.f == self.d * der(self.s_rel)


class Fixed(Model):
    """Immovable ground point"""

    s0 = parameter(Position, 0, doc="Position it is fixed at")
    flange = Flange()

    def equations(self):
        yield self.flange.s == self.s0


PACKAGE = package(
    "Translational",
    __name__,
    "Minimal 1-D translational mechanics, after Modelica.Mechanics.Translational",
)

if __name__ == "__main__":
    print(to_modelica(PACKAGE), end="")
