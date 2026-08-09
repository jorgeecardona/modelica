"""The Python twin of ``Translational.mo``."""

from modelica.build import (
    Connector,
    Function,
    Model,
    Real,
    alias,
    assign,
    call,
    der,
    flow,
    if_statement,
    input_,
    output,
    package,
    parameter,
    to_modelica,
)

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


class contactForce(Function):  # noqa: N801
    """Force from a one-sided elastic stop, zero until s_rel passes gap"""

    s_rel = input_(Real(unit="m"))
    gap = input_(Real(unit="m"))
    k = input_(Real(unit="N/m"))
    f = output(Real(unit="N"))

    def algorithm(self):
        yield assign(self.f, 0.0)
        yield if_statement(self.s_rel > self.gap, assign(self.f, self.k * (self.s_rel - self.gap)))


class contactEnergy(Function):  # noqa: N801
    """Strain energy stored in that stop -- the integral of contactForce"""

    s_rel = input_(Real(unit="m"))
    gap = input_(Real(unit="m"))
    k = input_(Real(unit="N/m"))
    u = output(Real(unit="J"))

    def algorithm(self):
        yield assign(self.u, 0.0)
        yield if_statement(
            self.s_rel > self.gap,
            assign(self.u, 0.5 * self.k * (self.s_rel - self.gap) ** 2),
        )


class Stop(Compliant):
    """One-sided elastic contact, after Modelica.Mechanics.Translational.ElastoGap"""

    k = parameter(Real, 100, unit="N/m", doc="Contact stiffness")
    gap = parameter(Position, 0.5, doc="Free travel before the stop is met")

    def equations(self):
        yield self.f == call("contactForce", self.s_rel, self.gap, self.k)


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
