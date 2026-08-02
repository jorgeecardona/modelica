"""The Python twin of ``Rotational.mo``."""

from modelica.build import Connector, Model, Real, alias, der, flow, package, parameter, to_modelica

Angle = alias("Angle", Real, unit="rad")
AngularVelocity = alias("AngularVelocity", Real, unit="rad/s")
AngularAcceleration = alias("AngularAcceleration", Real, unit="rad/s2")
Torque = alias("Torque", Real, unit="N.m")


class Flange(Connector):
    """Rotational flange: angle is shared, torque sums to zero"""

    phi = Angle("Absolute angle")
    tau = flow(Torque("Torque exerted on the flange"))


class Inertia(Model):
    """Newton's second law for rotation -- introduces two states, phi and w"""

    J = parameter(Real, 1, unit="kg.m2", doc="Moment of inertia")
    flange_a = Flange("Left flange")
    flange_b = Flange("Right flange")
    phi = Angle("Absolute angle of the body")
    w = AngularVelocity()
    a = AngularAcceleration()

    def equations(self):
        yield self.flange_a.phi == self.phi
        yield self.flange_b.phi == self.phi
        yield self.w == der(self.phi)
        yield self.a == der(self.w)
        yield self.J * self.a == self.flange_a.tau + self.flange_b.tau


class Damper(Model):
    """Viscous friction against a fixed housing"""

    d = parameter(Real, 0.1, unit="N.m.s/rad", doc="Damping constant")
    flange = Flange("Flange the friction acts on")

    def equations(self):
        yield self.flange.tau == self.d * der(self.flange.phi)


class Fixed(Model):
    """Immovable mounting point"""

    phi0 = parameter(Angle, 0, doc="Angle it is held at")
    flange = Flange()

    def equations(self):
        yield self.flange.phi == self.phi0


PACKAGE = package(
    "Rotational",
    __name__,
    "Minimal 1-D rotational mechanics, after Modelica.Mechanics.Rotational",
)

if __name__ == "__main__":
    print(to_modelica(PACKAGE), end="")
