"""The Python twin of ``MagLev.mo``."""

from modelica.build import Model, Real, der, parameter, sqrt, to_modelica


class MagLev(Model):
    """A steel ball levitated under an electromagnet: an unstable plant held at a fixed gap by PD feedback, the magnetic force pulling up as the inverse square of the gap"""

    m = parameter(Real, 0.05, unit="kg", doc="Ball mass")
    g = parameter(Real, 9.81, unit="m/s2", doc="Gravity")
    R = parameter(Real, 1, unit="Ohm", doc="Coil resistance")
    L = parameter(Real, 0.01, unit="H", doc="Coil inductance")
    km = parameter(Real, 0.001, unit="N.m2/A2", doc="Magnetic force constant")
    z_ref = parameter(Real, 0.01, unit="m", doc="Gap the controller holds")
    Kp = parameter(Real, 200, unit="V/m", doc="Proportional gain")
    Kd = parameter(Real, 5, unit="V.s/m", doc="Derivative gain")
    z = Real("Gap between the ball and the magnet", unit="m", start=0.012, fixed=True)
    v = Real("Rate the gap changes", unit="m/s", start=0, fixed=True)
    # The coil starts at the current that holds the ball at z_ref, z_ref*sqrt(m*g/km).
    i = Real("Coil current", unit="A", start=0.22147234590350104, fixed=True)
    i_ref = Real("Current that balances gravity at the target gap", unit="A")
    V = Real("Coil voltage the controller commands", unit="V")
    F = Real("Magnetic force lifting the ball", unit="N")

    def equations(self):
        yield self.i_ref == self.z_ref * sqrt(self.m * self.g / self.km)
        yield self.V == self.R * self.i_ref + self.Kp * (self.z - self.z_ref) + self.Kd * self.v
        yield self.F == self.km * (self.i / self.z) ** 2
        yield der(self.z) == self.v
        yield self.m * der(self.v) == self.m * self.g - self.F
        yield self.L * der(self.i) == self.V - self.R * self.i


if __name__ == "__main__":
    print(to_modelica(MagLev), end="")
