"""The Python twin of ``DoublePendulum.mo``."""

from modelica.build import Model, Real, cos, der, parameter, sin, to_modelica


class DoublePendulum(Model):
    """Two rods, two masses, released from horizontal: the textbook chaotic machine, in angles"""

    m1 = parameter(Real, 1, unit="kg", doc="Upper mass")
    m2 = parameter(Real, 1, unit="kg", doc="Lower mass")
    L1 = parameter(Real, 1, unit="m", doc="Upper rod")
    L2 = parameter(Real, 1, unit="m", doc="Lower rod")
    g = parameter(Real, 9.81, unit="m/s2", doc="Gravity")
    th1 = Real(
        start=1.5707963267948966, fixed=True, unit="rad", doc="Upper angle from the vertical, down"
    )
    th2 = Real(
        start=1.5707963267948966, fixed=True, unit="rad", doc="Lower angle from the vertical, down"
    )
    w1 = Real(start=0, fixed=True, unit="rad/s", doc="Upper angular velocity")
    w2 = Real(start=0, fixed=True, unit="rad/s", doc="Lower angular velocity")
    d = Real(unit="rad", doc="The angle between the rods")
    den = Real(doc="The common denominator of both accelerations; never zero for positive masses")
    E = Real(unit="J", doc="Total energy, kinetic plus potential: zero at the start and conserved")

    def equations(self):
        yield self.d == self.th1 - self.th2
        yield self.den == 2 * self.m1 + self.m2 - self.m2 * cos(2 * self.d)
        yield der(self.th1) == self.w1
        yield der(self.th2) == self.w2
        yield der(self.w1) == (
            -self.g * (2 * self.m1 + self.m2) * sin(self.th1)
            - self.m2 * self.g * sin(self.th1 - 2 * self.th2)
            - 2
            * sin(self.d)
            * self.m2
            * (self.w2**2 * self.L2 + self.w1**2 * self.L1 * cos(self.d))
        ) / (self.L1 * self.den)
        yield der(self.w2) == (
            2
            * sin(self.d)
            * (
                self.w1**2 * self.L1 * (self.m1 + self.m2)
                + self.g * (self.m1 + self.m2) * cos(self.th1)
                + self.w2**2 * self.L2 * self.m2 * cos(self.d)
            )
        ) / (self.L2 * self.den)
        yield self.E == (
            (self.m1 + self.m2) / 2 * self.L1**2 * self.w1**2
            + self.m2 / 2 * self.L2**2 * self.w2**2
            + self.m2 * self.L1 * self.L2 * self.w1 * self.w2 * cos(self.d)
            - (self.m1 + self.m2) * self.g * self.L1 * cos(self.th1)
            - self.m2 * self.g * self.L2 * cos(self.th2)
        )


if __name__ == "__main__":
    print(to_modelica(DoublePendulum), end="")
