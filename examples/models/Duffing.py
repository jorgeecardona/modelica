"""The Python twin of ``Duffing.mo``."""

from modelica.build import Model, Real, cos, der, parameter, time, to_modelica


class Duffing(Model):
    """A beam between two magnets, shaken: one degree of freedom, driven, and chaotic"""

    delta = parameter(Real, 0.3, doc="Damping")
    alpha = parameter(
        Real, -1, doc="Linear stiffness; negative, so the centre is a hilltop between two wells"
    )
    beta = parameter(Real, 1, doc="Cubic stiffness, which makes the wells")
    gamma = parameter(Real, 0.5, doc="Amplitude of the drive")
    omega = parameter(Real, 1.2, doc="Angular frequency of the drive")
    x = Real(start=1, fixed=True, doc="Displacement")
    v = Real(start=0, fixed=True, doc="Velocity")

    def equations(self):
        yield der(self.x) == self.v
        yield der(self.v) == (
            self.gamma * cos(self.omega * time)
            - self.delta * self.v
            - self.alpha * self.x
            - self.beta * self.x**3
        )


if __name__ == "__main__":
    print(to_modelica(Duffing), end="")
