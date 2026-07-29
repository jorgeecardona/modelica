"""The Python twin of ``Pendulum.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class Pendulum(Model):
    """A point mass on a rigid rod, in Cartesian coordinates -- an index-3 DAE"""

    m = parameter(Real, 1, unit="kg", doc="Point mass")
    L = parameter(Real, 1, unit="m", doc="Rod length")
    g = parameter(Real, 9.81, unit="m/s2", doc="Gravity")
    x = Real("Horizontal position", unit="m", start=1, fixed=True)
    y = Real("Vertical position, downward negative", unit="m", start=0, fixed=True)
    vx = Real(unit="m/s", start=0, fixed=True)
    vy = Real(unit="m/s", start=0, fixed=True)
    F = Real("Rod force -- the constraint multiplier", unit="N")

    def equations(self):
        yield der(self.x) == self.vx
        yield der(self.y) == self.vy
        yield self.m * der(self.vx) == -self.F * self.x / self.L
        yield self.m * der(self.vy) == -self.F * self.y / self.L - self.m * self.g
        yield self.x**2 + self.y**2 == self.L**2


if __name__ == "__main__":
    print(to_modelica(Pendulum), end="")
