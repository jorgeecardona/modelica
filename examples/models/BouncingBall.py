"""The Python twin of ``BouncingBall.mo``."""

from modelica.build import Model, Real, der, parameter, pre, reinit, to_modelica, when


class BouncingBall(Model):
    """A ball dropped onto the floor -- one state event per bounce"""

    g = parameter(Real, 9.81, unit="m/s2", doc="Gravity")
    e = parameter(Real, 0.8, doc="Coefficient of restitution (fraction of speed kept)")
    h = Real("Height above the floor", unit="m", start=1, fixed=True)
    v = Real("Vertical velocity", unit="m/s", start=0, fixed=True)

    def equations(self):
        yield der(self.h) == self.v
        yield der(self.v) == -self.g
        yield when(self.h <= 0, reinit(self.v, -self.e * pre(self.v)))


if __name__ == "__main__":
    print(to_modelica(BouncingBall), end="")
