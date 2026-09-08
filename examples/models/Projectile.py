"""The Python twin of ``Projectile.mo``."""

from modelica.build import Boolean, Model, Real, call, der, if_equation, parameter, to_modelica


class Projectile(Model):
    """A ball thrown straight up, its rise cut short and its fall made slow by air drag"""

    m = parameter(Real, 0.145, unit="kg", doc="Mass of the ball")
    g = parameter(Real, 9.81, unit="m/s2", doc="Gravitational acceleration")
    cd = parameter(Real, 0.0013, unit="kg/m", doc="Quadratic-drag coefficient, 0.5*rho*Cd*A")
    drag = parameter(Boolean, value=True, doc="Whether air drag acts; false is the vacuum shot")
    y = Real("Height above the launch point", unit="m", start=0, fixed=True)
    vy = Real("Vertical velocity", unit="m/s", start=30, fixed=True)

    def equations(self):
        yield der(self.y) == self.vy
        # A structural if-equation chooses the force law once, at flatten time, from the `drag`
        # parameter: with drag the ball feels quadratic air resistance opposing its motion
        # (cd*vy*abs(vy), always against vy), without it only gravity -- the textbook vacuum
        # parabola. The choice is not a run-time event; flipping `drag` reflattens the model.
        yield if_equation(
            self.drag,
            self.m * der(self.vy) == -self.m * self.g - self.cd * self.vy * call("abs", self.vy),
            otherwise=[self.m * der(self.vy) == -self.m * self.g],
        )


if __name__ == "__main__":
    print(to_modelica(Projectile), end="")
