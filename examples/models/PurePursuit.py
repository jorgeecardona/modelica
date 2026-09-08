"""The Python twin of ``PurePursuit.mo``."""

from modelica.build import Model, Real, atan2, cos, der, parameter, sin, to_modelica


class PurePursuit(Model):
    """A pursuer that always steers toward a target: pure-pursuit guidance with atan2"""

    xt = parameter(Real, 10, unit="m", doc="Target x position")
    yt = parameter(Real, 0, unit="m", doc="Target y position")
    V = parameter(Real, 1, unit="m/s", doc="Pursuer speed, constant")
    k = parameter(Real, 3, unit="1/s", doc="Heading turn-rate gain")
    x = Real("Pursuer x position", unit="m", start=0, fixed=True)
    y = Real("Pursuer y position", unit="m", start=8, fixed=True)
    theta = Real("Pursuer heading", unit="rad", start=0, fixed=True)
    thd = Real("Desired heading, the bearing to the target", unit="rad")

    def equations(self):
        # The bearing to the target from the pursuer's own position: atan2 of the relative
        # position, so the heading law depends on the state and its jacobian needs atan2's rule.
        yield self.thd == atan2(self.yt - self.y, self.xt - self.x)
        yield der(self.theta) == self.k * (self.thd - self.theta)
        yield der(self.x) == self.V * cos(self.theta)
        yield der(self.y) == self.V * sin(self.theta)


if __name__ == "__main__":
    print(to_modelica(PurePursuit), end="")
