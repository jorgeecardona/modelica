"""The Python twin of ``BouncingBallRest.mo``."""

from modelica.build import (
    Model,
    Real,
    assign,
    der,
    discrete,
    if_,
    parameter,
    pre,
    reinit,
    to_modelica,
    when,
)


class BouncingBallRest(Model):
    """A bouncing ball that comes to rest -- the well-posed companion to BouncingBall"""

    g = parameter(Real, 9.81, unit="m/s2", doc="Gravity")
    e = parameter(Real, 0.8, doc="Coefficient of restitution (fraction of speed kept)")
    v_min = parameter(
        Real, 0.1, unit="m/s", doc="Stick to the floor once the rebound speed would fall below this"
    )
    h = Real("Height above the floor", unit="m", start=1, fixed=True)
    v = Real("Vertical velocity", unit="m/s", start=0, fixed=True)
    resting = discrete(
        Real(start=0, fixed=True, doc="1 once the ball is at rest on the floor, 0 while it moves")
    )

    def equations(self):
        # ``resting`` is a discrete flag, 0 while the ball moves and 1 once it has stuck; it scales
        # the free-fall dynamics to zero rather than switching them with a branch, so a bare
        # discrete real (not a Boolean, which the Rust backend keeps as a float) carries it and
        # both backends read it the same way. It is what makes the whole run defined: BouncingBall
        # accumulates infinitely many bounces at a finite time and then falls through the floor,
        # and this one sticks once a rebound would be smaller than ``v_min``.
        yield der(self.h) == (1 - self.resting) * self.v
        yield der(self.v) == -(1 - self.resting) * self.g
        yield when(
            self.h <= 0,
            assign(self.resting, if_(-self.e * pre(self.v) < self.v_min, 1, 0)),
            reinit(self.v, if_(-self.e * pre(self.v) < self.v_min, 0, -self.e * pre(self.v))),
            reinit(self.h, 0),
        )


if __name__ == "__main__":
    print(to_modelica(BouncingBallRest), end="")
