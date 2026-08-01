"""The Python twin of ``AtwoodPendulum.mo``."""

from modelica.build import Model, Real, cos, der, parameter, sin, to_modelica


class AtwoodPendulum(Model):
    """A swinging Atwood machine on an elastic string, carrying a double pendulum -- five degrees of freedom, and chaos"""

    g = parameter(Real, 9.81, unit="m/s2", doc="Gravity")
    M = parameter(
        Real, 2.1, unit="kg", doc="Counterweight -- vertical only, on the other side of the pulley"
    )
    m0 = parameter(Real, 1.0, unit="kg", doc="The swinging body, at the end of the string")
    m1 = parameter(Real, 0.6, unit="kg", doc="First bob")
    m2 = parameter(Real, 0.4, unit="kg", doc="Second bob")
    l1 = parameter(Real, 0.35, unit="m", doc="First rod")
    l2 = parameter(Real, 0.3, unit="m", doc="Second rod")
    k = parameter(
        Real,
        60.0,
        unit="N/m",
        doc="String stiffness -- this is what makes it a *spring* Atwood machine",
    )
    L = parameter(Real, 1.5, unit="m", doc="String length, unstretched")
    track = parameter(
        Real, -0.45, unit="m", doc="Where the counterweight's line sits, for the picture"
    )

    z = Real("Counterweight descent below the pulley", unit="m", start=0.94335, fixed=True)
    r = Real("Length of the swinging segment", unit="m", start=0.9, fixed=True)
    th = Real("Its angle from the downward vertical", unit="rad", start=0.5, fixed=True)
    p1 = Real("First rod, from the downward vertical", unit="rad", start=0.9, fixed=True)
    p2 = Real("Second rod, from the downward vertical", unit="rad", start=-0.6, fixed=True)
    vz = Real(unit="m/s", start=0, fixed=True)
    vr = Real(unit="m/s", start=0, fixed=True)
    om = Real(unit="rad/s", start=0, fixed=True)
    w1 = Real(unit="rad/s", start=0, fixed=True)
    w2 = Real(unit="rad/s", start=0, fixed=True)
    stretch = Real("How much longer the string is than it wants to be", unit="m")
    tension = Real("String tension -- the only thing the two sides say to each other", unit="N")
    c1 = Real("cos(p1 - th), and the five that follow it: the geometry, named once")
    s1 = Real()
    c2 = Real()
    s2 = Real()
    c12 = Real()
    s12 = Real()
    xc = Real("The counterweight, and then the four joints, for drawing", unit="m")
    yc = Real(unit="m")
    xp = Real(unit="m")
    yp = Real(unit="m")
    x1 = Real(unit="m")
    y1 = Real(unit="m")
    x2 = Real(unit="m")
    y2 = Real(unit="m")
    energy = Real("Total energy -- conserved, and the only check on the algebra", unit="J")

    def equations(self):
        yield der(self.z) == self.vz
        yield der(self.r) == self.vr
        yield der(self.th) == self.om
        yield der(self.p1) == self.w1
        yield der(self.p2) == self.w2

        yield self.c1 == cos(self.p1 - self.th)
        yield self.s1 == sin(self.p1 - self.th)
        yield self.c2 == cos(self.p2 - self.th)
        yield self.s2 == sin(self.p2 - self.th)
        yield self.c12 == cos(self.p1 - self.p2)
        yield self.s12 == sin(self.p1 - self.p2)

        yield self.stretch == self.z + self.r - self.L
        yield self.tension == self.k * self.stretch

        # The counterweight only ever hears the string.
        yield self.M * der(self.vz) == self.M * self.g - self.tension

        swinging = self.m0 + self.m1 + self.m2
        lower = self.m1 + self.m2

        # Radial, and then the three angular equations, straight off the Lagrangian. Each
        # is linear in the accelerations and none is solved for one, which is the point:
        # the flattener finds the 4x4 block and solves it, so nothing here had to be
        # inverted by hand.
        yield swinging * (der(self.vr) - self.om**2 * self.r) == self.g * swinging * cos(
            self.th
        ) - self.tension + self.l1 * lower * (
            self.w1**2 * self.c1 + der(self.w1) * self.s1
        ) + self.l2 * self.m2 * (self.w2**2 * self.c2 + der(self.w2) * self.s2)

        yield swinging * (
            self.r * der(self.om) + 2 * self.om * self.vr
        ) == -self.g * swinging * sin(self.th) - self.l1 * lower * (
            self.c1 * der(self.w1) - self.s1 * self.w1**2
        ) - self.l2 * self.m2 * (self.c2 * der(self.w2) - self.s2 * self.w2**2)

        yield (
            self.l1 * lower * der(self.w1)
            + lower
            * (
                self.r * self.c1 * der(self.om)
                - self.s1 * der(self.vr)
                + self.om**2 * self.r * self.s1
                + 2 * self.om * self.vr * self.c1
            )
            + self.l2 * self.m2 * (self.c12 * der(self.w2) + self.s12 * self.w2**2)
            + self.g * lower * sin(self.p1)
            == 0
        )

        yield (
            self.l2 * der(self.w2)
            + self.r * self.c2 * der(self.om)
            - self.s2 * der(self.vr)
            + self.om**2 * self.r * self.s2
            + 2 * self.om * self.vr * self.c2
            + self.l1 * (self.c12 * der(self.w1) - self.s12 * self.w1**2)
            + self.g * sin(self.p2)
            == 0
        )

        yield self.xc == self.track
        yield self.yc == -self.z
        yield self.xp == self.r * sin(self.th)
        yield self.yp == -self.r * cos(self.th)
        yield self.x1 == self.xp + self.l1 * sin(self.p1)
        yield self.y1 == self.yp - self.l1 * cos(self.p1)
        yield self.x2 == self.x1 + self.l2 * sin(self.p2)
        yield self.y2 == self.y1 - self.l2 * cos(self.p2)

        # Kinetic plus potential plus what the string is storing. Nothing forces this to
        # hold: it holds only if every term above was derived correctly, which is why it
        # is in the model rather than in a comment.
        yield self.energy == (
            self.M * self.vz**2 / 2
            - self.M * self.g * self.z
            + self.m0 * (self.vr**2 + self.om**2 * self.r**2) / 2
            - self.g * self.m0 * self.r * cos(self.th)
            + self.m1
            * (
                self.vr**2
                + self.om**2 * self.r**2
                + self.l1**2 * self.w1**2
                + 2 * self.l1 * self.om * self.w1 * self.r * self.c1
                - 2 * self.l1 * self.vr * self.w1 * self.s1
            )
            / 2
            - self.g * self.m1 * (self.r * cos(self.th) + self.l1 * cos(self.p1))
            + self.m2
            * (
                self.vr**2
                + self.om**2 * self.r**2
                + self.l1**2 * self.w1**2
                + self.l2**2 * self.w2**2
                + 2 * self.l1 * self.l2 * self.w1 * self.w2 * self.c12
                + 2 * self.l1 * self.om * self.w1 * self.r * self.c1
                + 2 * self.l2 * self.om * self.w2 * self.r * self.c2
                - 2 * self.l1 * self.vr * self.w1 * self.s1
                - 2 * self.l2 * self.vr * self.w2 * self.s2
            )
            / 2
            - self.g
            * self.m2
            * (self.r * cos(self.th) + self.l1 * cos(self.p1) + self.l2 * cos(self.p2))
            + self.k * self.stretch**2 / 2
        )


if __name__ == "__main__":
    print(to_modelica(AtwoodPendulum), end="")
