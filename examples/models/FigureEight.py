"""The Python twin of ``FigureEight.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class FigureEight(Model):
    """The figure-eight three-body choreography: three equal masses on one closed curve"""

    gm = parameter(Real, 1, unit="m3/s2", doc="G times the common mass, with the mass taken as 1")
    x1 = Real("Body 1 position, x", unit="m", start=0.97000436, fixed=True)
    y1 = Real("Body 1 position, y", unit="m", start=-0.24308753, fixed=True)
    x2 = Real("Body 2 position, x", unit="m", start=-0.97000436, fixed=True)
    y2 = Real("Body 2 position, y", unit="m", start=0.24308753, fixed=True)
    x3 = Real("Body 3 position, x", unit="m", start=0, fixed=True)
    y3 = Real("Body 3 position, y", unit="m", start=0, fixed=True)
    vx1 = Real("Body 1 velocity, x", unit="m/s", start=0.46620369, fixed=True)
    vy1 = Real("Body 1 velocity, y", unit="m/s", start=0.43236573, fixed=True)
    vx2 = Real("Body 2 velocity, x", unit="m/s", start=0.46620369, fixed=True)
    vy2 = Real("Body 2 velocity, y", unit="m/s", start=0.43236573, fixed=True)
    vx3 = Real("Body 3 velocity, x", unit="m/s", start=-0.93240737, fixed=True)
    vy3 = Real("Body 3 velocity, y", unit="m/s", start=-0.86473146, fixed=True)

    def equations(self):
        # Newtonian gravity among three equal masses, the inverse-square law written per
        # component: each body is pulled toward the other two, the acceleration
        # gm * sum_j (r_j - r_i) / |r_j - r_i|^3. r12, r13, r23 are the squared pairwise
        # separations, so a denominator is r^1.5 = |r_j - r_i|^3. Started from the
        # Chenciner-Montgomery initial condition, the three bodies chase one another along a
        # single figure-eight curve, a periodic choreography. It is a conservative Hamiltonian
        # system: the energy 0.5*sum|v|^2 - gm*(1/|r12| + 1/|r13| + 1/|r23|), the total linear
        # momentum, the angular momentum (which is zero here) and the centre of mass are all
        # constant, held as invariants -- how far they drift over one period is the test.
        r12 = (self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2
        r13 = (self.x1 - self.x3) ** 2 + (self.y1 - self.y3) ** 2
        r23 = (self.x2 - self.x3) ** 2 + (self.y2 - self.y3) ** 2
        yield der(self.x1) == self.vx1
        yield der(self.y1) == self.vy1
        yield der(self.x2) == self.vx2
        yield der(self.y2) == self.vy2
        yield der(self.x3) == self.vx3
        yield der(self.y3) == self.vy3
        yield der(self.vx1) == self.gm * (
            (self.x2 - self.x1) / r12**1.5 + (self.x3 - self.x1) / r13**1.5
        )
        yield der(self.vy1) == self.gm * (
            (self.y2 - self.y1) / r12**1.5 + (self.y3 - self.y1) / r13**1.5
        )
        yield der(self.vx2) == self.gm * (
            (self.x1 - self.x2) / r12**1.5 + (self.x3 - self.x2) / r23**1.5
        )
        yield der(self.vy2) == self.gm * (
            (self.y1 - self.y2) / r12**1.5 + (self.y3 - self.y2) / r23**1.5
        )
        yield der(self.vx3) == self.gm * (
            (self.x1 - self.x3) / r13**1.5 + (self.x2 - self.x3) / r23**1.5
        )
        yield der(self.vy3) == self.gm * (
            (self.y1 - self.y3) / r13**1.5 + (self.y2 - self.y3) / r23**1.5
        )


if __name__ == "__main__":
    print(to_modelica(FigureEight), end="")
