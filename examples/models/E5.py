"""The Python twin of ``E5.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class E5(Model):
    """Chemical pyrolysis -- four species, thirteen decades of time, and badly scaled"""

    A = parameter(Real, 7.89e-10, doc="k1, the slow decay of A1")
    B = parameter(Real, 1.1e7, doc="k3")
    C = parameter(Real, 1.13e3, doc="k4")
    M = parameter(Real, 1e6, doc="k2 / k4, the ratio that carries the stiffness")
    y1 = Real("Concentration of A1", start=1.76e-3, fixed=True)
    y2 = Real("Concentration of A2", start=0, fixed=True)
    y3 = Real("Concentration of A3", start=0, fixed=True)
    y4 = Real("Concentration of A4", start=0, fixed=True)

    def equations(self):
        yield der(self.y1) == -self.A * self.y1 - self.B * self.y1 * self.y3
        yield der(self.y2) == self.A * self.y1 - self.M * self.C * self.y2 * self.y3
        yield der(self.y3) == der(self.y2) - der(self.y4)
        yield der(self.y4) == self.B * self.y1 * self.y3 - self.C * self.y4


if __name__ == "__main__":
    print(to_modelica(E5), end="")
