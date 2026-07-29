"""The Python twin of ``Robertson.mo``."""

from modelica.build import Model, Real, der, to_modelica


class Robertson(Model):
    """Robertson's autocatalytic reaction -- the standard stiff benchmark"""

    y1 = Real("Concentration of A", start=1, fixed=True)
    y2 = Real("Concentration of B", start=0, fixed=True)
    y3 = Real("Concentration of C", start=0, fixed=True)

    def equations(self):
        yield der(self.y1) == -0.04 * self.y1 + 1e4 * self.y2 * self.y3
        yield der(self.y2) == 0.04 * self.y1 - 1e4 * self.y2 * self.y3 - 3e7 * self.y2**2
        yield der(self.y3) == 3e7 * self.y2**2


if __name__ == "__main__":
    print(to_modelica(Robertson), end="")
