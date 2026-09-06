"""The Python twin of ``Brusselator.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class Brusselator(Model):
    """Two chemical concentrations that feed each other into a steady oscillation -- a limit cycle"""

    A = parameter(Real, 1, doc="Feed of the first species, held constant")
    B = parameter(Real, 3, doc="Feed of the second; the mixture oscillates once B exceeds 1 + A^2")
    x = Real(start=1, fixed=True, doc="Concentration of the autocatalytic species")
    y = Real(start=1, fixed=True, doc="Concentration of the species it consumes")

    def equations(self):
        yield der(self.x) == self.A + self.x * self.x * self.y - (self.B + 1) * self.x
        yield der(self.y) == self.B * self.x - self.x * self.x * self.y


if __name__ == "__main__":
    print(to_modelica(Brusselator), end="")
