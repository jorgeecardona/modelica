"""The Python twin of ``VanDerPol.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class VanDerPol(Model):
    """The van der Pol oscillator -- a relaxation limit cycle; large mu makes it stiff"""

    mu = parameter(Real, 1000, doc="Nonlinearity, and the stiffness ratio")
    x = Real(start=2, fixed=True)
    y = Real(start=0, fixed=True)

    def equations(self):
        yield der(self.x) == self.y
        yield der(self.y) == self.mu * ((1 - self.x**2) * self.y - self.x)


if __name__ == "__main__":
    print(to_modelica(VanDerPol), end="")
