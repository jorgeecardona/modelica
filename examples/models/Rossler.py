"""The Python twin of ``Rossler.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class Rossler(Model):
    """The Rössler system: a spiral that folds, one nonlinear term, chaotic -- and gentler than Lorenz"""

    a = parameter(Real, 0.2, doc="Spiral growth in the x-y plane")
    b = parameter(Real, 0.2, doc="Lift of the fold")
    c = parameter(Real, 5.7, doc="Where the fold begins; chaotic past about 5.7")
    x = Real(start=1, fixed=True)
    y = Real(start=1, fixed=True)
    z = Real(start=1, fixed=True)

    def equations(self):
        yield der(self.x) == -self.y - self.z
        yield der(self.y) == self.x + self.a * self.y
        yield der(self.z) == self.b + self.z * (self.x - self.c)


if __name__ == "__main__":
    print(to_modelica(Rossler), end="")
