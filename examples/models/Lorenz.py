"""The Python twin of ``Lorenz.mo``."""

from modelica import ir
from modelica.build import Expression, Model, Real, der, parameter, to_modelica


class Lorenz(Model):
    """The Lorenz system: three coupled equations, deterministic and chaotic -- the butterfly"""

    sigma = parameter(Real, 10, doc="Prandtl number")
    rho = parameter(Real, 28, doc="Rayleigh number, past the onset of chaos at 24.74")
    # Written as the division, not its value: the .mo says `8 / 3`, and the two front ends
    # have to build the same tree, not merely the same number.
    beta = parameter(Real, Expression(ir.IntLiteral(8)) / 3, doc="Geometric factor")
    x = Real(start=1, fixed=True)
    y = Real(start=1, fixed=True)
    z = Real(start=1, fixed=True)

    def equations(self):
        yield der(self.x) == self.sigma * (self.y - self.x)
        yield der(self.y) == self.x * (self.rho - self.z) - self.y
        yield der(self.z) == self.x * self.y - self.beta * self.z


if __name__ == "__main__":
    print(to_modelica(Lorenz), end="")
