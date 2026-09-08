"""The Python twin of ``HenonHeiles.mo``."""

from modelica.build import Model, Real, der, to_modelica


class HenonHeiles(Model):
    """The Henon-Heiles system: a 2-DOF Hamiltonian, regular at low energy and mixing to chaos"""

    x = Real("Position, first coordinate", start=0, fixed=True)
    y = Real("Position, second coordinate", start=0, fixed=True)
    px = Real("Momentum conjugate to x (unit mass, so a velocity)", start=0.45, fixed=True)
    py = Real("Momentum conjugate to y (unit mass, so a velocity)", start=0.15, fixed=True)

    def equations(self):
        # Hamilton's equations for H = 0.5*(px^2 + py^2) + 0.5*(x^2 + y^2) + x^2*y - y^3/3 at unit
        # mass, the potential Henon and Heiles used for a star in an axisymmetric galaxy. The cubic
        # term couples the two coordinates: below the escape energy 1/6 the motion is bounded inside
        # a triangular well, quasi-periodic when the energy is low and mixing into chaos as it
        # rises. The energy is conserved for all time, held here as an invariant.
        yield der(self.x) == self.px
        yield der(self.y) == self.py
        yield der(self.px) == -self.x - 2 * self.x * self.y
        yield der(self.py) == -self.y - self.x**2 + self.y**2


if __name__ == "__main__":
    print(to_modelica(HenonHeiles), end="")
