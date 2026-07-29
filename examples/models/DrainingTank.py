"""The Python twin of ``DrainingTank.mo``."""

from modelica.build import Model, Real, der, if_, parameter, sqrt, to_modelica


class DrainingTank(Model):
    """Torricelli outflow through an orifice -- the tank empties, then stays empty"""

    A = parameter(Real, 1, unit="m2", doc="Tank cross-section")
    a = parameter(Real, 0.02, unit="m2", doc="Orifice area")
    g = parameter(Real, 9.81, unit="m/s2", doc="Gravity")
    level = Real("Water level", unit="m", start=1, fixed=True)
    outflow = Real("Volumetric outflow", unit="m3/s")

    def equations(self):
        yield self.outflow == if_(self.level > 0, self.a * sqrt(2 * self.g * self.level), 0)
        yield self.A * der(self.level) == -self.outflow


if __name__ == "__main__":
    print(to_modelica(DrainingTank), end="")
