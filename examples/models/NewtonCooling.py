"""The Python twin of ``NewtonCooling.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class NewtonCooling(Model):
    """A hot object cooling toward ambient -- the smallest useful model"""

    T_amb = parameter(Real, 25, unit="degC", doc="Ambient temperature")
    k = parameter(Real, 0.5, unit="1/s", doc="Cooling coefficient")
    T = Real("Object temperature", unit="degC", start=90, fixed=True)

    def equations(self):
        yield der(self.T) == -self.k * (self.T - self.T_amb)


if __name__ == "__main__":
    print(to_modelica(NewtonCooling), end="")
