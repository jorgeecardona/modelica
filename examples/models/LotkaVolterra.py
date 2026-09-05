"""The Python twin of ``LotkaVolterra.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class LotkaVolterra(Model):
    """Prey and predators: two populations that chase each other around a closed orbit, forever"""

    alpha = parameter(Real, 1.1, doc="Prey birth rate, per unit time")
    beta = parameter(Real, 0.4, doc="How fast predators eat prey, per predator")
    delta = parameter(Real, 0.1, doc="How much of what is eaten becomes predators")
    gamma = parameter(Real, 0.4, doc="Predator death rate, per unit time")
    x = Real(start=10, fixed=True, doc="Prey")
    y = Real(start=10, fixed=True, doc="Predators")

    def equations(self):
        yield der(self.x) == self.alpha * self.x - self.beta * self.x * self.y
        yield der(self.y) == self.delta * self.x * self.y - self.gamma * self.y


if __name__ == "__main__":
    print(to_modelica(LotkaVolterra), end="")
