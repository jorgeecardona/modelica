"""The Python twin of ``Thermostat.mo``."""

from modelica.build import Model, Real, assign, der, discrete, parameter, pre, to_modelica, when


class Thermostat(Model):
    """A heater with hysteresis: the regime is remembered, not re-decided"""

    T_low = parameter(Real, 19, unit="degC", doc="Switch the heater on at or below this")
    T_high = parameter(Real, 21, unit="degC", doc="Switch it off at or above this")
    T_amb = parameter(Real, 5, unit="degC", doc="Outside, which the room leaks toward")
    k = parameter(Real, 0.15, unit="1/s", doc="How fast the room leaks")
    q = parameter(Real, 3, unit="K/s", doc="What the heater adds while it is on")
    T = Real(start=20, fixed=True, unit="degC", doc="Room temperature")
    heating = discrete(Real(start=0, fixed=True, doc="1 while the heater is on, 0 while it is off"))
    cycles = discrete(Real(start=0, fixed=True, doc="How many times the heater has come on"))

    def equations(self):
        yield der(self.T) == self.k * (self.T_amb - self.T) + self.q * self.heating
        yield when(
            self.T <= self.T_low,
            assign(self.heating, 1),
            assign(self.cycles, pre(self.cycles) + 1),
        )
        yield when(self.T >= self.T_high, assign(self.heating, 0))


if __name__ == "__main__":
    print(to_modelica(Thermostat), end="")
