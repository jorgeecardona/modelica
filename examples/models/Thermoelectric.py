"""The Python twin of ``Thermoelectric.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class Thermoelectric(Model):
    """A thermoelectric generator: a hot junction between a source and a sink drives a Seebeck current into a load, and the current carries heat back the way a Peltier element does"""

    T_src = parameter(Real, 500, unit="K", doc="Hot-source temperature")
    T_c = parameter(Real, 300, unit="K", doc="Cold-sink temperature")
    G_h = parameter(Real, 2, unit="W/K", doc="Conductance from the source to the hot junction")
    K = parameter(Real, 0.5, unit="W/K", doc="Conductance leaking through the module to the sink")
    S = parameter(Real, 0.05, unit="V/K", doc="Seebeck coefficient")
    R_int = parameter(Real, 1, unit="Ohm", doc="Module internal resistance")
    R_load = parameter(Real, 1, unit="Ohm", doc="Load resistance")
    C = parameter(Real, 10, unit="J/K", doc="Heat capacity of the hot junction")
    T_h = Real("Hot-junction temperature", unit="K", start=300, fixed=True)
    V_oc = Real("Open-circuit Seebeck voltage", unit="V")
    i = Real("Current through the load", unit="A")
    P_load = Real("Electrical power delivered to the load", unit="W")
    Q_peltier = Real("Heat the current carries off the hot junction", unit="W")

    def equations(self):
        yield self.V_oc == self.S * (self.T_h - self.T_c)
        yield self.i == self.V_oc / (self.R_int + self.R_load)
        yield self.P_load == self.i**2 * self.R_load
        yield self.Q_peltier == self.S * self.T_h * self.i
        yield self.C * der(self.T_h) == (
            self.G_h * (self.T_src - self.T_h)
            - self.K * (self.T_h - self.T_c)
            - self.Q_peltier
            + 0.5 * self.R_int * self.i**2
        )


if __name__ == "__main__":
    print(to_modelica(Thermoelectric), end="")
