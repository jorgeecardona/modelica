"""The Python twin of ``Thermal.mo``."""

from modelica.build import (
    Connector,
    Model,
    Real,
    alias,
    der,
    enumeration,
    flow,
    package,
    parameter,
    to_modelica,
)

Temperature = alias("Temperature", Real, unit="K")
HeatFlowRate = alias("HeatFlowRate", Real, unit="W")
Schedule = enumeration("Schedule", "day", "night", doc="Which of two setpoints is in force")


class HeatPort(Connector):
    """Thermal port: temperature is shared, heat flow sums to zero"""

    T = Temperature("Port temperature")
    Q_flow = flow(HeatFlowRate("Heat flowing into the port"))


class HeatCapacitor(Model):
    """A lumped thermal mass -- introduces one state, T"""

    C = parameter(Real, 1, unit="J/K", doc="Heat capacity")
    T = Temperature("Temperature of the mass")
    port = HeatPort()

    def equations(self):
        yield self.port.T == self.T
        yield self.C * der(self.T) == self.port.Q_flow


class ThermalConductor(Model):
    """Fourier conduction: Q = G*(T_a - T_b)"""

    G = parameter(Real, 1, unit="W/K", doc="Thermal conductance")
    port_a = HeatPort()
    port_b = HeatPort()

    def equations(self):
        yield self.port_a.Q_flow == self.G * (self.port_a.T - self.port_b.T)
        yield self.port_a.Q_flow + self.port_b.Q_flow == 0


class FixedTemperature(Model):
    """Ideal reservoir holding a fixed temperature"""

    T = parameter(Temperature, 293.15)
    port = HeatPort()

    def equations(self):
        yield self.port.T == self.T


PACKAGE = package(
    "Thermal", __name__, "Minimal lumped thermal components, after Modelica.Thermal.HeatTransfer"
)

if __name__ == "__main__":
    print(to_modelica(PACKAGE), end="")
