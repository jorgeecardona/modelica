"""The Python twin of ``TwoTank.mo``."""

from modelica.build import Model, Real, der, parameter, sqrt, to_modelica


class TwoTank(Model):
    """Two interacting tanks draining by gravity, their levels coupled through a valve"""

    area1 = parameter(Real, 1.5, unit="m2", doc="Cross-sectional area of tank 1")
    area2 = parameter(Real, 1.0, unit="m2", doc="Cross-sectional area of tank 2")
    q_in = parameter(Real, 0.5, unit="m3/s", doc="Constant inflow into tank 1")
    k1 = parameter(Real, 0.6, unit="m2.5/s", doc="Coupling-valve coefficient, tank 1 to tank 2")
    k2 = parameter(Real, 0.4, unit="m2.5/s", doc="Outlet-valve coefficient, tank 2 to atmosphere")
    h1 = Real("Liquid level in tank 1", unit="m", start=0.5, fixed=True)
    h2 = Real("Liquid level in tank 2", unit="m", start=0.1, fixed=True)
    q1 = Real("Flow from tank 1 into tank 2 through the coupling valve", unit="m3/s")
    q2 = Real("Outflow from tank 2 to atmosphere", unit="m3/s")

    def equations(self):
        # Torricelli through each valve: the flow is proportional to the square root of the head
        # driving it. The coupling flow depends on the level DIFFERENCE, which is what makes the two
        # tanks interact -- tank 2's own level feeds back into the rate at which tank 1 drains.
        yield self.q1 == self.k1 * sqrt(self.h1 - self.h2)
        yield self.q2 == self.k2 * sqrt(self.h2)
        yield self.area1 * der(self.h1) == self.q_in - self.q1
        yield self.area2 * der(self.h2) == self.q1 - self.q2


if __name__ == "__main__":
    print(to_modelica(TwoTank), end="")
