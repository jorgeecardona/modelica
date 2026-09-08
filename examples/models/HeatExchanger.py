"""The Python twin of ``HeatExchanger.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class HeatExchanger(Model):
    """Two fluid streams trading heat through a conducting wall, a lumped exchanger with wall inertia"""

    c_hot = parameter(Real, 4000, unit="J/K", doc="Heat capacity of the hot control volume")
    c_wall = parameter(Real, 8000, unit="J/K", doc="Heat capacity of the separating wall")
    c_cold = parameter(Real, 4000, unit="J/K", doc="Heat capacity of the cold control volume")
    q_hot = parameter(Real, 200, unit="W/K", doc="Hot-stream flow heat-capacity rate, mdot*cp")
    q_cold = parameter(Real, 150, unit="W/K", doc="Cold-stream flow heat-capacity rate, mdot*cp")
    ua_hot = parameter(Real, 500, unit="W/K", doc="Conductance between the hot stream and the wall")
    ua_cold = parameter(
        Real, 500, unit="W/K", doc="Conductance between the wall and the cold stream"
    )
    th_in = parameter(Real, 360, unit="K", doc="Hot-stream inlet temperature")
    tc_in = parameter(Real, 300, unit="K", doc="Cold-stream inlet temperature")
    t_hot = Real("Hot-stream outlet temperature", unit="K", start=300, fixed=True)
    t_wall = Real("Wall temperature", unit="K", start=300, fixed=True)
    t_cold = Real("Cold-stream outlet temperature", unit="K", start=300, fixed=True)

    def equations(self):
        # Each control volume balances the enthalpy its flow carries in and out against the heat
        # conducted across the wall. The wall couples the two streams: heat the hot stream gives up
        # crosses into the wall, and the wall passes it on to the cold stream, so the two states are
        # linked only through t_wall -- and the wall's own capacity delays that transfer.
        yield self.c_hot * der(self.t_hot) == self.q_hot * (
            self.th_in - self.t_hot
        ) - self.ua_hot * (self.t_hot - self.t_wall)
        yield self.c_wall * der(self.t_wall) == self.ua_hot * (
            self.t_hot - self.t_wall
        ) - self.ua_cold * (self.t_wall - self.t_cold)
        yield self.c_cold * der(self.t_cold) == self.q_cold * (
            self.tc_in - self.t_cold
        ) + self.ua_cold * (self.t_wall - self.t_cold)


if __name__ == "__main__":
    print(to_modelica(HeatExchanger), end="")
