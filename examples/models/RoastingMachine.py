"""The Python twin of ``RoastingMachine.mo``."""

from modelica.build import Model, Real, der, if_, parameter, time, to_modelica


class RoastingMachine(Model):
    """A drum coffee roaster over one batch -- the gas schedule makes time events"""

    T_amb = parameter(Real, 293.15, unit="K", doc="Ambient temperature")
    T_boil = parameter(Real, 373.15, unit="K", doc="Boiling point of water")
    C_drum = parameter(Real, 40000, unit="J/K", doc="Drum and internals")
    C_bean = parameter(Real, 20400, unit="J/K", doc="12 kg of green beans")
    G_loss = parameter(Real, 60, unit="W/K", doc="Drum to ambient, mostly the exhaust")
    G_bean = parameter(Real, 200, unit="W/K", doc="Drum air to the bean pile")
    L = parameter(Real, 2.26e6, unit="J/kg", doc="Latent heat of vaporisation")
    k_dry = parameter(Real, 1e-4, unit="1/(K.s)", doc="Drying rate above boiling")
    t_cut = parameter(Real, 480, unit="s", doc="Gas cut to low, at first crack")
    t_off = parameter(Real, 640, unit="s", doc="Flame out, the batch is dropped")
    Q_dry = parameter(Real, 28000, unit="W", doc="Burner through drying and ramp")
    Q_dev = parameter(Real, 10000, unit="W", doc="Burner through development")

    T_drum = Real("Drum temperature", unit="K", start=473.15, fixed=True)
    T_bean = Real("Bean temperature", unit="K", start=293.15, fixed=True)
    m_w = Real("Water left in the beans", unit="kg", start=1.44, fixed=True)
    Q_burner = Real("Burner heat into the drum", unit="W")
    evap = Real("Evaporation rate", unit="kg/s")

    def equations(self):
        yield self.Q_burner == if_(
            time < self.t_cut, self.Q_dry, if_(time < self.t_off, self.Q_dev, 0)
        )
        yield self.C_drum * der(self.T_drum) == self.Q_burner - self.G_loss * (
            self.T_drum - self.T_amb
        ) - self.G_bean * (self.T_drum - self.T_bean)
        yield self.evap == if_(
            self.T_bean > self.T_boil, self.k_dry * self.m_w * (self.T_bean - self.T_boil), 0
        )
        yield self.C_bean * der(self.T_bean) == self.G_bean * (
            self.T_drum - self.T_bean
        ) - self.L * self.evap
        yield der(self.m_w) == -self.evap


if __name__ == "__main__":
    print(to_modelica(RoastingMachine), end="")
