"""The Python twin of ``PhaseChange.mo``."""

from modelica.build import Model, Real, and_, der, if_, parameter, time, to_modelica


class PhaseChange(Model):
    """A steel bar heated, held and quenched: four phases, and the latent heat that puts them back into the temperature"""

    T_amb = parameter(Real, 300, unit="K", doc="The shop floor")
    m = parameter(Real, 1.0, unit="kg", doc="The bar")
    cp_body = parameter(
        Real, 490, unit="J/(kg.K)", doc="Specific heat of ferrite, pearlite and martensite"
    )
    cp_aust = parameter(
        Real,
        620,
        unit="J/(kg.K)",
        doc="Specific heat of austenite -- the close-packed phase holds more",
    )
    Q_furnace = parameter(Real, 1400, unit="W", doc="The furnace, while the bar is in it")
    t_out = parameter(Real, 520, unit="s", doc="When the bar comes out of the furnace")
    t_quench = parameter(Real, 700, unit="s", doc="When it goes into the oil")
    UA_air = parameter(Real, 1.1, unit="W/K", doc="Loss to the shop floor")
    UA_oil = parameter(Real, 34, unit="W/K", doc="Loss into the quenchant")
    Ac1 = parameter(Real, 996, unit="K", doc="Where austenite starts to form on heating")
    Ps = parameter(Real, 923, unit="K", doc="Pearlite start on cooling")
    Ms = parameter(Real, 596, unit="K", doc="Martensite start")
    n_aust = parameter(Real, 2, doc="Avrami exponent for austenitisation")
    k_aust = parameter(Real, 0.021, unit="1/s", doc="JMAK rate constant for austenitisation")
    n_pear = parameter(Real, 2, doc="Avrami exponent for the pearlite reaction")
    k_pear = parameter(Real, 0.0032, unit="1/s", doc="JMAK rate constant for pearlite")
    alpha = parameter(Real, 0.011, unit="1/K", doc="Koistinen-Marburger constant")
    L_aust = parameter(Real, 84000.0, unit="J/kg", doc="Absorbed forming austenite")
    L_mart = parameter(
        Real,
        21000.0,
        unit="J/kg",
        doc=(
            "Released forming martensite. See the note in the .toml: alpha * L_mart must"
            " stay under cp_body or the energy balance inverts"
        ),
    )
    HV_body = parameter(Real, 170, unit="HV", doc="Vickers of the as-delivered ferrite-pearlite")
    HV_aust = parameter(Real, 250, unit="HV", doc="Vickers of austenite")
    HV_pear = parameter(Real, 220, unit="HV", doc="Vickers of the pearlite formed on cooling")
    HV_mart = parameter(Real, 720, unit="HV", doc="Vickers of martensite")
    T = Real("Bar temperature", unit="K", start=300, fixed=True)
    x_body = Real("Ferrite and pearlite, as delivered", start=1.0, fixed=True)
    x_aust = Real("Austenite", start=0.0, fixed=True)
    x_pear = Real("Pearlite formed on cooling", start=0.0, fixed=True)
    x_mart = Real("Martensite", start=0.0, fixed=True)
    tau_aust = Real("Time spent above Ac1", unit="s", start=0.0, fixed=True)
    tau_pear = Real("Time spent in the pearlite window", unit="s", start=0.0, fixed=True)
    r_aust = Real("Austenitisation rate", unit="1/s")
    r_pear = Real("Pearlite rate", unit="1/s")
    r_mart = Real("Martensite rate", unit="1/s")
    cp = Real("Specific heat of the mixture", unit="J/(kg.K)")
    Q = Real("Net heat into the bar", unit="W")
    total = Real("The four fractions. Physics says one; the solver is asked to agree")
    hardness = Real("What the bar comes out at", unit="HV")
    celsius = Real("The same temperature, for reading", unit="degC")

    def equations(self):
        window = and_(self.T < self.Ps, self.T > self.Ms)
        yield self.Q == (
            if_(time < self.t_out, self.Q_furnace, 0)
            - if_(time < self.t_quench, self.UA_air, self.UA_oil) * (self.T - self.T_amb)
        )
        yield (
            self.cp
            == self.cp_body * (self.x_body + self.x_pear + self.x_mart) + self.cp_aust * self.x_aust
        )
        yield der(self.tau_aust) == if_(self.T > self.Ac1, 1, 0)
        yield der(self.tau_pear) == if_(window, 1, 0)
        yield self.r_aust == (
            self.n_aust
            * self.k_aust**self.n_aust
            * self.tau_aust ** (self.n_aust - 1)
            * self.x_body
        )
        yield self.r_pear == if_(
            window,
            self.n_pear
            * self.k_pear**self.n_pear
            * self.tau_pear ** (self.n_pear - 1)
            * self.x_aust,
            0,
        )
        yield self.r_mart == if_(self.T < self.Ms, -self.alpha * self.x_aust * der(self.T), 0)
        yield der(self.x_body) == -self.r_aust
        yield der(self.x_aust) == self.r_aust - self.r_pear - self.r_mart
        yield der(self.x_pear) == self.r_pear
        yield der(self.x_mart) == self.r_mart
        yield self.m * self.cp * der(self.T) == (
            self.Q - self.m * self.L_aust * self.r_aust + self.m * self.L_mart * self.r_mart
        )
        yield self.total == self.x_body + self.x_aust + self.x_pear + self.x_mart
        yield self.hardness == (
            self.HV_body * self.x_body
            + self.HV_aust * self.x_aust
            + self.HV_pear * self.x_pear
            + self.HV_mart * self.x_mart
        )
        yield self.celsius == self.T - 273.15


if __name__ == "__main__":
    print(to_modelica(PhaseChange), end="")
