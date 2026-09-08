"""The Python twin of ``CSTR.mo``."""

from modelica.build import Model, Real, der, exp, parameter, to_modelica


class CSTR(Model):
    """An exothermic reaction in a cooled stirred tank: heat and concentration feed back"""

    q = parameter(Real, 100, unit="L/min", doc="Volumetric feed and outflow rate")
    V = parameter(Real, 100, unit="L", doc="Reactor volume")
    Caf = parameter(Real, 1, unit="mol/L", doc="Feed concentration of A")
    Tf = parameter(Real, 350, unit="K", doc="Feed temperature")
    Tc = parameter(Real, 300, unit="K", doc="Coolant temperature")
    k0 = parameter(Real, 7.2e10, unit="1/min", doc="Arrhenius pre-exponential factor")
    EoR = parameter(Real, 8750, unit="K", doc="Activation energy over the gas constant")
    H_rxn = parameter(Real, -5e4, unit="J/mol", doc="Heat of reaction, negative is exothermic")
    rho = parameter(Real, 1000, unit="g/L", doc="Mixture density")
    Cp = parameter(Real, 0.239, unit="J/(g.K)", doc="Mixture heat capacity")
    UA = parameter(Real, 5e4, unit="J/(min.K)", doc="Cooling-jacket heat-transfer coefficient")
    Ca = Real("Concentration of A in the reactor", unit="mol/L", start=1, fixed=True)
    T = Real("Reactor temperature", unit="K", start=350, fixed=True)
    r = Real("Reaction rate", unit="mol/(L.min)")

    def equations(self):
        yield self.r == self.k0 * exp(-self.EoR / self.T) * self.Ca
        yield der(self.Ca) == self.q / self.V * (self.Caf - self.Ca) - self.r
        yield self.rho * self.Cp * der(self.T) == self.rho * self.Cp * self.q / self.V * (
            self.Tf - self.T
        ) - self.H_rxn * self.r + self.UA / self.V * (self.Tc - self.T)


if __name__ == "__main__":
    print(to_modelica(CSTR), end="")
