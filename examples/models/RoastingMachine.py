"""The Python twin of ``RoastingMachine.mo``."""

from modelica import ir
from modelica.build import (
    Expression,
    Model,
    Real,
    der,
    if_,
    parameter,
    time,
    to_modelica,
)

GAS = (
    (66.7, 20),
    (93.0, 27),
    (526.9, 85),
    (618.0, 79),
    (688.5, 70),
    (714.6, 61),
    (754.4, 50),
    (780.8, 37),
)
"""The batch's recorded gas trace: hold this level until that second."""

IDLE = 20
"""Where the burner sits once the schedule has run out -- the same idle it charged on."""

def staged(steps: tuple[tuple[float, int], ...], last: int) -> Expression | int:
    """Fold a schedule into one conditional on ``time``.

    Written as a table and folded, rather than typed out as a chain, because the table is
    the thing that came off the machine and the chain is only how Modelica spells it. The
    two are the same model -- ``unparse`` prints this as ``if ... elseif ... else``, which
    is exactly what stands in the ``.mo``.
    """
    if not steps:
        return last
    (until, level), rest = steps[0], steps[1:]
    return if_(time < until, level, staged(rest, last))


def ref(name: str) -> Expression:
    """A bare reference to another declaration in the same model.

    Inside ``equations`` a name is reached as ``self.x``, but a *parameter* that is
    computed from other parameters is bound in the class body, before any instance
    exists. In Modelica that is an ordinary reference -- ``m_dry = m_green*(1 - moisture)``
    -- so this is what spells one from Python.
    """
    return Expression(ir.Ref((name,)))


class RoastingMachine(Model):
    """A Loring S15 coffee roaster over one batch -- gas on a clock, water on a threshold"""

    T_amb = parameter(Real, 293.15, unit="K", doc="The roastery, 20 degC")

    m_green = parameter(Real, 13.8, unit="kg", doc="The charge, weighed")
    moisture = parameter(Real, 0.13, doc="Water in the green coffee, by mass")
    m_dry = parameter(Real, ref("m_green") * (1 - ref("moisture")), unit="kg", doc="Dry matter")
    cp_dry = parameter(Real, 1800, unit="J/(kg.K)", doc="Dry coffee")
    cp_water = parameter(Real, 4186, unit="J/(kg.K)", doc="Water")
    L = parameter(Real, 2.26e6, unit="J/kg", doc="Latent heat of vaporisation")
    k_dry = parameter(Real, 2.02e-5, unit="1/(K.s)", doc="Drying rate above the onset")
    T_dry = parameter(Real, 313.15, unit="K", doc="Below this, drying is too slow to matter")

    C_mach = parameter(Real, 5.2e5, unit="J/K", doc="Machine hot mass: steel, lining, loop air")
    UA = parameter(Real, 140, unit="W/K", doc="Air-to-bean coupling over the whole bed")
    G_loss = parameter(Real, 384, unit="W/K", doc="Loop to ambient: the bleed and the shell")
    Q_max = parameter(Real, 1.32e5, unit="W", doc="Burner at 100 percent")
    tau_probe = parameter(Real, 18.1, unit="s", doc="Bean probe time constant")

    T_mach = Real("Machine temperature", unit="K", start=510.53, fixed=True)
    T_bean = Real("Bean temperature -- the coffee itself", unit="K", start=293.15, fixed=True)
    T_probe = Real("Bean probe temperature", unit="K", start=489.98, fixed=True)
    m_w = Real(
        "Water still in the beans",
        unit="kg",
        start=ref("m_green") * ref("moisture"),
        fixed=True,
    )

    C_bean = Real("Heat capacity of the charge -- falls as it dries", unit="J/K")
    evap = Real("Evaporation rate", unit="kg/s")
    burner = Real("Gas modulation, per cent -- the machine's own signal")
    Q_burner = Real("Burner heat into the loop", unit="W")
    bean = Real("What the coffee is at, in degC", unit="degC")
    probe = Real("What the roaster reads, in degC", unit="degC")
    lag = Real("How far the coffee runs ahead of its own probe", unit="K")
    loss = Real("Weight lost as steam, per cent of the charge")

    def equations(self):
        yield self.burner == staged(GAS, IDLE)
        yield self.Q_burner == self.burner / 100 * self.Q_max
        yield self.C_mach * der(self.T_mach) == self.Q_burner - self.UA * (
            self.T_mach - self.T_bean
        ) - self.G_loss * (self.T_mach - self.T_amb)
        yield self.evap == if_(
            self.T_bean > self.T_dry, self.k_dry * self.m_w * (self.T_bean - self.T_dry), 0
        )
        yield self.C_bean == self.m_dry * self.cp_dry + self.m_w * self.cp_water
        yield self.C_bean * der(self.T_bean) == self.UA * (
            self.T_mach - self.T_bean
        ) - self.L * self.evap
        yield der(self.m_w) == -self.evap
        yield self.tau_probe * der(self.T_probe) == self.T_bean - self.T_probe
        yield self.bean == self.T_bean - 273.15
        yield self.probe == self.T_probe - 273.15
        yield self.lag == self.T_bean - self.T_probe
        yield self.loss == 100 * (self.m_green * self.moisture - self.m_w) / self.m_green


if __name__ == "__main__":
    print(to_modelica(RoastingMachine), end="")
