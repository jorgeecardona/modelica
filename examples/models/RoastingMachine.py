"""The Python twin of ``RoastingMachine.mo``."""

from modelica.build import Expression, Model, Real, der, if_, parameter, time, to_modelica

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


class RoastingMachine(Model):
    """A Loring S15 coffee roaster over one batch -- the gas schedule makes time events"""

    T_amb = parameter(Real, 293.15, unit="K", doc="The roastery, 20 degC")
    C_mach = parameter(Real, 6e5, unit="J/K", doc="Machine hot mass: steel, lining, loop air")
    C_bean = parameter(Real, 3e4, unit="J/K", doc="15 kg of green coffee, drying folded in")
    UA = parameter(Real, 222, unit="W/K", doc="Air-to-bean coupling over the whole bed")
    G_loss = parameter(Real, 164, unit="W/K", doc="Loop to ambient: the bleed and the shell")
    Q_max = parameter(Real, 1.25e5, unit="W", doc="Burner at 100 percent")

    T_mach = Real("Machine temperature", unit="K", start=424.45, fixed=True)
    T_bean = Real("Bean temperature", unit="K", start=293.15, fixed=True)
    bean = Real("Bean temperature in degC -- the number the roaster reads", unit="degC")
    burner = Real("Gas modulation, per cent -- the machine's own signal")
    Q_burner = Real("Burner heat into the loop", unit="W")

    def equations(self):
        yield self.burner == staged(GAS, IDLE)
        yield self.Q_burner == self.burner / 100 * self.Q_max
        yield self.C_mach * der(self.T_mach) == self.Q_burner - self.UA * (
            self.T_mach - self.T_bean
        ) - self.G_loss * (self.T_mach - self.T_amb)
        yield self.C_bean * der(self.T_bean) == self.UA * (self.T_mach - self.T_bean)
        yield self.bean == self.T_bean - 273.15


if __name__ == "__main__":
    print(to_modelica(RoastingMachine), end="")
