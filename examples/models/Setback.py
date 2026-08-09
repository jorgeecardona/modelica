"""The Python twin of ``Setback.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Thermal

from modelica.build import Model, Real, der, equal, if_, parameter, time, to_modelica


class Setback(Model):
    """A room on a night setback: an enumeration says which setpoint is in force"""

    k = parameter(Real, 0.5, unit="1/h", doc="How fast the room follows its setpoint")
    T_day = parameter(Real, 21, unit="degC", doc="Occupied setpoint")
    T_night = parameter(Real, 16, unit="degC", doc="Setback setpoint")
    switch_at = parameter(Real, 4.0, unit="h", doc="When the setback starts")
    schedule = Thermal.Schedule("Which setpoint the clock has selected")
    T_set = Real("The setpoint in force", unit="degC")
    T = Real("Room temperature", unit="degC", start=21, fixed=True)

    def equations(self):
        day, night = Thermal.Schedule.day, Thermal.Schedule.night
        yield self.schedule == if_(time < self.switch_at, day, night)
        yield self.T_set == if_(equal(self.schedule, day), self.T_day, self.T_night)
        yield der(self.T) == self.k * (self.T_set - self.T)


if __name__ == "__main__":
    print(to_modelica(Setback), end="")
