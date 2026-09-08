"""The Python twin of ``BuckConverter.mo``."""

from modelica.build import Model, Real, assert_, der, parameter, to_modelica


class BuckConverter(Model):
    """An averaged buck converter under PI voltage control: step 24 V down to a 12 V setpoint"""

    Vin = parameter(Real, 24, unit="V", doc="Input voltage")
    L = parameter(Real, 1e-3, unit="H", doc="Filter inductance")
    C = parameter(Real, 1e-3, unit="F", doc="Filter capacitance")
    Rload = parameter(Real, 6, unit="Ohm", doc="Load resistance")
    vref = parameter(Real, 12, unit="V", doc="Output voltage setpoint")
    Kp = parameter(Real, 0.05, unit="1/V", doc="Proportional gain")
    Ki = parameter(Real, 5, unit="1/(V.s)", doc="Integral gain")
    i_ind = Real("Inductor current", unit="A", start=0, fixed=True)
    v_out = Real("Output voltage across the capacitor", unit="V", start=0, fixed=True)
    x = Real("Integral of the voltage error", unit="V.s", start=0, fixed=True)
    e = Real("Voltage error, setpoint minus output", unit="V")
    D = Real("Duty cycle the controller commands", unit="1")

    def equations(self):
        yield self.e == self.vref - self.v_out
        yield self.D == self.Kp * self.e + self.Ki * self.x
        yield der(self.x) == self.e
        yield self.L * der(self.i_ind) == self.D * self.Vin - self.v_out
        yield self.C * der(self.v_out) == self.i_ind - self.v_out / self.Rload
        # The averaged switch only makes sense with a duty cycle in [0, 1]: a buck cannot invert
        # its output or boost above the input. The guard documents the model's valid range and
        # would fire if a parameter change pushed the loop out of it.
        yield assert_(self.D >= 0, "duty cycle went negative: the buck cannot invert its output")
        yield assert_(self.D <= 1, "duty cycle saturated: the buck cannot boost above the input")


if __name__ == "__main__":
    print(to_modelica(BuckConverter), end="")
