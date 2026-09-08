"""The Python twin of ``SaturatingServo.mo``."""

from modelica.build import Model, Real, der, maximum, minimum, parameter, to_modelica


class SaturatingServo(Model):
    """A position servo whose actuator voltage saturates: a step drives it to the limit first"""

    xref = parameter(Real, 1, unit="m", doc="Position setpoint")
    Kp = parameter(Real, 50, unit="V/m", doc="Proportional gain on position error")
    Kd = parameter(Real, 5, unit="V.s/m", doc="Derivative gain on speed")
    Vmax = parameter(Real, 12, unit="V", doc="Actuator voltage limit")
    R = parameter(Real, 1, unit="Ohm", doc="Coil resistance")
    L = parameter(Real, 0.05, unit="H", doc="Coil inductance")
    km = parameter(Real, 0.5, unit="N/A", doc="Motor constant, force and back-EMF")
    J = parameter(Real, 0.02, unit="kg", doc="Moving mass")
    b = parameter(Real, 0.1, unit="N.s/m", doc="Viscous damping")
    x = Real("Position", unit="m", start=0, fixed=True)
    w = Real("Speed", unit="m/s", start=0, fixed=True)
    i = Real("Coil current", unit="A", start=0, fixed=True)
    e = Real("Position error, setpoint minus position", unit="m")
    u = Real("Unsaturated voltage the PD law asks for", unit="V")
    v = Real("Actuator voltage after the plus/minus Vmax limit", unit="V")

    def equations(self):
        yield self.e == self.xref - self.x
        yield self.u == self.Kp * self.e - self.Kd * self.w
        # The actuator can only swing +/-Vmax; a large step asks for far more, so it saturates.
        # The min/max is what makes the response nonlinear and needs the Jacobian rules for them.
        yield self.v == maximum(-self.Vmax, minimum(self.Vmax, self.u))
        yield der(self.x) == self.w
        yield self.J * der(self.w) == self.km * self.i - self.b * self.w
        yield self.L * der(self.i) == self.v - self.R * self.i - self.km * self.w


if __name__ == "__main__":
    print(to_modelica(SaturatingServo), end="")
