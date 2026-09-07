"""The Python twin of ``ServoDrive.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class ServoDrive(Model):
    """A DC motor under PI speed control: the loop drives the shaft to a speed setpoint"""

    w_ref = parameter(Real, 100, unit="rad/s", doc="Speed setpoint")
    Kp = parameter(Real, 0.1, unit="V.s/rad", doc="Proportional gain")
    Ki = parameter(Real, 1, unit="V/rad", doc="Integral gain")
    R = parameter(Real, 0.5, unit="Ohm", doc="Armature resistance")
    L = parameter(Real, 0.05, unit="H", doc="Armature inductance")
    km = parameter(Real, 0.1, unit="N.m/A", doc="Motor constant, torque and back-EMF")
    J = parameter(Real, 0.001, unit="kg.m2", doc="Rotor and load inertia")
    b = parameter(Real, 0.0002, unit="N.m.s/rad", doc="Load damping")
    i = Real("Armature current", unit="A", start=0, fixed=True)
    w = Real("Shaft speed", unit="rad/s", start=0, fixed=True)
    x = Real("Integral of the speed error", unit="rad", start=0, fixed=True)
    e = Real("Speed error, setpoint minus measurement", unit="rad/s")
    v = Real("Armature voltage the controller commands", unit="V")
    tau = Real("Electromagnetic torque", unit="N.m")

    def equations(self):
        yield self.e == self.w_ref - self.w
        yield self.v == self.Kp * self.e + self.Ki * self.x
        yield self.tau == self.km * self.i
        yield der(self.x) == self.e
        yield self.L * der(self.i) == self.v - self.R * self.i - self.km * self.w
        yield self.J * der(self.w) == self.tau - self.b * self.w


if __name__ == "__main__":
    print(to_modelica(ServoDrive), end="")
