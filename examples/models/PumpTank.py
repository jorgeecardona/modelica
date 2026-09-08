"""The Python twin of ``PumpTank.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class PumpTank(Model):
    """A DC motor drives a pump that fills a tank against the head it builds"""

    V = parameter(Real, 12, unit="V", doc="Supply voltage")
    R = parameter(Real, 0.5, unit="Ohm", doc="Armature resistance")
    L = parameter(Real, 0.01, unit="H", doc="Armature inductance")
    k = parameter(Real, 0.1, unit="N.m/A", doc="Motor constant, torque and back-EMF")
    J = parameter(Real, 0.005, unit="kg.m2", doc="Rotor and pump inertia")
    b = parameter(Real, 0.001, unit="N.m.s/rad", doc="Bearing damping")
    kq = parameter(Real, 4e-4, unit="m3/rad", doc="Pump displacement, flow per shaft angle")
    c_load = parameter(Real, 0.675, unit="N.m/m", doc="Head-to-torque coefficient")
    A = parameter(Real, 1, unit="m2", doc="Tank cross-section")
    Rv = parameter(Real, 100, unit="s/m2", doc="Outlet valve resistance, Q_out = h/Rv")
    i = Real("Armature current", unit="A", start=0, fixed=True)
    w = Real("Shaft speed", unit="rad/s", start=0, fixed=True)
    h = Real("Tank head", unit="m", start=0, fixed=True)
    Q = Real("Pump flow into the tank", unit="m3/s")
    tau_pump = Real("Load torque the head imposes on the shaft", unit="N.m")

    def equations(self):
        yield self.Q == self.kq * self.w
        yield self.tau_pump == self.c_load * self.h
        yield self.L * der(self.i) == self.V - self.R * self.i - self.k * self.w
        yield self.J * der(self.w) == self.k * self.i - self.b * self.w - self.tau_pump
        yield self.A * der(self.h) == self.Q - self.h / self.Rv


if __name__ == "__main__":
    print(to_modelica(PumpTank), end="")
