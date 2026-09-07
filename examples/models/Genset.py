"""The Python twin of ``Genset.mo``."""

from modelica.build import Model, Real, der, if_, parameter, time, to_modelica


class Genset(Model):
    """A governed generator set: a PI speed governor holds the shaft at its setpoint, and when an electrical load is switched on the speed dips and the governor brings it back"""

    w_ref = parameter(Real, 100, unit="rad/s", doc="Speed setpoint")
    k = parameter(Real, 0.5, unit="N.m/A", doc="Generator constant, torque and back-EMF")
    R_gen = parameter(Real, 0.5, unit="Ohm", doc="Generator internal resistance")
    R_open = parameter(Real, 10000, unit="Ohm", doc="Load resistance before the switch")
    R_closed = parameter(Real, 2, unit="Ohm", doc="Load resistance after the switch")
    t_load = parameter(Real, 1, unit="s", doc="Time the load is switched on")
    b = parameter(Real, 0.01, unit="N.m.s/rad", doc="Bearing friction")
    J = parameter(Real, 0.05, unit="kg.m2", doc="Shaft inertia")
    Kp = parameter(Real, 1, unit="N.m.s/rad", doc="Governor proportional gain")
    Ki = parameter(Real, 10, unit="N.m/rad", doc="Governor integral gain")
    w = Real("Shaft speed", unit="rad/s", start=0, fixed=True)
    x = Real("Integral of the speed error", unit="rad", start=0, fixed=True)
    R_load = Real("Load resistance", unit="Ohm")
    v_gen = Real("Generator voltage", unit="V")
    i = Real("Load current", unit="A")
    tau_gen = Real("Braking torque from the generator", unit="N.m")
    tau_eng = Real("Engine torque the governor commands", unit="N.m")

    def equations(self):
        yield self.R_load == if_(time < self.t_load, self.R_open, self.R_closed)
        yield self.v_gen == self.k * self.w
        yield self.i == self.v_gen / (self.R_gen + self.R_load)
        yield self.tau_gen == self.k * self.i
        yield self.tau_eng == self.Kp * (self.w_ref - self.w) + self.Ki * self.x
        yield der(self.x) == self.w_ref - self.w
        yield self.J * der(self.w) == self.tau_eng - self.b * self.w - self.tau_gen


if __name__ == "__main__":
    print(to_modelica(Genset), end="")
