"""The Python twin of ``SyncGenerator.mo``."""

import math

from modelica.build import Model, Real, assert_, der, parameter, sin, to_modelica


class SyncGenerator(Model):
    """A synchronous generator on an infinite bus: the swing equation damps to its angle"""

    M = parameter(Real, 0.02, unit="s2", doc="Inertia, 2H over the synchronous speed")
    D = parameter(Real, 0.1, unit="s", doc="Damping coefficient")
    Pm = parameter(Real, 0.8, unit="1", doc="Mechanical power in, per unit")
    Pmax = parameter(Real, 1.5, unit="1", doc="Peak transferable electrical power, E*V/X")
    delta_max = parameter(Real, math.pi, unit="rad", doc="Synchronism limit, one pole slip")
    delta = Real("Rotor angle relative to the infinite bus", unit="rad", start=0.2, fixed=True)
    w = Real("Speed deviation from synchronous", unit="rad/s", start=0, fixed=True)
    Pe = Real("Electrical power delivered to the bus, per unit", unit="1")

    def equations(self):
        yield self.Pe == self.Pmax * sin(self.delta)
        yield der(self.delta) == self.w
        yield self.M * der(self.w) == self.Pm - self.Pe - self.D * self.w
        # Past a pole slip the machine has lost synchronism and the model no longer describes it.
        # The stable swing here peaks near 0.7 rad, well inside the limit; a larger disturbance or
        # a weaker line would carry delta past it and stop the run with the reason.
        yield assert_(
            self.delta < self.delta_max, "loss of synchronism: rotor angle slipped a pole"
        )


if __name__ == "__main__":
    print(to_modelica(SyncGenerator), end="")
