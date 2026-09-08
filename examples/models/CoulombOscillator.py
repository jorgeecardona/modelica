"""The Python twin of ``CoulombOscillator.mo``."""

from modelica.build import Model, Real, call, der, parameter, to_modelica


class CoulombOscillator(Model):
    """A mass on a spring damped by dry friction, its amplitude falling linearly not exponentially"""

    m = parameter(Real, 1, unit="kg", doc="Mass")
    k = parameter(Real, 1, unit="N/m", doc="Spring stiffness")
    fc = parameter(Real, 0.1, unit="N", doc="Coulomb friction force magnitude")
    v_eps = parameter(Real, 0.05, unit="m/s", doc="Velocity scale that smooths the friction sign")
    x = Real("Position", unit="m", start=1.5, fixed=True)
    v = Real("Velocity", unit="m/s", start=0, fixed=True)

    def equations(self):
        # Dry (Coulomb) friction opposes the motion with a constant magnitude, so it is fc*sign(v).
        # sign has no derivative at v = 0, which the Jacobian needs, so the sign is smoothed to
        # tanh(v/v_eps): the same force away from zero, differentiable through it. The result is the
        # signature of Coulomb damping -- the amplitude falls by 2*fc/k each half-swing, a straight
        # envelope, where a viscous damper would fall off exponentially.
        yield der(self.x) == self.v
        yield self.m * der(self.v) == -self.k * self.x - self.fc * call("tanh", self.v / self.v_eps)


if __name__ == "__main__":
    print(to_modelica(CoulombOscillator), end="")
