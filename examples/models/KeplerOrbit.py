"""The Python twin of ``KeplerOrbit.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class KeplerOrbit(Model):
    """The Kepler two-body orbit: an eccentric ellipse conserving energy and angular momentum"""

    mu = parameter(Real, 1, unit="m3/s2", doc="Gravitational parameter G*M of the central body")
    x = Real("Position, x component", unit="m", start=0.4, fixed=True)
    y = Real("Position, y component", unit="m", start=0, fixed=True)
    vx = Real("Velocity, x component", unit="m/s", start=0, fixed=True)
    vy = Real("Velocity, y component", unit="m/s", start=2, fixed=True)

    def equations(self):
        # Newtonian gravity toward the origin: the acceleration is -mu * r_vec / |r|^3, the
        # inverse-square law written per component. r = sqrt(x^2 + y^2) never reaches zero because
        # the angular momentum x*vy - y*vx is nonzero, so the orbit clears the centre at perihelion.
        # A conservative Hamiltonian system: energy 0.5*(vx^2 + vy^2) - mu/r and angular momentum
        # x*vy - y*vx are constant, held here as invariants -- a benchmark for how far a
        # non-symplectic integrator's conserved quantities drift over several orbits.
        yield der(self.x) == self.vx
        yield der(self.y) == self.vy
        yield der(self.vx) == -self.mu * self.x / (self.x**2 + self.y**2) ** 1.5
        yield der(self.vy) == -self.mu * self.y / (self.x**2 + self.y**2) ** 1.5


if __name__ == "__main__":
    print(to_modelica(KeplerOrbit), end="")
