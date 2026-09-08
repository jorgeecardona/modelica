"""The Python twin of ``SchwarzschildOrbit.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class SchwarzschildOrbit(Model):
    """A test mass orbiting a Schwarzschild black hole: a precessing relativistic orbit"""

    bigm = parameter(Real, 1, doc="Black-hole mass in geometrized units G = c = 1 (a length)")
    ell = parameter(
        Real, 4, doc="Specific angular momentum, conserved; > sqrt(12)*M for a bound orbit"
    )
    r = Real("Areal radius, in units of the mass", start=20, fixed=True)
    phi = Real("Azimuthal angle in the equatorial plane", start=0, fixed=True)
    vr = Real("Radial rate dr/dtau, per unit proper time", start=0, fixed=True)

    def equations(self):
        # An equatorial timelike geodesic of the Schwarzschild metric, in proper time tau. Angular
        # momentum ell = r^2 * dphi/dtau is a constant of the motion (a Killing vector), imposed by
        # the phi equation. The radial acceleration is -0.5 dV/dr of the effective potential
        # V = (1 - 2M/r)(1 + ell^2/r^2): the Newtonian -M/r^2 and centrifugal +ell^2/r^3, plus the
        # purely relativistic -3*M*ell^2/r^4 that makes the orbit precess instead of closing. The
        # energy E, with E^2 = vr^2 + V(r), is the other constant, held here as an invariant.
        yield der(self.r) == self.vr
        yield der(self.phi) == self.ell / self.r**2
        yield der(self.vr) == (
            -self.bigm / self.r**2
            + self.ell**2 / self.r**3
            - 3 * self.bigm * self.ell**2 / self.r**4
        )


if __name__ == "__main__":
    print(to_modelica(SchwarzschildOrbit), end="")
