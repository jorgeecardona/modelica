"""The Python twin of ``SphericalPendulum.mo``."""

from modelica.build import Model, Real, cos, der, parameter, sin, to_modelica


class SphericalPendulum(Model):
    """A pendulum free to swing in 3D: a bob on a rigid rod tracing a path on a sphere"""

    g = parameter(Real, 9.81, unit="m/s2", doc="Gravity")
    rod = parameter(Real, 1, unit="m", doc="Rod length, the radius of the sphere the bob moves on")
    r = Real("Radial coordinate of the bob, held at the rod length", unit="m")
    theta = Real("Polar angle from the downward vertical", unit="rad", start=1.0, fixed=True)
    phi = Real("Azimuth about the vertical", unit="rad", start=0, fixed=True)
    w_theta = Real("Rate of the polar angle", unit="rad/s", start=0, fixed=True)
    w_phi = Real("Rate of the azimuth", unit="rad/s", start=2, fixed=True)

    def equations(self):
        # The bob sits at a fixed radius (the rod is rigid), so its coordinate on the sphere is the
        # pair (theta, phi); r is carried as a constant coordinate so the trajectory reads directly
        # in the spherical view. The two angular equations are the Lagrangian ones: theta feels
        # gravity and the centrifugal push of the azimuthal motion, and phi is driven by the
        # Coriolis term that conserves the vertical angular momentum sin(theta)^2 * w_phi.
        yield self.r == self.rod
        yield der(self.theta) == self.w_theta
        yield der(self.phi) == self.w_phi
        yield der(self.w_theta) == (
            sin(self.theta) * cos(self.theta) * self.w_phi**2 - self.g / self.rod * sin(self.theta)
        )
        yield der(self.w_phi) == -2 * cos(self.theta) / sin(self.theta) * self.w_theta * self.w_phi


if __name__ == "__main__":
    print(to_modelica(SphericalPendulum), end="")
