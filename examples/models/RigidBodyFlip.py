"""The Python twin of ``RigidBodyFlip.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class RigidBodyFlip(Model):
    """A torque-free rigid body about its principal axes: Euler's equations and the tennis-racket flip"""

    inertia1 = parameter(
        Real, 1, unit="kg.m2", doc="Moment of inertia about principal axis 1 (small)"
    )
    inertia2 = parameter(
        Real, 2, unit="kg.m2", doc="Moment of inertia about principal axis 2 (intermediate)"
    )
    inertia3 = parameter(
        Real, 3, unit="kg.m2", doc="Moment of inertia about principal axis 3 (large)"
    )
    w1 = Real("Angular velocity about axis 1", unit="rad/s", start=0.05, fixed=True)
    w2 = Real("Angular velocity about axis 2", unit="rad/s", start=1, fixed=True)
    w3 = Real("Angular velocity about axis 3", unit="rad/s", start=0, fixed=True)

    def equations(self):
        # Euler's equations for a torque-free rigid body, one per principal axis:
        #   I1 w1' = (I2 - I3) w2 w3, and the two cyclic permutations. Spun almost about the
        # intermediate axis (axis 2, whose inertia lies between the other two), the motion is
        # unstable: w2 holds steady for a while, then flips sign in a quick tumble, over and over --
        # the tennis-racket theorem, or the Dzhanibekov effect a cosmonaut filmed in orbit. It is a
        # conservative, integrable system with two constants of the motion: the rotational kinetic
        # energy 0.5*(I1 w1^2 + I2 w2^2 + I3 w3^2) and the squared angular momentum
        # (I1 w1)^2 + (I2 w2)^2 + (I3 w3)^2, held here as invariants.
        yield self.inertia1 * der(self.w1) == (self.inertia2 - self.inertia3) * self.w2 * self.w3
        yield self.inertia2 * der(self.w2) == (self.inertia3 - self.inertia1) * self.w3 * self.w1
        yield self.inertia3 * der(self.w3) == (self.inertia1 - self.inertia2) * self.w1 * self.w2


if __name__ == "__main__":
    print(to_modelica(RigidBodyFlip), end="")
