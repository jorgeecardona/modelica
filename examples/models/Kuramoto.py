"""The Python twin of ``Kuramoto.mo``."""

from modelica.build import (
    Integer,
    Model,
    Real,
    array,
    cos,
    der,
    parameter,
    ref,
    sin,
    sqrt,
    to_modelica,
)


class Kuramoto(Model):
    """The Kuramoto model: four coupled phase oscillators pulling into synchrony"""

    n = parameter(Integer, 4, doc="Number of oscillators")
    k = parameter(
        Real, 1.5, unit="rad/s", doc="Coupling strength (above the synchronising threshold)"
    )
    omega = parameter(
        Real,
        array(-0.6, -0.2, 0.2, 0.6),
        shape=(ref("n"),),
        unit="rad/s",
        doc="Natural frequency of each oscillator (symmetric, mean zero)",
    )
    theta = Real(
        "Phase of each oscillator",
        shape=(ref("n"),),
        unit="rad",
        start=array(0, 1.5, 3, 4.5),
        fixed=True,
    )
    r = Real("Order parameter: 0 fully incoherent, 1 fully synchronised")

    def equations(self):
        # Each oscillator runs at its own natural frequency and is pulled toward every other by a
        # sine of their phase difference, der(theta_i) = omega_i + k/n * sum_j sin(theta_j -
        # theta_i). With the coupling k above the critical value the phases lock: they stop drifting
        # apart and hold a fixed pattern, rotating together at the mean frequency (zero here, so the
        # locked cluster stands still). The order parameter r = |mean(exp(i*theta))|, written out
        # as sqrt(mean(cos)^2 + mean(sin)^2), measures how tight the cluster is: it climbs from near
        # zero (phases spread around the circle) to near one (phases bunched) as they synchronise.
        for i in (1, 2, 3, 4):
            coupling = (
                sin(self.theta[1] - self.theta[i])
                + sin(self.theta[2] - self.theta[i])
                + sin(self.theta[3] - self.theta[i])
                + sin(self.theta[4] - self.theta[i])
            )
            yield der(self.theta[i]) == self.omega[i] + self.k / self.n * coupling
        c = (
            cos(self.theta[1]) + cos(self.theta[2]) + cos(self.theta[3]) + cos(self.theta[4])
        ) / self.n
        s = (
            sin(self.theta[1]) + sin(self.theta[2]) + sin(self.theta[3]) + sin(self.theta[4])
        ) / self.n
        yield self.r == sqrt(c**2 + s**2)


if __name__ == "__main__":
    print(to_modelica(Kuramoto), end="")
