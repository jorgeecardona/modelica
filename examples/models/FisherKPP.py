"""The Python twin of ``FisherKPP.mo``."""

from modelica.build import (
    Integer,
    Model,
    Real,
    der,
    for_,
    parameter,
    range_,
    ref,
    to_modelica,
)


class FisherKPP(Model):
    """The Fisher-KPP equation: a logistic-growth invasion front by the method of lines"""

    n = parameter(Integer, 16, doc="Interior grid points")
    d = parameter(Real, 1, unit="m2/s", doc="Diffusion coefficient")
    r = parameter(Real, 1, unit="1/s", doc="Logistic growth rate")
    length = parameter(Real, 10, unit="m", doc="Domain length")
    h = parameter(Real, ref("length") / (ref("n") + 1), unit="m", doc="Grid spacing")
    a = parameter(Real, ref("d") / ref("h") ** 2, unit="1/s", doc="Discrete-Laplacian stiffness")
    u_left = parameter(Real, 1, doc="Population density held at the left boundary (invaded)")
    u = Real(
        shape=(ref("n"),), start=0, fixed=True, doc="Population density at each interior point"
    )

    def equations(self):
        # Fisher-KPP: u_t = d * u_xx + r * u * (1 - u), diffusion plus logistic growth. Discretised
        # in space, each interior point gets the second difference a * (u[i-1] - 2 u[i] + u[i+1])
        # plus the local reaction. The left end is held invaded (u[0] = u_left = 1) and the right
        # end empty (u[n+1] = 0, so its term drops); the interior starts empty. An invasion front
        # forms at the left and travels right at the asymptotic speed 2 * sqrt(r * d). The
        # comparison principle keeps every u[i] in [0, 1] for all time.
        i = ref("i")
        yield der(self.u[1]) == self.a * (
            self.u_left - 2 * self.u[1] + self.u[2]
        ) + self.r * self.u[1] * (1 - self.u[1])
        yield for_(
            "i",
            range_(2, self.n - 1),
            der(self.u[i])
            == self.a * (self.u[i - 1] - 2 * self.u[i] + self.u[i + 1])
            + self.r * self.u[i] * (1 - self.u[i]),
        )
        yield der(self.u[self.n]) == self.a * (
            self.u[self.n - 1] - 2 * self.u[self.n]
        ) + self.r * self.u[self.n] * (1 - self.u[self.n])


if __name__ == "__main__":
    print(to_modelica(FisherKPP), end="")
