"""The Python twin of ``WaveString.mo``."""

from modelica.build import (
    Integer,
    Model,
    Real,
    constant,
    der,
    for_,
    parameter,
    range_,
    ref,
    sin,
    to_modelica,
)


class WaveString(Model):
    """The wave equation on a string with fixed ends: a standing wave by the method of lines"""

    n = parameter(Integer, 8, doc="Interior grid points")
    c = parameter(Real, 1, unit="m/s", doc="Wave speed along the string")
    h = parameter(Real, 1 / (ref("n") + 1), unit="m", doc="Grid spacing; the string has length 1")
    k = parameter(
        Real, ref("c") ** 2 / ref("h") ** 2, unit="1/s2", doc="Discrete-Laplacian stiffness"
    )
    pi = constant(Real, 3.141592653589793, doc="The ratio of a circle's circumference to diameter")
    u = Real(shape=(ref("n"),), unit="m", doc="Displacement of each interior point")
    v = Real(
        shape=(ref("n"),), unit="m/s", start=0, fixed=True, doc="Velocity of each interior point"
    )

    def initial_equations(self):
        # The fundamental mode sin(pi*i/(n+1)): an exact eigenvector of the discrete Laplacian with
        # fixed ends, so the string keeps its shape and swings at one frequency, a standing wave.
        i = ref("i")
        yield for_("i", range_(1, self.n), self.u[i] == sin(self.pi * i / (self.n + 1)))

    def equations(self):
        # der(u) = v as one array equation, then u'' = k * (second difference) at each point. The
        # ends are held at zero (u[0] = u[n+1] = 0), so the boundary rows drop the missing neighbour.
        i = ref("i")
        yield der(self.u) == self.v
        yield der(self.v[1]) == self.k * (-2 * self.u[1] + self.u[2])
        yield for_(
            "i",
            range_(2, self.n - 1),
            der(self.v[i]) == self.k * (self.u[i - 1] - 2 * self.u[i] + self.u[i + 1]),
        )
        yield der(self.v[self.n]) == self.k * (self.u[self.n - 1] - 2 * self.u[self.n])


if __name__ == "__main__":
    print(to_modelica(WaveString), end="")
