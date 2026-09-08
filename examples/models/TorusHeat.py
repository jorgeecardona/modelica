"""The Python twin of ``TorusHeat.mo``."""

from modelica.build import (
    Integer,
    Model,
    Real,
    call,
    constant,
    cos,
    der,
    for_,
    parameter,
    range_,
    ref,
    to_modelica,
)


class TorusHeat(Model):
    """The heat equation on a flat torus: a 2-D periodic grid by the method of lines"""

    n = parameter(Integer, 8, doc="Cells along each side of the square grid")
    a = parameter(Real, 1, unit="1/s", doc="Diffusivity over the square of the cell width")
    pi = constant(Real, 3.141592653589793, doc="The ratio of a circle's circumference to diameter")
    T = Real(shape=(ref("n"), ref("n")), doc="Cell temperatures over the torus")

    def initial_equations(self):
        # A single grid Fourier mode, cos(2*pi*(i-1)/n)*cos(2*pi*(j-1)/n): an exact eigenvector of
        # the discrete periodic Laplacian, so the whole field decays uniformly and stays this shape.
        i, j = ref("i"), ref("j")
        yield for_(
            "i",
            range_(1, self.n),
            for_(
                "j",
                range_(1, self.n),
                self.T[i, j]
                == cos(2 * self.pi * (i - 1) / self.n) * cos(2 * self.pi * (j - 1) / self.n),
            ),
        )

    def equations(self):
        # The 5-point Laplacian stencil with periodic wrap-around: on a torus the last cell's
        # neighbour is the first. The wrap is `mod` on the index -- i-1 becomes mod(i-2,n)+1 and i+1
        # becomes mod(i,n)+1 -- evaluated to a fixed neighbour for each cell when the model flattens.
        i, j = ref("i"), ref("j")
        im, ip = call("mod", i - 2, self.n) + 1, call("mod", i, self.n) + 1
        jm, jp = call("mod", j - 2, self.n) + 1, call("mod", j, self.n) + 1
        yield for_(
            "i",
            range_(1, self.n),
            for_(
                "j",
                range_(1, self.n),
                der(self.T[i, j])
                == self.a
                * (
                    self.T[im, j] + self.T[ip, j] + self.T[i, jm] + self.T[i, jp] - 4 * self.T[i, j]
                ),
            ),
        )


if __name__ == "__main__":
    print(to_modelica(TorusHeat), end="")
