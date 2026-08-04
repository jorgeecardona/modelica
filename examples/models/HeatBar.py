"""The Python twin of ``HeatBar.mo``."""

from modelica.build import Integer, Model, Real, der, for_, parameter, range_, ref, to_modelica


class HeatBar(Model):
    """A rod discretised into n cells: the method of lines, written by hand"""

    n = parameter(Integer, 8, doc="Number of cells")
    a = parameter(Real, 0.4, unit="1/s", doc="Diffusivity over the square of the cell width")
    T_left = parameter(Real, 100, unit="degC", doc="The hot end, held")
    T_0 = parameter(Real, 20, unit="degC", doc="What the rod starts at, everywhere")
    T = Real(
        shape=ref("n"),
        start=ref("T_0"),
        fixed=True,
        unit="degC",
        doc="Cell temperatures, left to right",
    )

    def equations(self):
        # `i` is the loop index and not a component, so it is written with `ref` rather than
        # reached through `self`. Writing the loop out with a Python `for` would give the
        # same *flat* model and a different `.mo`, and the corpus holds the two front ends
        # to the same text as well as to the same value.
        i = ref("i")
        yield der(self.T[1]) == self.a * (self.T_left - 2 * self.T[1] + self.T[2])
        yield for_(
            "i",
            range_(2, self.n - 1),
            der(self.T[i]) == self.a * (self.T[i - 1] - 2 * self.T[i] + self.T[i + 1]),
        )
        yield der(self.T[self.n]) == self.a * (self.T[self.n - 1] - self.T[self.n])


if __name__ == "__main__":
    print(to_modelica(HeatBar), end="")
