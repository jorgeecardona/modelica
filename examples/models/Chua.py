"""The Python twin of ``Chua.mo``."""

from modelica import ir
from modelica.build import Expression, Model, Real, call, der, parameter, to_modelica


class Chua(Model):
    """Chua's circuit: two capacitors, an inductor and a kinked resistor -- the double scroll"""

    alpha = parameter(Real, 15.6, doc="Ratio of the two capacitances, scaled")
    beta = parameter(Real, 28, doc="The inductor, scaled")
    # The slopes are fractions, written as the division and not its value: the .mo says
    # `-(8 / 7)`, and the two front ends have to build the same tree, not merely the same number.
    m0 = parameter(
        Real, -(Expression(ir.IntLiteral(8)) / 7), doc="Slope of the resistor's middle segment"
    )
    m1 = parameter(Real, -(Expression(ir.IntLiteral(5)) / 7), doc="Slope of its outer segments")
    x = Real(start=0.7, fixed=True, doc="Voltage across the first capacitor")
    y = Real(start=0, fixed=True, doc="Voltage across the second capacitor")
    z = Real(start=0, fixed=True, doc="Current through the inductor")
    h = Real(doc="The resistor's characteristic: three straight lines joined at x = -1 and x = 1")

    def equations(self):
        yield self.h == self.m1 * self.x + 0.5 * (self.m0 - self.m1) * (
            call("abs", self.x + 1) - call("abs", self.x - 1)
        )
        yield der(self.x) == self.alpha * (self.y - self.x - self.h)
        yield der(self.y) == self.x - self.y + self.z
        yield der(self.z) == -self.beta * self.y


if __name__ == "__main__":
    print(to_modelica(Chua), end="")
