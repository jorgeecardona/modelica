"""The Python twin of ``HIRES.mo``."""

from modelica.build import Model, Real, der, to_modelica


class HIRES(Model):
    """High Irradiance RESponse of photomorphogenesis -- eight reactants, mildly stiff"""

    y1 = Real("P_r, the red-absorbing form of phytochrome", start=1, fixed=True)
    y2 = Real("P_fr, the far-red-absorbing form", start=0, fixed=True)
    y3 = Real("P_rX, P_r bound to receptor X", start=0, fixed=True)
    y4 = Real("P_frX, P_fr bound to receptor X", start=0, fixed=True)
    y5 = Real("P_rX', P_r bound to receptor X'", start=0, fixed=True)
    y6 = Real("P_frX', P_fr bound to receptor X'", start=0, fixed=True)
    y7 = Real("P_frX'E, the enzyme complex", start=0, fixed=True)
    y8 = Real("E, the free enzyme", start=0.0057, fixed=True)

    def equations(self):
        yield der(self.y1) == -1.71 * self.y1 + 0.43 * self.y2 + 8.32 * self.y3 + 0.0007
        yield der(self.y2) == 1.71 * self.y1 - 8.75 * self.y2
        yield der(self.y3) == -10.03 * self.y3 + 0.43 * self.y4 + 0.035 * self.y5
        yield der(self.y4) == 8.32 * self.y2 + 1.71 * self.y3 - 1.12 * self.y4
        yield der(self.y5) == -1.745 * self.y5 + 0.43 * self.y6 + 0.43 * self.y7
        yield (
            der(self.y6)
            == -280 * self.y6 * self.y8
            + 0.69 * self.y4
            + 1.71 * self.y5
            - 0.43 * self.y6
            + 0.69 * self.y7
        )
        yield der(self.y7) == 280 * self.y6 * self.y8 - 1.81 * self.y7
        yield der(self.y8) == -280 * self.y6 * self.y8 + 1.81 * self.y7


if __name__ == "__main__":
    print(to_modelica(HIRES), end="")
