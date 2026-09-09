"""The Python twin of ``SIR.mo``."""

from modelica.build import Model, Real, der, parameter, to_modelica


class SIR(Model):
    """The Kermack-McKendrick SIR epidemic: susceptible, infected and recovered fractions"""

    beta = parameter(Real, 0.4, unit="1/s", doc="Infection rate per susceptible-infected contact")
    gamma = parameter(
        Real, 0.1, unit="1/s", doc="Recovery rate; 1/gamma is the mean infectious time"
    )
    s = Real("Susceptible fraction of the population", start=0.99, fixed=True)
    i = Real("Infected fraction of the population", start=0.01, fixed=True)
    r = Real("Recovered (or removed) fraction of the population", start=0, fixed=True)

    def equations(self):
        # The Kermack-McKendrick compartmental model, in population fractions. Susceptibles meet
        # infectives and become infected at rate beta*s*i; infectives recover at rate gamma*i.
        # Nobody is created or destroyed, so s + i + r is conserved at 1 -- the invariant that
        # makes this a compartmental model rather than three unrelated rates. With R0 = beta/gamma
        # above 1 the infected fraction rises to a peak and falls; the peak is exactly where the
        # susceptible fraction crosses gamma/beta = 1/R0, the point at which each case no longer
        # replaces itself. A fraction s_inf is never infected, and the epidemic ends of its own
        # accord with susceptibles still left, not because it ran out of them.
        yield der(self.s) == -self.beta * self.s * self.i
        yield der(self.i) == self.beta * self.s * self.i - self.gamma * self.i
        yield der(self.r) == self.gamma * self.i


if __name__ == "__main__":
    print(to_modelica(SIR), end="")
