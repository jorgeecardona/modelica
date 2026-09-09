"""The Python twin of ``MixingRing.mo``."""

from modelica.build import (
    Integer,
    Model,
    Real,
    array,
    assert_,
    call,
    der,
    for_,
    parameter,
    range_,
    ref,
    to_modelica,
)


class MixingRing(Model):
    """A closed ring of stirred tanks pumped one way: a tracer circulates and mixes to uniform, total mass conserved"""

    n = parameter(Integer, 4, doc="Number of tanks around the ring")
    k = parameter(Real, 1, unit="1/s", doc="Pumped flow rate over the tank volume")
    slack = parameter(
        Real, 1e-6, unit="mol/m3", doc="Round-off the positivity guard allows below 0"
    )
    c = Real(
        shape=(ref("n"),),
        unit="mol/m3",
        start=array(4, 0, 0, 0),
        fixed=True,
        doc="Tracer concentration in each tank",
    )
    total = Real(unit="mol/m3", doc="Total tracer over the ring, conserved")

    def equations(self):
        # Each tank is fed from its predecessor around the loop and drains to its successor at the
        # same rate, so der(c[i]) = k * (c[i-1] - c[i]); the predecessor wraps periodically, i-1
        # becoming mod(i-2, n)+1. Every row sums to zero, so the total is conserved for all time.
        i = ref("i")
        pred = call("mod", i - 2, self.n) + 1
        yield for_("i", range_(1, self.n), der(self.c[i]) == self.k * (self.c[pred] - self.c[i]))
        # The conserved total, as a reduction over the whole ring: constant at its starting value.
        yield self.total == call("sum", self.c)
        # A well-posed mixing model never drives a concentration negative -- the system is positive
        # (off-diagonal inflows are non-negative) -- and this guards that with a reduction over the
        # ring, the minimum concentration, which stays at or above zero.
        yield assert_(call("min", self.c) > -self.slack, "a tank concentration went negative")


if __name__ == "__main__":
    print(to_modelica(MixingRing), end="")
