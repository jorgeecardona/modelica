"""The Python twin of ``Blocks.mo``."""

from modelica.build import (
    Block,
    Connector,
    Real,
    der,
    if_,
    input_,
    output,
    package,
    parameter,
    to_modelica,
)


class InPort(Connector):
    """A signal arriving at a block"""

    signal = input_(Real("Value carried on the wire"))


class OutPort(Connector):
    """A signal leaving a block"""

    signal = output(Real("Value carried on the wire"))


class SISO(Block, partial=True):
    """One signal in, one signal out"""

    u = InPort("Input signal")
    y = OutPort("Output signal")


class Constant(Block):
    """Holds one value for the whole run"""

    k = parameter(Real, 1, doc="The value")
    y = OutPort("Output signal")

    def equations(self):
        yield self.y.signal == self.k


class Gain(SISO):
    """Multiplies the input by a constant"""

    k = parameter(Real, 1, doc="Gain")

    def equations(self):
        yield self.y.signal == self.k * self.u.signal


class Feedback(Block):
    """Subtracts the measurement from the setpoint"""

    u1 = InPort("Setpoint")
    u2 = InPort("Measurement")
    y = OutPort("Error")

    def equations(self):
        yield self.y.signal == self.u1.signal - self.u2.signal


class PI(SISO):
    """Proportional and integral controller -- introduces one state, x"""

    k = parameter(Real, 1, doc="Proportional gain")
    T = parameter(Real, 1, unit="s", doc="Integral time constant")
    x = Real("The integrator state, in units of the input")

    def equations(self):
        yield der(self.x) == self.u.signal / self.T
        yield self.y.signal == self.k * (self.x + self.u.signal)


class FirstOrder(SISO):
    """A plant that lags its input -- introduces one state, y.signal"""

    k = parameter(Real, 1, doc="Steady-state gain")
    T = parameter(Real, 1, unit="s", doc="Time constant")

    def equations(self):
        yield self.T * der(self.y.signal) == self.k * self.u.signal - self.y.signal


class Limiter(SISO):
    """Holds the signal between two values"""

    minimum = parameter(Real, -1, doc="Lower limit")
    maximum = parameter(Real, 1, doc="Upper limit")

    def equations(self):
        yield self.y.signal == if_(
            self.u.signal < self.minimum,
            self.minimum,
            if_(self.u.signal > self.maximum, self.maximum, self.u.signal),
        )


PACKAGE = package("Blocks", __name__, "Minimal causal signal blocks, after Modelica.Blocks")

if __name__ == "__main__":
    print(to_modelica(PACKAGE), end="")
