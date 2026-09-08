"""The Python twin of ``CoupledOscillators.mo``."""

from modelica.build import Model, Real, array, der, parameter, to_modelica


class CoupledOscillators(Model):
    """Two masses coupled by a spring, in state-space form der(z) = A*z: the energy beats between them"""

    # The companion matrix of two unit masses, each on a wall spring k = 1, coupled by a spring
    # kc = 0.5. The state is z = [x1, x2, v1, v2], so the top two rows are der(position) = velocity
    # and the bottom two der(velocity) = -(k + kc) * this position + kc * the other. It is a plain
    # linear system der(z) = A*z, the state-space form -- the array product that writes it.
    a = parameter(
        Real,
        array(
            array(0, 0, 1, 0),
            array(0, 0, 0, 1),
            array(-1.5, 0.5, 0, 0),
            array(0.5, -1.5, 0, 0),
        ),
        shape=(4, 4),
        doc="State matrix: unit masses, wall springs 1, coupling 0.5",
    )
    z = Real("State [x1, x2, v1, v2]", shape=(4,), start=array(1, 0, 0, 0), fixed=True)

    def equations(self):
        yield der(self.z) == self.a * self.z


if __name__ == "__main__":
    print(to_modelica(CoupledOscillators), end="")
