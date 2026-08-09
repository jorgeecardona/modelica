"""The Python twin of ``ProbeRC.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Electrical

from modelica.build import Model, to_modelica


class ProbeRC(Model):
    """The probe with its default element: a resistor, so an RC charging curve"""

    probe = Electrical.Probe()


if __name__ == "__main__":
    print(to_modelica(ProbeRC), end="")
