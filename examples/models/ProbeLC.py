"""The Python twin of ``ProbeLC.mo``."""

import sys
from pathlib import Path

# The examples are not an installed package, so point Python at the library next door.
sys.path.append(str(Path(__file__).resolve().parent.parent / "library"))

import Electrical

from modelica.build import Model, mod, redeclare, to_modelica


class ProbeLC(Model):
    """The same probe with an inductor redeclared into the slot: an LC oscillation"""

    probe = Electrical.Probe(
        element=redeclare(Electrical.Inductor, L=0.5, i=mod(start=0, fixed=True))
    )


if __name__ == "__main__":
    print(to_modelica(ProbeLC), end="")
