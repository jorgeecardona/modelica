"""SSP 2.0: the protocol of the co-simulation master that steps a system of components.

Like :mod:`modelica.fmi`, the part that comes first is the protocol rather than the
numerics: :mod:`modelica.ssp.master` says what moves a co-simulation master may make and
what each one does, so that the algorithm can be checked before it is written and so that
what gets written has something to be checked *against*.

Reading `.ssp` archives, resolving the connection graph and doing the actual stepping are
not implemented yet.

SSP is a trademark of the Modelica Association; this is an independent implementation.
"""

from __future__ import annotations

from modelica.ssp.master import Component, Phase, Progress, moves, start

__all__ = ["Component", "Phase", "Progress", "moves", "start"]
