"""FMI 3.0: ``modelDescription.xml`` and the calling sequence of an FMU.

Two parts are implemented. :func:`modelica.fmi.describe` writes the ``modelDescription.xml``
of a `System`. :mod:`modelica.fmi.modes` holds the **calling sequence**. Reading, running and
exporting a whole FMU are not implemented. An FMU is a state machine before it is a set of
functions, and both directions of this layer (importing someone else's FMU and exporting
one of ours) have to obey the same machine. Writing it down before either exists is the
point, not an accident of ordering: it is what lets the exporter's guard, the importer's
driver and the model checker read one table instead of three descriptions of it.

FMI is a trademark of the Modelica Association; this is an independent implementation.
"""

from __future__ import annotations

from modelica.fmi.description import describe
from modelica.fmi.modes import Interface, Lifecycle, Mode, ModeError, table

__all__ = ["Interface", "Lifecycle", "Mode", "ModeError", "describe", "table"]
