"""FMI 3.0 — reading and writing Functional Mock-up Units.

Nothing is implemented here yet except the part that has to come first: the **calling
sequence**, in :mod:`modelica.fmi.modes`. An FMU is a state machine before it is a set of
functions, and both directions of this layer -- importing someone else's FMU and exporting
one of ours -- have to obey the same machine. Writing it down before either exists is the
point, not an accident of ordering: it is what lets the exporter's guard, the importer's
driver and the model checker read one table instead of three descriptions of it.

FMI is a trademark of the Modelica Association; this is an independent implementation.
"""

from __future__ import annotations

from modelica.fmi.modes import Interface, Lifecycle, Mode, ModeError, table

__all__ = ["Interface", "Lifecycle", "Mode", "ModeError", "table"]
