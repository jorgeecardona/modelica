"""``modelDescription.xml`` from a `System`: the half of an FMU that is not a binary.

An FMU is a zip of two things -- a description of what the model *has*, and a shared library
that computes it. This writes the first. It needs no compiler, which is why it comes
first: the variable table, the causality and variability of every name, the model
structure and the unit definitions are all facts the flattener already knows, and getting them
wrong is the failure mode that makes an importer reject an FMU before it has run a step.

**A value reference is a name's identity across the boundary.** Inside this package a variable
is a string; across FMI it is a 32-bit integer, and every later call -- ``fmi3GetFloat64``, the
model structure's dependency lists, an importer's plot -- refers to it by that number. They are
assigned here in `Code.observed` order, which is the order the emitted ``observe`` fills, so a
caller holding one array and one description does not have to build a second mapping.

**What this does not do**, and deliberately: no zip, no binary, no ``fmi3`` entry points. The
register entry `03-08-2026-fmi-export.md` records that the real deliverable is a C translation
unit, that a Python-backed FMU "is not an FMU in the sense anybody else means it", and that the
two design pages disagree about which is intended. None of that is settled by writing the XML,
and writing the XML does not pretend to settle it.

FMI is a trademark of the Modelica Association; this is an independent implementation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

from modelica.ir.evaluate import EvaluationError, evaluate
from modelica.ir.flat import FlatVariable
from modelica.sim.emit import emit

if TYPE_CHECKING:
    from modelica.ir.system import System

__all__ = ["describe"]

VERSION = "3.0"
"""The FMI version this writes. Not a knob: 2.0 has a different schema and a different
`modelica.fmi.modes` table, and supporting both would be two implementations."""

_CAUSALITY = {"none": "local", "input": "input", "output": "output"}
"""Modelica causality to FMI causality.

``none`` becomes ``local`` rather than being left out, and that is the whole of the mapping's
content: FMI has no "unspecified", so an internal variable is one an importer may read and may
not set. A `parameter` is overridden below, because in FMI causality and variability are not
independent -- a parameter is ``causality="parameter"`` whatever its Modelica causality says.
"""

_VARIABILITY = {
    "constant": "constant",
    "parameter": "fixed",
    "discrete": "discrete",
    "continuous": "continuous",
}
"""Modelica variability to FMI variability.

``parameter`` becomes ``fixed`` and not ``tunable``. A tunable parameter may be changed in
Initialization Mode *and* between communication points; a fixed one only before initialisation
ends. Modelica's `parameter` is the second by definition -- it is constant through a run -- and
calling it tunable would promise an importer something the equations do not support.
"""


_BASE = {
    "m": {"m": "1"},
    "kg": {"kg": "1"},
    "s": {"s": "1"},
    "A": {"A": "1"},
    "K": {"K": "1"},
    "rad": {"rad": "1"},
    "degC": {"K": "1", "offset": "273.15"},
    "N": {"kg": "1", "m": "1", "s": "-2"},
    "J": {"kg": "1", "m": "2", "s": "-2"},
    "W": {"kg": "1", "m": "2", "s": "-3"},
    "V": {"kg": "1", "m": "2", "s": "-3", "A": "-1"},
    "Ohm": {"kg": "1", "m": "2", "s": "-3", "A": "-2"},
    "F": {"kg": "-1", "m": "-2", "s": "4", "A": "2"},
    "H": {"kg": "1", "m": "2", "s": "-2", "A": "-2"},
    "1/s": {"s": "-1"},
    "m/s": {"m": "1", "s": "-1"},
    "m/s2": {"m": "1", "s": "-2"},
    "rad/s": {"rad": "1", "s": "-1"},
    "rad/s2": {"rad": "1", "s": "-2"},
    "m2": {"m": "2"},
    "m3/s": {"m": "3", "s": "-1"},
}
"""SI decompositions for the units this package is willing to claim it knows.

**Deliberately short.** The corpus uses 37 distinct unit strings and this covers 21 of them;
the rest -- ``W/K``, ``N.m.s/rad``, ``1/(K.s)`` -- get a ``<Unit>`` with a name and no
``<BaseUnit>``, which is legal FMI and says exactly what is true: this is the string the model
used, and nothing here claims to convert it. Parsing Modelica's unit grammar to derive the
exponents is a real piece of work and guessing them is worse than declining.

``degC`` is why the table exists at all rather than being skipped entirely. It is the one unit
in the corpus that is *not* a product of powers -- an importer converting it to kelvin has to
add 273.15, and the ``offset`` attribute is the only place that can be said. Getting it wrong
is a twenty-degree error in somebody else's plot.
"""


def _units(parent: ET.Element, used: set[str]) -> None:
    """One ``<Unit>`` per distinct unit string the model mentions.

    Every unit a variable names has to be declared, or an importer reading ``unit="degC"``
    has a dangling reference. The ones `_BASE` knows carry their decomposition; the rest carry
    their name alone.

    Declared from the *model's* variables rather than from the observed ones, so a unit that
    only appears on something alias elimination removed is still defined. Extra definitions
    are legal; a dangling reference is not.
    """
    if not used:
        return
    definitions = ET.SubElement(parent, "UnitDefinitions")
    for name in sorted(used):
        unit = ET.SubElement(definitions, "Unit")
        unit.set("name", name)
        if (base := _BASE.get(name)) is not None:
            ET.SubElement(unit, "BaseUnit", base)


def _number(node: object) -> float | None:
    """A start attribute as a number, or `None` if it needs the model to be solved.

    A start value may be written in terms of parameters -- ``T(start = T_amb)`` -- and FMI
    wants a literal. Rather than evaluate the parameter graph here, this returns `None` and
    the attribute is left out, which is legal: an importer then takes the type's default and
    the value it gets from initialisation, which is the one this package computes anyway.
    """
    try:
        value = evaluate(node, {})  # pyright: ignore[reportArgumentType]
    except (EvaluationError, AttributeError, TypeError):
        return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _variable(
    parent: ET.Element, reference: int, variable: FlatVariable, *, derivative: int | None
) -> None:
    """One ``<Float64>`` entry of the model variables."""
    element = ET.SubElement(parent, variable.base_type.replace("Real", "Float64"))
    element.set("name", variable.name)
    element.set("valueReference", str(reference))
    element.set(
        "causality", "parameter" if variable.is_constant else _CAUSALITY[variable.causality]
    )
    element.set("variability", _VARIABILITY[variable.variability])
    if variable.comment:
        element.set("description", variable.comment)
    if variable.unit:
        element.set("unit", variable.unit)
    if (start := _number(variable.start)) is not None or (
        variable.is_constant and (start := _number(variable.binding)) is not None
    ):
        element.set("start", repr(start))
    if derivative is not None:
        # FMI states a continuous state by pointing at *its derivative's* value reference,
        # which is the opposite direction from `Mode.derivative` and is the single place this
        # mapping is easy to write backwards.
        element.set("derivative", str(derivative))


def describe(system: System, *, name: str | None = None) -> str:
    """Write ``modelDescription.xml`` for ``system`` and return it as text.

    A `System` and not a `Kernel`, because everything here is a fact about the model: the
    units, the comments and the causality live on `FlatVariable` and never reach the emitted
    code. What the emitter *is* asked for is the observed order, so that a value reference
    means the same slot an ``observe`` call fills.

    ``name`` overrides the model name. The instantiation token is deliberately absent: FMI 3.0
    requires the binary and the description to agree on it, so inventing one here would be a
    promise this file cannot keep on its own.
    """
    code = emit(system, observed_jacobian=False)
    derivative = dict(system.modes[0].derivative)
    references = {variable: slot for slot, variable in enumerate(code.observed)}
    by_name = {variable.name: variable for variable in system.variables}

    root = ET.Element("fmiModelDescription")
    root.set("fmiVersion", VERSION)
    root.set("modelName", name or code.name)
    root.set("description", f"Generated by modelica from {code.name}")
    ET.SubElement(root, "ModelExchange").set("modelIdentifier", name or code.name)

    # Before `ModelVariables`, because FMI 3.0's schema fixes the order of these children --
    # the interface elements, then `UnitDefinitions`, then `TypeDefinitions`, then
    # `ModelVariables`, then `ModelStructure`. A document with the right content in the wrong
    # order is one a validating importer rejects, and it is not visible by reading the output.
    _units(root, {variable.unit for variable in by_name.values() if variable.unit})

    variables = ET.SubElement(root, "ModelVariables")
    for variable_name, reference in references.items():
        # A name the model observes but never declared is a derivative causalization invented
        # and then eliminated as an alias -- `VanDerPol`'s `der_x` is one, because `der(x)` is
        # `y`. It is still a variable of the *FMU*: `observe` fills a slot for it and a state
        # points at it, so leaving it out would give an importer a table with a hole in it
        # exactly where the model structure says to look.
        variable = by_name.get(variable_name) or FlatVariable(name=variable_name)
        _variable(
            variables,
            reference,
            variable,
            derivative=references.get(derivative.get(variable_name, "")),
        )

    _units(
        root, {by_name[name].unit for name in references if name in by_name and by_name[name].unit}
    )

    structure = ET.SubElement(root, "ModelStructure")
    for state in code.states:
        slot = references.get(derivative.get(state, ""))
        if slot is not None:
            ET.SubElement(structure, "ContinuousStateDerivative").set("valueReference", str(slot))
    for output in code.observed:
        variable = by_name.get(output)
        if variable is not None and variable.causality == "output":
            ET.SubElement(structure, "Output").set("valueReference", str(references[output]))

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
