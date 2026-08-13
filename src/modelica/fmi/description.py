"""``modelDescription.xml`` from a `System`: the half of an FMU that is not a binary.

An FMU is a zip of two things -- a description of what the model *has*, and a shared library
that computes it. This writes the first. It needs no compiler and no C emitter, which is why
it comes first: the variable table, the causality and variability of every name, the model
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
    root.set("description", f"Generated by pymodelica from {code.name}")
    ET.SubElement(root, "ModelExchange").set("modelIdentifier", name or code.name)

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
