"""Whatever you have, collected into one program.

:mod:`modelica.ir.flatten` insists on an :class:`~modelica.ir.nodes.Program`, because a
pass that resolves names should not also be in the business of guessing what it was
handed. This module does that guessing once, at the edge, so that every caller above it
-- tests, the CLI, FMU export, :mod:`modelica.diagram` -- can pass what it happens to
have: a parsed program, a single class, a :mod:`modelica.build` package, or a list mixing
all three.

It sits under :mod:`modelica.ir` and not under :mod:`modelica.sim` for one reason: it
must cost nothing to import. :mod:`modelica.sim` pulls in NumPy, and a caller that only
reads the structure of a model -- an editor, a diagram, a linter -- must not pay for a
numerical stack it does not use.

The list form is what makes libraries work. A model and the libraries it needs are
separate files and separate programs; naming resolution needs them side by side in one
program, and concatenating their classes is all "side by side" means.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelica import build
from modelica.ir import nodes as ir
from modelica.ir.flatten import flatten as _flatten

if TYPE_CHECKING:
    from collections.abc import Sequence

    from modelica.ir.flat import FlatModel

__all__ = ["AmbiguousModelError", "Source", "flatten", "only_model", "to_program"]

type Source = ir.Program | ir.Class | build.Package | build.TypeRef
"""Anything that names classes: an IR value, or a :mod:`modelica.build` class or package."""


def _classes(source: Source) -> tuple[ir.Class, ...]:
    if isinstance(source, ir.Program):
        return source.classes
    if isinstance(source, ir.Class):
        return (source,)
    return (build.to_ir(source),)


def to_program(source: Source | Sequence[Source]) -> ir.Program:
    """Collect ``source`` -- one value or several -- into a single program."""
    one = (ir.Program, ir.Class, build.Package, build.Type, type)
    sources: Sequence[Source] = [source] if isinstance(source, one) else source
    classes: list[ir.Class] = []
    for item in sources:
        classes.extend(_classes(item))
    return ir.Program(tuple(classes))


def flatten(source: Source | Sequence[Source], model: str) -> FlatModel:
    """Flatten the class named ``model``, found among ``source``.

    ``model`` is a dotted name, so a model nested in a package is reachable as
    ``"Electrical.Examples.Charging"``.
    """
    return _flatten(to_program(source), model)


class AmbiguousModelError(Exception):
    """The files hold no model, or more than one, and no model name was given."""


def only_model(programs: Sequence[ir.Program], where: str) -> str:
    """The name of the one model or block in ``programs``; ``where`` names them in the error.

    Raises:
        AmbiguousModelError: there is none, or more than one. The message lists the choices.
    """
    names = [
        cls.name
        for program in programs
        for cls in program.classes
        if cls.restriction in ("model", "block")
    ]
    if len(names) == 1:
        return names[0]
    if not names:
        raise AmbiguousModelError(f"no model is defined in {where}")
    raise AmbiguousModelError(f"{where} defines {len(names)} models; name one of {names}")
