"""The flat model as something you can actually run.

This is the layer users call. Below it, :mod:`modelica.ir` holds the passes -- flattening,
differentiation, causalization -- and each is usable on its own; here they are composed
into the two verbs that matter:

* :func:`flatten` -- source, class or package in, :class:`~modelica.ir.flat.FlatModel` out;
* :func:`causalize` -- flat model in, :class:`~modelica.ir.system.System` out: states and
  the blocks to evaluate, in the order to evaluate them;
* :func:`emit` and :func:`build` -- system in, generated source and callables out;
* :func:`diverges` -- two tiers of one model in, the first block they disagree on out;
* :func:`simulate` -- system in, a :class:`~modelica.sim.result.Result` out;
* :mod:`modelica.sim.radau` -- the same, by a Radau IIA loop compiled together with the
  model rather than handed to SciPy: much faster, and no events;
* :func:`sweep` -- system and a list of parameter values in, one result per value out;
* :func:`steady` -- system in, the state where it settles out.

Numerics live behind the ``sim`` extra. Importing this module does not pull in SciPy;
only :func:`simulate` does, which is why it is imported lazily below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelica.ir.causalize import CausalizationError as CausalizationError
from modelica.ir.causalize import causalize as causalize
from modelica.ir.source import Source as Source
from modelica.ir.source import flatten as flatten
from modelica.ir.source import to_program as to_program
from modelica.ir.system import Block as Block
from modelica.ir.system import System as System
from modelica.sim.compare import Divergence as Divergence
from modelica.sim.compare import diverges as diverges
from modelica.sim.emit import Code as Code
from modelica.sim.emit import EmitError as EmitError
from modelica.sim.emit import Trigger as Trigger
from modelica.sim.emit import emit as emit
from modelica.sim.emit import emit_expr as emit_expr
from modelica.sim.kernel import Kernel as Kernel
from modelica.sim.kernel import build as build
from modelica.sim.matfile import read as read
from modelica.sim.result import Event as Event
from modelica.sim.result import Result as Result
from modelica.sim.sensitivity import (
    Identifiability,
    Sensitivity,
    identifiability,
    sensitivity,
)
from modelica.sim.steady import Steady as Steady
from modelica.sim.steady import SteadyError as SteadyError
from modelica.sim.steady import steady as steady
from modelica.sim.sweep import grid as grid
from modelica.sim.sweep import sweep as sweep
from modelica.sim.sweep import vector as vector
from modelica.sim.view import View as View

if TYPE_CHECKING:
    from modelica.sim.solve import SimulationError as SimulationError
    from modelica.sim.solve import simulate as simulate

__all__ = [
    "Block",
    "CausalizationError",
    "Code",
    "Divergence",
    "EmitError",
    "Event",
    "Identifiability",
    "Kernel",
    "Result",
    "Sensitivity",
    "SimulationError",
    "Source",
    "Steady",
    "SteadyError",
    "System",
    "Trigger",
    "View",
    "build",
    "causalize",
    "diverges",
    "emit",
    "emit_expr",
    "flatten",
    "grid",
    "identifiability",
    "read",
    "sensitivity",
    "simulate",
    "steady",
    "sweep",
    "to_program",
    "vector",
]

_LAZY = frozenset({"SimulationError", "simulate"})


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a module __getattr__ is untyped by nature
    """Defer the SciPy-backed names so importing this package stays dependency-free."""
    if name in _LAZY:
        from modelica.sim import solve  # noqa: PLC0415 - the whole point of the indirection

        return getattr(solve, name)
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
