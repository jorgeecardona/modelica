"""Run a model: flatten it, causalize it, simulate it.

* `run`: ``.mo`` files in, a `Result` out, in one call; stop time from the model's
  ``experiment`` annotation when not given.
* `flatten`: source, class or package in, a `FlatModel` out.
* `causalize`: flat model in, a `System` out (states and blocks in evaluation order).
* `simulate`: system in, a `Result` out. It runs on the precompiled engine when installed,
  else on SciPy (`EngineFallbackWarning` says when an installed engine refused a model).
* `sweep`: one result per parameter point.
* `steady`: the state where the model settles.
* `build`: the model as callable Python functions (the SciPy path).

Experimental modules are imported by module path and are not part of this surface:
`modelica.sim.emit`, `kernel`, `compare`, `sensitivity`, `fv`, `fem`, `mesh`, `vm`,
`engine`, `radau`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelica.ir.causalize import CausalizationError as CausalizationError
from modelica.ir.causalize import causalize as causalize
from modelica.ir.source import flatten as flatten
from modelica.ir.system import System as System
from modelica.sim.emit import Trigger as Trigger
from modelica.sim.kernel import build as build
from modelica.sim.matfile import read as read
from modelica.sim.result import Event as Event
from modelica.sim.result import Result as Result
from modelica.sim.steady import Steady as Steady
from modelica.sim.steady import SteadyError as SteadyError
from modelica.sim.steady import steady as steady
from modelica.sim.sweep import grid as grid
from modelica.sim.sweep import sweep as sweep
from modelica.sim.view import View as View

if TYPE_CHECKING:
    from modelica.sim.files import run as run
    from modelica.sim.solve import EngineFallbackWarning as EngineFallbackWarning
    from modelica.sim.solve import SimulationError as SimulationError
    from modelica.sim.solve import simulate as simulate

__all__ = [
    "CausalizationError",
    "EngineFallbackWarning",
    "Event",
    "Result",
    "SimulationError",
    "Steady",
    "SteadyError",
    "System",
    "Trigger",
    "View",
    "build",
    "causalize",
    "flatten",
    "grid",
    "read",
    "run",
    "simulate",
    "steady",
    "sweep",
]

_LAZY = frozenset({"EngineFallbackWarning", "SimulationError", "simulate"})


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a module __getattr__ is untyped by nature
    """Defer the SciPy-backed names so importing this package stays dependency-free."""
    if name == "run":
        from modelica.sim.files import run  # noqa: PLC0415 - it imports the solver

        return run
    if name in _LAZY:
        from modelica.sim import solve  # noqa: PLC0415 - the whole point of the indirection

        return getattr(solve, name)
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
