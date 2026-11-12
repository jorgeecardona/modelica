"""One call from ``.mo`` files to a `Result`: parse, flatten, causalize and simulate."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelica.ir.causalize import causalize
from modelica.ir.source import flatten, only_model
from modelica.lang import parse
from modelica.sim.solve import simulate

if TYPE_CHECKING:
    from modelica.sim.result import Result


def run(*paths: str | Path, model: str | None = None, stop: float, **options: Any) -> Result:  # noqa: ANN401
    """Simulate a model read from ``paths`` and return its `Result`.

    Give the libraries as well as the model, as `modelica.diagram.of` takes them. ``model``
    names the class; leave it out when the files hold one model. ``options`` go to
    `simulate` unchanged (``start``, ``tolerance``, ``times``, ``backend``, ...).

    Raises:
        AmbiguousModelError: no ``model`` given, and the files hold none or several.
    """
    programs = [parse(Path(path).read_text(encoding="utf-8"), str(path)) for path in paths]
    if model is None:
        model = only_model(programs, ", ".join(str(path) for path in paths))
    return simulate(causalize(flatten(programs, model)), stop=stop, **options)
