"""One call from ``.mo`` files to a `Result`: parse, flatten, causalize and simulate."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelica.ir.causalize import causalize
from modelica.ir.evaluate import evaluate
from modelica.ir.source import flatten, only_model
from modelica.lang import parse
from modelica.sim.solve import SimulationError, simulate
from modelica.sim.sweep import vector

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from modelica.ir import nodes as ir
    from modelica.sim.result import Result

#: ``experiment`` annotation entry, and the `simulate` argument it gives a default for.
_EXPERIMENT = {"StartTime": "start", "StopTime": "stop", "Tolerance": "tolerance"}


def run(
    *paths: str | Path,
    model: str | None = None,
    stop: float | None = None,
    parameters: Mapping[str, float] | None = None,
    **options: Any,  # noqa: ANN401 - passed to `simulate` unchanged
) -> Result:
    """Simulate a model read from ``paths`` and return its `Result`.

    Give the libraries as well as the model, as `modelica.diagram.of` takes them. ``model``
    names the class; leave it out when the files hold one model. ``options`` go to
    `simulate` unchanged (``start``, ``tolerance``, ``times``, ``backend``, ...).

    ``stop``, ``start`` and ``tolerance`` not given are taken from the model's
    ``annotation(experiment(StopTime, StartTime, Tolerance))``. ``parameters`` overrides
    parameters by name (``{"r.R": 2.0}``); a parameter written in terms of another is
    recomputed from the new value.

    Raises:
        AmbiguousModelError: no ``model`` given, and the files hold none or several.
        SimulationError: no ``stop``, and the model has no ``experiment(StopTime)``.
        EmitError: a name in ``parameters`` is not a parameter; the message lists close ones.
    """
    programs = [parse(Path(path).read_text(encoding="utf-8"), str(path)) for path in paths]
    if model is None:
        model = only_model(programs, ", ".join(str(path) for path in paths))
    given = {**options, **({} if stop is None else {"stop": stop})}
    settings: dict[str, Any] = {**experiment(programs, model), **given}
    if "stop" not in settings:
        message = f"{model}: give stop=..., the model has no experiment(StopTime=...) annotation"
        raise SimulationError(message)
    system = causalize(flatten(programs, model))
    if parameters is not None:
        settings["parameters"] = vector(system, parameters)
    return simulate(system, **settings)


def experiment(programs: Sequence[ir.Program], model: str) -> dict[str, float]:
    """The `simulate` arguments ``model``'s ``experiment`` annotation sets, by argument name."""
    for program in programs:
        cls = program.find(model)
        if cls is None:
            continue
        entries = cls.annotation.entries.get("experiment")
        if entries is None:
            return {}
        out: dict[str, float] = {}
        for key, argument in _EXPERIMENT.items():
            entry = entries.entries.get(key)
            if entry is not None and entry.value is not None:
                out[argument] = float(evaluate(entry.value, {}))  # pyright: ignore[reportArgumentType]
        return out
    return {}
