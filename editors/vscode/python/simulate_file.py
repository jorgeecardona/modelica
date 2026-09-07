#!/usr/bin/env python
"""Simulate one .mo file and print its trajectory as JSON, for the VS Code Scope.

The VS Code extension calls this on the .mo you have open. It flattens the file with
`modelica.sim`, integrates it, and prints one JSON record the webview draws -- the same
reference tier as the notebook widget, no per-model compilation. A sibling `library/` folder
(and any `--library` paths) are loaded first so a model that uses components resolves.

On any failure it still prints JSON, with an `error` field, so the extension can show the
message in the panel instead of a silent nothing.
"""

from __future__ import annotations

# A CLI: it prints its JSON result to stdout for the extension to read.
# ruff: noqa: T201
import argparse
import json
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from modelica import ir
from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten
from modelica.sim.emit import settings
from modelica.sim.kernel import build
from modelica.sim.solve import (
    _Run,  # pyright: ignore[reportPrivateUsage]
    simulate,
)
from modelica.sim.sweep import vector

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.kernel import Kernel
    from modelica.sim.result import Result


def _douglas_peucker(
    x: NDArray[np.float64], y: NDArray[np.float64], eps: float
) -> NDArray[np.bool_]:
    """The points a polyline must keep to stay within ``eps`` of the original (Ramer-Douglas-Peucker).

    Iterative rather than recursive: a stiff run has tens of thousands of steps and the
    recursion would be as deep as the spike is sharp.
    """
    n = len(x)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        dx, dy = x[b] - x[a], y[b] - y[a]
        length = float(np.hypot(dx, dy)) or 1.0
        d = np.abs(dy * (x[a + 1 : b] - x[a]) - dx * (y[a + 1 : b] - y[a])) / length
        m = int(np.argmax(d))
        if d[m] > eps:
            idx = a + 1 + m
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return keep


_LOG_SPAN = 1e4
"""A horizon this many times the first positive instant is sampled, and read, on a log axis."""

_GRID = 4000
"""How many points the dense solution is read on for the main run. Generous, so a zoom still
has samples inside its window; the dense-output evaluation is cheap next to the integration."""


def _sample(
    time: NDArray[np.float64], columns: list[NDArray[np.float64]], limit: int = 4000
) -> list[int]:
    """Which instants to send the panel: evenly dense, and keeping every sharp feature.

    Two parts. A uniform grid along the axis the run is drawn on gives a consistent density, so
    a curve is a curve everywhere; and Douglas-Peucker over each series adds the sharp features
    an even grid would step over -- the apex of a spike between two grid points, the knee of a
    corner. The union is what is sent, with the event instants (a repeated time) always in, so a
    discontinuity draws as a vertical step.

    The uniform part is what a plain stride could not do on the solver's own steps: those cluster
    at a jump, so a stride kept an impulse's peak on some cycles and missed it on others (van der
    Pol's der(x), a 0.01 s jump to 1350, drew at a wandering height). Sampling by *time* rather
    than by step index makes the density even; Douglas-Peucker then puts the peak back. Earlier
    this function was Douglas-Peucker alone, which keeps only what a straight line cannot stand in
    for -- minimal, so a many-swing run like Lorenz drew as sawtooth across the gentle stretches.
    """
    n = len(time)
    if n <= limit:
        return list(range(n))
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    breaks = np.flatnonzero(np.diff(time) <= 0.0)
    keep[breaks] = True
    keep[np.minimum(breaks + 1, n - 1)] = True
    # The axis the run is read on. A stiff model over a long horizon -- E5 runs to 1e13 s and
    # finishes its kinetics by 1e3 -- has all its motion in the first decades, one point on a
    # linear axis, so when the horizon spans four decades or more the coordinate is log time,
    # which is the axis such a run is drawn on and where its samples should be even. The span is
    # measured against the 1st percentile of the positive instants, not the smallest: a solver
    # routinely takes a tiny first step (Lorenz's is ~1e-4 on a 40 s run), and comparing to that
    # made an ordinary run look like it spanned decades, so it was sampled dense-early and starved
    # late -- exactly the thin resolution a zoom into the late part then showed.
    positive = time[time > 0.0]
    floor = float(np.percentile(positive, 1)) if positive.size else 1.0
    logaxis = bool(positive.size) and float(time[-1]) / floor >= _LOG_SPAN
    coord = np.log10(np.maximum(time, floor)) if logaxis else time.astype(float)
    lo, hi = float(coord[0]), float(coord[-1])
    tx = (coord - lo) / ((hi - lo) or 1.0)
    # A uniform grid along that axis is most of the budget: consistent density, so a curve reads
    # as a curve everywhere and not only where a corner forced a point. Douglas-Peucker alone --
    # which keeps only what a straight line cannot stand in for -- left long chords across the
    # gentle stretches of an oscillation, and Lorenz's forty seconds of swings drew as sawtooth.
    base = max(2, int(limit * 0.75))
    keep[np.clip(np.searchsorted(coord, np.linspace(lo, hi, base)), 0, n - 1)] = True
    # The rest of the budget keeps the sharp features an even grid steps over: a spike between two
    # grid points (van der Pol's der(x), a 0.01 s jump to 1350), the knee of a corner. Those are
    # exactly the points a straight line cannot stand in for, at a tolerance of 0.2% of each
    # series' range, loosened only if the union overruns the cap.
    normalised: list[NDArray[np.float64]] = []
    for column in columns:
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        span = float(finite.max() - finite.min())
        if span == 0.0:  # one value, or a flat line: an even grid already spans it
            continue
        normalised.append(np.nan_to_num((column - float(finite.min())) / span))
    eps = 0.002
    picked = keep
    for _ in range(8):
        picked = keep.copy()
        for column in normalised:
            picked |= _douglas_peucker(tx, column, eps)
        if int(picked.sum()) <= limit:
            break
        eps *= 1.6
    return [int(i) for i in np.flatnonzero(picked)]


def _r7(value: object) -> float | None:
    try:
        return float(f"{float(value):.7g}")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _load(path: Path, name: str, libraries: list[str]) -> tuple[System, Kernel, dict[str, str]]:
    """Parse the .mo and its libraries once, so a probe can re-integrate without reloading.

    Also returns every variable's declared unit, read from the flat model *before* causalization:
    the causalizer folds ``probe = T_probe - 273.15`` into an alias, and an alias is no longer in
    ``system.variables`` -- so the panel would have shown the roaster's degC probe as unitless.
    """
    sources: list[object] = []
    search = [Path(lib) for lib in libraries]
    # A `library/` folder holding the components a model uses, found without configuration: beside
    # the model, and one level up, which is where the corpus keeps it -- `examples/library/` is a
    # sibling of `examples/models/`, so opening `examples/models/BumpStop.mo` resolves
    # `Translational` with no `pymodelica.library` set.
    for folder in (path.parent / "library", path.parent.parent / "library"):
        if folder.is_dir() and folder not in search:
            search.append(folder)
    for entry in search:
        if entry.is_dir():
            sources.extend(parse(f.read_text(encoding="utf-8")) for f in sorted(entry.glob("*.mo")))
        elif entry.exists():
            sources.append(parse(entry.read_text(encoding="utf-8")))
    sources.append(parse(path.read_text(encoding="utf-8")))
    flat = flatten(sources, name)  # type: ignore[arg-type]
    system = causalize(flat)
    units = {variable.name: variable.unit for variable in flat.variables}
    return system, build(system), units


@dataclass(frozen=True)
class _Solver:
    """How a model asks to be integrated: its sibling ``.toml``'s ``[simulate]`` settings.

    The corpus tests run each model at the tolerance and method its ``.toml`` names (VanDerPol
    at 1e-9 with LSODA, the roaster at 1e-11), and the panel used to run everything at
    `simulate`'s 1e-6 Radau default -- a different integrator from the one the model's reference
    points were checked against. Now the panel runs what the tests run, and says so.
    """

    method: str = "Radau"
    tolerance: float = 1e-6
    absolute: float | None = None

    @property
    def atol(self) -> float:
        """`simulate`'s rule: the absolute tolerance defaults to the relative one."""
        return self.tolerance if self.absolute is None else self.absolute


def _integrate(
    kernel: Kernel, stop: float, params: NDArray[np.float64] | None, solver: _Solver
) -> Result:
    """One run to ``stop`` on a 300-point grid, at the model's parameters or ``params``."""
    return simulate(
        kernel,
        stop=stop,
        times=np.linspace(0.0, stop, 300),
        parameters=params,
        tolerance=solver.tolerance,
        absolute=solver.absolute,
        method=solver.method,
    )


def _integrate_from(  # noqa: PLR0913, PLR0917 - the run's knobs, passed straight through
    kernel: Kernel,
    stop: float,
    params: NDArray[np.float64] | None,
    resume: dict[str, object] | None,
    states: list[str],
    solver: _Solver,
) -> tuple[Result, dict[str, object]]:
    """The panel's main run, with a checkpoint out and, optionally, a checkpoint in.

    Extending the horizon used to re-integrate from t = 0 every time -- O(t) per keystroke, so
    a held arrow key grew slower with every press. Instead the run's end is handed back as a
    checkpoint -- the instant, the state vector and the whole workspace, latches included --
    and the next call can pick up from it, integrating only the new stretch. That is the same
    loop `simulate` runs (`_Run`), started at the checkpoint's instant with its state and its
    workspace rather than at zero with `kernel.start`'s; the run loop never re-latches on
    entry, so a hysteresis latch or a discrete variable set by an earlier event carries over.
    Only valid when the parameters did not change and the stop grew, which the panel checks.
    """
    # The output grid the dense solution is read on. Evaluating the dense polynomial at these
    # points is cheap (it is not more integration), so it is generous: the panel then has enough
    # to stay smooth when zoomed in, where only the fraction of the samples inside the window is
    # seen. `_sample` thins this and the solver's own steps down to what the panel is sent.
    p, y, w = kernel.start(0.0, params)
    if resume is not None:
        start = float(resume["t"])  # type: ignore[arg-type]
        y = np.asarray(resume["y"], dtype=float)
        w = np.asarray(resume["w"], dtype=float)
        grid = np.linspace(start, stop, max(200, int(_GRID * (stop - start) / max(stop, 1e-12))))
    else:
        start = 0.0
        grid = np.linspace(0.0, stop, _GRID)
    plan = _Run(kernel, p, w, solver.tolerance, solver.atol, solver.method)
    result = plan.run(start, stop, y, grid)
    final = [float(np.asarray(result[state], dtype=float)[-1]) for state in states]
    checkpoint: dict[str, object] = {"t": stop, "y": final, "w": [float(v) for v in plan.w]}
    return result, checkpoint


def _settle_time(result: Result, states: set[str], eps: float = 0.01) -> float:
    """The earliest time after which every state stays within ``eps`` of its final value.

    "The signals stopped changing": for each state, the last instant it was more than ``eps`` of
    its own range away from where it ends up, taken over all states. A state that never moves
    contributes nothing. This is the automatic stop -- run to here and the transient is captured
    with no long flat tail.
    """
    time = np.asarray(result.time, dtype=float)
    settle = float(time[0])
    for state in states:
        column = np.asarray(result[state], dtype=float)
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        span = float(finite.max() - finite.min())
        if span == 0.0:
            continue
        moving = np.flatnonzero(np.abs(column - column[-1]) > eps * span)
        if moving.size:
            settle = max(settle, float(time[int(moving[-1])]))
    return settle


_PROBE_TOLERANCE = 1e-6
"""The relative tolerance the automatic stops probe at.

The question a probe answers is qualitative -- did anything fire past here, have the signals
stopped moving -- and the model's own tolerance is for the run that is shown, not for that
question. Measured before this: Lotka-Volterra's `⇥ settle` took 88 s and Rössler's 183 s to
say "not settled", each probe integrating to ten times the horizon at 1e-10. The absolute
tolerance is left as the model's, since a model like E5 lives below any coarse floor.
"""


def _auto_stop(  # noqa: PLR0913, PLR0917 - the knobs of one search, passed straight through
    kernel: Kernel,
    states: set[str],
    until: str,
    frm: float,
    default: float,
    maximum: float,
    params: NDArray[np.float64] | None,
    solver: _Solver,
) -> tuple[float, Result, str]:
    """Choose a stop from the signals: the next event past ``frm``, or where they settle.

    Probes a doubling horizon up to ``maximum`` -- the cap that keeps a model that never settles,
    or has no further event, from running away -- at `_PROBE_TOLERANCE`, then integrates the
    chosen stop at the model's own, and returns the chosen stop, the result
    integrated to it, and a note for the panel. Falls back to ``frm``, unchanged, when there is
    nothing to find before the cap, and the note says so with the cap it reached. A small tail is
    added past the instant found, so the event line and what follows it (or the flat settled
    stretch) is visible rather than cut off at the right edge.
    """
    maximum = max(maximum, frm)  # never below what is already on screen; a cap under the
    # model's own default is still honoured -- the user asked for it
    coarse = _Solver(
        method=solver.method,
        tolerance=max(solver.tolerance, _PROBE_TOLERANCE),
        absolute=solver.absolute,
    )
    horizon = max(frm * 1.5, frm + default, default) if until == "event" else max(frm, default, 1.0)
    horizon = min(horizon, maximum)
    while True:
        probe = _integrate(kernel, horizon, params, coarse)
        if until == "event":
            future = [e for e in probe.events if float(e.time) > frm + 1e-9]
            if future:
                nxt = min(future, key=lambda e: float(e.time))
                target = min(horizon, float(nxt.time) + 0.05 * max(float(nxt.time) - frm, 1.0))
                return (
                    target,
                    _integrate(kernel, target, params, solver),
                    f"next event: {nxt.cause}",
                )
        else:
            settle = _settle_time(probe, states)
            if settle < 0.85 * horizon:  # settled with room to spare, not merely cut off
                target = min(horizon, settle * 1.08)
                return target, _integrate(kernel, target, params, solver), "signals settled"
        if horizon >= maximum:
            break
        horizon = min(horizon * 2.0, maximum)
    fallback = frm if frm > 0 else default
    note = (
        f"no further event before t = {maximum:g}"
        if until == "event"
        else f"signals had not settled by t = {maximum:g}"
    )
    return fallback, _integrate(kernel, fallback, params, solver), note


def _response(result: Result, spec: str) -> float | None:
    """One number read off a run: ``final:<variable>`` or ``event:<cause>`` (its first time).

    ``None`` when there is no such number -- an event that never fired -- which the panel draws
    as a gap rather than a zero.
    """
    kind, _, target = spec.partition(":")
    if kind == "event":
        for event in result.events:
            if event.cause == target:
                return float(event.time)
        return None
    if kind == "final" and target in result.names:
        column = np.asarray(result[target], dtype=float)
        finite = column[np.isfinite(column)]
        return float(finite[-1]) if finite.size else None
    return None


def _sensitivity(  # noqa: PLR0913, PLR0917 - the knobs of one study, passed straight through
    system: System,
    kernel: Kernel,
    stop: float,
    sets: dict[str, float],
    spec: str,
    step: float,
    base: Result,
    solver: _Solver,
) -> dict[str, object]:
    """How much each parameter moves the response, one at a time, by ``step`` either way.

    A finite difference, not the model's analytic sensitivities: the response may be the time
    an event fires, which is not a smooth function the emitted Jacobians know about. Every
    editable parameter (a literal binding; a derived one follows whatever it depends on) is
    run at (1 - step) and (1 + step) times its value with everything else as it is, and the
    response read off each. 2N runs, so this is an explicit action in the panel, not a live one.
    """
    values = settings(system, sets)
    rows: list[dict[str, object]] = []
    for variable in system.parameters:
        derived = variable.binding is not None and not type(variable.binding).__name__.endswith(
            "Literal"
        )
        value = values.get(variable.name)
        if derived or value is None or not isinstance(value, (int, float)) or float(value) == 0.0:
            continue
        ends: list[float | None] = []
        for factor in (1.0 - step, 1.0 + step):
            trial = dict(sets)
            trial[variable.name] = float(value) * factor
            ends.append(_response(_integrate(kernel, stop, vector(system, trial), solver), spec))
        rows.append(
            {
                "name": variable.name,
                "unit": variable.unit,
                "value": _r7(value),
                "minus": _r7(ends[0]) if ends[0] is not None else None,
                "plus": _r7(ends[1]) if ends[1] is not None else None,
            }
        )

    def span(row: dict[str, object]) -> float:
        plus, minus = row["plus"], row["minus"]
        if isinstance(plus, (int, float)) and isinstance(minus, (int, float)):
            return abs(float(plus) - float(minus))
        return -1.0

    rows.sort(key=span, reverse=True)
    return {"response": spec, "step": step, "base": _response(base, spec), "rows": rows}


def _parse_sets(items: list[str]) -> dict[str, float]:
    """``--set name=value`` pairs as a mapping; a malformed one is an error the panel shows."""
    out: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            message = f"--set expects name=value, got {item!r}"
            raise ValueError(message)
        out[name.strip()] = float(value)
    return out


def _per_second(unit: str) -> str:
    """A rate's unit, composed as Modelica writes it: ``K`` to ``K/s``, ``rad/s`` to ``rad/s2``."""
    if not unit:
        return "1/s"
    if unit.endswith("/s2"):
        return unit[:-3] + "/s3"
    if unit.endswith("/s"):
        return unit[:-2] + "/s2"
    return unit + "/s"


def _fill_units(units: dict[str, str], system: System) -> None:
    """Give the derivatives and the aliases the units the flat model leaves blank.

    A derivative carries the state's unit over time: der(T_bean) is K/s, not dimensionless,
    which is also what keeps a °C display from wrongly subtracting 273.15 from a *rate* (only a
    bare "K" is converted, never "K/s"). The flattener leaves a derivative's unit blank, so it is
    filled from the variable it differentiates -- the state named by the mode's derivative map,
    and also any ``der_<v>`` the model observes without ``v`` being a state in this mode
    (Pendulum's ``der_vx`` while it integrates ``[y, vy]``). When the derivative *is* another
    state (``der(th1) = w1``), ``der_th1`` survives only as an alias of it, and any alias of a
    bare variable (``a = b``) carries that variable's unit.
    """
    for state, derivative in system.modes[0].derivative.items():
        units[derivative] = _per_second(units.get(state, ""))
    # Observed derivatives are in the system's variables, not necessarily in the flat model's.
    for variable in sorted({*units, *(v.name for v in system.variables)}):
        if not units.get(variable) and variable.startswith("der_") and units.get(variable[4:]):
            units[variable] = _per_second(units[variable[4:]])
    for alias, expr in system.aliases.items():
        if not units.get(alias) and isinstance(expr, ir.Ref) and units.get(expr.name):
            units[alias] = units[expr.name]


def _run(  # noqa: PLR0913, PLR0917 - the CLI's knobs, passed straight through
    path: Path,
    name: str,
    stop: float,
    libraries: list[str],
    until: str = "none",
    frm: float = 0.0,
    maximum: float = 0.0,
    sets: dict[str, float] | None = None,
    sensitivity: str | None = None,
    sstep: float = 0.05,
    resume: dict[str, object] | None = None,
    tolerance: float | None = None,
) -> dict[str, object]:
    system, kernel, units = _load(path, name, libraries)
    # The model's own integrator, unless the panel asked for a display tolerance. A model's
    # `.toml` names the tolerance that reproduces its reference points -- 1e-10 for the chaotic
    # double pendulum -- which is far tighter than looking at a trajectory needs and, on a
    # chaotic model, several times slower. `--tolerance` overrides the relative tolerance for
    # the run the panel draws, keeping the model's method and its absolute tolerance (E5 lives
    # below any coarse floor and needs its own). "reference" in the panel sends no override.
    solver = _solver(path)
    if tolerance is not None and tolerance > 0:
        solver = replace(solver, tolerance=tolerance)
    states = set(system.modes[0].states)
    sets = sets or {}
    # Overrides go through `vector`, which recomputes everything downstream of a changed
    # parameter (m_dry from m_green), so the run is the model the caller asked for.
    params = vector(system, sets) if sets else None
    # Every parameter with the value this run uses, for the panel to show and edit. One whose
    # binding is an expression (not a literal) is derived from others and marked so: it is
    # shown read-only, since overriding it would silently detach it from what it depends on.
    values = settings(system, sets)
    parameters = [
        {
            "name": variable.name,
            "value": _r7(values.get(variable.name)),
            "unit": variable.unit,
            "doc": variable.comment,
            "derived": variable.binding is not None
            and not type(variable.binding).__name__.endswith("Literal"),
        }
        for variable in system.parameters
    ]
    note = ""
    checkpoint: dict[str, object] | None = None
    if until in ("event", "settle"):
        default = _default_stop(path)
        # No cap given: ten times what is on screen, far enough to find something and bounded
        # enough that a model that never settles comes back in seconds, not minutes.
        cap = maximum if maximum > 0 else 10.0 * max(frm, default)
        stop, result, note = _auto_stop(kernel, states, until, frm, default, cap, params, solver)
    else:
        result, checkpoint = _integrate_from(
            kernel, stop, params, resume, list(system.modes[0].states), solver
        )
    time = np.asarray(result.time, dtype=float)
    # A parameter is one flat value; keep the ones that move, plus the states so a constant
    # state (a body at rest) still shows. Decided on the full column, then every kept series
    # has a say in which instants are sent, so each is drawn to its true shape.
    full: dict[str, NDArray[np.float64]] = {}
    for variable in result.names:
        column = np.asarray(result[variable], dtype=float)
        finite = column[np.isfinite(column)]
        if variable in states or (finite.size and float(finite.max() - finite.min()) > 0.0):
            full[variable] = column
    keep = _sample(time, list(full.values()))
    series: dict[str, list[float | None]] = {
        variable: [_r7(v) for v in column[keep]] for variable, column in full.items()
    }
    events = [
        {"t": _r7(event.time), "cause": str(getattr(event, "cause", ""))}
        for event in (getattr(result, "events", []) or [])
    ]

    # `units` came from the flat model in `_load`, aliases included; derivatives and aliases of
    # a bare variable get theirs here.
    _fill_units(units, system)
    # The sensitivity study, when asked for: 2N more runs, so only on request.
    sens = (
        _sensitivity(system, kernel, stop, sets, sensitivity, sstep, result, solver)
        if sensitivity
        else None
    )
    return {
        "model": name,
        "summary": str(path),
        "tags": [],
        "group": "",
        "stop": stop,
        "t": [_r7(v) for v in time[keep]],
        "series": series,
        "units": {nm: units.get(nm, "") for nm in series},
        "refs": [],
        "events": events,
        "nstates": len(states),
        "states": list(system.modes[0].states),
        "autoNote": note,
        "parameters": parameters,
        "overrides": sets,
        "sensitivity": sens,
        # The integrator this run used -- the model's, from its .toml, unless the panel asked for
        # a display tolerance -- so the panel can say. `tolerance` echoes the override the panel
        # sent (None for the model's reference run), which is what the panel compares a continued
        # run against: a checkpoint holds only at the tolerance it was taken at.
        "solver": {
            "method": solver.method,
            "tolerance": solver.tolerance,
            "absolute": solver.absolute,
        },
        "tolerance": tolerance,
        # Where this run ended, for the next call to continue from; and, when this call did
        # continue, where it took over -- so the panel appends the tail to what it has.
        "checkpoint": checkpoint,
        "resumedFrom": float(resume["t"]) if resume else None,  # type: ignore[arg-type]
    }


def _section(path: Path) -> dict[str, object]:
    """The model's ``[simulate]`` table from its sibling ``<name>.toml``, or nothing.

    A missing or malformed file is not an error here: the model still simulates, at defaults.
    """
    toml = path.with_suffix(".toml")
    if not toml.exists():
        return {}
    try:
        spec = tomllib.loads(toml.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return {}
    section: object = spec.get("simulate", {})
    return cast("dict[str, object]", section) if isinstance(section, dict) else {}


def _default_stop(path: Path) -> float:
    """The model's own stop time: ``[simulate].stop``, else 10."""
    try:
        return float(_section(path).get("stop", 10.0))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 10.0


def _solver(path: Path) -> _Solver:
    """The model's own integrator: ``[simulate].method``, ``tolerance``, ``absolute``.

    The same keys `tests/test_examples.py` reads when it checks the model against its reference
    points, so the panel reproduces the corpus run rather than a looser default.
    """
    section = _section(path)
    try:
        return _Solver(
            method=str(section.get("method", "Radau")),
            tolerance=float(section.get("tolerance", 1e-6)),  # type: ignore[arg-type]
            absolute=(
                float(section["absolute"])  # type: ignore[arg-type]
                if section.get("absolute") is not None
                else None
            ),
        )
    except (ValueError, TypeError):
        return _Solver()


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate one .mo file to JSON.")
    ap.add_argument("path", help="the .mo file")
    ap.add_argument("--stop", type=float, default=None, help="stop time (default: the model's own)")
    ap.add_argument("--model", default=None, help="model class name (default: the file stem)")
    ap.add_argument("--library", action="append", default=[], help="extra .mo file or folder")
    ap.add_argument(
        "--until",
        choices=["none", "event", "settle"],
        default="none",
        help="pick the stop from the signals: the next event past --from, or where they settle",
    )
    ap.add_argument(
        "--from", dest="frm", type=float, default=0.0, help="the current stop, for --until event"
    )
    ap.add_argument(
        "--max",
        dest="maximum",
        type=float,
        default=0.0,
        help="the farthest --until may probe (0: ten times the current stop)",
    )
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a parameter (repeatable); everything derived from it is recomputed",
    )
    ap.add_argument(
        "--sensitivity",
        default=None,
        metavar="SPEC",
        help="also run every editable parameter a step either way and read off this response: "
        "'final:<variable>' or 'event:<cause>'",
    )
    ap.add_argument(
        "--sensitivity-step",
        dest="sensitivity_step",
        type=float,
        default=0.05,
        help="the relative step for --sensitivity (0.05 is +-5 percent)",
    )
    ap.add_argument(
        "--resume",
        default=None,
        metavar="FILE",
        help="a checkpoint JSON {t, y, w} from an earlier run's output: continue from it to "
        "--stop instead of starting at t = 0 (same parameters, larger stop)",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="the relative tolerance for the run the panel draws, overriding the model's own "
        "(its method and absolute tolerance are kept); omit to use the model's reference tolerance",
    )
    args = ap.parse_args()

    path = Path(args.path)
    name = args.model or path.stem
    stop = args.stop if args.stop is not None else _default_stop(path)
    try:
        record = _run(
            path,
            name,
            stop,
            args.library,
            until=args.until,
            frm=args.frm,
            maximum=args.maximum,
            sets=_parse_sets(args.sets),
            sensitivity=args.sensitivity,
            sstep=args.sensitivity_step,
            resume=json.loads(Path(args.resume).read_text(encoding="utf-8"))
            if args.resume
            else None,
            tolerance=args.tolerance,
        )
        print(json.dumps({"models": [record]}, separators=(",", ":"), allow_nan=False))
    except Exception as error:  # noqa: BLE001 - report every failure to the panel, never crash silently
        print(
            json.dumps(
                {"error": f"{type(error).__name__}: {error}", "path": str(path), "model": name}
            )
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
