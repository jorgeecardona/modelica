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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten
from modelica.sim.emit import settings
from modelica.sim.kernel import build
from modelica.sim.solve import simulate
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


def _sample(
    time: NDArray[np.float64], columns: list[NDArray[np.float64]], limit: int = 700
) -> list[int]:
    """Which instants to send the panel: shape-preserving, so a spike keeps its apex every cycle.

    The solver's own steps cluster at a jump -- that is where it had to work -- and a uniform
    stride over them kept the sample at the peak on some cycles and dropped it on others, so a
    periodic impulse (van der Pol's der(x), a 0.01 s jump to 1350) rendered at a wandering
    height. Douglas-Peucker per series keeps exactly the points a straight line cannot stand
    in for -- the apex and the two shoulders of every spike, the knee of every corner -- and
    the union over the series is what is kept, usually fewer points than the stride was. Event
    instants (a repeated time) always stay, so a discontinuity draws as a vertical step.
    """
    n = len(time)
    if n <= limit:
        return list(range(n))
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    breaks = np.flatnonzero(np.diff(time) <= 0.0)
    keep[breaks] = True
    keep[np.minimum(breaks + 1, n - 1)] = True
    tx = (time - time[0]) / (float(time[-1] - time[0]) or 1.0)
    normalised: list[NDArray[np.float64]] = []
    for column in columns:
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        span = float(finite.max() - finite.min())
        if span == 0.0:  # one value, or a flat line: nothing a straight line cannot stand in for
            continue
        normalised.append(np.nan_to_num((column - float(finite.min())) / span))
    # A tolerance of 0.4% of each series' own range keeps every feature a reader could see;
    # it is loosened only if the union still overruns the cap.
    eps = 0.004
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
    sibling = path.parent / "library"
    if sibling.is_dir():
        search.append(sibling)
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


def _integrate(kernel: Kernel, stop: float, params: NDArray[np.float64] | None = None) -> Result:
    """One run to ``stop`` on a 300-point grid, at the model's parameters or ``params``."""
    return simulate(kernel, stop=stop, times=np.linspace(0.0, stop, 300), parameters=params)


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


def _auto_stop(  # noqa: PLR0913, PLR0917 - the knobs of one search, passed straight through
    kernel: Kernel,
    states: set[str],
    until: str,
    frm: float,
    default: float,
    maximum: float,
    params: NDArray[np.float64] | None,
) -> tuple[float, Result, str]:
    """Choose a stop from the signals: the next event past ``frm``, or where they settle.

    Probes a doubling horizon up to ``maximum`` -- the cap that keeps a model that never settles,
    or has no further event, from running away -- and returns the chosen stop, the result
    integrated to it, and a note for the panel. Falls back to ``frm``, unchanged, when there is
    nothing to find before the cap, and the note says so with the cap it reached. A small tail is
    added past the instant found, so the event line and what follows it (or the flat settled
    stretch) is visible rather than cut off at the right edge.
    """
    maximum = max(maximum, frm)  # never below what is already on screen; a cap under the
    # model's own default is still honoured -- the user asked for it
    horizon = max(frm * 1.5, frm + default, default) if until == "event" else max(frm, default, 1.0)
    horizon = min(horizon, maximum)
    while True:
        probe = _integrate(kernel, horizon, params)
        if until == "event":
            future = [e for e in probe.events if float(e.time) > frm + 1e-9]
            if future:
                nxt = min(future, key=lambda e: float(e.time))
                target = min(horizon, float(nxt.time) + 0.05 * max(float(nxt.time) - frm, 1.0))
                return target, _integrate(kernel, target, params), f"next event: {nxt.cause}"
        else:
            settle = _settle_time(probe, states)
            if settle < 0.85 * horizon:  # settled with room to spare, not merely cut off
                target = min(horizon, settle * 1.08)
                return target, _integrate(kernel, target, params), "signals settled"
        if horizon >= maximum:
            break
        horizon = min(horizon * 2.0, maximum)
    fallback = frm if frm > 0 else default
    note = (
        f"no further event before t = {maximum:g}"
        if until == "event"
        else f"signals had not settled by t = {maximum:g}"
    )
    return fallback, _integrate(kernel, fallback, params), note


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
            ends.append(_response(_integrate(kernel, stop, vector(system, trial)), spec))
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
) -> dict[str, object]:
    system, kernel, units = _load(path, name, libraries)
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
    if until in ("event", "settle"):
        default = _default_stop(path)
        # No cap given: ten times what is on screen, far enough to find something and bounded
        # enough that a model that never settles comes back in seconds, not minutes.
        cap = maximum if maximum > 0 else 10.0 * max(frm, default)
        stop, result, note = _auto_stop(kernel, states, until, frm, default, cap, params)
    else:
        result = _integrate(kernel, stop, params)
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
    # `units` came from the flat model in `_load`, aliases included.
    # A derivative carries the state's unit over time: der(T_bean) is K/s, not dimensionless,
    # which is also what keeps a °C display from wrongly subtracting 273.15 from a *rate* (only
    # a bare "K" is converted, never "K/s"). The flattener leaves the derivative variable's unit
    # blank, so fill it from the state it differentiates, named by the mode's derivative map.
    for state, derivative in system.modes[0].derivative.items():
        base = units.get(state, "")
        units[derivative] = f"{base}/s" if base else "1/s"
    # The sensitivity study, when asked for: 2N more runs, so only on request.
    sens = (
        _sensitivity(system, kernel, stop, sets, sensitivity, sstep, result)
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
    }


def _default_stop(path: Path) -> float:
    """The model's own stop time: a sibling ``<name>.toml``'s ``[simulate].stop``, else 10."""
    toml = path.with_suffix(".toml")
    if toml.exists():
        try:
            spec = tomllib.loads(toml.read_text(encoding="utf-8"))
            stop = spec.get("simulate", {}).get("stop")
            if stop is not None:
                return float(stop)
        except (tomllib.TOMLDecodeError, ValueError, TypeError):
            pass
    return 10.0


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
