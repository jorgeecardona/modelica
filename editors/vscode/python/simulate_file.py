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


def _thin(time: list[float], limit: int = 700) -> list[int]:
    t = np.asarray(time, dtype=float)
    n = len(t)
    if n <= limit:
        return list(range(n))
    stride = int(np.ceil(n / limit))
    breaks = np.flatnonzero(np.diff(t) <= 0.0)
    keep = np.union1d(
        np.arange(0, n, stride), np.union1d(np.union1d(breaks, breaks + 1), [0, n - 1])
    )
    return [int(i) for i in keep]


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
    keep = _thin(list(time))
    series: dict[str, list[float | None]] = {}
    for variable in result.names:
        column = np.asarray(result[variable])
        # A parameter is one flat value; keep the ones that move, plus the states so a
        # constant state (a body at rest) still shows. This is what the widget opens on.
        values = [_r7(v) for v in column[keep]]
        if variable in states or len({v for v in values if v is not None}) > 1:
            series[variable] = values
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
