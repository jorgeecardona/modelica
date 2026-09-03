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

import numpy as np

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten
from modelica.sim.kernel import build
from modelica.sim.solve import simulate


def _thin(time: list[float], limit: int = 700) -> list[int]:
    t = np.asarray(time, dtype=float)
    n = len(t)
    if n <= limit:
        return list(range(n))
    stride = int(np.ceil(n / limit))
    breaks = np.flatnonzero(np.diff(t) <= 0.0)
    keep = np.union1d(np.arange(0, n, stride), np.union1d(np.union1d(breaks, breaks + 1), [0, n - 1]))
    return [int(i) for i in keep]


def _r7(value: object) -> float | None:
    try:
        return float(f"{float(value):.7g}")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _run(path: Path, name: str, stop: float, libraries: list[str]) -> dict[str, object]:
    sources = []
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

    system = causalize(flatten(sources, name))
    kernel = build(system)
    grid = np.linspace(0.0, stop, 300)
    result = simulate(kernel, stop=stop, times=grid)
    time = np.asarray(result.time, dtype=float)
    keep = _thin(list(time))
    states = set(system.modes[0].states)
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
    units = {v.name: v.unit for v in system.variables}
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
    args = ap.parse_args()

    path = Path(args.path)
    name = args.model or path.stem
    stop = args.stop if args.stop is not None else _default_stop(path)
    try:
        record = _run(path, name, stop, args.library)
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
