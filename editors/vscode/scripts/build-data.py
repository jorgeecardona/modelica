#!/usr/bin/env python
"""Regenerate the Simulation Scope's data snapshot from the corpus.

Run from the repo root:  uv run --extra sim python editors/vscode/scripts/build-data.py

Every trajectory here comes from `modelica.sim` -- the Python simulator, the reference tier.
Nothing is compiled per model and no external compiler is invoked: the model is flattened,
integrated, and its trajectory written to `media/scope.data.json`, which the VS Code webview
(and the notebook widget it shares an engine with) draws. Refresh the corpus, rerun this, and
the chart updates.
"""

# A build script: printing progress to the terminal is the point.
# ruff: noqa: T201
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten
from modelica.sim.kernel import build
from modelica.sim.solve import simulate

if TYPE_CHECKING:
    from modelica.ir.system import System

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples"
OUT = Path(__file__).resolve().parents[1] / "media" / "scope.data.json"

#: Which shelf of the rail each model sits on. Anything unlisted falls to "Other".
GROUP = {
    "RLCCircuit": "Circuits",
    "RCCircuit": "Circuits",
    "ProbeLC": "Circuits",
    "ProbeRC": "Circuits",
    "DCMotor": "Circuits",
    "PIControl": "Circuits",
    "MassSpringDamper": "Mechanical",
    "Pendulum": "Mechanical",
    "AtwoodPendulum": "Mechanical",
    "TwoInertias": "Mechanical",
    "TwoFalls": "Mechanical",
    "HeatBar": "Thermal",
    "TwoBodyHeat": "Thermal",
    "NewtonCooling": "Thermal",
    "Robertson": "Stiff benchmarks",
    "VanDerPol": "Stiff benchmarks",
    "HIRES": "Stiff benchmarks",
    "BouncingBall": "Hybrid / events",
    "Thermostat": "Hybrid / events",
    "RoastingMachine": "Hybrid / events",
    "DrainingTank": "Hybrid / events",
    "Setback": "Hybrid / events",
    "BumpStop": "Hybrid / events",
    "PhaseChange": "Hybrid / events",
    "E5": "Stiff benchmarks",
    "Lorenz": "Chaos and orbits",
    "Rossler": "Chaos and orbits",
    "Duffing": "Chaos and orbits",
    "LotkaVolterra": "Chaos and orbits",
    "Brusselator": "Chaos and orbits",
    "Chua": "Chaos and orbits",
}

#: Rail order. Listed models come first in this order; the rest follow alphabetically.
ORDER = [
    "RLCCircuit",
    "RCCircuit",
    "DCMotor",
    "PIControl",
    "ProbeLC",
    "ProbeRC",
    "MassSpringDamper",
    "Pendulum",
    "AtwoodPendulum",
    "TwoInertias",
    "TwoFalls",
    "HeatBar",
    "TwoBodyHeat",
    "NewtonCooling",
    "Robertson",
    "VanDerPol",
    "HIRES",
    "E5",
    "BouncingBall",
    "Thermostat",
    "RoastingMachine",
    "DrainingTank",
    "Setback",
    "BumpStop",
    "PhaseChange",
    "Lorenz",
    "Rossler",
    "Duffing",
    "LotkaVolterra",
    "Brusselator",
    "Chua",
]

LIMIT = 700
"""Samples kept per model after thinning -- a stride, plus both sides of every event."""


def _system(stem: str) -> tuple[System, dict]:
    spec = tomllib.loads((EXAMPLES / "models" / f"{stem}.toml").read_text(encoding="utf-8"))
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse((EXAMPLES / "models" / f"{stem}.mo").read_text(encoding="utf-8")))
    return causalize(flatten(sources, stem)), spec


def _thin(time: np.ndarray) -> np.ndarray:
    n = len(time)
    if n <= LIMIT:
        return np.arange(n)
    stride = int(np.ceil(n / LIMIT))
    breaks = np.flatnonzero(np.diff(time) <= 0.0)
    ends = np.array([0, n - 1])
    return np.union1d(np.arange(0, n, stride), np.union1d(np.union1d(breaks, breaks + 1), ends))


def _r7(value: object) -> float | None:
    try:
        return float(f"{float(value):.7g}")
    except (TypeError, ValueError):
        return None


def _trajectory(stem: str) -> dict | None:
    system, spec = _system(stem)
    if "simulate" not in spec or "stop" not in spec["simulate"]:
        return None
    stop = float(spec["simulate"]["stop"])
    # The model's own integrator, as the corpus tests run it and as the live panel does: E5's
    # states live at 1e-20 and it does not integrate at all without its absolute tolerance
    # (`Required step size is less than spacing between numbers`), and VanDerPol asks for LSODA.
    section = spec["simulate"]
    method = str(section.get("method", "Radau"))
    tolerance = float(section.get("tolerance", 1e-6))
    absolute = float(section["absolute"]) if section.get("absolute") is not None else None
    kernel = build(system)
    result = simulate(
        kernel,
        stop=stop,
        times=np.linspace(0.0, stop, 300),
        tolerance=tolerance,
        absolute=absolute,
        method=method,
    )
    time = np.asarray(result.time, dtype=float)
    keep = _thin(time)
    names = list(system.modes[0].states)
    series = {name: [_r7(v) for v in np.asarray(result[name])[keep]] for name in names}
    if not series:
        return None
    events = [
        {"t": _r7(event.time), "cause": str(getattr(event, "cause", ""))}
        for event in (getattr(result, "events", []) or [])
    ]
    refs = [
        {"t": float(point["t"]), "values": {k: _r7(v) for k, v in point["values"].items()}}
        for point in spec.get("simulate", {}).get("point", [])
        if point.get("source") == "reference" and "values" in point
    ]
    units = {v.name: v.unit for v in system.variables}
    return {
        "model": stem,
        "summary": spec.get("summary", ""),
        "tags": spec.get("teaches", []),
        "group": GROUP.get(stem, "Other"),
        "stop": stop,
        "t": [_r7(v) for v in time[keep]],
        "series": series,
        "units": {nm: units.get(nm, "") for nm in series},
        "refs": refs,
        "events": events,
        "nstates": len(names),
        "states": list(names),
    }


def main() -> None:
    stems = [p.stem for p in sorted((EXAMPLES / "models").glob("*.mo"))]
    order = [m for m in ORDER if m in stems] + [m for m in stems if m not in ORDER]
    models = []
    for stem in order:
        try:
            record = _trajectory(stem)
        except Exception as error:  # noqa: BLE001 - one bad model must not sink the snapshot
            print(f"  skip {stem}: {type(error).__name__}: {error}")
            continue
        if record is None:
            continue
        finite = all(
            v is not None and np.isfinite(v) for ser in record["series"].values() for v in ser
        )
        if not finite:
            print(f"  skip {stem}: non-finite trajectory")
            continue
        models.append(record)
        print(f"  ok   {stem}: {record['nstates']} states, {len(record['events'])} events")
    OUT.write_text(json.dumps({"models": models}, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(models)} models to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
