"""``modelica.sim`` exports the user surface only; experimental modules stay importable by path."""

from __future__ import annotations

import importlib

import modelica.sim

SURFACE = {
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
    "simulate",
    "steady",
    "sweep",
}


def test_the_surface_is_the_listed_names() -> None:
    assert set(modelica.sim.__all__) == SURFACE


def test_every_name_imports() -> None:
    for name in SURFACE:
        assert getattr(modelica.sim, name) is not None


def test_experimental_modules_import_by_path() -> None:
    for module in ("emit", "kernel", "compare", "sensitivity", "fv", "fem", "mesh", "vm"):
        importlib.import_module(f"modelica.sim.{module}")
