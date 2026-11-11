"""The front-end benchmark runs, and ``vm.lower`` does not add per-cell calls."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "frontend_bench.py"


def _bench() -> Any:
    spec = importlib.util.spec_from_file_location("frontend_bench", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["frontend_bench"] = module  # a dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


def test_lower_makes_no_per_cell_calls_on_the_box() -> None:
    """Each subscript reads one loop index, so `_read_plan` evaluates it along that loop only.

    What is left is 4 ``at_cell`` calls (the two end cells of each loop), at every size.
    """
    bench = _bench()
    assert bench.measure(4, run=False).cell_calls_in_lower == 4
    assert bench.measure(6, run=False).cell_calls_in_lower == 4


def test_alias_detection_does_not_differentiate_a_grid_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each box cell reads seven unknowns, so `_affine` gives up before any ``d_dx``."""
    from modelica.ir import causalize as module  # noqa: PLC0415 - patched below
    from modelica.lang import parse  # noqa: PLC0415
    from modelica.sim import flatten  # noqa: PLC0415

    flat = flatten([parse(_bench().BOX.format(n=4), "Box.mo")], "Box")
    calls = 0
    original = module.d_dx

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "d_dx", counted)
    work = module._start(flat)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    module._eliminate_aliases(work)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    assert calls == 0
