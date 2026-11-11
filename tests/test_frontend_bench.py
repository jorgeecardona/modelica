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


def test_lower_makes_at_most_todays_per_cell_calls_on_a_small_box() -> None:
    """216 cells, 14 subscripts each: 3024 ``indices_at`` calls today, plus 4 ``at_cell``.

    The array-node plan lowers this number toward a constant; it must never grow.
    """
    stages = _bench().measure(6, run=False)
    assert stages.cell_calls_in_lower <= 14 * 6**3 + 4


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
