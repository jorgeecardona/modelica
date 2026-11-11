"""The front-end benchmark runs, and ``vm.lower`` does not add per-cell calls."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

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
