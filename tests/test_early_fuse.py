"""``vm.lower`` joins loads and stores early, and the tape is the one late joining gives."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelica.lang import parse
from modelica.sim import causalize, flatten, vm

MODELS = Path(__file__).resolve().parents[1] / "examples" / "models"


@pytest.mark.parametrize("name", ["Lorenz", "Robertson", "HeatBar", "WaveString", "PhaseChange"])
def test_early_joining_gives_the_same_tape(name: str) -> None:
    """`_preload`, `_hoist` and `_compact` see the joined runs; the result does not change."""
    source = (MODELS / f"{name}.mo").read_text()
    system = causalize(flatten([parse(source, f"{name}.mo")], name))
    assert vm.lower(system) == vm.lower(system, early=False)


def test_lorenz_jacobian_loads_its_states_in_one_instruction() -> None:
    """The three states are read by one ``VLOAD_Y``, as the late `_fuse` writes it."""
    source = (MODELS / "Lorenz.mo").read_text()
    jacobian = vm.lower(causalize(flatten([parse(source, "Lorenz.mo")], "Lorenz"))).jacobian[0]
    assert len(jacobian.code) == 25
    assert [i.op for i in jacobian.code].count(vm.Op.VLOAD_Y) == 1
    assert vm.Op.LOAD_Y not in [i.op for i in jacobian.code]
