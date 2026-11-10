"""The shaped opcodes in `vm.evaluate`, each against NumPy on a 3 x 4 array."""

from __future__ import annotations

import dataclasses
import functools
import operator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from modelica.lang import parse
from modelica.sim import causalize, flatten, vm

if TYPE_CHECKING:
    from collections.abc import Callable

LORENZ = Path(__file__).resolve().parent.parent / "examples" / "models" / "Lorenz.mo"
SHAPE = (3, 4)
VALUES = np.arange(1.0, 13.0).reshape(SHAPE) * 0.37 - 2.0


@functools.cache
def _program() -> vm.Program:
    system = causalize(flatten([parse(LORENZ.read_text(), str(LORENZ))], "Lorenz"))
    return vm.lower(system)


def _run(views: tuple[vm.View, ...], code: tuple[vm.Instr, ...], size: int) -> np.ndarray:
    """Load ``VALUES`` into view 0 (registers ``s`` on), run ``code``, return ``out[:size]``."""
    program = _program()
    s = program.registers
    program = dataclasses.replace(program, registers=s + 256 + len(program.constants), views=views)
    load = vm.Instr(vm.Op.VLOAD_Y, s, 0, 0, VALUES.size)
    out = [0.0] * size
    w = [0.0] * 64
    p = [0.0] * len(program.layout.parameters)
    vm.evaluate(program, vm.Function((load, *code)), 0.0, VALUES.ravel().tolist(), p, w, out)
    return np.asarray(out)


def _base() -> int:
    return _program().registers


def _dense(base: int, shape: tuple[int, ...]) -> vm.View:
    strides = tuple(int(np.prod(shape[k + 1 :])) for k in range(len(shape)))
    return vm.View(base, shape, strides)


def _out(register: int, size: int) -> vm.Instr:
    return vm.Instr(vm.Op.VSTORE_OUT, 0, 0, register, size)


def test_elements_are_row_major_with_strides() -> None:
    view = vm.View(10, (2, 3), (1, 2))  # a transpose of a 3 x 2 block
    assert vm.elements(view) == [10, 12, 14, 11, 13, 15]
    assert len(view) == 6


def test_binary_with_a_broadcast_scalar() -> None:
    s = _base()
    views = (_dense(s, SHAPE), vm.View(s, SHAPE, (0, 0)), _dense(s + 100, SHAPE))
    code = (vm.Instr(vm.Op.TBINARY, 2, 0, 1, int(vm.Op.SUB)), _out(s + 100, 12))
    np.testing.assert_array_equal(_run(views, code, 12), (VALUES - VALUES[0, 0]).ravel())


def test_a_transposed_view_reads_columns() -> None:
    s = _base()
    transposed = vm.View(s, (4, 3), (1, 4))
    views = (transposed, _dense(s + 100, (4, 3)))
    code = (vm.Instr(vm.Op.TBINARY, 1, 0, 0, int(vm.Op.MUL)), _out(s + 100, 12))
    np.testing.assert_array_equal(_run(views, code, 12), (VALUES.T * VALUES.T).ravel())


def test_a_slice_moves_the_base_and_shrinks_the_shape() -> None:
    s = _base()
    interior = vm.View(s + 1, (3, 2), (4, 1))  # VALUES[:, 1:3]
    left = vm.View(s, (3, 2), (4, 1))  # VALUES[:, 0:2]
    views = (interior, left, _dense(s + 100, (3, 2)))
    code = (vm.Instr(vm.Op.TBINARY, 2, 0, 1, int(vm.Op.SUB)), _out(s + 100, 6))
    expected = VALUES[:, 1:3] - VALUES[:, 0:2]
    np.testing.assert_array_equal(_run(views, code, 6), expected.ravel())


@pytest.mark.parametrize(("axis", "offset"), [(0, 1), (1, -1), (1, 2)])
def test_roll_is_numpy_roll(axis: int, offset: int) -> None:
    s = _base()
    views = (_dense(s, SHAPE), _dense(s + 100, SHAPE))
    code = (vm.Instr(vm.Op.ROLL, 1, 0, axis, offset), _out(s + 100, 12))
    expected = np.roll(VALUES, offset, axis=axis)
    np.testing.assert_array_equal(_run(views, code, 12), expected.ravel())


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    ("kind", "fold"),
    [
        (vm.Reduce.SUM, operator.add),
        (vm.Reduce.PRODUCT, operator.mul),
        (vm.Reduce.MIN, min),
        (vm.Reduce.MAX, max),
    ],
)
def test_reduce_folds_one_axis_in_order(
    axis: int, kind: vm.Reduce, fold: Callable[[float, float], float]
) -> None:
    s = _base()
    kept = SHAPE[1 - axis]
    views = (_dense(s, SHAPE), _dense(s + 100, (kept,)))
    code = (vm.Instr(vm.Op.TREDUCE, 1, 0, axis, int(kind)), _out(s + 100, kept))
    lines = VALUES.T if axis == 0 else VALUES
    expected = [functools.reduce(fold, line.tolist()) for line in lines]
    np.testing.assert_array_equal(_run(views, code, kept), expected)


def test_index_counts_from_one_along_an_axis() -> None:
    s = _base()
    views = (_dense(s, SHAPE), _dense(s + 100, SHAPE))
    code = (vm.Instr(vm.Op.TINDEX, 1, 1), _out(s + 100, 12))
    expected = np.broadcast_to(np.arange(1.0, 5.0), SHAPE)
    np.testing.assert_array_equal(_run(views, code, 12), expected.ravel())


def test_select_takes_then_where_the_condition_holds() -> None:
    s = _base()
    zero = s + 200  # a register the tape leaves at 0.0
    views = (
        _dense(s, SHAPE),
        vm.View(zero, SHAPE, (0, 0)),
        _dense(s + 100, SHAPE),
        _dense(s + 120, SHAPE),
    )
    code = (
        vm.Instr(vm.Op.TBINARY, 2, 0, 1, int(vm.Op.GT)),
        vm.Instr(vm.Op.TUNARY, 3, 0, 0, int(vm.Op.NEG)),
        vm.Instr(vm.Op.TSELECT, 2, 2, 0, 3),
        _out(s + 100, 12),
    )
    np.testing.assert_array_equal(_run(views, code, 12), np.abs(VALUES).ravel())


def test_unary_builtin_and_binary_builtin() -> None:
    s = _base()
    views = (_dense(s, SHAPE), _dense(s + 100, SHAPE), _dense(s + 120, SHAPE))
    code = (
        vm.Instr(vm.Op.TUNARY, 1, 0, int(vm.Builtin.SIN), int(vm.Op.CALL1)),
        vm.Instr(vm.Op.TCALL2, 2, 0, 1, int(vm.Builtin.MAX)),
        _out(s + 120, 12),
    )
    np.testing.assert_array_equal(_run(views, code, 12), np.maximum(VALUES, np.sin(VALUES)).ravel())
