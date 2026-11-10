"""Linear finite elements on triangles, written as array opcodes (plan step 14).

The mesh, the element matrices and their assembly are computed once in Python. Stiffness is
assembled into entries in row order (CSR without the row pointers, which the tape does not
need); mass is lumped to a diagonal, the vertex share of each triangle's area. The tape reads
the entries from ``p`` and multiplies with `vm.spmv`. Nothing is generated or compiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from modelica.sim.vm import Instr, Op, spmv

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass(frozen=True)
class Triangles:
    """A triangle mesh: ``points`` (x, y) and ``triangles`` as three point indices, anticlockwise."""

    points: tuple[tuple[float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    boundary: frozenset[int]
    """The points on the domain boundary."""


def square(n: int) -> Triangles:
    """The unit square cut into ``n x n`` cells, each split into two triangles by a diagonal."""
    points = tuple((i / n, j / n) for j in range(n + 1) for i in range(n + 1))

    def at(i: int, j: int) -> int:
        return i + (n + 1) * j

    triangles: list[tuple[int, int, int]] = []
    for j in range(n):
        for i in range(n):
            triangles.append((at(i, j), at(i + 1, j), at(i + 1, j + 1)))
            triangles.append((at(i, j), at(i + 1, j + 1), at(i, j + 1)))
    boundary = frozenset(
        at(i, j) for j in range(n + 1) for i in range(n + 1) if i in (0, n) or j in (0, n)
    )
    return Triangles(points, tuple(triangles), boundary)


@dataclass(frozen=True)
class Assembled:
    """The P1 stiffness ``K`` as entries in row order, and the lumped mass per point."""

    rows: tuple[int, ...]
    columns: tuple[int, ...]
    values: tuple[float, ...]
    mass: tuple[float, ...]


def assemble(mesh: Triangles) -> Assembled:
    """``K[a, b] = sum over triangles of grad phi_a . grad phi_b * area``; mass lumped."""
    points = np.asarray(mesh.points, dtype=float)
    entries: dict[tuple[int, int], float] = {}
    mass = np.zeros(len(points))
    for triangle in mesh.triangles:
        corners: NDArray[np.float64] = points[list(triangle)]
        edges = np.array([corners[1] - corners[0], corners[2] - corners[0]])
        area = 0.5 * float(np.linalg.det(edges))
        # Gradients of the three hat functions: constant on the triangle.
        inverse = np.linalg.inv(edges)
        gradients = np.vstack([-inverse.sum(axis=1), inverse[:, 0], inverse[:, 1]])
        local = area * gradients @ gradients.T
        for p, a in enumerate(triangle):
            mass[a] += area / 3.0
            for q, b in enumerate(triangle):
                entries[a, b] = entries.get((a, b), 0.0) + float(local[p, q])
    ordered = sorted(entries)
    return Assembled(
        tuple(a for a, _ in ordered),
        tuple(b for _, b in ordered),
        tuple(entries[key] for key in ordered),
        tuple(mass.tolist()),
    )


@dataclass(frozen=True)
class Tape:
    """The instructions of ``M du/dt = -K u + M s`` on the interior points, and their ``p``."""

    code: tuple[Instr, ...]
    tables: tuple[int, ...]
    parameters: tuple[float, ...]
    unknowns: tuple[int, ...]
    """The mesh point of each state."""
    mass: tuple[float, ...]
    """The lumped mass of each state, for `vm.Program.mass`."""


def heat(
    mesh: Triangles,
    source: Callable[[float, float], float],
    register: Callable[[int], int],
) -> Tape:
    """``du/dt = laplacian(u) + s`` with ``u = 0`` on the boundary, as a tape writing ``f``.

    The states are the interior points. ``rhs`` writes ``-K u + M s`` and `Tape.mass` is
    ``M``, so Radau solves ``M u' = f`` (`vm.Program.mass`).
    """
    assembled = assemble(mesh)
    unknowns = tuple(k for k in range(len(mesh.points)) if k not in mesh.boundary)
    index = {point: state for state, point in enumerate(unknowns)}
    kept = [
        (index[a], index[b], -value)
        for a, b, value in zip(assembled.rows, assembled.columns, assembled.values, strict=True)
        if a in index and b in index
    ]
    n = len(unknowns)
    values: list[float] = []
    code: list[Instr] = []
    tables: list[int] = []

    def load(numbers: list[float]) -> int:
        first = register(len(numbers))
        for k, value in enumerate(numbers):
            code.append(Instr(Op.LOAD_P, first + k, len(values)))
            values.append(value)
        return first

    entries = load([value for _, _, value in kept])
    zero = load([0.0])
    load_vector = load([assembled.mass[point] * source(*mesh.points[point]) for point in unknowns])
    state = register(n)
    code.extend(Instr(Op.LOAD_Y, state + k, k) for k in range(n))
    product = register(n)
    rows = [row for row, _, _ in kept]
    columns = [column for _, column, _ in kept]
    code.extend(spmv(rows, columns, product, state, entries, zero, tables, register))
    total = register(n)
    code.append(Instr(Op.VADD, total, product, load_vector, n))
    code.extend(Instr(Op.STORE_OUT, 0, k, total + k) for k in range(n))
    return Tape(
        tuple(code),
        tuple(tables),
        tuple(values),
        unknowns,
        tuple(assembled.mass[point] for point in unknowns),
    )
