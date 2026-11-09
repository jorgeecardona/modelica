"""Finite-volume heat conduction on a `Mesh`, written as array opcodes (plan steps 9 and 11).

The discretisation is computed once in Python: an LDU Laplacian over the internal faces and
one coefficient per boundary face. Boundary rules are data (`FixedValue`, `FixedGradient`,
`ZeroGradient`), applied on the tape by gathers, products and scatter-adds over each patch's
faces. Nothing is generated or compiled. An unknown rule is an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from modelica.sim.vm import Faces, Instr, Op, ldu_amul

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from modelica.sim.mesh import Mesh


@dataclass(frozen=True, slots=True)
class FixedValue:
    """The boundary value is ``value`` (OpenFOAM ``fixedValue``)."""

    value: float


@dataclass(frozen=True, slots=True)
class FixedGradient:
    """The outward normal gradient is ``gradient`` (OpenFOAM ``fixedGradient``)."""

    gradient: float


@dataclass(frozen=True, slots=True)
class ZeroGradient:
    """No flux through the patch (OpenFOAM ``zeroGradient``)."""


Rule = FixedValue | FixedGradient | ZeroGradient


@dataclass(frozen=True)
class Laplacian:
    """``k`` times the Laplacian on a mesh, as LDU coefficients over its faces.

    For an internal face ``f`` the coefficient is ``k |S_f| / |d_f|``, with ``d_f`` from the
    owner's centre to the neighbour's. It is both ``upper[f]`` and ``lower[f]``, and it comes
    off both cells' ``diag``. ``boundary[f - internal]`` is ``k |S_f| / |d_f|`` for a boundary
    face, with ``d_f`` from the cell centre to the face centre. Orthogonal meshes only.
    """

    faces: Faces
    diag: tuple[float, ...]
    upper: tuple[float, ...]
    boundary: tuple[float, ...]
    areas: tuple[float, ...]
    """``k |S_f|`` for each boundary face, for a fixed gradient."""
    volumes: tuple[float, ...]


def laplacian(mesh: Mesh, conductivity: float) -> Laplacian:
    """The Laplacian of ``mesh`` with constant ``conductivity``, before any boundary rule."""
    faces = mesh.addressing()
    internal = len(faces.neighbour)
    centres, face_centres = mesh.cell_centres, mesh.face_centres
    magnitude: NDArray[np.float64] = np.linalg.norm(mesh.face_areas, axis=1)
    diag = np.zeros(faces.cells)
    upper: list[float] = []
    for f, (o, n) in enumerate(zip(faces.owner, faces.neighbour, strict=True)):
        gap: NDArray[np.float64] = centres[n] - centres[o]
        c = conductivity * float(magnitude[f]) / float(np.linalg.norm(gap))
        upper.append(c)
        diag[o] -= c
        diag[n] -= c
    boundary: list[float] = []
    for f, o in enumerate(mesh.owner[internal:], start=internal):
        half: NDArray[np.float64] = face_centres[f] - centres[o]
        boundary.append(conductivity * float(magnitude[f]) / float(np.linalg.norm(half)))
    return Laplacian(
        faces,
        tuple(diag.tolist()),
        tuple(upper),
        tuple(boundary),
        tuple((conductivity * magnitude[internal:]).tolist()),
        tuple(mesh.volumes.tolist()),
    )


@dataclass(frozen=True)
class Tape:
    """The instructions of ``der(T)`` and the values they read from ``p``."""

    code: tuple[Instr, ...]
    tables: tuple[int, ...]
    parameters: tuple[float, ...]
    """What ``p`` must hold: coefficients, boundary values and inverse volumes."""


def heat(  # noqa: PLR0915 - one block per boundary rule
    mesh: Mesh,
    conductivity: float,
    rules: Mapping[str, Rule],
    register: Callable[[int], int],
) -> Tape:
    """``V dT/dt = k laplacian(T)`` on ``mesh`` with ``rules`` per patch, as a tape.

    ``T`` is the state vector, one per cell; ``der(T)`` goes to ``out``. Every number is read
    from ``p`` (`Tape.parameters`), so a boundary value is a parameter a run can change. Every
    patch needs a rule.

    Raises:
        ValueError: a patch has no rule, or a rule names no patch.
    """
    missing = {p.name for p in mesh.patches} ^ set(rules)
    if missing:
        message = f"patches and rules do not match: {sorted(missing)}"
        raise ValueError(message)
    lap = laplacian(mesh, conductivity)
    n, internal = lap.faces.cells, len(lap.faces.neighbour)
    values: list[float] = []
    code: list[Instr] = []
    tables: list[int] = []

    def load(numbers: list[float]) -> int:
        first = register(len(numbers))
        for k, value in enumerate(numbers):
            code.append(Instr(Op.LOAD_P, first + k, len(values)))
            values.append(value)
        return first

    diag = load(list(lap.diag))
    upper = load(list(lap.upper))
    inverse = load([1.0 / v for v in lap.volumes])
    state = register(n)
    code.extend(Instr(Op.LOAD_Y, state + k, k) for k in range(n))
    flux = register(n)
    code.extend(ldu_amul(lap.faces, flux, state, diag, upper, upper, tables, register))
    for patch in mesh.patches:
        rule = rules[patch.name]
        rows = range(patch.start - internal, patch.start - internal + patch.count)
        owners = [mesh.owner[patch.start + k] for k in range(patch.count)]
        count = patch.count

        def scatter_add(run: int, owners: list[int] = owners, count: int = count) -> None:
            offset = len(tables)
            tables.extend(flux + c for c in owners)
            code.append(Instr(Op.VSCATTER_ADD, 0, offset, run, count))

        match rule:
            case ZeroGradient():
                continue
            case FixedValue(value=value):
                # c (T_b - T_P): the cell's own value through a gather, the wall's broadcast.
                coefficient = load([lap.boundary[r] for r in rows])
                wall = load([value])
                offset = len(tables)
                tables.extend(state + c for c in owners)
                cells = register(count)
                code.append(Instr(Op.VGATHER, cells, offset, 0, count))
                walls = register(count)
                code.append(Instr(Op.VBROADCAST, walls, wall, 0, count))
                difference = register(count)
                code.append(Instr(Op.VSUB, difference, walls, cells, count))
                product = register(count)
                code.append(Instr(Op.VMUL, product, coefficient, difference, count))
                scatter_add(product)
            case FixedGradient(gradient=gradient):
                area = load([lap.areas[r] for r in rows])
                slope = load([gradient])
                slopes = register(count)
                code.append(Instr(Op.VBROADCAST, slopes, slope, 0, count))
                product = register(count)
                code.append(Instr(Op.VMUL, product, area, slopes, count))
                scatter_add(product)
    rate = register(n)
    code.append(Instr(Op.VMUL, rate, flux, inverse, n))
    code.extend(Instr(Op.STORE_OUT, 0, k, rate + k) for k in range(n))
    return Tape(tuple(code), tuple(tables), tuple(values))
