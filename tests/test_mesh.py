"""`modelica.sim.mesh`: a box mesh, its geometry, and the OpenFOAM polyMesh files (plan step 10)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

np = pytest.importorskip("numpy")

from modelica.sim import mesh  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = (3, 2, 2)
SIZE = (3.0, 1.0, 0.5)


def test_a_box_has_the_addressing_of_its_grid() -> None:
    """Internal faces join each cell to its +x, +y and +z neighbours, owner first, in order."""
    box = mesh.box(SHAPE, SIZE)
    nx, ny, nz = SHAPE
    grid = sorted(
        (c, c + step)
        for k in range(nz)
        for j in range(ny)
        for i in range(nx)
        for c in [i + nx * (j + ny * k)]
        for step, inside in ((1, i + 1 < nx), (nx, j + 1 < ny), (nx * ny, k + 1 < nz))
        if inside
    )
    faces = box.addressing()
    pairs = list(zip(faces.owner, faces.neighbour, strict=True))
    assert pairs == grid  # already in OpenFOAM's upper-triangular order
    assert faces.cells == nx * ny * nz
    assert [p.name for p in box.patches] == ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    assert sum(p.count for p in box.patches) == 2 * (ny * nz + nx * nz + nx * ny)


def test_a_box_has_the_geometry_of_its_grid() -> None:
    """Every cell has volume ``dx dy dz`` and its centre; every cell is closed."""
    box = mesh.box(SHAPE, SIZE)
    dx, dy, dz = (s / n for s, n in zip(SIZE, SHAPE, strict=True))
    np.testing.assert_allclose(box.volumes, dx * dy * dz, rtol=1e-14)
    assert box.cell_centres[0] == pytest.approx([dx / 2, dy / 2, dz / 2], rel=1e-14)
    # Internal normals point owner to neighbour, boundary normals point out, so the area
    # vectors around each cell sum to zero.
    closed = np.zeros((box.cells, 3))
    for f, c in enumerate(box.owner):
        closed[c] += box.face_areas[f]
    for f, c in enumerate(box.neighbour):
        closed[c] -= box.face_areas[f]
    np.testing.assert_allclose(closed, 0.0, atol=1e-14)
    first = box.patches[0]  # xmin: area dy dz, pointing to -x
    np.testing.assert_allclose(box.face_areas[first.start], [-dy * dz, 0, 0], atol=1e-15)


def test_a_one_dimensional_box_is_heat_bars_faces() -> None:
    """``n`` cells in a row give faces ``i`` to ``i + 1``, as the heat bar in test_vm uses."""
    faces = mesh.box((8, 1, 1)).addressing()
    assert faces.owner == tuple(range(7))
    assert faces.neighbour == tuple(range(1, 8))


def test_a_polymesh_reads_back_as_written(tmp_path: Path) -> None:
    """`write_polymesh` then `read_polymesh` gives the same mesh."""
    box = mesh.box(SHAPE, SIZE)
    mesh.write_polymesh(box, tmp_path / "polyMesh")
    back = mesh.read_polymesh(tmp_path / "polyMesh")
    assert back == box
    np.testing.assert_array_equal(back.volumes, box.volumes)
