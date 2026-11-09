"""Finite-volume meshes as data: points, faces, ``owner`` and ``neighbour``, boundary patches.

The layout is OpenFOAM's ``polyMesh``. Internal faces come first, ordered by owner and then by
neighbour, each with ``owner < neighbour`` and its normal pointing from owner to neighbour.
Boundary faces follow, grouped by patch, with outward normals. `Mesh.faces` gives the
addressing `modelica.sim.vm.ldu_amul` reads. Nothing here is compiled: a program ships these
arrays (plan step 10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from modelica.sim.vm import Faces

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Patch:
    """A named run of boundary faces: ``count`` faces from face ``start``."""

    name: str
    start: int
    count: int


@dataclass(frozen=True)
class Mesh:
    """A polyhedral mesh in OpenFOAM's ``polyMesh`` layout."""

    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    """Each face as its point indices, in order around it."""
    owner: tuple[int, ...]
    """The cell on the side the face normal points away from; one per face."""
    neighbour: tuple[int, ...]
    """The cell the normal points into; one per internal face."""
    patches: tuple[Patch, ...]

    @property
    def cells(self) -> int:
        """The number of cells: one past the largest cell any face names."""
        return max(*self.owner, *self.neighbour, -1) + 1

    def addressing(self) -> Faces:
        """The internal faces as `vm.Faces`, for `vm.ldu_amul`."""
        internal = len(self.neighbour)
        return Faces(self.cells, self.owner[:internal], self.neighbour)

    @cached_property
    def _geometry(self) -> tuple[NDArray[np.float64], ...]:
        points = np.asarray(self.points, dtype=float)
        centres = np.empty((len(self.faces), 3))
        areas = np.empty((len(self.faces), 3))
        for f, face in enumerate(self.faces):
            centres[f], areas[f] = _face(points[list(face)])
        guess = np.zeros((self.cells, 3))
        count = np.zeros(self.cells)
        sides = [(self.owner, 1.0), (self.neighbour, -1.0)]
        for cells, _ in sides:
            for f, c in enumerate(cells):
                guess[c] += centres[f]
                count[c] += 1
        guess /= count[:, None]
        volumes = np.zeros(self.cells)
        moments = np.zeros((self.cells, 3))
        for cells, sign in sides:
            for f, c in enumerate(cells):
                # The pyramid from the face to the estimated centre, as OpenFOAM's
                # primitiveMeshCellCentresAndVols builds it.
                pyramid = sign * float(areas[f] @ (centres[f] - guess[c])) / 3.0
                volumes[c] += pyramid
                moments[c] += pyramid * (0.75 * centres[f] + 0.25 * guess[c])
        return centres, areas, volumes, moments / volumes[:, None]

    @property
    def face_centres(self) -> NDArray[np.float64]:
        """The centroid of each face, shape ``(faces, 3)``."""
        return self._geometry[0]

    @property
    def face_areas(self) -> NDArray[np.float64]:
        """Each face's area vector (normal times area, owner to neighbour or outward)."""
        return self._geometry[1]

    @property
    def volumes(self) -> NDArray[np.float64]:
        """The volume of each cell."""
        return self._geometry[2]

    @property
    def cell_centres(self) -> NDArray[np.float64]:
        """The centroid of each cell, shape ``(cells, 3)``."""
        return self._geometry[3]


def _face(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A polygon's centroid and area vector, from triangles fanned around its vertex mean."""
    middle = points.mean(axis=0)
    following = np.roll(points, -1, axis=0)
    triangles = np.cross(points - middle, following - middle) / 2.0
    weights = np.linalg.norm(triangles, axis=1)
    centroids = (points + following + middle) / 3.0
    total = weights.sum()
    centre = (weights[:, None] * centroids).sum(axis=0) / total if total > 0 else middle
    return centre, triangles.sum(axis=0)


def box(  # noqa: C901 - one loop per side of the box
    shape: tuple[int, int, int], size: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Mesh:
    """A structured box of ``nx * ny * nz`` equal cells, from the origin to ``size``.

    Cell ``(i, j, k)`` is number ``i + nx * (j + ny * k)``. The patches are ``xmin``,
    ``xmax``, ``ymin``, ``ymax``, ``zmin`` and ``zmax``. A 2-D problem uses ``nz = 1``.
    """
    nx, ny, nz = shape
    dx, dy, dz = (s / n for s, n in zip(size, shape, strict=True))

    def point(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    def cell(i: int, j: int, k: int) -> int:
        return i + nx * (j + ny * k)

    # Each face normal is +x, +y or +z; the vertex order makes it so by the right-hand rule.
    def plus_x(i: int, j: int, k: int) -> tuple[int, ...]:
        return (point(i, j, k), point(i, j + 1, k), point(i, j + 1, k + 1), point(i, j, k + 1))

    def plus_y(i: int, j: int, k: int) -> tuple[int, ...]:
        return (point(i, j, k), point(i, j, k + 1), point(i + 1, j, k + 1), point(i + 1, j, k))

    def plus_z(i: int, j: int, k: int) -> tuple[int, ...]:
        return (point(i, j, k), point(i + 1, j, k), point(i + 1, j + 1, k), point(i, j + 1, k))

    points = tuple(
        (i * dx, j * dy, k * dz)
        for k in range(nz + 1)
        for j in range(ny + 1)
        for i in range(nx + 1)
    )
    faces: list[tuple[int, ...]] = []
    owner: list[int] = []
    neighbour: list[int] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Neighbours in increasing number: +x, then +y, then +z.
                for inside, face, other in (
                    (i + 1 < nx, plus_x(i + 1, j, k), (i + 1, j, k)),
                    (j + 1 < ny, plus_y(i, j + 1, k), (i, j + 1, k)),
                    (k + 1 < nz, plus_z(i, j, k + 1), (i, j, k + 1)),
                ):
                    if inside:
                        faces.append(face)
                        owner.append(cell(i, j, k))
                        neighbour.append(cell(*other))
    patches: list[Patch] = []
    sides = (
        ("xmin", [(plus_x(0, j, k)[::-1], cell(0, j, k)) for k in range(nz) for j in range(ny)]),
        ("xmax", [(plus_x(nx, j, k), cell(nx - 1, j, k)) for k in range(nz) for j in range(ny)]),
        ("ymin", [(plus_y(i, 0, k)[::-1], cell(i, 0, k)) for k in range(nz) for i in range(nx)]),
        ("ymax", [(plus_y(i, ny, k), cell(i, ny - 1, k)) for k in range(nz) for i in range(nx)]),
        ("zmin", [(plus_z(i, j, 0)[::-1], cell(i, j, 0)) for j in range(ny) for i in range(nx)]),
        ("zmax", [(plus_z(i, j, nz), cell(i, j, nz - 1)) for j in range(ny) for i in range(nx)]),
    )
    for name, entries in sides:
        patches.append(Patch(name, len(faces), len(entries)))
        for face, c in entries:
            faces.append(face)
            owner.append(c)
    return Mesh(points, tuple(faces), tuple(owner), tuple(neighbour), tuple(patches))


# --------------------------------------------------------------------------------------
# OpenFOAM polyMesh files (ASCII)
# --------------------------------------------------------------------------------------

_HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {name};
}}

"""


def write_polymesh(mesh: Mesh, directory: Path) -> None:
    """Write ``mesh`` as the ASCII files of an OpenFOAM ``constant/polyMesh`` directory."""
    directory.mkdir(parents=True, exist_ok=True)

    def write(name: str, cls: str, rows: list[str]) -> None:
        body = f"{len(rows)}\n(\n" + "".join(f"{row}\n" for row in rows) + ")\n"
        text = _HEADER.format(cls=cls, name=name) + body
        (directory / name).write_text(text, encoding="utf-8")

    write("points", "vectorField", [f"({x!r} {y!r} {z!r})" for x, y, z in mesh.points])
    write("faces", "faceList", [f"{len(f)}({' '.join(map(str, f))})" for f in mesh.faces])
    write("owner", "labelList", [str(c) for c in mesh.owner])
    write("neighbour", "labelList", [str(c) for c in mesh.neighbour])
    write(
        "boundary",
        "polyBoundaryMesh",
        [
            f"{p.name}\n{{\n    type patch;\n    nFaces {p.count};\n    startFace {p.start};\n}}"
            for p in mesh.patches
        ],
    )


def _body(path: Path) -> str:
    """The list in an OpenFOAM file: the text from the first ``(`` after the header."""
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    after = text[text.index("}", text.index("FoamFile")) + 1 :]
    return after[after.index("(") + 1 : after.rindex(")")]


def read_polymesh(directory: Path) -> Mesh:
    """Read the ASCII ``points``, ``faces``, ``owner``, ``neighbour`` and ``boundary`` files.

    Raises `ValueError` for a binary file or a face list in compact form.
    """
    number = r"[-+0-9.eE]+"
    points = tuple(
        (float(x), float(y), float(z))
        for x, y, z in re.findall(
            rf"\(\s*({number})\s+({number})\s+({number})\s*\)", _body(directory / "points")
        )
    )
    faces = tuple(
        tuple(int(v) for v in labels.split())
        for _, labels in re.findall(r"(\d+)\s*\(([\d\s]*)\)", _body(directory / "faces"))
    )
    owner = tuple(int(v) for v in _body(directory / "owner").split())
    neighbour = tuple(int(v) for v in _body(directory / "neighbour").split())
    patches = tuple(
        Patch(name, int(start), int(count))
        for name, fields in re.findall(r"(\w+)\s*\{([^}]*)\}", _body(directory / "boundary"))
        for count in re.findall(r"nFaces\s+(\d+)", fields)
        for start in re.findall(r"startFace\s+(\d+)", fields)
    )
    if len(owner) != len(faces) or not points:
        message = f"{directory}: not an ASCII polyMesh with one owner per face"
        raise ValueError(message)
    return Mesh(points, faces, owner, neighbour, patches)
