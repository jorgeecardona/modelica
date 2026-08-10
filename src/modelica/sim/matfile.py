"""The Dymola/OpenModelica result format, written and read.

One `Result` is readable only by this package. The `.mat` this module writes is readable by
ModelicaRes, DyMat and BuildingsPy -- every plotting script, regression harness and analysis
notebook that already exists in this field -- for one emitter and no new dependency.

The format is a MATLAB level-4 file with five matrices, and the shapes are not obvious:

``Aclass``      four rows: ``Atrajectory``, ``1.1``, empty, ``binTrans``. The last one is the
                whole convention -- **binTrans** means every matrix below is stored
                transposed, so a name is a *column* of characters and a trajectory is a
                *row* of numbers.
``name``        characters by variables. Read by joining each column.
``description`` the same shape, the same way.
``dataInfo``    four rows by variables. Row 0 is which data block the variable is in, row 1
                is its **one-based, signed** index within that block.
``data_1``      the constants. Row 0 is ``[t_start, t_stop]``; every other row is one
                parameter's value, written twice so it plots as a flat line.
``data_2``      the trajectories. Row 0 is time; every other row is one variable.

**The sign in ``dataInfo`` row 1 is the part worth reading twice.** A variable stored as the
negative of another is packed by pointing at that other one's row with a negative index --
which is how a flat model's ``a = -b`` aliases avoid being stored twice. Nothing in a round
trip through this module alone would catch getting it backwards, so `tests/test_matfile.py`
checks it against DyMat, a reader this project did not write.

``scipy.io`` does the container. It is already available under the ``sim`` extra, where the
numerics live, and hand-rolling a level-4 writer to avoid it would be re-implementing the one
part that is not Modelica-specific.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from modelica.sim.result import Result

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["read", "write"]

_ACLASS = ("Atrajectory", "1.1", "", "binTrans")

_CONSTANTS = 1
"""The data block holding values that do not move. ``data_1``."""

_TRAJECTORIES = 2
"""The data block holding values that do. ``data_2``."""


def _characters(strings: list[str], *, down: bool) -> np.ndarray:
    """A list of strings as a character matrix, padded to the longest.

    ``down`` says which way, and it is not a detail: ``Aclass`` is read row by row while
    ``name`` and ``description`` are read *column* by column -- in the same file, by the same
    reader. Writing both the natural way round produces a header that reads ``Atra``,
    ``ject``, ``ory1``, and a file no tool opens. Measured against DyMat, which answered
    "File structure not supported!".
    """
    width = max((len(text) for text in strings), default=1)
    grid = np.array([list(text.ljust(width)) for text in strings], dtype="U1")
    # `ascontiguousarray` and not a bare `.T`, and this one is a trap worth naming.
    # `scipy.io.savemat` writes the array's *buffer*, so a transposed view goes to disk in the
    # original order under a header claiming the new shape -- a file that opens, reports the
    # right number of variables, and names them things like `Te  v  .`. Nothing about it looks
    # like a bug until a reader written by somebody else prints the names.
    return np.ascontiguousarray(grid.T) if down else grid


def write(result: Result, path: str | PathLike[str]) -> None:
    """Write ``result`` as a Dymola-format ``.mat``.

    Every variable becomes a trajectory, including the ones that never move. Sorting the
    constants into ``data_1`` is what a Modelica tool does and it saves space, but it also
    decides -- from this run's samples -- that a parameter is a parameter, and a variable
    that happened to be flat over one interval is not one. This writes the honest thing and
    leaves ``data_1`` holding only the time range.
    """
    # `scipy.io` is untyped, so everything crossing this boundary is `Any` and saying so once
    # is more honest than an ignore on each line it touches.
    from scipy.io import (  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        savemat,  # pyright: ignore[reportUnknownVariableType]
    )

    names = ["Time", *result.names]
    descriptions = ["Time in [s]", *("" for _ in result.names)]
    info = np.zeros((4, len(names)), dtype=np.int32)
    info[0, 0], info[1, 0] = 0, 1  # the abscissa: block 0, and column 1 means "this is time"
    info[0, 1:] = _TRAJECTORIES
    info[1, 1:] = np.arange(2, len(names) + 1)  # one-based, and row 1 of `data_2` is time
    info[3, :] = -1
    savemat(
        path,
        {
            "Aclass": _characters(list(_ACLASS), down=False),
            "name": _characters(names, down=True),
            "description": _characters(descriptions, down=True),
            "dataInfo": info,
            "data_1": np.array([[float(result.time[0]), float(result.time[-1])]], dtype=np.float64),
            "data_2": np.vstack([result.time, result.values]).astype(np.float64),
        },
        format="4",
    )


def read(path: str | PathLike[str], name: str = "") -> Result:
    """Read a Dymola-format ``.mat`` back into a `Result`.

    Written to read what *other* tools write, not only what `write` writes: the sign and the
    one-based index of ``dataInfo`` are honoured, so a variable packed as the negative of
    another comes back negated, and a constant in ``data_1`` comes back as a flat trajectory
    over the run's own time range rather than as two points.
    """
    from scipy.io import (  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        loadmat,  # pyright: ignore[reportUnknownVariableType]
    )

    raw = cast("dict[str, Any]", loadmat(path, chars_as_strings=False))
    if "Aclass" not in raw:
        message = f"{path}: no `Aclass`, so this is not a Dymola result file"
        raise ValueError(message)
    header: list[str] = ["".join(row).rstrip() for row in raw["Aclass"]]
    if header[3] != "binTrans":
        message = f"{path}: only the `binTrans` layout is read, which is what every tool writes"
        raise ValueError(message)

    labels: list[str] = ["".join(column).rstrip() for column in zip(*raw["name"], strict=True)]
    info: Any = raw["dataInfo"]
    blocks: dict[int, Any] = {
        _CONSTANTS: raw.get("data_1"),
        _TRAJECTORIES: raw.get("data_2"),
    }
    time: np.ndarray = np.asarray(blocks[_TRAJECTORIES])[0]

    names: list[str] = []
    rows: list[np.ndarray] = []
    for index, label in enumerate(labels):
        block, signed = int(info[0][index]), int(info[1][index])
        column = abs(signed) - 1
        if column == 0:
            continue  # the abscissa, which comes back as `Result.time`
        data = blocks.get(block)
        if data is None:
            message = f"{path}: {label!r} points at data_{block}, which the file does not have"
            raise ValueError(message)
        values: np.ndarray = np.asarray(data)[column]
        # A constant is two points spanning the run. It becomes a flat trajectory, because a
        # `Result` has one time axis and a caller asking for a parameter wants an array they
        # can plot beside everything else.
        if block == _CONSTANTS:
            values = np.full(len(time), float(values[0]))
        names.append(label)
        rows.append(-values if signed < 0 else values)

    return Result(
        name=name or str(path),
        time=np.asarray(time, dtype=float),
        names=tuple(names),
        values=np.vstack(rows).astype(float) if rows else np.empty((0, len(time))),
    )
