"""What a simulation hands back.

A trajectory, indexed by the names the modeller wrote -- ``result["c.v"]``, not
``result.y[3]``. Every variable of the flat model is in here, aliases included, because
the whole point of keeping the alias table through causalization was so that a user who
asks for ``r.p.v`` gets a number rather than an explanation of why it was eliminated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from modelica.sim.view import View, view

if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike

    from numpy.typing import NDArray

__all__ = ["Event", "Result"]


@dataclass(frozen=True)
class Event:
    """A moment where the model changed discontinuously."""

    time: float
    index: int
    """Which crossing fired, as an index into the emitted indicators."""

    cause: str
    """What crossed, in Modelica, for reading: a ``when`` branch or a relation."""

    def __str__(self) -> str:
        return f"t = {self.time:.12g}: {self.cause}"


@dataclass(frozen=True)
class Result:
    """Time, one row per variable, and the events that interrupted the integration."""

    name: str
    time: NDArray[np.float64]
    names: tuple[str, ...]
    values: NDArray[np.float64]
    """Shape ``(len(names), len(time))`` -- one row per variable, so a row is a slice."""

    events: tuple[Event, ...] = field(default_factory=tuple[Event, ...])

    def write(self, path: str | PathLike[str]) -> None:
        """Write this run as a Dymola-format ``.mat``, which is what the field reads.

        ModelicaRes, DyMat and BuildingsPy all open it. See `modelica.sim.matfile` for the
        format and for the two ways of getting it subtly wrong.
        """
        from modelica.sim.matfile import write  # noqa: PLC0415 - `scipy.io` stays optional

        write(self, path)

    @cached_property
    def _row(self) -> dict[str, int]:
        return {name: row for row, name in enumerate(self.names)}

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        try:
            return self.values[self._row[name]]
        except KeyError:
            raise KeyError(f"{self.name} has no variable {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._row

    def __len__(self) -> int:
        """The number of samples, not the number of variables."""
        return len(self.time)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def at(self, t: float) -> dict[str, float]:
        """Every variable at time ``t``, interpolated linearly between samples.

        Linear interpolation, not the integrator's dense output: this is for reading a
        trajectory, and a caller who needs the solver's own accuracy between steps should
        ask for that time as an output point instead.
        """

        def sample(row: int) -> float:
            series: NDArray[np.float64] = self.values[row]
            return float(np.interp(t, self.time, series))

        return {name: sample(row) for name, row in self._row.items()}

    def plot(self, *names: str, **options: int) -> View:
        """An interactive chart of ``names`` -- of everything, if you name nothing.

        The result is a :class:`~modelica.sim.view.View`, which draws itself in a notebook
        and can be written to a self-contained HTML file with
        :meth:`~modelica.sim.view.View.save`. ``height``, ``limit`` and ``initial`` are
        forwarded to it.
        """
        return view(self, names, **options)

    def _repr_html_(self) -> str:
        """A notebook showing a result should show the trajectory, not the shape of it."""
        return view(self).embed()

    def __repr__(self) -> str:
        return (
            f"Result({self.name!r}, {len(self.time)} samples, "
            f"{len(self.names)} variables, {len(self.events)} events)"
        )
