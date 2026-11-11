"""Wall time of each front-end stage on a 3-D heat box, and the per-cell calls ``vm.lower`` makes.

Run it directly: ``uv run python scripts/frontend_bench.py 13 20``. It prints one line per size.
The goal (docs/issues/11-11-2026-an-explicit-loop-stays-one-array-node-from-flatten-to-the-tape.md)
is that the front end costs less than integration, and that the ``at_cell`` and ``indices_at``
calls during ``lower`` do not grow with the number of cells.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelica.lang import parse
from modelica.sim import causalize, flatten, simulate, vm

if TYPE_CHECKING:
    from collections.abc import Callable

    from modelica.ir.system import System

BOX = """
model Box "Heat in an insulated cube: a neighbour past a face is the cell itself"
  parameter Integer n = {n};
  parameter Real a = 2.5 * n ^ 2;
  Real T[n, n, n];
initial equation
  for i in 1:n loop
    for j in 1:n loop
      for k in 1:n loop
        T[i, j, k] = i + 0.5 * j - 0.25 * k;
      end for;
    end for;
  end for;
equation
  for i in 1:n loop
    for j in 1:n loop
      for k in 1:n loop
        der(T[i, j, k]) = a * (T[max(i - 1, 1), j, k] + T[min(i + 1, n), j, k]
          + T[i, max(j - 1, 1), k] + T[i, min(j + 1, n), k]
          + T[i, j, max(k - 1, 1)] + T[i, j, min(k + 1, n)] - 6 * T[i, j, k]);
      end for;
    end for;
  end for;
end Box;
"""


@dataclass(frozen=True)
class Stages:
    """Wall seconds per stage for one box, and the per-cell calls ``lower`` made.

    ``simulate`` lowers again, so it includes a second ``lower``.
    """

    n: int
    flatten: float
    causalize: float
    lower: float
    simulate: float
    cell_calls_in_lower: int


def lower_counting(system: System) -> tuple[vm.Program, int]:
    """``vm.lower(system)`` and the number of ``at_cell`` and ``indices_at`` calls it made."""
    names = ("at_cell", "indices_at")
    originals: dict[str, Callable[..., Any]] = {name: getattr(vm, name) for name in names}
    calls = 0

    def counted(original: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        return call

    for name, original in originals.items():
        setattr(vm, name, counted(original))
    try:
        program = vm.lower(system)
    finally:
        for name, original in originals.items():
            setattr(vm, name, original)
    return program, calls


def measure(n: int, *, stop: float = 0.01, run: bool = True) -> Stages:
    """Time each stage on the box with ``n`` cells per side (``n ** 3`` states)."""
    t0 = time.perf_counter()
    flat = flatten([parse(BOX.format(n=n), "Box.mo")], "Box")
    t1 = time.perf_counter()
    system = causalize(flat)
    t2 = time.perf_counter()
    _, calls = lower_counting(system)
    t3 = time.perf_counter()
    if run:
        simulate(system, stop=stop, backend="engine")
    t4 = time.perf_counter()
    return Stages(n, t1 - t0, t2 - t1, t3 - t2, t4 - t3, calls)


def main() -> None:
    """Print one line per box size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes", nargs="*", type=int, default=[13, 20])
    parser.add_argument("--stop", type=float, default=0.01)
    args = parser.parse_args()
    for n in args.sizes:
        s = measure(n, stop=args.stop)
        print(
            f"n={s.n:3d} states={s.n**3:6d} flatten={s.flatten:6.2f}s"
            f" causalize={s.causalize:6.2f}s lower={s.lower:6.2f}s"
            f" simulate={s.simulate:6.2f}s cell calls in lower={s.cell_calls_in_lower}"
        )


if __name__ == "__main__":
    main()
