"""How far each Modelica Standard Library example gets through what a user calls: `run`.

Each example is loaded with `modelica.lang.library.closure` (only the files its names
reach), then flattened, causalized and simulated on the default backend. One row per
example records the last stage reached, the first refusal, the load time and the backend
that ran it. The table at the end groups refusals by construct (`coverage.normalise`), most
examples blocked first, which is the order to fix them in.

    uv run python scripts/reach.py              # every example, 4 worker processes
    uv run python scripts/reach.py --limit 20   # a quick sample

The MSL is the cached 4.1.0 tree that ``scripts/coverage.py`` downloads. Output:
``reach.json`` beside this script's repository root.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage import (
    CACHE,
    FLATTEN_TIMEOUT,
    SIMULATE_TIMEOUT,
    deadline,
    enumerate_examples,
    normalise,
)

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "reach.json"
WORKERS = 4


@dataclass
class Row:
    """One example's outcome."""

    example: str
    stage: str = ""
    """The last stage that succeeded: "load", "flatten", "causalize" or "simulate"."""
    error: str = ""
    load_seconds: float = 0.0
    backend: str = ""


def _first_line(error: BaseException) -> str:
    text = str(error).strip() or type(error).__name__
    return text.splitlines()[0]


def measure(qualified: str, stop: float, tolerance: float) -> Row:
    """Take one example as far as it goes; never raises."""
    from modelica.ir.causalize import causalize  # noqa: PLC0415 - imported in the worker
    from modelica.lang import parse  # noqa: PLC0415
    from modelica.lang.library import closure  # noqa: PLC0415
    from modelica.sim import flatten, simulate  # noqa: PLC0415

    row = Row(qualified)
    model = f"model ReachProbe\n  extends {qualified};\nend ReachProbe;\n"
    roots = [CACHE / "Modelica", CACHE / "ModelicaServices"]
    try:
        started = time.perf_counter()
        with deadline(FLATTEN_TIMEOUT):
            library = closure(roots, [model])
        row.load_seconds = round(time.perf_counter() - started, 2)
        row.stage = "load"
        with deadline(FLATTEN_TIMEOUT):
            flat = flatten([library, parse(model)], "ReachProbe")
        row.stage = "flatten"
        with deadline(FLATTEN_TIMEOUT):
            system = causalize(flat)
        row.stage = "causalize"
        with deadline(SIMULATE_TIMEOUT):
            result = simulate(system, stop=stop, tolerance=tolerance)
        row.stage = "simulate"
        row.backend = getattr(result, "backend", "")
    except BaseException as error:  # noqa: BLE001 - every refusal is a row, not a crash
        row.error = _first_line(error)
    return row


def main() -> None:
    """Measure, write ``reach.json``, print the stage counts and the blockers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="measure only the first N examples")
    args = parser.parse_args()
    examples = enumerate_examples(CACHE)[: args.limit]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(
            pool.map(
                measure,
                [e.qualified for e in examples],
                [e.stop for e in examples],
                [e.tolerance for e in examples],
            )
        )
    OUTPUT.write_text(json.dumps([asdict(r) for r in rows], indent=1))
    stages = Counter(r.stage or "none" for r in rows)
    print(f"{len(rows)} examples")
    for stage in ("none", "load", "flatten", "causalize", "simulate"):
        print(f"  last stage {stage:10s} {stages.get(stage, 0)}")
    loads = sorted(r.load_seconds for r in rows if r.stage)
    if loads:
        print(f"  median load {loads[len(loads) // 2]:.2f} s")
    print(f"  backends {dict(Counter(r.backend for r in rows if r.backend))}")
    blockers = Counter(normalise(r.error.split("; read because")[0]) for r in rows if r.error)
    print("blockers, most examples first:")
    for reason, count in blockers.most_common(15):
        print(f"  {count:4d}  {reason[:110]}")


if __name__ == "__main__":
    main()
