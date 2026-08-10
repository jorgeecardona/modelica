"""A work-precision diagram: achieved error against CPU time, swept over tolerances.

`docs/ideas/benchmarks.md:8` gives the argument for this shape rather than any other. A
throughput number cannot compare a stiff solver with an explicit one -- on `Robertson` an
explicit method takes 10^7 to 10^8 steps where a stiff one takes a few hundred -- so the only
comparison that means anything is *how much time did it cost to reach how much error*.

Run it with ``moon run :bench``, or directly. It prints a table and writes the same numbers as
JSON so a later run can be diffed against an earlier one.

**The error axis needs a reference, and where the reference comes from decides what the
diagram is worth.** The Bari IVP test set publishes high-precision solutions for both problems
here, and using those would make the error axis absolute. This harness does not have them: it
computes its own reference at `REFERENCE` tolerance, with a *different* method from the ones
under test, and cross-checks two methods against each other at that tolerance before using it.
That measures agreement rather than truth. It is enough to compare three solvers on one
problem and not enough to claim conformance to a published figure, and every table it prints
says so.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten
from modelica.sim.solve import simulate

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

#: The tolerance the reference solution is computed at, and the two methods that have to
#: agree there before it is believed.
REFERENCE = 1e-12
WITNESSES = ("Radau", "LSODA")

#: What is swept. Coarse enough to be run often; the shape of a work-precision curve is
#: visible over four decades and does not need eight.
TOLERANCES = (1e-4, 1e-6, 1e-8, 1e-10)
METHODS = ("Radau", "BDF", "LSODA")

#: Stage 1 of `docs/ideas/benchmarks.md:84`: the two problems that are in a published test set
#: *and* already in this corpus, with the horizon each is swept over.
#:
#: `Robertson` runs to 1e4 rather than the test set's 1e11 -- the model has no scaling for the
#: long tail and the transient is over long before. It is cheap: 0.02 s at rtol 1e-4 and
#: 0.23 s at 1e-10.
#:
#: `VanDerPol` runs to 20 and not to its own `.toml`'s 3000, and that horizon is a measurement
#: rather than a preference. At mu = 1000 the period is about 1600, so 3000 is two relaxation
#: cycles and it costs 118 s at rtol 1e-4 alone; at 1e-10 a *tenth* of one cycle costs 265 s.
#: A sweep of three methods over four tolerances at that horizon is hours. Twenty seconds of
#: simulated time is the first sharp corner and it is 0.75 s to 16.5 s, which is a sweep
#: somebody will actually run.
PROBLEMS: dict[str, float] = {"Robertson": 1.0e4, "VanDerPol": 20.0}


def system(stem: str) -> System:
    model = EXAMPLES / "models" / f"{stem}.mo"
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    return causalize(flatten(sources, stem))


def reference(model: System, stop: float, grid: np.ndarray) -> np.ndarray:
    """The answer the error axis is measured against, and a check that it is one.

    Two methods at `REFERENCE`, compared. If they disagree by more than the loosest tolerance
    under test, the reference is not a reference and the run says so rather than quoting an
    error against a number nobody should trust.
    """
    answers = [
        sampled(
            simulate(
                model, stop=stop, times=grid, tolerance=REFERENCE, absolute=REFERENCE, method=method
            ),
            model,
            grid,
        )
        for method in WITNESSES
    ]
    spread = float(np.abs(answers[0] - answers[1]).max())
    if spread > min(TOLERANCES):
        message = (
            f"the two reference methods disagree by {spread:.2e}, which is more than the"
            f" loosest tolerance under test ({min(TOLERANCES):.0e}) -- there is no error axis"
        )
        raise SystemExit(message)
    return answers[0]


def sampled(result: Any, model: System, grid: np.ndarray) -> np.ndarray:
    """The states at the requested instants, as one array, in `System.states` order."""
    at = np.searchsorted(result.time, grid)
    return np.stack([result[name][at] for name in model.states])


def measure(
    model: System,
    stop: float,
    grid: np.ndarray,
    method: str,
    tolerance: float,
    truth: np.ndarray,
    repeats: int,
) -> dict[str, float]:
    """One point of one curve: the time it took and the error it reached.

    Best of ``repeats`` rather than the mean. A benchmark on a shared laptop measures the
    machine's worst interruptions as much as the code; the minimum is the closest thing to
    the run with nothing else happening.
    """
    times: list[float] = []
    started = time.perf_counter()
    answer = simulate(
        model, stop=stop, times=grid, tolerance=tolerance, absolute=tolerance, method=method
    )
    times.append(time.perf_counter() - started)
    for _ in range(repeats - 1):
        started = time.perf_counter()
        simulate(
            model, stop=stop, times=grid, tolerance=tolerance, absolute=tolerance, method=method
        )
        times.append(time.perf_counter() - started)
    got = sampled(answer, model, grid)
    scale = np.maximum(np.abs(truth), 1e-30)
    return {
        "tolerance": tolerance,
        "seconds": min(times),
        "relative_error": float(np.abs((got - truth) / scale).max()),
        "absolute_error": float(np.abs(got - truth).max()),
        "samples": len(answer.time),
    }


def run(repeats: int, points: int) -> dict[str, Any]:
    machine = {
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    print(f"machine   {machine['processor']}, python {machine['python']}")
    print(
        f"reference {WITNESSES[0]} and {WITNESSES[1]} at rtol = atol = {REFERENCE:.0e},"
        " computed here -- not the Bari published solution"
    )
    print(f"timing    best of {repeats}\n")

    collected: dict[str, Any] = {"machine": machine, "reference": REFERENCE, "problems": {}}
    for stem, stop in PROBLEMS.items():
        model = system(stem)
        grid = np.linspace(0.0, stop, points)
        truth = reference(model, stop, grid)
        print(f"{stem}, to t = {stop:g}, {len(model.states)} states")
        print(f"  {'method':8s} {'rtol':>8s} {'seconds':>10s} {'rel error':>11s} {'steps':>7s}")
        collected["problems"][stem] = {"stop": stop, "curves": {}}
        for method in METHODS:
            curve: list[dict[str, float]] = []
            for tolerance in TOLERANCES:
                point = measure(model, stop, grid, method, tolerance, truth, repeats)
                curve.append(point)
                print(
                    f"  {method:8s} {tolerance:8.0e} {point['seconds']:9.4f}s"
                    f" {point['relative_error']:11.2e} {point['samples']:7d}"
                )
            collected["problems"][stem]["curves"][method] = curve
        print()
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--points", type=int, default=51)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()
    collected = run(arguments.repeats, arguments.points)
    if arguments.out:
        arguments.out.write_text(json.dumps(collected, indent=2) + "\n", encoding="utf-8")
        print(f"written to {arguments.out}")


if __name__ == "__main__":
    main()
