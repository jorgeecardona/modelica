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
import importlib
import json
import platform
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten, radau
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
SCIPY = ("Radau", "BDF", "LSODA")

#: The fused loop is `modelica.sim.radau`: the same Radau IIA method as the first of those,
#: with the step compiled together with the model instead of calling into it from Python. It
#: is on this diagram and not on a table of its own because that is the only comparison that
#: settles anything -- it is a *different implementation of the same method*, so if fusion is
#: worth having it shows up as the same curve shifted left, and if it is not, the two curves
#: land on top of each other.
#:
#: numba is in no dependency group (`pyproject.toml` says why), so the column appears when it
#: is importable and the table says so when it is not. `moon run :bench-fused` installs it.
FUSED = "fused"


def available() -> tuple[str, ...]:
    """The methods this environment can actually run, which is not a constant."""
    try:
        importlib.import_module("numba")
    except ImportError:
        return SCIPY
    return (*SCIPY, FUSED)


METHODS = available()

#: A reference has to be this many times tighter than the tightest point it is the reference
#: *for*. Ten is modest and it is the rule that decides how deep each sweep below can go.
MARGIN = 10.0

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
#: How deep each sweep goes is measured, not chosen, and the two problems differ. A reference
#: computed here agrees with itself to 4.7e-11 relative on `Robertson` and only 1.6e-8 on
#: `VanDerPol` -- whose states are order 2 and 4.5, so that is not a scaling artefact. Divided
#: by `MARGIN` that supports 1e-8 and 1e-6 respectively, and no further. Anything deeper on
#: either needs the Bari published values.
PROBLEMS: dict[str, tuple[float, tuple[float, ...]]] = {
    "Robertson": (1.0e4, (1e-4, 1e-6, 1e-8)),
    "VanDerPol": (20.0, (1e-4, 1e-5, 1e-6)),
    # Stage 2 of `docs/ideas/benchmarks.md`. HIRES is *mildly* stiff -- three decades of
    # eigenvalue spread against Robertson's nine -- so it is where an explicit method is still
    # in the race, which is the comparison neither stage-1 problem offers. Its horizon is the
    # one the Bari test set uses, and unlike the other two its end point has a published
    # answer, so this row's error axis can be checked against somebody else's arithmetic.
    "HIRES": (321.8122, (1e-4, 1e-6, 1e-8)),
}


def system(stem: str) -> System:
    model = EXAMPLES / "models" / f"{stem}.mo"
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    return causalize(flatten(sources, stem))


def reference(
    model: System, stop: float, grid: np.ndarray, tolerances: tuple[float, ...]
) -> np.ndarray:
    """The answer the error axis is measured against, and a check that it is one.

    Two methods at `REFERENCE`, compared *relative to the size of the states*, and against the
    **tightest** tolerance the sweep will ask for divided by `MARGIN`. The first version of
    this compared an absolute difference against the loosest tolerance, which is wrong twice:
    a reference has to be better than the best point it grades, not than the worst, and a
    difference of 7e-8 means nothing until you know whether the state is 1 or 1e6.
    """
    answers = [
        sampled(once(model, stop, grid, method, REFERENCE)[0], model, grid) for method in WITNESSES
    ]
    scale = np.maximum(np.abs(answers[0]), 1.0)
    spread = float(np.abs((answers[0] - answers[1]) / scale).max())
    allowed = min(tolerances) / MARGIN
    if spread > allowed:
        message = (
            f"{model.name}: the two reference methods disagree by {spread:.2e} relative,"
            f" and the tightest point asked for is {min(tolerances):.0e} -- a reference has to"
            f" be {MARGIN:.0f}x better than that ({allowed:.1e}). Loosen the sweep or bring a"
            " published reference."
        )
        raise SystemExit(message)
    return answers[0]


def sampled(result: Any, model: System, grid: np.ndarray) -> np.ndarray:
    """The states at the requested instants, as one array, in `System.states` order."""
    at = np.searchsorted(result.time, grid)
    return np.stack([result[name][at] for name in model.states])


def once(
    model: System, stop: float, grid: np.ndarray, method: str, tolerance: float
) -> tuple[Any, int]:
    """One run, by whichever path the method names, and how many steps it took.

    The step count is exact for the fused loop, which counts them, and approximate for the
    SciPy path, where the only thing available is the length of the returned trajectory --
    the union of the solver's own steps with the requested output grid. At 51 output points
    against hundreds of steps the union is the steps and the column is honest; at a coarse
    tolerance where a method takes fewer steps than there are output points, it is not, and
    that is why the fused column is the one that reports the real number.
    """
    if method == FUSED:
        answer, stats = radau.simulate(
            model, stop=stop, times=grid, tolerance=tolerance, absolute=tolerance, jit=True
        )
        return answer, stats.steps
    result = simulate(
        model, stop=stop, times=grid, tolerance=tolerance, absolute=tolerance, method=method
    )
    return result, len(result.time)


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

    The minimum is also what keeps the fused column honest in the other direction. Its first
    call compiles -- tens of seconds -- and every call after it does not, so the mean of three
    would report a compile that a real run pays once per process and never inside a sweep.
    The compile is a real cost and it is reported separately rather than smeared into the
    per-run number.
    """
    times: list[float] = []
    started = time.perf_counter()
    answer, steps = once(model, stop, grid, method, tolerance)
    times.append(time.perf_counter() - started)
    for _ in range(repeats - 1):
        started = time.perf_counter()
        once(model, stop, grid, method, tolerance)
        times.append(time.perf_counter() - started)
    got = sampled(answer, model, grid)
    scale = np.maximum(np.abs(truth), 1e-30)
    return {
        "tolerance": tolerance,
        "seconds": min(times),
        "first": times[0],
        "relative_error": float(np.abs((got - truth) / scale).max()),
        "absolute_error": float(np.abs(got - truth).max()),
        "steps": steps,
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
    print(f"timing    best of {repeats}")
    if FUSED not in METHODS:
        print("fused     not measured -- numba is absent; run `moon run :bench-fused`")
    print()

    collected: dict[str, Any] = {"machine": machine, "reference": REFERENCE, "problems": {}}
    for stem, (stop, tolerances) in PROBLEMS.items():
        model = system(stem)
        grid = np.linspace(0.0, stop, points)
        truth = reference(model, stop, grid, tolerances)
        print(f"{stem}, to t = {stop:g}, {len(model.states)} states")
        print(f"  {'method':8s} {'rtol':>8s} {'seconds':>10s} {'rel error':>11s} {'steps':>7s}")
        collected["problems"][stem] = {"stop": stop, "curves": {}}
        for method in METHODS:
            curve: list[dict[str, float]] = []
            for tolerance in tolerances:
                point = measure(model, stop, grid, method, tolerance, truth, repeats)
                curve.append(point)
                print(
                    f"  {method:8s} {tolerance:8.0e} {point['seconds']:9.4f}s"
                    f" {point['relative_error']:11.2e} {point['steps']:7.0f}"
                    + (
                        f"   (first call {point['first']:.1f}s: compiling)"
                        if point["first"] > 2.0 * point["seconds"]
                        else ""
                    )
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
