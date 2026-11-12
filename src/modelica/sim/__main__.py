"""Simulate a model from the command line.

    python -m modelica.sim examples/models/RLCCircuit.mo examples/library/Electrical.mo --stop 0.01

Give the model file and the libraries it uses. Without ``-o`` the trajectory goes to the
standard output as CSV (a ``time`` column, then one column per variable)::

    python -m modelica.sim Decay.mo --stop 1 --set k=2 -o decay.mat
    python -m modelica.sim Decay.mo --points 11 > decay.csv

``--stop`` defaults to the model's ``experiment(StopTime=...)`` annotation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np

from modelica.sim.files import run

if TYPE_CHECKING:
    from modelica.sim.result import Result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modelica.sim", description="Simulate a Modelica model."
    )
    parser.add_argument("files", nargs="+", type=Path, help="the model, then its libraries")
    parser.add_argument("--model", help="the class to simulate; needed when the files hold several")
    parser.add_argument("--stop", type=float, help="end time; default: experiment(StopTime)")
    parser.add_argument("--points", type=int, help="output points, evenly spaced")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a parameter; repeat for several",
    )
    parser.add_argument("-o", "--output", type=Path, help="a .mat or .csv file; default: CSV out")
    return parser


def _parameters(pairs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            message = f"--set needs NAME=VALUE, got {pair!r}"
            raise SystemExit(message)
        out[name.strip()] = float(value)
    return out


def _csv(result: Result, stream: TextIO) -> None:
    table = np.vstack([result.time, result.values]).T
    header = ",".join(("time", *result.names))
    np.savetxt(stream, table, delimiter=",", header=header, comments="")


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv``, simulate, and write the result."""
    args = _parser().parse_args(argv)
    options: dict[str, Any] = {}
    if args.points is not None:
        options["points"] = args.points
    result = run(
        *args.files,
        model=args.model,
        stop=args.stop,
        parameters=_parameters(args.set) or None,
        **options,
    )
    if args.output is None:
        _csv(result, sys.stdout)
    elif args.output.suffix == ".mat":
        result.write(args.output)
    else:
        with args.output.open("w", encoding="utf-8") as stream:
            _csv(result, stream)


if __name__ == "__main__":
    main()
