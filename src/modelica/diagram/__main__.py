"""Draw a model from the command line.

    python -m modelica.diagram examples/models/RLCCircuit.mo examples/library/Electrical.mo

Give the model file and the libraries it uses. The command writes SVG to the standard
output, so send it to a file or to a viewer::

    python -m modelica.diagram RLCCircuit.mo Electrical.mo > rlc.svg
    python -m modelica.diagram RLCCircuit.mo Electrical.mo --format dot | dot -Tpng > rlc.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modelica.diagram import blocks, incidence, of


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modelica.diagram",
        description="Draw a Modelica model as a graph.",
    )
    parser.add_argument("files", nargs="+", type=Path, help="the model, then its libraries")
    parser.add_argument("--model", help="the class to draw; needed when the files hold several")
    parser.add_argument(
        "--view",
        default="schematic",
        choices=["schematic", "incidence", "blocks"],
        help="schematic: components and connections. incidence: equations and variables. "
        "blocks: the order the solver uses.",
    )
    parser.add_argument(
        "--format", default="svg", choices=["svg", "dot", "mermaid"], help="the output format"
    )
    parser.add_argument("--depth", type=int, default=1, help="how many levels to open")
    parser.add_argument("-o", "--out", type=Path, help="write to this file instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command. Return 0 after success and 1 after an error."""
    # `modelica.sim` brings NumPy with it, and the schematic -- what this command is
    # asked for most -- needs neither. So the numerical half is imported when a view that
    # uses it is asked for, and a reader with no NumPy can still draw a model.
    from modelica.ir.flatten import FlattenError  # noqa: PLC0415 - see above
    from modelica.lang import ModelicaSyntaxError, parse  # noqa: PLC0415 - see above
    from modelica.sim import causalize, flatten  # noqa: PLC0415 - see above

    arguments = _parser().parse_args(argv)
    try:
        graph = of(*arguments.files, model=arguments.model, depth=arguments.depth)
        if arguments.view != "schematic":
            programs = [
                parse(path.read_text(encoding="utf-8"), str(path)) for path in arguments.files
            ]
            flat = flatten(programs, graph.name)
            graph = incidence(flat) if arguments.view == "incidence" else blocks(causalize(flat))
    except (FlattenError, ModelicaSyntaxError, OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    writers = {"svg": graph.to_svg, "dot": graph.to_dot, "mermaid": graph.to_mermaid}
    text = writers[arguments.format]()
    if arguments.out is not None:
        arguments.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
