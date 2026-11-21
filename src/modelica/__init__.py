"""Modelica toolchain for Python: parse, flatten, simulate and draw models.

The package is layered, and each layer is usable without the ones above it:

* ``modelica.lang``: Modelica source -> tokens -> AST (hand-written lexer/parser).
* ``modelica.build``: Python classes -> the same AST values. A second front end.
* ``modelica.ir``: AST -> flat model: variables, equations, an expression IR;
  then matching/BLT sorting and index reduction down to a solvable system.
* ``modelica.sim``: simulation of that system on the precompiled engine or on SciPy,
  with event (zero-crossing) handling.
* ``modelica.diagram``: any of the above to a picture: components, incidence, or blocks.
* ``modelica.fmi``: FMI 3.0: writes ``modelDescription.xml`` and holds the calling
  sequence (experimental). It does not read, run or export an FMU.
* ``modelica.ssp``: SSP 2.0: the co-simulation master's protocol (experimental). It
  does not read a ``.ssp`` archive or step components.

``modelica.run("model.mo", stop=1.0)`` reads files and returns the simulated result in one
call (`modelica.sim.run`). It is loaded on first use: ``import modelica`` pulls in no
third-party dependency, so tooling which only reads model *structure* stays cheap.

Modelica(R) is a registered trademark of the Modelica Association, as are FMI and SSP;
they are used here descriptively, to name the language this package parses and the
standards it implements. This is an independent open-source implementation, not
affiliated with or endorsed by the Modelica Association.
"""

__version__ = "0.1.78"

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelica.sim.files import run as run

__all__ = ["__version__", "run"]


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a module __getattr__ is untyped by nature
    """Load `run` on first use, so importing the package loads no NumPy."""
    if name == "run":
        from modelica.sim.files import run  # noqa: PLC0415 - the whole point of the indirection

        return run
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
