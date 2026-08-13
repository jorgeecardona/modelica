"""Pure-Python Modelica toolchain: parse, flatten, simulate, exchange.

The package is layered, and each layer is usable without the ones above it:

* ``modelica.lang``  -- Modelica source -> tokens -> AST (hand-written lexer/parser).
* ``modelica.ir``    -- AST -> flat model: variables, equations, an expression IR;
  then matching/BLT sorting and index reduction down to a solvable system.
* ``modelica.sim``   -- the flat model as an ODE/DAE problem plus solver backends and
  event (zero-crossing) handling.
* ``modelica.fmi``   -- FMI 3.0: read/simulate an existing FMU, and export a flat model
  as one.
* ``modelica.ssp``   -- SSP 2.0: system composition (.ssd/.ssv/.ssb inside a .ssp) and a
  co-simulation master over the components it names.

``import modelica`` pulls in no third-party dependency; numerics live behind the ``sim``
extra so that tooling which only reads model *structure* stays cheap to install.

Modelica(R) is a registered trademark of the Modelica Association, as are FMI and SSP;
they are used here descriptively, to name the language this package parses and the
standards it implements. This is an independent open-source implementation, not
affiliated with or endorsed by the Modelica Association.
"""

__version__ = "0.0.85"

__all__ = ["__version__"]
