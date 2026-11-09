# modelica

[![pipeline status](https://gitlab.com/cardona-oss/pymodelica/badges/main/pipeline.svg)](https://gitlab.com/cardona-oss/pymodelica/-/pipelines)
[![PyPI version](https://img.shields.io/pypi/v/modelica.svg)](https://pypi.org/project/modelica/)
[![Python versions](https://img.shields.io/pypi/pyversions/modelica.svg)](https://pypi.org/project/modelica/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://gitlab.com/cardona-oss/pymodelica/-/blob/main/LICENSE)

A Modelica toolchain for Python: parse `.mo` source, flatten it to a DAE, simulate it, and
draw it, with **FMI 3.0** and **SSP 2.0** support. Every stage up to simulation is an
ordinary Python value you can read and change. For engineers and researchers who want
Modelica semantics inside Python rather than behind a separate compiler.

Simulation runs on a precompiled engine: a virtual machine written in Rust that ships in the
platform wheels (Linux x86_64 and arm64, Windows, macOS arm64) and interprets the model as
bytecode. Nothing is compiled per model. Where no engine wheel exists, the pure-Python wheel
simulates with SciPy instead. See [the engine](https://gitlab.com/cardona-oss/pymodelica/-/blob/main/docs/engine.md).

> **Install name is `modelica`.** The git repository keeps its older name, `pymodelica`,
> because PyPI's `pymodelica` holds an abandoned Modelon/JModelica.org upload. The import
> package and the distribution are both `modelica`.

## Why modelica

Python has the two ends of a Modelica toolchain and little of the middle.
[FMPy](https://github.com/CATIA-Systems/FMPy) and
[PyFMI](https://github.com/modelon-community/PyFMI) *run* an FMU that some other tool
compiled. The tools that compile Modelica — **OpenModelica** (and its `OMPython` bridge),
**Dymola**, **JModelica** — are large programs, and none of them is Python. If you want a
mature, near-complete Modelica compiler for production, reach for one of those; this is not
that yet.

So the gap is narrow and specific, and it is two things:

- **Nothing in Python takes Modelica source to a trajectory the way a *hybrid* model needs**
  — adaptive stiff integration, index reduction, and events the solver stops at.
- **Nothing treats the model itself as an ordinary Python value** — the flat equations, the
  incidence structure, the index — that you can read, inspect and change.

That gap is the project. Choose `modelica` when you want Modelica semantics *as a library*:
to inspect the flat DAE, script the toolchain, or embed a solver in your own Python. Choose
OpenModelica or Dymola when you need breadth of language coverage today.

## Install

```bash
pip install modelica
```

That is all: parsing, diagrams, FMI and SSP, and simulation on the engine. It brings NumPy and
SciPy.

**Status: pre-alpha.** A full rewrite of a 2010 Python 2 experiment; no code of it is left,
only the goal. Python 3.12 or later.

## Quickstart

From Modelica source to a trajectory in four values. Each line is a stage you can stop at.

```python
from modelica.lang import parse
from modelica.sim import flatten, causalize, simulate

source = """
model NewtonCooling "A hot object cooling toward ambient"
  parameter Real T_amb = 25;
  parameter Real k = 0.5;
  Real T(start = 90, fixed = true);
equation
  der(T) = -k * (T - T_amb);
end NewtonCooling;
"""

flat = flatten([parse(source)], "NewtonCooling")   # AST -> flat DAE
system = causalize(flat)                            # matching, BLT sort, index reduction
result = simulate(system, stop=10.0)                # integrate (Radau by default)

print(system.states)   # ('T',)
print(result.time)     # the solver's own steps
print(result["T"])     # the trajectory, indexed by the name you wrote
```

`result.plot()` returns an interactive chart that writes itself to a self-contained HTML
page with no CDN behind it (`result.plot().save("cooling.html")`), and in a notebook it
renders when it is the last line of a cell. `result.write("cooling.mat")` writes a
Dymola-format `.mat` that ModelicaRes, DyMat and BuildingsPy read.

The layers compose from source to picture the same way. This draws a model to SVG with no
other program:

```bash
python -m modelica.diagram examples/models/RLCCircuit.mo examples/library/Electrical.mo > rlc.svg
```

```python
from modelica import diagram

picture = diagram.of("examples/models/RLCCircuit.mo", "examples/library/Electrical.mo")
picture.write("rlc.svg")          # an image, light- and dark-safe (it names no colour)
print(picture.to_mermaid())       # or text for a Markdown page
print(picture.to_dot())           # or source for Graphviz
```

## Key features

- **Two front ends, one IR.** A hand-written lexer and recursive-descent parser carry source
  positions, so an error says where; a Python builder API (`modelica.build`) produces the
  same AST values. The two are cross-checked against each other.
- **Dependency-free core.** `import modelica` pulls in nothing. Numerics arrive as the `sim`
  extra (numpy + scipy), so tooling that only reads model structure stays cheap to install.
- **The flat model is a value.** Flattening resolves `extends`, modifications and `connect()`
  into scalar variables and acausal equations; causalization does alias elimination, matching,
  Tarjan BLT sorting and Pantelides index reduction. States, candidate state sets, index and
  blocks are all readable off the `System`.
- **Hybrid simulation.** Adaptive stiff integration (Radau, LSODA via SciPy), state and time
  events with `reinit`, a sparse Jacobian, and asserts checked along the trajectory. The IR is
  compilable by construction: `emit` writes the model as straight-line arithmetic.
- **Diagrams with no extra program.** Any layer draws to SVG, Mermaid or DOT — components and
  connections, the incidence of equations against variables, or blocks in solution order.
- **FMI 3.0.** Read and simulate an existing FMU, or export a flat model as one
  (`modelica.fmi`).
- **SSP 2.0.** Read the `.ssd`/`.ssv`/`.ssb` files inside a `.ssp` and run a co-simulation
  master over the components (`modelica.ssp`).
- **A bundled Modelica Standard Library slice** under the real `Modelica.*` names: `Units.SI`,
  `Electrical.Analog`, `Mechanics.Translational` and `Rotational`, `Thermal.HeatTransfer`, and
  a piece of `Blocks`. A curated subset, not the whole library.

## The layers

Each layer is usable without the ones above it.

| Module | What it does |
| --- | --- |
| `modelica.lang` | Modelica source to tokens to an AST, with source positions. |
| `modelica.build` | Python classes to the same AST values. A second front end. |
| `modelica.ir` | AST to a flat model; then matching, BLT sorting and index reduction. |
| `modelica.sim` | Flat model to an ODE/DAE problem; solver backends and events. |
| `modelica.diagram` | Any of the above to a picture: components, incidence, or blocks. |
| `modelica.fmi` | FMI 3.0: read an FMU and simulate it, or export a flat model as one. |
| `modelica.ssp` | SSP 2.0: read a `.ssp` and run a co-simulation master. |

## What of Modelica it handles

The language is large and this is pre-alpha, so the honest summary is a table. About 95 % of
the Modelica Standard Library example models the project measures against parse; fewer
simulate end to end. The per-model
[coverage notes](https://cardona-oss.gitlab.io/pymodelica/ideas/coverage/) track the frontier.

| Area | Handled | Not yet |
| --- | --- | --- |
| Equations | `der`, algebraic, `connect` (flow/potential and signal), `when` with `reinit`, `terminate`, structural `if`-equations, `for`, `assert` | an `if`-equation on a *continuous* condition |
| Expressions | full arithmetic and relations, `if`-expressions, every built-in math function, `smooth` / `noEvent` / `pre` / `homotopy`, `der(der(x))` | `spatialDistribution`, `delay` |
| Structure | components, `extends` with modification, conditional components, `redeclare`, `import`, nested packages | operator overloading, external (C) functions, `impure` |
| Types | `Real` / `Integer` / `Boolean`, discrete variables, enumerations, unit and `min`/`max` attributes, type aliases | |
| Functions | user functions with `algorithm` bodies, `if` and `for` statements | |
| Arrays | indexed access, array parameters and starts, element-wise array equations and products (dot, matrix-vector, matrix-matrix), `for` loops, multi-dimensional grids | array `/` and `^` |
| Numerics | index reduction (Pantelides), alias elimination, BLT sorting, adaptive stiff integration (Radau, LSODA), state events, sparse Jacobian | a symplectic integrator |
| Standards | FMI 3.0 (import and export), SSP 2.0 co-simulation | |

## Links

- **Repository:** <https://gitlab.com/cardona-oss/pymodelica>
- **PyPI:** <https://pypi.org/project/modelica/>
- **Documentation:** <https://cardona-oss.gitlab.io/pymodelica/>
- **Changelog:** [`CHANGELOG.md`](https://gitlab.com/cardona-oss/pymodelica/-/blob/main/CHANGELOG.md)

## Contributing

```bash
make install-hooks   # once after cloning: pre-commit bumps the version and runs ruff
make check           # lint, typecheck and test. This is what CI runs.
make docs            # build the documentation site, strictly
```

`.prototools` pins the toolchain (moon and uv); `proto install` brings it up. `moon.yml`
holds every task and CI runs one command, `moon run :ci`. See
[`AGENTS.md`](https://gitlab.com/cardona-oss/pymodelica/-/blob/main/AGENTS.md) for the full build/test/convention detail and
[`CONTRIBUTING.md`](https://gitlab.com/cardona-oss/pymodelica/-/blob/main/CONTRIBUTING.md) for how to send a change.

## License

MIT © Jorge Cardona. See [LICENSE](https://gitlab.com/cardona-oss/pymodelica/-/blob/main/LICENSE).

## Trademarks

Modelica® is a registered trademark of the [Modelica Association](https://modelica.org/).
FMI and SSP are also its trademarks. This package is an independent open-source
implementation, not affiliated with or endorsed by the Modelica Association; the names
identify the language it reads and the standards it implements.
