# modelica

[![pipeline status](https://gitlab.com/jorgeecardona/pymodelica/badges/main/pipeline.svg)](https://gitlab.com/jorgeecardona/pymodelica/-/pipelines)
[![PyPI version](https://img.shields.io/pypi/v/modelica.svg)](https://pypi.org/project/modelica/)
[![Python versions](https://img.shields.io/pypi/pyversions/modelica.svg)](https://pypi.org/project/modelica/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-gitlab%20pages-blue.svg)](https://jorgeecardona.gitlab.io/pymodelica/)

A Modelica toolchain written only in Python. It reads Modelica source, flattens it to a
DAE, simulates it, and draws it. It also speaks the two exchange standards of the Modelica
Association: **FMI 3.0** for one component, and **SSP 2.0** for a system of them.

> The distribution is named `modelica`. The git repository keeps its older name,
> `pymodelica`. PyPI holds the name `pymodelica` for an abandoned Modelon/JModelica.org
> upload.

```bash
pip install modelica          # the core: parse, model structure, diagrams. No dependencies.
pip install "modelica[sim]"   # adds numpy and scipy, and therefore simulation
```

**Documentation: <https://jorgeecardona.gitlab.io/pymodelica/>** — the architecture, the
example corpus, and seven notebooks that are executed when the site is built.

## Status

Pre-alpha. This is a full rewrite of a Python 2 experiment from 2010. No code of that
experiment is left. Only the goal is. Python 3.12 or later.

## Why this exists

Python has the two ends of a Modelica toolchain and very little of the middle.
[FMPy](https://github.com/CATIA-Systems/FMPy) and
[PyFMI](https://github.com/modelon-community/PyFMI) *run* an FMU that some other tool
compiled. The tools that compile Modelica — Dymola, OpenModelica, JModelica — are large
programs, and none of them is Python.

The middle is not empty. [pymoca](https://github.com/pymoca/pymoca) parses Modelica and
flattens it, and Deltares runs it under [RTC-Tools](https://github.com/rtc-tools/rtc-tools)
to control water systems: that is a `.mo` file to a trajectory, in Python, in industrial
use. It gets there with a fixed step and smooth residuals, because its consumer is an
optimizer.

So the gap is narrower than "nothing does this", and it is two things. Nothing in Python
takes Modelica source to a trajectory the way a *hybrid* model needs — adaptive stiff
integration, index reduction, and events the solver stops at. And nothing treats the model
itself — the flat equations, the incidence structure, the index — as an ordinary Python
value that you can read and change.

[What else exists](https://jorgeecardona.gitlab.io/pymodelica/ideas/landscape/) is the
evidence for both halves of that, tool by tool.

That gap is the project. Every stage is a value you can hold.

## The layers

You can use each layer without the layers above it.

| Module | What it does |
| --- | --- |
| `modelica.lang` | Modelica source to tokens to an AST. The lexer and the parser are written by hand. They carry source positions, so an error can say where. |
| `modelica.build` | Python classes to the same AST values. A second front end, not a wrapper around the first. |
| `modelica.ir` | AST to a flat model: variables, equations, an expression IR. Then matching, BLT sorting and index reduction. |
| `modelica.sim` | Flat model to an ODE or DAE problem. Solver backends, and events. |
| `modelica.diagram` | Any of the above to a picture: components and connections, equations and variables, or blocks in solution order. |
| `modelica.fmi` | FMI 3.0. Read an FMU and simulate it, or export a flat model as one. |
| `modelica.ssp` | SSP 2.0. Read the `.ssd`, `.ssv` and `.ssb` files in a `.ssp`, and run a co-simulation master over the components. |

Read [the architecture notes](https://jorgeecardona.gitlab.io/pymodelica/architecture/)
for the data model and the order the layers were built in.

## What of Modelica it handles

The language is large and this is pre-alpha, so the honest summary is a table, not a claim of
completeness. The per-model measurement against the full Modelica Standard Library lives in
[the coverage notes](https://jorgeecardona.gitlab.io/pymodelica/ideas/coverage/): about 95 % of
its example models parse, fewer simulate end to end. The frontier there is one column below.

| Area | Handled | Not yet |
| --- | --- | --- |
| Equations | `der`, algebraic, `connect` (flow/potential and signal), `when` with `reinit`, `terminate`, structural `if`-equations (a parameter or constant condition), `for`, `assert` | an `if`-equation on a *continuous* condition (a run-time structural switch, which is a `when`) |
| Expressions | full arithmetic and relations, `if`-expressions, every built-in math function, `smooth` / `noEvent` / `pre` / `homotopy`, a second derivative written `der(der(x))` | `spatialDistribution`, `delay` |
| Structure | components, `extends` with modification, conditional components, `redeclare`, `import` (all three forms), nested packages | operator overloading, external (C) functions, `impure` |
| Types | `Real` / `Integer` / `Boolean`, discrete variables, enumerations, unit and `min`/`max` attributes, type aliases | |
| Functions | user-defined functions with `algorithm` bodies, `if` and `for` statements | |
| Arrays | indexed access, array parameters and starts, element-wise array equations both vectorised (`der(x) = -x`, a scalar broadcast, an element-wise function call) and written out with a `for` loop, multi-dimensional grids | an array *product* or contraction such as `der(x) = A * x` (write it over the elements in a `for` loop) |
| Numerics | index reduction (Pantelides), alias elimination, BLT sorting, adaptive stiff integration (Radau, LSODA), state events, a sparse Jacobian | a symplectic integrator |
| Standards | FMI 3.0 (import and export), SSP 2.0 co-simulation | |

The bundled Modelica Standard Library slice carries `Units.SI`, `Electrical.Analog`,
`Mechanics.Translational` and `Rotational`, `Thermal.HeatTransfer`, and a piece of `Blocks`,
under their real names, so a model that references `Modelica.*` resolves without configuration. It
is a curated subset, not the whole library.

## See a model

A Modelica model is a graph. The components are the boxes and `connect()` gives the lines.
The source text holds that graph, but a person cannot see it there. After flattening it is
gone: the equations keep every piece of the information and none of the shape.

```bash
python -m modelica.diagram examples/models/DCMotor.mo examples/library/Electrical.mo \
    examples/library/Rotational.mo examples/library/Machines.mo --depth 2 > motor.svg
```

The picture needs no other program. This package draws the SVG itself. The image names no
colour, so it reads correctly on a light page and on a dark one.

```python
from modelica import diagram

picture = diagram.of("examples/models/RLCCircuit.mo", "examples/library/Electrical.mo")
picture.write("rlc.svg")  # an image
print(picture.to_mermaid())  # or text for a Markdown page
print(picture.to_dot())  # or source for Graphviz
```

A notebook shows the picture when you put it on the last line of a cell.

## Development

```bash
make install-hooks   # once, after you clone. A pre-commit hook bumps the version and runs ruff.
make check           # lint, typecheck and test. This is what CI runs.
make docs            # build the documentation site, strictly.
```

`.prototools` pins the toolchain, which is moon and uv. `proto install` brings up all of
it. `moon.yml` holds the order of every task, and CI runs one command: `moon run :ci`.

## Releasing

There is no manual release step. The `version` in `pyproject.toml` is the only source of
truth.

* The pre-commit hook increases the patch version when a file under `src/**/*.py` changes.
* `scripts/ci-version-guard` fails the pipeline if the shipped source moved and the version
  did not. A hook can be bypassed. This cannot.
* On `main`, `moon run :release` publishes to PyPI, but only if that version is not there
  yet. It uses GitLab OIDC Trusted Publishing, so no API token is stored anywhere.

The same pipeline publishes the documentation to GitLab Pages.

## The documentation

The documentation is at **<https://jorgeecardona.gitlab.io/pymodelica/>**. The same
pipeline that tests the code builds it, and it executes every notebook on the way, so a
page that stopped working fails the build.

The pages follow ASD-STE100 Simplified Technical English. [How these pages are
written](https://jorgeecardona.gitlab.io/pymodelica/style/) lists the rules and the two
places this project departs from them.

## License

MIT © Jorge Cardona

## Trademarks

Modelica® is a registered trademark of the [Modelica Association](https://modelica.org/).
FMI and SSP are also its trademarks. This page uses those names to identify the language
this software reads and the standards it implements.
