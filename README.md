<h1 align="center">modelica</h1>

<p align="center"><strong>Read Modelica source, flatten it to a DAE and simulate it from Python, with no OpenModelica or Dymola install.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/modelica/"><img alt="PyPI" src="https://img.shields.io/pypi/v/modelica?style=flat-square"></a>
  <a href="https://pypi.org/project/modelica/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/modelica?style=flat-square"></a>
  <a href="https://cardona-oss.gitlab.io/modelica/"><img alt="Docs" src="https://img.shields.io/badge/docs-cardona--oss.gitlab.io%2Fmodelica-blue?style=flat-square"></a>
  <a href="https://gitlab.com/cardona-oss/modelica/-/pipelines"><img alt="Pipeline" src="https://img.shields.io/gitlab/pipeline-status/cardona-oss%2Fmodelica?branch=main&style=flat-square"></a>
  <a href="https://gitlab.com/cardona-oss/modelica/-/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="https://cardona-oss.gitlab.io/modelica/notebooks/simulating/">Simulating a model</a> ·
  <a href="https://cardona-oss.gitlab.io/modelica/engine/">The engine</a> ·
  <a href="https://cardona-oss.gitlab.io/modelica/examples/">Examples</a> ·
  <a href="https://cardona-oss.gitlab.io/modelica/ideas/coverage/">Coverage</a> ·
  <a href="https://gitlab.com/cardona-oss/modelica/-/blob/main/CHANGELOG.md">Changelog</a>
</p>

---

## Overview

`modelica` parses `.mo` source, flattens it to a DAE, simulates it and draws it. Each stage
before simulation (the AST, the flat model, the sorted system) is a Python value you can read
and change. It needs no OpenModelica or Dymola install.

Simulation runs on a precompiled engine: a virtual machine written in Rust that ships in the
platform wheels and interprets the model as bytecode. Nothing is compiled per model. Where no
engine wheel exists, the pure-Python wheel simulates with SciPy. See
[the engine](https://cardona-oss.gitlab.io/modelica/engine/).

Status: pre-alpha. Of the 430 example models in the Modelica Standard Library 4.1.0, 420 parse
and 89 (21 %) simulate end to end (measured 2026-11-15, see the
[coverage notes](https://cardona-oss.gitlab.io/modelica/ideas/coverage/)). The API can still
change. Use OpenModelica or Dymola for full language coverage.

## Install

```bash
pip install modelica
```

It brings NumPy and SciPy. Python 3.12 or later.

| Platform | Simulation runs on |
| --- | --- |
| Linux x86_64, Linux aarch64, Windows x86_64, macOS arm64 | the precompiled engine |
| any other | SciPy |

`Result.backend` says which one ran. The engine runs a `System` with `method="Radau"` (the
default); other calls run on SciPy. See
[which backend to use](https://cardona-oss.gitlab.io/modelica/engine/#which-backend-to-use).

Parsing, flattening, causalization, simulation and diagrams are tested and give the same numbers on
the engine and on SciPy. Experimental and not promised: `modelica.fmi`, `modelica.ssp`, and the
`modelica.sim` modules not exported from `modelica.sim` (`fv`, `fem`, `mesh`, `sensitivity`,
`compare`).

## Quickstart

One call from a `.mo` file to a trajectory. Save this as `Cooling.mo`:

```modelica
model NewtonCooling "A hot object cooling toward ambient"
  parameter Real T_amb = 25;
  parameter Real k = 0.5;
  Real T(start = 90, fixed = true);
equation
  der(T) = -k * (T - T_amb);
  annotation(experiment(StopTime = 10));
end NewtonCooling;
```

```python
import modelica

result = modelica.run("Cooling.mo", parameters={"k": 1.0}, times=[0, 10])
print(result["T"])   # [90.         25.00295111]
```

The stop time comes from `experiment(StopTime = 10)`; pass `stop=` to change it. The same run
from the shell, as CSV (`-o out.mat` writes a Dymola `.mat`):

```bash
python -m modelica.sim Cooling.mo --points 3 --set k=1
```

```text
time,T_amb,k,T,der_T
0,25,1,90,-65
5,25,1,25.43796545,-0.4379654537
10,25,1,25.00295111,-0.00295111351
```

Under `run` are four stages, and each is a value you can stop at and inspect:

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

Components from the Modelica Standard Library work under their real names. The bundled slice
ships in the wheel, so nothing else needs to be passed:

```python
source = """
model RC "A charged capacitor discharging through a resistor"
  Modelica.Electrical.Analog.Basic.Resistor r(R = 1000);
  Modelica.Electrical.Analog.Basic.Capacitor c(C = 1e-3, v(start = 1));
  Modelica.Electrical.Analog.Basic.Ground g;
equation
  connect(r.p, c.p);
  connect(r.n, c.n);
  connect(c.n, g.p);
end RC;
"""

result = simulate(causalize(flatten([parse(source)], "RC")), stop=1.0, times=[0.0, 1.0])
print(result["c.v"])   # [1.         0.36787944], exp(-t / RC) at t = 1
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

## Why modelica

[FMPy](https://github.com/CATIA-Systems/FMPy) and
[PyFMI](https://github.com/modelon-community/PyFMI) run an FMU that another tool compiled.
The tools that compile Modelica (OpenModelica and its `OMPython` bridge, Dymola, JModelica)
are large programs, and none of them is Python. `modelica` does two things those do not:

- It takes Modelica source to a trajectory inside Python, with what a hybrid model needs:
  adaptive stiff integration, index reduction, and events the solver stops at.
- It gives you the model as a Python value (the flat equations, the incidence structure, the
  index) that you can read and change.

Choose `modelica` to inspect the flat DAE, script the toolchain, or embed a solver in your own
Python. Choose OpenModelica or Dymola for a mature compiler with near-complete language
coverage.

## Key features

- **Two front ends, one IR.** A hand-written lexer and recursive-descent parser carry source
  positions, so an error says where; a Python builder API (`modelica.build`) produces the
  same AST values. The two are cross-checked against each other.
- **The flat model is a value.** Flattening resolves `extends`, modifications and `connect()`
  into scalar variables and acausal equations; causalization does alias elimination, matching,
  Tarjan BLT sorting and Pantelides index reduction. States, candidate state sets, index and
  blocks are all readable off the `System`.
- **Hybrid simulation.** Adaptive stiff integration (Radau on the engine; Radau, BDF and LSODA via
  SciPy), state and time events with `reinit`, a sparse Jacobian, and asserts checked along the
  trajectory.
- **Diagrams with no extra program.** Any layer draws to SVG, Mermaid or DOT: components and
  connections, the incidence of equations against variables, or blocks in solution order.
- **FMI 3.0, started.** `modelica.fmi` writes a model's `modelDescription.xml` and holds the
  FMI calling sequence as a state machine. Importing or exporting a whole FMU is not done.
- **A bundled Modelica Standard Library slice** under the real `Modelica.*` names: `Units.SI`,
  `Electrical.Analog`, `Mechanics.Translational` and `Rotational`, `Thermal.HeatTransfer`, and
  a piece of `Blocks`. A curated subset, not the whole library. It ships in the wheel and
  `flatten` adds it on its own, so `Modelica.Electrical.Analog.Basic.Resistor` resolves after
  `pip install modelica`; a `Modelica` package you pass yourself is used instead.

## The layers

Each layer is usable without the ones above it.

| Module | What it does |
| --- | --- |
| `modelica.lang` | Modelica source to tokens to an AST, with source positions. |
| `modelica.build` | Python classes to the same AST values. A second front end. |
| `modelica.ir` | AST to a flat model; then matching, BLT sorting and index reduction. |
| `modelica.sim` | Flat model to an ODE/DAE problem; solver backends and events. |
| `modelica.diagram` | Any of the above to a picture: components, incidence, or blocks. |
| `modelica.fmi` | FMI 3.0: `modelDescription.xml` and the calling sequence (experimental). |
| `modelica.ssp` | SSP 2.0: the co-simulation master's protocol (experimental). |

## What of Modelica it handles

The table lists what is handled. Of the 430 example models in the Modelica Standard Library
4.1.0, 420 parse and 89 simulate end to end (measured 2026-11-15). The
[coverage notes](https://cardona-oss.gitlab.io/modelica/ideas/coverage/) list each
measurement and what blocks the rest.

| Area | Handled | Not yet |
| --- | --- | --- |
| Equations | `der`, algebraic, `connect` (flow/potential and signal), `when` with `reinit`, `terminate`, structural `if`-equations, `for`, `assert` | an `if`-equation on a *continuous* condition |
| Expressions | full arithmetic and relations, `if`-expressions, every built-in math function, `smooth` / `noEvent` / `pre` / `homotopy`, `der(der(x))` | `spatialDistribution`, `delay` |
| Structure | components, `extends` with modification, conditional components, `redeclare`, `import`, nested packages | operator overloading, external (C) functions, `impure` |
| Types | `Real` / `Integer` / `Boolean`, discrete variables, enumerations, unit and `min`/`max` attributes, type aliases | |
| Functions | user functions with `algorithm` bodies, `if` and `for` statements | |
| Arrays | indexed access, array parameters and starts, element-wise array equations and products (dot, matrix-vector, matrix-matrix), `for` loops, multi-dimensional grids | array `/` and `^` |
| Numerics | index reduction (Pantelides), alias elimination, BLT sorting, adaptive stiff integration (Radau, LSODA), state events, sparse Jacobian | a symplectic integrator |
| Standards | FMI 3.0 `modelDescription.xml` | FMU import and export, SSP archives |

## Links

- **Repository:** <https://gitlab.com/cardona-oss/modelica>
- **PyPI:** <https://pypi.org/project/modelica/>
- **Documentation:** <https://cardona-oss.gitlab.io/modelica/>
- **Changelog:** [`CHANGELOG.md`](https://gitlab.com/cardona-oss/modelica/-/blob/main/CHANGELOG.md)

## Contributing

```bash
make install-hooks   # once after cloning: pre-commit bumps the version and runs ruff
make check           # lint, typecheck and test. This is what CI runs.
make docs            # build the documentation site, strictly
```

`.prototools` pins the toolchain (moon and uv); `proto install` brings it up. `moon.yml`
holds every task and CI runs one command, `moon run :ci`. See
[`AGENTS.md`](https://gitlab.com/cardona-oss/modelica/-/blob/main/AGENTS.md) for the build,
test and convention detail and
[`CONTRIBUTING.md`](https://gitlab.com/cardona-oss/modelica/-/blob/main/CONTRIBUTING.md) for
how to send a change.

## License

MIT © Jorge Cardona. See [LICENSE](https://gitlab.com/cardona-oss/modelica/-/blob/main/LICENSE).

## Trademarks

Modelica® is a registered trademark of the [Modelica Association](https://modelica.org/).
FMI and SSP are also its trademarks. This package is an independent open-source
implementation, not affiliated with or endorsed by the Modelica Association; the names
identify the language it reads and the standards it implements.
