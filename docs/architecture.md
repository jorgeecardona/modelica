# Architecture

The design rule for this rewrite: **every stage is a value you can hold.** Parsing returns
an AST, flattening returns a flat model, causalization returns a sorted block structure,
and simulation returns trajectories — each one an ordinary, inspectable Python object
rather than state hidden inside a compiler. The 2010 prototype failed the other way: the
grammar (pyparsing) *was* the program, so there was nothing to look at between "source" and
"result".

Nothing here is implemented yet. This is the target, and the order it gets built in.

## The pipeline

```
Modelica source
   │  modelica.lang            hand-written lexer + recursive-descent parser
   ▼
AST                            positions carried on every node (diagnostics)
   │  modelica.ir              instantiation / flattening
   ▼
FlatModel                      variables + equations + expression IR
   │  modelica.ir              matching, BLT sorting, index reduction
   ▼
CausalizedModel                sorted blocks, known index, states chosen
   │  modelica.sim             residual/RHS construction
   ▼
Problem  ──►  Solver  ──►  Result
```

Two exchange formats cut across it:

* **FMI 3.0** (`modelica.fmi`) attaches at both ends — read an FMU someone else compiled
  and simulate it through the same `Problem`/`Solver` interface, or export a
  `CausalizedModel` as an FMU.
* **SSP 2.0** (`modelica.ssp`) sits above — a `.ssp` names components (usually FMUs) and
  wires them, so it drives a co-simulation master over several `Problem`s at once.

## Layer notes

### `modelica.lang` — source → AST

Hand-written lexer and recursive-descent parser against the Modelica Language
Specification 3.x grammar. No parser-generator and no `pyparsing`: the grammar is stable
and small enough to write out, and only a hand-written parser gives usable errors
("expected `end <name>;` at 12:5, to close `model Foo` opened at 3:1"). AST nodes are
frozen dataclasses carrying source spans.

### `modelica.ir` — AST → solvable system

The interesting layer, and the one the whole project stands on.

* **Flattening** — resolve `extends`, expand components and `for`-equations, apply
  modifications, connect `connect()` equations into their potential/flow forms. Output:
  a `FlatModel` of typed variables and scalar equations over an expression IR.
* **Expression IR** — separate from the AST on purpose. The AST mirrors what was written;
  the IR is what gets differentiated and evaluated, so it is small, typed, and closed under
  symbolic differentiation.
* **Matching + BLT** — bipartite matching of equations to unknowns, then Tarjan SCC to get
  block-lower-triangular form: which equations must be solved simultaneously and which are
  a straight assignment sequence.
* **Index reduction** — Pantelides to find the structurally singular subsets, symbolic
  differentiation of those equations, and the dummy-derivative method to select states.
  A high-index DAE is the normal case for a physical model written with `connect()`; a
  toolchain that cannot reduce the index cannot simulate a mechanical model at all.

### `modelica.sim` — running it

`Problem` is the interface every backend consumes: an explicit ODE where causalization
achieved one, a residual DAE otherwise. Backends:

* **scipy** (`sim` extra) — `solve_ivp` for ODE form, plus root-finding for the algebraic
  blocks. The default, because it installs everywhere.
* **SUNDIALS IDA** via [scikit-SUNDAE](https://github.com/NREL/scikit-sundae) (`sundials`
  extra) — real index-1 DAE integration with consistent initialization. Optional because it
  needs a compiled SUNDIALS.

Events (zero crossings from `if`/`when`, and state events generally) are the part that
makes this harder than "call an integrator": the solver must stop at the crossing, the
model is re-initialized, and integration restarts. That is designed in from the start, not
bolted on.

### `modelica.fmi` — FMI 3.0

`modelDescription.xml` parsed into dataclasses (variables, causality/variability, model
structure, unit definitions), the FMU's shared library loaded through `ctypes`, and both
interface types wrapped: Model Exchange (the FMU supplies the derivatives, we integrate)
and Co-Simulation (the FMU integrates itself between communication points). Export goes the
other way: a `CausalizedModel` plus a generated `modelDescription.xml`, with the FMU's C
entry points backed by the Python evaluator.

### `modelica.ssp` — SSP 2.0

A `.ssp` is a zip: `SystemStructure.ssd` (components, connectors, connections, possibly
nested systems), `.ssv` parameter sets, `.ssb` bindings. Parsed into a composition graph,
resolved against the FMUs it references, and executed by a co-simulation master that steps
the components and exchanges signals at communication points. SSP 2.0 specifics that matter
to the data model: FMI 3.0 component support, the `structuralParameter` connector kind,
clocked connectors, and arrays in connectors/connections.

## Build order

1. `lang` + `ir` far enough to flatten a trivial `model` into scalar equations.
2. `sim` on the explicit-ODE case, scipy backend — first end-to-end simulation.
3. Matching/BLT, then Pantelides + dummy derivatives — first physical model with
   `connect()`.
4. `fmi` import (read + simulate an existing FMU) — validates the variable/model-structure
   data model against real files.
5. Events.
6. `fmi` export.
7. `ssp` composition + co-simulation master.
