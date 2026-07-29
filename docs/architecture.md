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
Modelica source                Python model
   │  modelica.lang               │  modelica.build
   │  lexer + recursive descent   │  descriptors + operator overloading
   ▼                              ▼
AST ─────────────────────────► modelica.ir.Class
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

## Two front-ends, one IR

A model can arrive as a `.mo` file or as Python. `modelica.build` is **not** a wrapper that
generates Modelica text and parses it back; it is a peer of `modelica.lang` that constructs
`modelica.ir` values directly. Both roads end at the same `ir.Class`, and everything
downstream — flattening, index reduction, code generation, FMU export — sees only that.

```python
class NewtonCooling(Model):
    """A hot object cooling toward ambient"""

    k = parameter(Real, 0.5, unit="1/s", doc="Cooling coefficient")
    T = Real(unit="degC", start=90, fixed=True, doc="Object temperature")

    def equations(self):
        yield der(self.T) == -self.k * self.T
```

Four mappings carry the whole front-end: a declaration is a **descriptor** (so `self.T`
reads as a reference without any instance existing), `==` builds an **equation** rather
than a bool (Modelica's `=` is acausal, so this is the honest reading), Python inheritance
is **`extends`**, and a Python module is a **`package`**.

The acceptance test is equality of values, not of text: every example in `examples/` is
written twice, once per front-end, and the pair must flatten to the identical `FlatModel`.
Until the parser lands the comparison runs through `unparse` — render the Python model as
Modelica and diff it against the handwritten file — which is weaker, and is replaced by the
value comparison the moment `modelica.lang.parse` exists.

Why bother with a second front-end at all: Modelica text is the interchange format, but a
model that is *generated* — swept over parameters, assembled from a topology, built inside
a notebook — wants to be a Python value. Making the two peers rather than layering one on
the other means neither can quietly become the poor relation.

## Performance: where the time actually goes

The goal is a Python engine that stands next to LSODA, DASSL, RADAU5 and CVODE. That is
achievable, and the reason is that a stiff solver spends almost none of its time in the
parts written in Python.

Split the work in two:

* **Control** — step-size and order selection, error norms, the Newton loop, event
  location, Jacobian reuse heuristics. This runs *once per step*, on scalars and small
  arrays. Python is fine here; scipy's `Radau` and `BDF` are pure Python + NumPy and are
  used in production every day.
* **Kernel** — evaluating the residual/RHS and the Jacobian, and the linear solve. This
  runs several times per step and dominates.

The dominant factor is algorithmic, not linguistic. On Robertson, an explicit method needs
on the order of 10⁷–10⁸ steps and a good stiff method needs a few hundred; that ~10⁶ ratio
is entirely independent of the implementation language. Getting the order/step-size control
and the index reduction right buys more than any amount of micro-optimisation, and getting
them wrong cannot be recovered by rewriting the inner loop in C.

That leaves the per-step constant, and it is closed by **generating code**:

> **The IR must be compilable.** `modelica.sim` emits source for the RHS, the Jacobian and
> the event indicator functions, and hands it to a compiler — Numba `@njit` first, C as a
> fallback path for FMU export. It does **not** build a tree of closures and walk it at
> every evaluation.

This is a constraint on the IR design, which is why it is recorded here and not in a
performance-tuning note later. The expression IR is small, typed, closed under symbolic
differentiation, and free of Python objects that cannot be spelled as arithmetic —
precisely so that a straightforward emitter can turn a `FlatModel` into a nopython
function. An interpreter over the same IR stays as the reference implementation: it is what
the compiled kernel is tested against, and what runs when Numba is not installed.

The naive alternative — a closure tree walked from Python — is 10–100× worse, and no amount
of solver quality hides it.

### Matching is the floor, not the ceiling

"As fast as the Fortran codes" is the wrong ambition to stop at, because a compiler that
owns the model can do three things a precompiled library structurally cannot:

* **Inline the RHS into the integrator loop.** RADAU5 is a library: it calls the user's `F`
  through a function pointer, across a boundary the optimiser cannot see through, with the
  parameters living in memory it has to load. A JIT compiles the model and the step
  together — the RHS is inlined and the parameters are constants folded once at compile
  time, not reloaded once per stage per step.
* **Differentiate the model.** The equations are held symbolically, so an analytic *and
  sparse* Jacobian is free, with the pattern falling straight out of the incidence matrix
  matching already built. A black-box integrator must finite-difference it: *n* extra RHS
  evaluations per Jacobian, plus truncation error injected exactly where it damages Newton
  convergence.
* **Shrink the system before the solver sees it.** Tearing and BLT turn a simultaneous
  block of *n* equations into a much smaller torn system wrapped in an assignment sequence.
  That is a symbolic transformation on the model; a bare ODE library receives the system
  already assembled and cannot perform it.

So the target is to be *faster* on the classic problems, not merely close — and the claim
is only worth making against a fixed harness. [Benchmarks](ideas/benchmarks.md) records
which problem sets that means and why the measurement is a work-precision diagram rather
than a timing. The compiler that makes the three points above possible is written for this
workload rather than adopted from a general-purpose one; [backends](ideas/backends.md)
covers that decision and the debuggability it is meant to buy.

## Layer notes

### `modelica.lang` — source → AST

Hand-written lexer and recursive-descent parser against the Modelica Language
Specification 3.x grammar. No parser-generator and no `pyparsing`: the grammar is stable
and small enough to write out, and only a hand-written parser gives usable errors
("expected `end <name>;` at 12:5, to close `model Foo` opened at 3:1"). AST nodes are
frozen dataclasses carrying source spans.

### `modelica.build` — Python → IR

Descriptors, operator overloading and a metaclass, and nothing else. Every value in it is a
frozen dataclass: `flow(...)`, `parameter(...)` and friends return a *new* declaration
rather than mutating one, so a builder expression means the same thing wherever it appears.
The single mutation in the module is `__set_name__`, which is unavoidable — a descriptor
learns its own name only after it exists.

One consequence of Python's rules is worth stating, because it shaped the design.
Rich comparison gives priority to the *right* operand when its type is a proper subclass of
the left's, with none of the "differs from the base" guard the numeric operators have. Had
`Declaration` subclassed `Expression`, `der(self.h) == self.v` would have run
`self.v.__eq__` and produced `v = der(h)` — the same equation acausally, but not the same
text, and a mismatch with no visible cause. `Declaration` is therefore a descriptor that
*returns* an `Expression`, which removes the possibility instead of documenting it.

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

0. `ir` + `build` + `lang.unparse` — the value types, the Python front-end, and a way to
   read them back as text. **Done**; the example corpus is written twice against it.
1. `lang.parse` + `ir` far enough to flatten a trivial `model` into scalar equations.
   The parser is **done**: `lang.lexer` and `lang.parse` read the whole example corpus, and
   `parse(unparse(x)) == x` holds over it because both directions climb the same
   precedence table.
2. `sim` on the explicit-ODE case, scipy backend — first end-to-end simulation.
3. Matching/BLT, then Pantelides + dummy derivatives — first physical model with
   `connect()`.
4. Events.
5. Code generation from the `FlatModel` (Numba), with work-precision diagrams against
   scipy's Fortran-backed integrators on Robertson and van der Pol — see
   [benchmarks](ideas/benchmarks.md) for the harness this has to answer to.
6. `fmi` import (read + simulate an existing FMU) — validates the variable/model-structure
   data model against real files.
7. `fmi` export.
8. `ssp` composition + co-simulation master.
