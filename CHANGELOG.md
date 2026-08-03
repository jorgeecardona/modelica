# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org/); the version in `pyproject.toml` is the single
source of truth and is bumped automatically when shipped source changes.

## Unreleased

- `modelica.sim.emit`: **`observe_jacobian`** — `d(observed)/dy`, exact, by the same forward
  mode that writes `jacobian`. This is what a state estimator calls `H`, and until now every
  caller had to finite-difference it: one extra `observe` per state, and truncation error
  injected straight into the Kalman gain. It reaches code `jacobian` does not — the aliases,
  which causalization eliminates before the solver sees them and which are exactly what a
  model is usually measured *through* (a probe temperature, a branch current, a lag). Named
  `observe_jacobian` rather than `sensitivity` deliberately: *sensitivity* means the
  derivative with respect to a **parameter** in this field, that is the same transformation
  with a different seed, and it will want the name.
- `modelica.ir.derive`: **a conditional whose two answers are the same is folded away**, and
  that one line is what makes a *schedule* compilable. A chain of constant segments has
  derivative zero on every segment, so differentiating it produced
  `0.0 if w[30] > 0.5 else 0.0 if w[31] > 0.5 else ...` — nine branches, one answer — and
  `simplify` had no fold for it. Every block Jacobian carried the chain, so it reached `rhs`,
  `jacobian`, `observe`, `indicators` and `transition` alike. Compiling `RoastingMachine`
  under Numba, measured before and after: `jacobian` went from 15.6 s and 4.4 GB *without
  finishing* to 2.6 s and 0.21 GB, `rhs` from 5.6 s to 0.2 s, and the whole model from
  uncompilable to 3.9 s. `tests/test_jit.py` covers it now, and the note there that said "a
  model whose behaviour is a long schedule against `time` is effectively not compilable" is
  corrected: it was a missing fold, not a limit of Numba and not a fault of the model.
- `modelica.diagram`: **a model is a graph, and now you can see it.** Three pictures, one
  per stage: `schematic` reads the classes before flattening and shows components and
  connections; `incidence` reads a `FlatModel` and shows which equation touches which
  variable; `blocks` reads a `System` and shows the order the solver runs in. Each returns
  a `Graph`, which writes itself as SVG, Graphviz or Mermaid, and renders itself in a
  notebook. There is a CLI: `python -m modelica.diagram <model.mo> <libraries...>`. No new
  dependency — the SVG, including the layout, is drawn here.
  - The schematic resolves names with `modelica.ir.flatten`'s own machinery rather than a
    second copy, which is why `Scope`, `Found` and `effective` are public now. A test holds
    the two together: no box may name a component the flat model does not have.
  - The SVG names no colour. Every line and character is `currentColor`, so one image reads
    correctly on a light page and a dark one — no second copy, no script, no CDN.
  - The layout starts from a circle, never from random positions, so a rebuild of the docs
    site produces no diff.
- `modelica.ir.flatten`: **connectors are compared by structure, not by class name.**
  `connect(source.y, error.u1)` joining an `output` connector to an `input` one was
  rejected as "mixes connector types"; Modelica compares the variables the two carry, and
  an output joined to an input is the ordinary case. The opposite check arrived with it:
  two outputs on one wire is a real error, and the message says which two.
- `modelica.ir.flatten`: **a modification on an inner component now gets the enclosing
  instance's path.** `winding(R = Rw)` inside a component bound `R` to a variable that did
  not exist, and code generation died on "nothing computes 'Rw'". At the top level the
  prefix is empty and nothing moves, which is why no earlier example could catch it.
- `modelica.build.ref`: promoted from a private helper in one example. A parameter computed
  from other parameters, and a component modification that reads one, are both written in
  the class body, where `self` does not exist yet.
- `modelica.ir.source`: `to_program` and `flatten` moved out of `modelica.sim`. They have no
  numerics in them, and reaching them through `modelica.sim` pulled NumPy into a package
  whose promise is that reading a model's structure costs nothing. `modelica.sim` re-exports
  both, so no caller changes.
- `examples/`: `Blocks`, `Rotational` and `Machines` libraries, and two models.
  - `PIControl` — a block diagram: `input`/`output` connectors, a directed signal, and a
    closed loop. It found both flattener issues above and one in the corpus harness, which
    read the documented `after` key on an invariant and then ignored it.
  - `DCMotor` — the first model in two domains at once, and the first whose component holds
    components: `Machines.Motor` has the winding, coil, converter and rotor inside it, with
    three connectors wired from both sides.
  - Both agree with an independently integrated reference to 1e-10.
- `tests/test_diagram.py`: 63 tests. Every model in the corpus draws, the SVG parses as XML,
  the same model draws the same bytes twice, and no drawn component is missing from the flat
  model.
- **Docs rewritten in ASD-STE100 Simplified Technical English.** Short sentences, active
  voice, one idea per sentence, one topic per paragraph, and one meaning per word. Modelica
  is used worldwide and most readers here do not read English first; the standard costs the
  writer some effort and saves every reader some. `docs/style.md` records the rules followed
  and the two places this project departs from them. No claim, number or reason was dropped
  in the rewrite.
- Docs: *Drawing a model*, a new executed notebook — the three pictures, what each one is
  for, and what keeps them honest. The architecture pipeline diagram is a Mermaid graph
  instead of ASCII art, and the site now renders `mermaid` fences.

- `modelica.ir`: the expression, equation and class value types the whole toolchain is
  built on. Frozen dataclasses throughout.
- `modelica.lang.lexer`: a hand-written tokenizer. Strings arrive decoded, comments and
  whitespace are dropped, and every token carries its line and column so errors can point
  at the source rather than describe it.
- `modelica.lang.parse`: a recursive-descent parser producing `modelica.ir` values. It
  refuses the constructs it does not implement by name (`for`, `while`, `algorithm`,
  arrays, …) instead of misreading them, because a parser that quietly mangles a model is
  worse than one that has never heard of it.
- `modelica.lang.unparse`: IR back to Modelica source, with correct operator precedence and
  associativity.
- `modelica.lang.precedence`: one table, climbed by both the parser and the printer, which
  is why `parse(unparse(x)) == x` is structural rather than coincidental. The single
  deliberate deviation from the spec — the unary sign binding tighter than `*` — is
  recorded there with its justification.
- `tests/test_lang.py`: 139 tests. Round trips over the whole example corpus in both
  directions, the precedence deviations pinned as trees rather than as strings, and the
  refusals asserted so an unimplemented construct cannot start silently half-working.
- `modelica.build`: a Python front-end that constructs `modelica.ir` values directly — a
  peer of `modelica.lang`, not a wrapper around it. Declarations are descriptors, `==`
  builds an equation, class inheritance is `extends`, a module is a `package`.
- `examples/`: 3 component libraries and 10 models, each written twice — once as `.mo` and
  once through the Python front-end — plus a TOML behaviour spec per model
  (reference points, invariants, expected events).
- `tests/test_examples.py`: runs the corpus. The spec guard and the front-end equivalence
  check run today; parse, flatten and simulate skip until those stages exist.
- Architecture: recorded that the IR must be *compilable* — `modelica.sim` emits code for
  the RHS and Jacobian rather than walking a closure tree, which is what makes a Python
  engine competitive with the Fortran solvers. Matching them is now stated as the floor
  rather than the target, with the three things a model-owning compiler can do that a
  precompiled library structurally cannot.
- Docs: *Backends* — two evaluators over one IR (the interpreter stays as the reference
  tier and makes differential debugging possible), backends as emitters rather than
  rewrites, and why the compiler is written for this workload instead of adopted from a
  general-purpose one.
- Docs: *Benchmarks* — the problem sets any performance claim here has to answer to (Bari
  IVP test set, Hairer's, SciMLBenchmarks, the Modelica ScalableTestSuite) and why the
  measurement is a work-precision diagram rather than a FLOPS number.
- **Python floor raised to 3.12** (was 3.11). The `sim` extra decides it: NumPy 2.4 is
  itself `Requires-Python: >=3.12`, and its stubs spell `ArrayLike` with
  `collections.abc.Buffer`, a 3.12 name — type-checked as 3.11, every array argument in the
  package reads as partially unknown. Supporting a version the numerics cannot be installed
  on was a claim about nothing. `requires-python`, the ruff target, the pyright version and
  the floor CI job now all say 3.12.

- `modelica.sim.solve`: **event iteration**. An event no longer ends the instant — applying
  it can leave a second relation disagreeing with its own latch, and the loop now runs to a
  fixed point before the integrator restarts. This was a correctness hole, not a
  performance one: the integrator could never have caught that second relation, because on
  the next segment its indicator does not *cross* zero, it starts out already past it, so
  the stale value was carried for the rest of the run.
- `modelica.sim.emit`: `Trigger.latch` puts the discrete structure of a model on the public
  record — which workspace slot each crossing owns, `None` for a `when` branch. Two things
  above the emitter need it, and both used to have no way to ask.
- `modelica.verify`: model-checking the discrete half, behind the new `verify` extra
  ([musil](https://gitlab.com/jorgeecardona/musil) as the checker). At one instant the
  continuous state is frozen, so the event iteration is a *finite* machine: this enumerates
  it from every latch configuration and in every order the pending events could be applied,
  and proves that it settles (liveness) and that it settles in one place (confluence). The
  emitted `indicators` and `transition` are the functions being called, so a counterexample
  is a real execution rather than an artefact of an approximation.
- `tests/test_verify.py`: the whole corpus checked at `t = 0` and again at every instant a
  real run stopped at — the bounce, the switch, the moment the tank runs dry.
- `modelica.ir.causalize`: **every admissible state set**, on `System.candidates`. A set can
  be the states exactly when holding it back still leaves an equation for every remaining
  unknown, so the test is the matching itself, run once per candidate with that set
  forbidden. The pendulum answers `(y, vy)` and `(x, vx)` — the two the literature names,
  which are *two* exchanges apart, so they are enumerated rather than found by perturbing the
  chosen set one variable at a time. This is the menu dynamic state selection chooses from.
- `modelica.sim.emit`: **dynamic state selection**, and with it the pendulum simulates. Each
  candidate set is causalized into its own blocks and scored by the product of its 1×1 pivots
  — `4x²` against `4y²` for the pendulum, so the crossover is at 45°, which is the answer the
  literature gives. The emitter writes one `rhs`/`observe`/`indicators`/`transition` per set
  plus a picker that reads which one is live from a workspace slot appended after the latches,
  so nothing above the emitter needs an extra argument and numba still compiles the lot.
  Crossings come one per *ordered* pair (a pair compares with one smooth expression; a maximum
  does not) and read `−1` outside their origin, because a set the run is not using is not a set
  it can leave. A 20% margin on the scores is what makes the switch terminate rather than
  chatter. The crossing itself costs no accuracy — every state of every set is an unknown of
  the model, so the destination's states are already computed when the transition rewrites the
  vector. Pendulum over two seconds: four switches, `x² + y² = 1` held to 2·10⁻¹⁶, energy drift
  1.6·10⁻¹⁰.
- `modelica.sim.emit`: **the analytic state Jacobian**, as an eighth generated function.
  Forward mode by source transformation over the same blocks the right-hand side runs, one
  row vector per variable so the emitted source stays the size of the model rather than its
  square. It is not a speedup and is not offered as one — measured against SciPy's own
  difference quotient the step counts are identical and the wall clock moves by less than
  the noise, because for systems this small a difference quotient is two right-hand sides.
  It is *exactness*: a block that Newton iterates on has no derivative a difference quotient
  can see (perturbing the input re-runs the loop, so the answer is only as smooth as the
  tolerance it stopped at) while its residual differentiates into one linear solve against
  the matrix the block was already solved with; and a latched relation is frozen, so the
  exact derivative steps across a switch that a difference quotient would straddle. Both are
  why FMI asks for `fmi3GetDirectionalDerivative` rather than differencing outputs itself.
  Wired into `solve_ivp` for the implicit methods and to those only, since SciPy warns about
  arguments a solver ignores, and handed back in a fresh array each call because Radau keeps
  the last Jacobian it was given across steps.
- `tests/test_examples.py`: the emitted Jacobian against central differences of the emitted
  right-hand side, at twelve points along each model's own simulated trajectory rather than
  around the initial point — which is where models are at their most degenerate, every rate
  zero. Only the pendulum exercises the implicit path, and it is the one that catches a sign
  flip in either the 1×1 or the 3×3 tangent solve.
- `modelica.sim.kernel`: **the compiled path is executed rather than claimed.** `jit=True` has
  handed the generated module to Numba since the emitter existed, and nothing had ever run it —
  Numba is in no dependency group, so "there is a Numba backend" was a docstring rather than a
  fact. `tests/test_jit.py` now checks every generated function, compiled against interpreted,
  bit for bit, on five models chosen to force every shape the emitter writes; plus a whole
  simulation, bounces and all, agreeing to 1e-12.
- `modelica.sim.kernel`: **`error_model="numpy"`**. Numba defaults to raising
  `ZeroDivisionError` where CPython would, while the interpreted kernel divides numpy scalars
  and gets an infinity — a real semantic fork between the two tiers, and the interpreted one is
  the reference tier that differential debugging depends on. Defensive rather than a fix for an
  observed failure, and said that way on purpose: no model in the corpus reaches a division by
  exactly zero (the pendulum at `y = -L` comes closest and its Newton lands near 1e-16, not on
  0.0). But an exact zero pivot is reachable in principle exactly where this engine has
  something to say — a state set going singular, which dynamic state selection survives by
  switching rather than stopping. An infinity there is information; an exception is a crash.
- Measured, because the reason to compile is not the same for every model: on `AtwoodPendulum`
  (10 states, a 4×4 implicit block) Numba is **21× on `rhs` and 44× on `jacobian`** — 33.6 µs to
  1.61 µs, 180.6 µs to 4.07 µs. On a two-state model it is 1.0×, the dispatch costing what the
  arithmetic does. And put back in context it is a **1.55× on the simulation**, because 63% of
  that run is SciPy's Radau, which is written in Python: compiling the model is nearly free and
  nearly finished, and it is not where the time is.
- `moon run :test-jit`, and deliberately **no `jit` extra** — which is a finding, not an
  oversight. Numba pins `numpy<2.5` as far as its current release (0.66) and this project runs
  2.5. Declaring the extra was tried and reverted: uv locks one resolution across every extra,
  so naming Numba anywhere in `pyproject.toml` rolled the whole lock back to numpy 2.4.6, dev
  environment included — every other test would have run against a numpy nobody installs, to
  support an optional backend most callers never ask for. `[tool.uv] conflicts` would separate
  them at the price of declaring `sim` and `jit` mutually exclusive, which is false. So the task
  installs Numba into a throwaway environment where numpy resolves down on its own, and
  `tests/test_jit.py` skips itself everywhere else.
- `modelica.verify.check_selection`: the switching modelled with the linear algebra as the
  adversary — which set is usable is not the engine's to choose, and it only finds out by
  failing to factor one. The two assumptions that makes necessary are surfaced through
  `check_open` rather than swallowed, and the load-bearing one is executable:
  `quiescent=False` drops it and produces the run that breaks it.
- `modelica.fmi.modes`: the **FMI 3.0 calling sequence**, as one table, written before the FMI
  layer that has to obey it. Read three times — an exported FMU's guard (`Lifecycle.call`
  refuses an illegal call, which is what becomes `fmi3Error`), an importer's driver, and
  `check_modes`. Checking it immediately found Event Mode sitting in the Scheduled Execution
  table, where the standard has Clock Activation Mode and Clock Update Mode instead.
- `modelica.ssp.master`: the co-simulation master's protocol as pure transitions, and the
  finding that motivates writing it first — `canGetAndSetFMUState` is optional, so a
  conforming FMU may decline to save its state, and a master that rolls back on a discarded
  step then has no move left. Four moves from the start, and invisible to any test whose FMUs
  all happen to support state saving.
- Docs: *Verification* — what a model checker can and cannot say about a simulation engine,
  and the four places it pays. Written before any of it existed and corrected where building
  it disagreed with the plan, including one plan that turned out to be unnecessary.
- Docs: two **executable notebooks** (`docs/notebooks/`), run at build time by mkdocs-jupyter
  with `allow_errors: false`. A page whose code stopped working fails `mkdocs build --strict`
  in CI rather than sitting there showing output from a version that no longer exists. They
  are `.py` in percent format rather than `.ipynb` so a diff of one is readable.
- Docs: three **`.ipynb` notebooks** with matplotlib figures and ipywidgets sliders, meant to
  be opened and run rather than only read — *Turning the knobs* (parameter studies by
  rewriting a `FlatVariable.binding`, a damping sweep, overshoot and settling time against
  critical damping), *The shape of the equations* (the incidence matrix of a generated RC
  ladder, what alias elimination deletes from it, the block-lower-triangular staircase, and
  the pendulum's 3×3 block that cannot be sorted away), and *Where the smoothness breaks*
  (bounce events against the geometric series they should follow, the solver's step size
  collapsing into each event, both state sets' pivots crossing at 45°, and constraint drift
  never exceeding one machine epsilon). Committed with outputs stripped, so they diff like
  the percent files and cannot render results from an older version of the code; the site
  executes them like the others. matplotlib and ipywidgets joined the `docs` extra — the
  library itself still draws its own charts with no plotting dependency.

## 0.0.2

- Trademark notice: Modelica® / FMI / SSP belong to the Modelica Association; this is an
  independent implementation, not affiliated or endorsed. Stated in the README (so it is
  on the PyPI page), the docs, and the package docstring.

## 0.0.1

Project restart. The 2010 Python 2 / pyparsing prototype is retired (it remains in git
history); nothing from it is carried forward.

- Claim the `modelica` distribution name on PyPI.
- Repository moved from GitHub to GitLab (`gitlab.com/jorgeecardona/pymodelica`).
- Toolchain: uv + moon pinned via proto; ruff (`select = ["ALL"]`) and pyright strict.
- Single-job GitLab CI driven by `moon run :ci`; PyPI releases via OIDC Trusted
  Publishing; docs on GitLab Pages.
- Python 3.11+ only.
