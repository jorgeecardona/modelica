# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org/); the version in `pyproject.toml` is the single
source of truth and is bumped automatically when shipped source changes.

## Unreleased

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
  chosen set one variable at a time. Nothing switches between them yet; this is the menu the
  eventual dynamic state selection chooses from.
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
