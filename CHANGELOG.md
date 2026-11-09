# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org/); the version in `pyproject.toml` is the single
source of truth and is bumped automatically when shipped source changes.

## Unreleased

- `when sample(start, interval)` is a time event: the integrator steps to each sample instant exactly, with no root search, on SciPy and on the engine.
- The observe tape uses array opcodes too; on a 50 x 50 TorusHeat it is 21 instructions, down from 22506, and lowering takes 1.7 s instead of 4.1 s.
- The Jacobian tape is built from the array opcodes, not from a second scalar lowering; TorusHeat Jacobian is 414 instructions, down from 1090.
- The engine loads and stores runs of states with one instruction; TorusHeat `rhs` is 18 instructions, down from 144.
- **`pip install modelica` is enough to simulate.** NumPy and SciPy are now plain dependencies;
  `modelica[sim]` still installs and adds nothing.
- **Large array models flatten and causalize faster.** A `for` loop's equations are processed
  once instead of once per cell. On a 32 x 32 heat grid, flattening goes from 1.07 s to 0.19 s and
  causalization from 1.74 s to 0.53 s.
- **The engine computes time-only work once per time.** It caches what depends only on `time`
  for the last four times. vowel_i needs 16% fewer instructions, with bit-identical results.
- **The engine factors in a banded state order.** It reorders the states so the Radau matrix is
  banded and skips the zeros outside the band. vowel_i needs less than half the instructions.
- **A precompiled engine runs a model with no compiler per model.** `simulate(..., backend="engine")`
  lowers the model to a register tape (`modelica.sim.vm`) and runs it in a Rust engine with the
  same Radau integrator as `rustsim`. The Linux x86_64 wheel carries the engine as
  `modelica/_engine.so`; the pure wheel builds it once with `rustc` on first use. `simulate()` uses the
  engine by default (`backend="auto"`) when it is installed and can run the call, else SciPy.
- **A sweep on the engine uses every core.** 16 vowel_i runs take 4.7 s instead of 27 s on 16
  logical cores, with the same results bit for bit.
- **The engine runs about a quarter fewer instructions.** The tape folds constant subexpressions
  and keeps constants in registers set once, as compiled code does.
- **The engine runs without an output grid, and returns the rows SciPy returns.** With no `times`
  or `points` it returns every accepted step. With a grid it adds the start, the stop and both
  sides of each event, as SciPy does.
- **Engine wheels for Linux aarch64, Windows x86_64 and macOS arm64**, beside Linux x86_64. The page
  "The precompiled engine" in the docs says when `simulate()` uses it.
- **A parameter sweep runs on the engine.** `sweep()` loads the model once and runs every point
  on it, with the same `backend` choice as `simulate()`.
- **The engine runs from any `start` time**, not only from 0.
- **`Result.backend` says who integrated** (`"engine"` or `"python"`).
- **A requested output grid returns only the grid.** With `times` or `points`, SciPy no longer
  adds its own steps to the result. The two sides of each event stay.
- **A parameter with no binding takes its `start` value.** The specification lets a parameter stand
  on its `start` attribute (with a tool warning) when nothing binds it, and the machines rely on it:
  a synchronous machine declares `parameter Boolean useDamperCage(start = true)` with no binding and
  gates its damper-cage heat port on it, so `start` is the value the condition must read. An unbound
  parameter now evaluates its `start` expression against the known values and records the result, so
  a conditional component keyed on such a parameter resolves.
- **The elementwise operators `.*`, `./`, `.^` (and `.+`, `.-`).** Modelica's dotted operators are
  elementwise between two arrays: `x .* y` is the elementwise product `*` is not (a `*` between two
  arrays is a contraction), and `./` and `.^` are the elementwise divide and power that plain `/`
  and `^` refuse. The parser now accepts them (they were tokenised but rejected); `.+`/`.-` fold to
  `+`/`-`, which are already elementwise for arrays, and `.*`/`./`/`.^` keep their own operator that
  the flattener reads out elementwise -- each element is its plain scalar sibling, so causalization,
  the Jacobian and every emitter (the Rust backend included) see ordinary `*`/`/`/`^` and need no
  new rule. A plain `/` or `^` between two arrays now names its dotted form in the refusal.
- **`transpose(A)` swaps a matrix's two indices.** `transpose(A)[i, j]` is `A[j, i]`, whether it is
  written whole (`B = transpose(A)`) or read through a product (`transpose(A) * x`, the linear form
  a state-space model and a quadratic form use). It is expanded where an array equation scalarises,
  so causalization, the Jacobian and every emitter (the Rust backend included) see the swapped
  element; a non-square `A[m, n]` transposes to `A[n, m]`.
- **`+` and `-` between two arrays are elementwise.** `x + y` and `x - y` for two same-shaped arrays
  are the elementwise sum and difference, as `der(x) = p - x` (an array settling to a constant array)
  needs. The shape pass had refused any non-`*` operator between two arrays outright, so this had to
  be written as a `for` loop, even though the element pass already read it correctly; now only `/`
  and `^` (which Modelica spells `./` and `.^`) are refused between two arrays. `*` stays the
  contraction it was.
- **Array constructors work in an equation, not only a declaration.** `der(x) = A * x + ones(n)`,
  `der(x) = fill(2, n) - x`: a constructor written in an equation is expanded to its literal array
  the same way a declaration's is, reading its shape off its own size arguments (structural, known
  at flatten). This completes the constructors added just above, which were scoped to declarations.
- **The array-constructor functions in a declaration: `zeros`, `ones`, `fill`, `identity`,
  `linspace`.** `Real x[n](start = zeros(n))`, `parameter Real A[n, n] = identity(n)`,
  `parameter Real g[n] = linspace(0, 1, n)` -- the ordinary way to give an array declaration its
  start or its value. Each is array-valued and structural, and is now expanded at flatten into the
  literal array of the declared shape, exactly as a written `{...}` would be; before, the whole call
  reached the emitter as an untranslated function and the model would not build. `zeros`/`ones`/
  `fill` cover any shape (a vector or a matrix); `identity(n)` is the square unit matrix (a
  non-square shape is refused); `linspace(a, b, n)` steps evenly with the endpoints kept exact. The
  fold happens where an array declaration already scalarises, so causalization, the Jacobian and
  every emitter (the Rust backend included) see the plain per-element values.
- `examples/MixingRing`: **a closed ring of stirred tanks, a conservation law written with array
  reductions.** N tanks are pumped one way around a loop, each fed from its predecessor and draining
  to its successor at the same rate, `der(c[i]) = k * (c[i-1] - c[i])` with the predecessor wrapping
  around. Every row of the transport matrix sums to zero, so the total `sum(c)` is conserved for all
  time; the model holds that total as an algebraic variable and it stays at its starting value. A
  tracer injected in one tank circulates around the loop as it disperses (the transport matrix has a
  complex-conjugate eigenvalue pair) and settles to the common mean. It is the first corpus model to
  put the array reductions on a physical law: `sum(c)` for the conserved total, and `min(c)` in an
  assert that guards the model's positivity. Linear, so its reference points are exact (scipy DOP853
  at 1e-12), and both backends hold the total to 1e-9.
- **Array reductions work in a `when` condition and an assert, not only an equation.** A
  reduction is folded as an equation is scalarised, but a `when` condition and an assert never
  pass through that split, so `when sum(x) < 1` reached the emitter as an untranslated `sum` and
  failed, and `assert(sum(x) < top, ...)` was quietly skipped (its condition named the array `x`,
  which the run does not report as a variable, only its elements `x[1]`, `x[2]`). Now the fold
  runs in the flattener's expression rewrite, so a reduction in either context becomes the
  arithmetic it stands for: the `when` is an ordinary crossing of `x[1] + x[2]`, and the assert
  guards the real total and stops the run when it is violated.
- **Array reductions fold in a parameter binding and an attribute too.** `parameter Real total =
  sum(x0)`, `Real y(start = sum(x0))` -- a leaf is not an equation, so it never passes through the
  equation split; the fold now runs where the leaf is built, once the array it reduces is declared
  and its shape is known. `sum`, `min` and `product` each reduce their array parameter to the scalar
  the binding stands for, in both backends. (An array declared after the leaf that reduces it is the
  one case left: its shape is not yet known there, so that reduction still needs an equation.)
- **Array reductions fold to a scalar: `sum`, `product`, `min`, `max` over a whole array.**
  `total = sum(x)` for a vector `x` is `x[1] + x[2] + ... + x[n]`, `product(x)` the same with
  `*`, and `min(x)` / `max(x)` fold through the pairwise `min(a, b)` / `max(a, b)`. Before this
  the scalariser treated a reduction as elementwise and mangled `sum(x)` into `sum(x[1])`, one
  copy per equation element, so a conservation sum or an extremum over an array could not be
  written. The fold happens at flatten, so causalization, the Jacobian and every emitter (the
  Rust backend included) see ordinary arithmetic and need no reduction rule of their own; a
  two-argument `min(a, b)` stays the pairwise operator it was. A 2-D array folds every element.
  This completes the array arc that began with whole-array equations and array products.
- **An inconsistent algebraic loop is named, not a `ZeroDivisionError`.** `a = b + 1` and
  `b = a + 1` say `a - b = 1` and `a - b = -1` at once: no solution. Alias elimination reduces this
  to a single equation whose unknown has a zero coefficient, and the emitted code used to divide by
  that literal zero, so a real model with a contradictory pair of equations failed at runtime with a
  bare `ZeroDivisionError` and no clue which equations were at fault. Causalization now checks the
  pivot before it emits: a coefficient that folds to zero is refused with the model name, the
  offending equation as Modelica source, and the words "singular algebraic loop". A consistent loop
  (`a = 2*b + x`, `b = a - 1`) still solves.
- `examples/WaveString`: **the wave equation on a string with fixed ends, the second PDE in the
  corpus.** A string of length 1 held at both ends, discretised in space by the method of lines --
  eight interior points, each obeying `u'' = c^2 * (second difference of its neighbours)`, the same
  recipe `HeatBar` and `TorusHeat` use but for a hyperbolic equation, so the string oscillates
  instead of decaying. The initial shape is the fundamental mode `sin(pi*x)`, an exact eigenvector
  of the discrete Laplacian, so the string swings at one frequency: a standing wave with the exact
  solution `u_i(t) = sin(pi*i/9) cos(omega t)`, `omega = 18 sin(pi/18)`. Linear and undamped, it has
  that closed form (the reference points are analytic) and a conserved energy, the wave analogue of
  the Hamiltonian benchmarks. At sixteen states with a tridiagonal Jacobian it is the second model
  on the sparse-Jacobian path, alongside the 2-D `TorusHeat`.
- `examples/CoupledOscillators`: **two masses coupled by a spring, written as a state-space system
  `der(z) = A * z`.** The first corpus model in the matrix-ODE form the array-product feature added:
  the whole right-hand side is one matrix-vector multiply, the companion matrix `A` carrying the
  springs. Started with one mass pulled aside, the motion is a beat -- the in-phase mode at
  frequency 1 and the out-of-phase mode at `sqrt(2)` interfere and pass the energy back and forth,
  `x1 = (cos(t) + cos(sqrt(2) t))/2`. Linear and undamped, so it has a closed-form solution (the
  reference points are analytic) and a conserved energy, a benchmark with an exact reference like
  the Hamiltonian ones. The `build` DSL's `parameter`/`constant` gained a `shape=` argument to
  declare the array parameter its twin needs.
- **Array products scalarise: the dot product, matrix-vector, and matrix-matrix.** A `*` between
  two arrays is a contraction, not an elementwise operation, and is now expanded into the sums it
  stands for: `der(x) = A * x` (the linear system the feature is named for) becomes
  `der(x[i]) = sum_j A[i, j] * x[j]`, a dot product `a * x` becomes a scalar sum, and a
  matrix-matrix `A * B` becomes the matrix product. The last dimension of the left runs against the
  first of the right and both drop out, so a non-conforming product is refused by name. Array `/`
  and `^` still have no elementwise reading and are refused with the `for` loop as the way to write
  them. This closes the last array-equation gap, leaving elementwise equations (already supported)
  and products handled together.
- `examples/SphericalPendulum`: **a pendulum free to swing in 3D, the first genuine
  spherical-coordinate trajectory.** A bob on a rigid rod traces a path on a sphere: in the angles
  (`theta` from the downward vertical, `phi` around it) the motion is the Lagrangian one, with the
  Coriolis coupling that conserves the vertical angular momentum keeping the bob off the poles.
  Started at `theta = 1` rad with a pure azimuthal rate, it oscillates in `theta` between about 0.47
  and 1.0 rad while `phi` precesses about 5.5 turns, a rosette wrapped on the sphere. Two quantities
  are conserved -- the energy and the vertical angular momentum `sin(theta)^2 * w_phi` -- and grade
  the integrator as in the Kepler, Henon-Heiles and Schwarzschild benchmarks; both hold to about
  1e-12. The rigid rod keeps the radius constant, so the state reads directly as `(r, theta, phi)`
  for the new 3-D spherical view. Six scipy reference points at 1e-12, matched to 1e-10.
- **`homotopy(actual, simplified)` is taken at its actual value.** The standard operator that
  helps an initialisation converge by walking a path from a simplified model to the actual one is
  now supported: with no continuation run, its value is the actual (the first argument), which is
  what the expression is worth once initialisation is done. It evaluates, differentiates and emits
  (interpreter and Rust backend alike) as that first argument -- the opposite end from `noEvent`
  and `smooth`, which are transparent to their last -- and the simplified branch is discarded,
  including any relations in it, so they do not become events nothing reads. This was the last
  operator still refused.
- **`terminate(message)` ends a run at the instant its `when` fires.** The standard Modelica
  operator that stops the analysis when a model is defined only up to some condition -- a tank
  that empties, a decay past a threshold -- is now supported end to end, in the interpreter and
  the compiled Rust backend alike. It is a state event like any other: the `when` condition is a
  zero crossing located the ordinary way, and the only new thing is the action, "stop and report"
  instead of `reinit`. The run keeps the trajectory up to the located instant and nothing past it,
  and records the message as the event's cause. Written as ``when cond then terminate("reason");
  end when;``.
- `examples/BouncingBallRest`: **a bouncing ball that comes to rest, the well-posed companion to
  BouncingBall.** BouncingBall accumulates infinitely many bounces at a finite time (Zeno) and
  then, past the event limit, falls through the floor. This one carries a discrete flag that
  sticks the ball to the floor once a rebound would be smaller than `v_min`, so the whole run is
  defined: the height stays in `[0, 1]` and the ball is provably at rest for good after the last
  bounce, checked with `after` invariants. It shows the standard way to resolve Zeno -- a rest
  mode -- rather than a terminal time.
- **Array equations scalarise element by element.** An equation written between whole arrays --
  `der(x) = -x` for `Real x[2]`, `der(T) = -T` for `Real T[2, 2]` -- is now expanded into one scalar
  equation per element, the same split the declaration already does for the variables, so a vector
  field no longer has to be written as an explicit `for` loop over its elements (`TorusHeat` still
  uses the loop; it now has the shorter form beside it). An array-valued `start` is split the same
  way, so `Real x[2](start = {1, 2})` gives its elements two different starts. A scalar broadcasts
  across the array (`der(x) = -a * x + 1`, `start = x0`), and an elementwise function call
  (`-sin(x)`) is read out per element. What is not elementwise is refused rather than mis-expanded:
  an array product like `x * y` (a contraction, not a per-element operation) says so and names the
  `for` loop as the way to write it.
- `examples/TorusHeat`: **the heat equation on a flat torus, a 2-D method-of-lines PDE.** Diffusion
  over an 8x8 periodic grid (64 states), the space discretised by hand and the edges wrapped with
  `mod` so there is no boundary -- HeatBar's one-dimensional recipe on the simplest manifold. The
  initial condition is a single grid Fourier mode, an exact eigenvector of the discrete periodic
  Laplacian, so the field keeps its shape and decays in closed form, `T(t) = T(0)*exp(lambda*t)`
  with `lambda = 4*a*(cos(2*pi/n) - 1)`. The reference is that exact decay, checked at three cells,
  and a node of the mode is pinned at zero to show the discretisation preserves the eigenmode. It
  exercises 2-D arrays, an `initial equation` over the grid indices, and the sparse Jacobian.
- `examples/SchwarzschildOrbit`: **a relativistic orbit, the reachable general-relativity rung.**
  A test mass on an equatorial timelike geodesic of the Schwarzschild metric -- the Kepler problem
  with the relativistic `-3*M*ell^2/r^4` term that makes a bound orbit precess into a rosette rather
  than close. Started at `r = 20M` with `ell = 4M`, it swings between perihelion `8.2M` and aphelion
  `20M`, advancing about 2.78 rad of precession per radial cycle (Mercury's 43"/century, in the
  strong field). The specific energy `E^2 = vr^2 + (1 - 2M/r)(1 + ell^2/r^2) = 0.936` is conserved,
  held to 1e-6 (Radau drifts it by 1e-14), the same conserved-quantity check as Kepler and
  Henon-Heiles, now in GR. Six scipy reference points at 1e-12 over nearly four orbits.
- `examples/HenonHeiles`: **the Henon-Heiles system, a second Hamiltonian benchmark.** A star in a
  galactic potential `0.5*(x^2 + y^2) + x^2*y - y^3/3`, two degrees of freedom, run at energy 0.1125
  (below the escape energy 1/6, a bounded regular orbit). Where Kepler is integrable and closed-form,
  this one is non-integrable; both carry the energy as the invariant that grades the integrator.
  Radau holds it to 1e-13 over fifty time units, and the orbit is regular enough that a 1e-11 run
  and a 1e-12 reference stay within 3e-6, so its six scipy points pin the trajectory point for point.
- `examples/KeplerOrbit`: **the Kepler two-body problem, the canonical Hamiltonian benchmark.**
  An eccentric orbit (e = 0.6, period 2*pi) under inverse-square gravity, run for three orbits.
  Its point is its two constants of motion, held as invariants rather than integrated: the energy
  `0.5*(vx^2 + vy^2) - mu/r = -0.5` and the angular momentum `x*vy - y*vx = 0.8`. Radau holds both
  to below 1e-12 over the three orbits; the invariants pin them at 1e-6, the margin a longer,
  drift-accumulating run would spend. Six scipy reference points at 1e-12.
- `modelica.ir.causalize`: **a second derivative may be written `der(der(x))`.** The pass that
  turns `der(v)` into the variable `der_v` insisted its argument be a plain variable, so a model
  that wrote the second derivative out was refused, even though index reduction already names
  `der_der_x` when it differentiates an equation twice. It now recurses into the argument: the
  inner `der(x)` becomes `der_x`, and this der is then `der(der_x)` -> `der_der_x`. So
  `der(der(x)) = -x` is a harmonic oscillator with `x` and `der_x` as its two states, to any depth,
  and a model need no longer introduce `v = der(x)` by hand.
- `modelica.ir.flatten`, `modelica.build`: **structural if-equations.** An `if ... else ... end if`
  in an equation section now selects, when the model is flattened, the branch whose condition holds
  -- the way a conditional component's `if` already does -- and keeps only that branch's equations;
  `elseif` chains and an `else` fall out. The condition has to be known at flatten time, a parameter
  or a constant; a condition on a continuous variable is refused, because switching which equations
  exist at run time is a `when`, a different machine. `build.if_equation` writes one from the Python
  DSL, and the `Projectile` example uses it to choose the air-drag law from a Boolean parameter.
- `modelica.ir.evaluate`, `modelica.ir.derive`: **`smooth(order, expr)` evaluates and
  differentiates.** The operator was already transparent to the emitter but unknown to the
  interpreter and the differentiator, so a model that wrapped a term in it ran through compiled
  code yet was refused at the Jacobian with "no derivative rule for 'smooth'". Both tiers now see
  through it to its expression argument, the way they do `noEvent`; the continuity order is a hint
  to the solver and drops.
- `examples`: **four cleanly-referenced prototypes.** `TwoTank` (interacting tank levels with
  Torricelli coupling, the textbook process-control plant), `HeatExchanger` (two fluid streams
  trading heat through a wall with its own inertia, the first flow-through thermal model),
  `CoulombOscillator` (a spring-mass damped by dry friction via `tanh`, whose amplitude decays
  linearly not exponentially), and `Projectile` (a ball thrown up with quadratic air drag, the law
  chosen by a structural if-equation). Each carries scipy reference points at 1e-12 and bounded
  invariants, and between them they exercise `tanh`, `atan2`, `sqrt` and the if-equation in real
  trajectories.
- `modelica.ir.flatten`: **causal signal connectors, so `Modelica.Blocks` flattens.** A
  connector written `connector RealInput = input Real` *is* a scalar, not a record of fields:
  it aliases a predefined type and carries a causality but holds no members. It used to hit the
  refusal that only a `type` may extend a predefined type. It now lays down as one leaf variable,
  the way a `type` alias does, and stays a connector, so `connect(out, in)` equates the two -- an
  output driving an input, the whole point of a signal wire; two outputs on one wire are refused
  by name. The existing connection machinery needed no change. The bundled standard-library slice
  gained `Modelica.Blocks` (`Sources.Constant`, `Math.Gain`/`Feedback`, `Continuous.Integrator`/
  `FirstOrder`), wired by the real `RealInput`/`RealOutput`.
- `modelica.sim.rustsim`: **the compiled run carries the events back in full.** `md_integrate`
  gained two output buffers it writes each located crossing's instant and indicator index into,
  threaded through the FFI entry point; `rustsim` reads the cause from the model by that index, so
  `Run.event_list` is the interpreter's `Result.events` -- instant, index and cause. The bouncing
  ball's six bounces, the roaster's nine and a genset's load switch match the interpreter's events
  exactly (instants to a microsecond); a smooth model carries none.
- `modelica.sim.rustsim`: **the Rust backend, packaged.** `rustsim.simulate(system, stop, ...)`
  takes the same knobs as `simulate` and returns the same `Result`, from a shared library:
  the model's unit and the radau runtime as one source behind a C entry point, compiled once
  with `rustc --crate-type cdylib`, kept under the source's content hash so an unchanged model
  is never compiled twice, and called through `ctypes`. A run the driver stops raises
  `SimulationError` with the driver's reason. Checked against the interpreter on a smooth
  model, an event model and one that switches state sets; the event tallies agree exactly.
- `modelica.sim.rust_radau`: **the chattering guard.** A relay with no hysteresis reaches its
  threshold and then crosses it at every step, each crossing carrying the run forward by the
  rearm nudge and nothing more; the Rust driver ran such a model to the horizon, or forever.
  It now keeps the interpreter's standstill counter, the same two numbers, and stops with the
  status `CHATTERING` where the interpreter raises, to within 1e-2 on the chattering model.
- `docs/rust-backend.md`: **the Rust backend, today**: a standing account of what exists (the
  dialect, `rustgen`, the `radau` port with its event loop, dispatcher and Newton), how it is
  checked (215 differential tests against the interpreter), how to use it now (emit, append the
  runtime, compile with `rustc`), and what is missing (the chattering guard, the packaging, the
  interpreter pivot). The issue page that planned it records where it stands.
- `modelica.sim.emit`: **an event's reported condition names the alias the model wrote.**
  Alias elimination folds `probe = T_probe - 273.15` into every read of `probe`, so the
  roaster's `when probe >= T_drop` reached the event list, the chattering report and the
  sweep's response menu as `when T_probe + -273.15 >= T_drop`. The reported text now writes an
  eliminated alias back by name wherever a subtree is exactly its value; a bare-name alias is
  left alone, since renaming every read of the other variable is a choice the model did not
  make. The indicators the integrator watches are untouched.
- `modelica.sim.rust_radau`: **observed variables at every grid point.** `md_integrate` takes
  a further output and fills it with `observe` in whichever state set the run is in, which is
  what a run that switches sets is compared on: Pendulum's coordinates across its four switches
  now agree with the interpreter at 1e-8, the last corpus trajectory the Rust backend could not
  check.
- `modelica.sim.rust_radau`: **a located root lands past the crossing.** The bisection returned
  the midpoint of its final bracket, a hair before the crossing, so a relation sharing the
  threshold with the one that fired (PhaseChange's `T > Ms` and `T < Ms`) was not judged stale
  and each re-crossed its rearm nudge: four events at one instant where the interpreter fires
  two. The far end of the bracket is returned now, and the counts agree, 12 and 12.
- `examples/LotkaVolterra`: **prey and predators on a closed orbit**, with the conserved
  quantity `delta x - gamma ln x + beta y - alpha ln y` held to 1e-7 over four periods as an
  invariant: exact at every instant, so its drift is the integrator's drift and nothing else.
- `examples/Rossler`: **chaos from one nonlinear term**, with a Lyapunov exponent thirteen
  times smaller than Lorenz's, so a reference point late in the run is still a fair test of the
  integrator rather than of the arithmetic. Both new models' `.mo` files are generated from
  their Python twins, so the two front ends build one tree by construction.
- `examples/TwoInertias`: **two bodies bolted to one shaft** — index 3 again, and for a
  different reason than the pendulum. The constraint `big.phi = small.phi` is between two
  *states*, so Pantelides differentiates it twice and the dummy-derivative method throws away
  one of each pair. What makes it worth a model is that the two candidate state sets are the
  same shape: `big` and `small` differ only in a number, so nothing in the structure prefers
  either and the answer comes from a tie-break. `candidates` is a new key in the `.toml`,
  ordered with the chosen set first, so that a change in the selection rule shows up as a diff
  in a spec file rather than as a silent change of behaviour. Every reference number is the
  closed form `w(t) = 10 exp(-t/10)`, and the conserved angular impulse `J w + d phi` is
  asserted flat at 25 over the whole run.
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
