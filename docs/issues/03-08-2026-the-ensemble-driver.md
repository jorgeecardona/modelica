---
title: The ensemble driver — one compiled model over a matrix of parameters
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
kind: story
scope: M
verify: test
---

## Measured

- `modelica.sim.simulate` takes one model and returns one `Result`
  (`src/modelica/sim/solve.py:101`). There is no way to ask for many trajectories of the same
  model at different parameter values other than calling it in a Python loop, which recompiles
  or at best re-enters the kernel once per run.
- `docs/ideas/backends.md:66` already names the compile mode this needs: "leave them as an
  argument so that one compiled artefact serves a whole sweep. Two compile modes, and the
  caller picks." Neither mode is selectable today —
  `grep -n "def build" src/modelica/sim/kernel.py` takes `jit` and nothing about parameters.
- `docs/ideas/vulkan.md:111` states the ordering explicitly: "The first GPU-worthy target is
  **not** the solver. It is the ensemble driver: run one compiled model over a matrix of
  parameters. Design it single-threaded first. The shape is already the shape a GPU wants."

## Inferred

Why this is the highest-value item in `docs/ideas/vulkan.md`'s list, inferred from what
Modelica models are used for after they first run: parameter identification against
measurements, uncertainty propagation, and Monte Carlo over component tolerances are all
thousands of independent integrations of one identical right-hand side
(`docs/ideas/vulkan.md:40`). The roasting work in `logs/2026-07-30.md` is the local instance —
five batches fitted jointly with per-batch initial conditions, done by hand in a scratch
script, which is an ensemble sweep written out longhand.

This is inference about *demand*, not a measurement. Nobody has yet been blocked by the missing
API; the sweeps so far were small enough to loop.

## Done

- `modelica.sim` builds a kernel once with parameters left as an argument, and runs it over an
  array of parameter vectors, returning results indexed by the sweep.
- The compile happens once. The observation that closes this: sweeping *n* parameter sets costs
  one compile and *n* integrations, and that is visible as wall time that does not grow with a
  per-run compile — measurable by comparing a 1-point and a 100-point sweep of the same model.
- Correctness is differential: every point of the sweep equals what `simulate` returns for that
  parameter value on its own, to the solver's own tolerance.
- The API is single-threaded and has no opinion about where the loop runs. That is what makes
  it the interface a GPU or a process pool can be put behind later.

## Deliberately not done

- No parallelism, no threads, no GPU. `docs/ideas/vulkan.md:112` says design it
  single-threaded first, and the SPIR-V spike names this issue in `blocked_by` rather than the
  other way around.
- No optimiser and no fitting loop on top. Least squares over a sweep is a caller's job; this
  issue is the sweep.

## What landed

`modelica.sim.sweep`, exported as `modelica.sim.sweep`, `.grid` and `.vector`.
`tests/test_sweep.py`, 9 tests. 1236 pass.

    from modelica.sim import sweep, grid
    results = sweep(system, grid(R=[1.0, 2.0], C=[0.5, 1.0]), stop=3.0)

Each Done bullet:

- **Built once, run over an array, indexed by the sweep.** `sweep(system, over, **options)`
  returns one `Result` per point, in the order given. `options` go straight to `simulate`, so
  the sweep adds no knobs of its own.
- **The compile happens once, and it is counted rather than timed.** `cache_info().misses` is
  1 after a hundred points, which is an integer and does not flake on a loaded machine the
  way a wall-clock comparison does. The wall time the entry asked for, on `RoastingMachine`
  to t = 200 s, i9-9880H:

  | points | total | each | compiles |
  | --- | --- | --- | --- |
  | 1 | 0.036 s | 36.2 ms | 1 |
  | 10 | 0.203 s | 20.3 ms | 1 |
  | 100 | 1.831 s | 18.3 ms | 1 |

  Per-point cost *falls* from 36 to 18 ms as the one compile is amortised, and then flattens.
  That flat 18 ms is the integration.
- **Correctness is differential.** Every point equals what `simulate` returns for that
  parameter value on its own, and the assertion is exact equality rather than a tolerance:
  the same kernel, the same vector, the same solver.
- **Single-threaded, no opinion about where the loop runs.** No threads, no pool, no device.
  There is a test that a sweep run backwards gives the same answers as one run forwards,
  which is the property a pool or a queue would need and which is checkable while the loop is
  still a `for`.

### The trap, which is where the work went

**A parameter is not an array slot.** Parameters are computed in dependency order --
`RoastingMachine` writes `m_dry = m_green * (1 - moisture)` -- so overriding `m_green` by
writing into the parameter vector leaves `m_dry` holding a value derived from the *old* one.
That produces a model which is nearly the one the caller asked for, integrates happily, and
draws a smooth plausible wrong curve.

`modelica.sim.emit.settings` recomputes every parameter downstream of a change, in the order
`Emitter.ordered` already computes for the emitted `parameters` function, so the two cannot
disagree about what a parameter is worth. There is a test that the Python side and the
generated module produce the same default vector on four models.

The same trap has a second half: **the initial conditions are computed from the parameters
too.** `m_w` starts at `m_green * moisture`, so a swept moisture has to move where the run
*begins*. `Kernel.start` takes the vector and computes `initial` and `latch` from it, and
there is a test that a swept moisture of 0.20 starts `m_w` at 2.76 kg rather than 1.794.

## Deliberately not done, found while doing it

- **`over`, not `points`.** `simulate` already has a `points` and it means the number of
  output samples of *one* run. Two meanings for one word in one call is how a caller writes
  `sweep(system, points=..., points=...)` and finds out at runtime. Named in the docstring so
  the next reader does not rename it back.
- **`simulate` grew a `parameters` argument**, which is useful on its own -- "run this model
  with R = 2" needed a hand-built kernel before -- and is what the sweep is a loop over.
- **`grid(**axes)` is the cross product** and nothing more. A sweep along a line, a Latin
  hypercube or a sampled distribution are all just iterables of mappings, and `sweep` takes
  any of them; only the cross product was worth a helper because `itertools.product` with the
  names zipped back on is the part people get subtly wrong.
- **No parallelism**, as the entry required, and no fitting loop on top.
- **A test had to stop `BouncingBall` at 1 second.** A swept restitution reaches its Zeno
  point at a different instant per point, and past about 2 s the lively ones raise the
  chatter guard rather than returning -- so a sweep over restitution is a sweep whose points
  do not all survive the same horizon. Worth knowing before somebody sweeps a parameter that
  changes *how long a run is valid for*.
