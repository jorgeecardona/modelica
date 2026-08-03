---
title: The ensemble driver — one compiled model over a matrix of parameters
status: open
area: sim
owner: null
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
