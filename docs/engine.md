# The precompiled engine

`simulate()` runs a model in a precompiled Rust engine when the engine is installed. No compiler
runs on your machine. This page says when the engine runs, how to check, and how to turn it off.

## When the engine runs

`simulate(..., backend="auto")` is the default. It uses the engine when all of these are true:

- The engine is installed: a platform wheel of `modelica` carries it as `modelica/_engine.*`.
- The model is a `System`, not a `Code` or `Kernel`.
- `method="Radau"` (the default) and no `jit`.

In every other case it uses SciPy. `backend="python"` always uses SciPy. `backend="engine"`
raises a `SimulationError` when the engine cannot run the call.

Platform wheels exist for Linux x86_64, Linux aarch64, Windows x86_64 and macOS arm64. On other platforms,
`pip install modelica` installs the pure wheel, and `simulate()` uses SciPy.

## Check which one ran

`Result.backend` is `"engine"` or `"python"`:

```python
from pathlib import Path

from modelica.lang import parse
from modelica.sim import causalize, flatten, simulate

source = Path("examples/models/BouncingBall.mo").read_text()
system = causalize(flatten([parse(source)], "BouncingBall"))
result = simulate(system, stop=3.0, points=7, backend="engine")
print(result.backend, len(result.time), len(result.events))
print(result.time.round(3))
```

```text
engine 19 6
[0.    0.452 0.452 0.5   1.    1.174 1.174 1.5   1.752 1.752 2.    2.214
 2.214 2.5   2.584 2.584 2.88  2.88  3.   ]
```

## What comes back

The engine returns the same rows as SciPy:

- With `times` or `points`: the grid instants, the start and stop instants, and each event
  instant twice (the value before and after the event).
- With no grid: every accepted step, and each event instant twice.

A sweep (`modelica.sim.sweep.sweep`) loads the model into the engine once and runs every point
on it.

## How it works

`modelica.sim.vm` lowers the model to a register tape: one list of three-address instructions
for each of `rhs`, `observe`, `indicators`, `transition` and `jacobian`. The engine
(`engine/src/lib.rs`) runs the tape inside the same Radau integrator and event loop as the
[Rust backend](rust-backend.md). The design is in
`docs/issues/08-11-2026-a-modelica-vm-runs-the-model-as-bytecode-in-a-precompiled-rust-engine.md`.

On the 80-state vowel_i model, the engine takes the same steps as the compiled Rust backend and
costs about 1.5 to 2 times more per step.

## From a source checkout

A source checkout has no installed engine, so `backend="auto"` uses SciPy. `backend="engine"`
builds the engine once with `rustc` from `engine/src` and keeps it in the cache directory.
