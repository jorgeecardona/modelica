# The Rust backend, today

A model flattened by this package can be written out as one Rust source file that carries its
own integrator, and compiled with nothing but `rustc` into a binary that runs the model: no
Python, no SciPy, no runtime library. This page says what of that exists, how it is checked,
how to use it now, and what is still missing. The reasoning behind the design is in
`docs/issues/03-09-2026-a-rust-backend-is-a-third-dialect-plus-a-port-of-radau.md` and the
general shape of backends in [Backends](ideas/backends.md).

## What exists

**A third dialect.** `modelica.sim.dialect.RUST` sits beside `PYTHON` and `C`: operator
spellings, precedence, and the builtins, most of which map to `f64` associated functions. Only
`md_min`, `md_max`, `md_sign`, `md_mod`, `md_rem` and the dense LU `md_solve` are helpers in the
preamble.

**A structural emitter, `modelica.sim.rustgen`.** `unit(system)` returns the model as Rust:
the five kernel functions `rhs`, `observe`, `indicators`, `transition` and `jacobian`, in the
same decomposition `cgen` writes for C and the interpreter runs in Python. Torn and iterated
blocks come out as a Newton loop, a model with several state sets gets one function per set
behind a dispatcher that reads the switching slot, and a mode switch is a transition that
rebases the state into the destination set. The emitter refuses nothing in the corpus: every
model in `examples/models` emits.

**A port of the integrator, `modelica.sim.rust_radau`.** Radau IIA of order five, the same
method `modelica.sim.radau` implements in Python from RADAU5, in six pieces meant to be
concatenated after the emitted unit: the linear algebra (a real and a complex dense LU, since
the method factorises one real and one complex matrix per step), the method constants, the
step with its embedded error estimate and Gustafsson controller, the simplified-Newton stage
iteration, the starting-step heuristic, and the driver `md_integrate`.

**The event loop, inside the driver.** After each accepted step the driver watches every
indicator over the step's dense polynomial, locates the earliest crossing by bisection (the
root is the far end of the final bracket, so the state handed to the transition has crossed),
applies the transition, iterates over every latched relation the transition left disagreeing
with its latch, rearms each fired crossing on the side it settled, and restarts the stepper.
`when` branches, latched relations and mode switches are all handled. A standstill counter,
the interpreter's two numbers, stops a run whose events stop carrying it forward, so a relay
with no hysteresis ends where the interpreter's does instead of running forever. The driver
fills the states at every grid point and, when asked, the observed variables too, which is
what a run that changes state set mid-way is compared on.

## How it is checked

Everything is differential: the Rust unit against the Python one, on the same `System`, with
no reference numbers of the emitter's own. The Rust tests are skipped when there is no `rustc`.

| Test | What it holds |
| --- | --- |
| `tests/test_rust_dialect.py` | 26 expressions compiled with `rustc -D warnings` equal the interpreter. |
| per-function tests in `tests/test_rustgen.py` | `rhs`, `observe`, `indicators`, `jacobian` and every `transition` equal the interpreter at each model's start point, Newton blocks and rebasing switches included. |
| reference points | The driver, run at the `.toml`'s tolerance, reproduces every state reference point of the event-free models to the point's own bar, the terms the interpreter is held to. |
| events | For every model with events, the driver fires exactly the interpreter's crossings -- the same instants (to a microsecond), indices and causes -- and its trajectory agrees on a 41-point grid (`rtol` 1e-4, `atol` 1e-3): the bouncing ball's six bounces to 1e-14, the roaster's schedule, the thermostat, PhaseChange's twelve. |
| mode switch | Pendulum switches state set four times in two seconds; its coordinates, read from the observed-variable output, agree with the interpreter at 1e-8. |

215 tests in the two files, at the time of writing.

## Using it now

`modelica.sim.rustsim` is the packaged entry point: the model and the runtime as one source
behind a C function, compiled once with `rustc --crate-type cdylib` into a shared library kept
under the source's content hash (`~/.cache/pymodelica/rust`, or `$PYMODELICA_CACHE/rust`),
loaded through `ctypes`, and called with the same knobs as `modelica.sim.simulate`:

```python
from modelica.sim import rustsim

run = rustsim.simulate(system, stop=3.0, points=300, tolerance=1e-8)
run.result["h"]      # the same Result the interpreter returns: states and observed, by name
run.events           # how many crossings fired
run.event_list       # each as the interpreter's Event: instant, indicator index, and cause
run.steps, run.evaluations, run.jacobians
```

A model that has not changed is never compiled twice; a change to the model, the emitter or
the runtime is a different hash and a fresh build. `rustc` is needed only when the cache
misses. A run the driver stops -- a step below the spacing of numbers, a Newton iteration that
does not converge, a chattering model -- raises `SimulationError` with the driver's reason,
as the interpreter would.

For the pieces themselves, `rustsim.program(system)` is the source, and `tests/test_rustgen.py`
shows the same runtime driven from a `main`.

## What is missing

- **The chattering message.** The driver stops a chattering model where the interpreter does,
  with the status `CHATTERING`, but it returns a code where the interpreter names the relation
  that kept firing; the caller has to work that out from the model.
- **The browser form.** The native half of the packaging is built: `rustsim` compiles to a
  `cdylib` keyed by content hash and calls it across the FFI boundary. WebAssembly is the same
  source under `--target wasm32-unknown-unknown` and is not built yet, and the VS Code Scope and
  the notebooks still run the Python tier.
- ~~**Event causes.**~~ Done: the compiled run carries each crossing's instant and indicator
  index back over the FFI, and `rustsim` reads the cause from the model by that index, so
  `run.event_list` is the interpreter's `Result.events` -- instant, index and cause -- checked
  against it on the bouncing ball, the roaster's nine and a genset's load switch.
- **Size.** The dense LU is written for the single-digit state counts of the corpus. A
  method-of-lines model with hundreds of states is a separate decision.
- **Direction.** The plan of record is an interpreter over the flat IR, so a model runs with no
  compilation at all; the per-model emitter stays as the path that specialises one model to a
  fast artifact. See the issue page for the decision.
