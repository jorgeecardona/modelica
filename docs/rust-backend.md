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
`when` branches, latched relations and mode switches are all handled. The driver fills the
states at every grid point and, when asked, the observed variables too, which is what a run
that changes state set mid-way is compared on.

## How it is checked

Everything is differential: the Rust unit against the Python one, on the same `System`, with
no reference numbers of the emitter's own. The Rust tests are skipped when there is no `rustc`.

| Test | What it holds |
| --- | --- |
| `tests/test_rust_dialect.py` | 26 expressions compiled with `rustc -D warnings` equal the interpreter. |
| per-function tests in `tests/test_rustgen.py` | `rhs`, `observe`, `indicators`, `jacobian` and every `transition` equal the interpreter at each model's start point, Newton blocks and rebasing switches included. |
| reference points | The driver reproduces every `.toml` state reference point of the event-free models to 1e-6. |
| events | For every model with events, the driver applies exactly the interpreter's number of crossings and its trajectory agrees on a 41-point grid (`rtol` 1e-4, `atol` 1e-3): the bouncing ball to 1e-14, the roaster's schedule, the thermostat, PhaseChange's twelve. |
| mode switch | Pendulum switches state set four times in two seconds; its coordinates, read from the observed-variable output, agree with the interpreter at 1e-8. |

215 tests in the two files, at the time of writing.

## Using it now

There is no packaged entry point yet. What the tests do is the way to use it: emit the unit,
append the runtime, add a `main`, compile.

```python
from modelica.sim.rustgen import unit
from modelica.sim.rust_radau import CONSTANTS, LINALG, STEP, NEWTON, FIRST_STEP, INTEGRATE

source = "\n".join([unit(system), *CONSTANTS, *LINALG, *STEP, *NEWTON, *FIRST_STEP, *INTEGRATE])
```

`main` builds the start point from `kernel.start(0.0)` (states, parameters, workspace), the
event directions and latch slots from `kernel.code.events`, a grid, and calls
`md_integrate(t0, t1, &y0, &p, &mut w, rtol, atol, &directions, &latches, &grid, &mut out,
&mut obs)`; `tests/test_rustgen.py` has the complete program.

## What is missing

- **The chattering guard.** The interpreter counts events that do not advance time and stops
  with a message naming what kept firing. The Rust driver has no such counter; a chattering
  model runs until the horizon or forever.
- **Packaging.** The execution model is decided (compile once to a `cdylib`, keyed by the
  system's content hash, loaded across an FFI boundary; WASM for the browser) and none of it is
  built. The VS Code Scope and the notebooks still run the Python tier.
- **Size.** The dense LU is written for the single-digit state counts of the corpus. A
  method-of-lines model with hundreds of states is a separate decision.
- **Direction.** The plan of record is an interpreter over the flat IR, so a model runs with no
  compilation at all; the per-model emitter stays as the path that specialises one model to a
  fast artifact. See the issue page for the decision.
