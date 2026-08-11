---
title: A TypeScript backend is a third dialect plus a port of radau.py, and it would put models in the browser
status: open
area: sim
owner: null
filed: 2026-08-11
kind: idea
scope: L
verify: probe
---

## Measured

Three pieces that would have to be built for this already exist.

- **`modelica.sim.dialect` parameterises the emitter by target language.** `PYTHON` at
  `dialect.py:129` and `C` at `dialect.py:234` are two `Dialect` values — operator spellings,
  precedence levels, function names — and `emit_expr(..., dialect=...)` threads one through
  every expression the emitter prints. A third entry is the shape the module already has.
- **`modelica.sim.cgen` is a working second backend.** So "the same emitter, pointed at a
  different set of templates" (`emit.py:8`) is not a plan; it has been done once.
- **`modelica.sim.radau.py` is a hand-written Radau IIA**, three-stage, order five, after
  Hairer and Wanner IV.8 — and its docstring says it is "written in a subset of Python that
  numba compiles, with `rhs` and `jacobian` taken as arguments". That subset is scalars,
  arrays, loops and no Python object model, which is close to the intersection of Python and
  TypeScript.

So the missing piece is not the front end, not the flattener, not the emitter, and not the
integration method. It is a port of one file and a third dialect table.

## Inferred

Inferred, not measured: the reason to want this is that it removes a whole class of problem
rather than solving one. A model that runs in the browser needs no kernel, no Pyodide, no
server, and no round trip — so the documentation could let a reader change a parameter and
watch the trajectory move, an example could be embedded in a page anywhere, and a model
could be shown to somebody who will not install anything.

Also inferred, and it is the honest cost: **the linear algebra is what a port cannot borrow.**
`radau.py` calls `np.linalg.solve` on the stage system, and the sparse path calls SuperLU
through SciPy. A browser has neither. For the corpus's sizes (states in single digits) a
dense LU written out is a hundred lines and no dependency; for the method-of-lines sizes it
is a real decision and probably the boundary of what this backend is for.

`Rumoca` emits JavaScript already (`docs/ideas/landscape.md`), which is evidence that the
target is reasonable rather than exotic — and it is one of the few things it emits that this
project cannot.

## What would turn this into a story

A spike, in this order, each answerable on its own:

1. **The kernel.** Add a `TYPESCRIPT` dialect and emit `rhs` for one corpus model. Compare
   its output against the interpreted tier on the same inputs, which is the standing rule for
   every backend here. If the dialect table is the only change, that is the finding.
2. **The integrator.** Port `radau.py` with a dense LU written in place of `np.linalg.solve`,
   and reproduce one corpus model's reference points from its `.toml` in `node`.
3. **The event loop.** `indicators`, root location and `transition` — this is where a port
   stops being mechanical, because the standstill counter and the event iteration are
   `solve.py`, not `radau.py`.

Only then is it worth asking about a build step, a package, or a page that uses it.

## Deliberately not done

- **Not needed for interactive documentation.** A precomputed sweep — run the ensemble
  driver at build time, ship the curves, let a slider pick one — gives an interactive figure
  with no runtime at all, and `modelica.sim.sweep` is already the driver for it. That is the
  cheaper answer to the question this idea arrived attached to, and it is what was chosen.
- No WebAssembly. Emitting C and compiling it with Emscripten is a different route with a
  build step; the point of the TypeScript one is that the artifact is source you can read.
