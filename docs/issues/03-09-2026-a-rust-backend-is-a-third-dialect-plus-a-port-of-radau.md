---
title: A Rust backend is a third dialect plus a port of radau.py, and it compiles a model to a self-contained native binary (and to WASM)
status: in-progress
kind: idea
scope: L
owner: jorgeecardona
filed: 2026-09-03
area: sim
verify: probe
blocked_by: null
fixed_in: null
---

## Measured

The same three pieces the TypeScript idea named
(`docs/issues/11-08-2026-a-typescript-backend-is-a-third-dialect-plus-a-port-of-radau.md`)
exist, and one of them has already been built for Rust.

- **`modelica.sim.dialect` parameterises the emitter by target language, and `RUST` is now
  a third `Dialect`.** `PYTHON`, `C` and `RUST` are three values -- operator spellings,
  precedence levels, function names -- threaded through `emit_expr(..., dialect=...)`. Most
  builtins map to `f64::` associated functions; only `md_min/max/sign/mod/rem` and an
  `md_solve` dense-LU are helpers in the preamble.
- **`tests/test_rust_dialect.py` already proves the expression tier.** 26 cases compiled with
  `rustc -D warnings`, run, and compared to the interpreted tier -- the standing rule for
  every backend here. So "the same emitter, pointed at a different set of templates" is done
  once for expressions in Rust, exactly as `cgen` did it for C.
- **`modelica.sim.cgen` is the structural template.** It shows the part a dialect table cannot
  hold -- the Newton loop, the function wrapper, the semicolon, the row-at-a-time Jacobian --
  written once for C. A Rust backend is a `rustgen` sibling that reuses the same `Emitter`
  decisions (blocks, schedule, `block_jacobian`, tangents) and writes Rust structure.
- **`modelica.sim.radau.py` is a hand-written Radau IIA** (three-stage, order five, RADAU5,
  Hairer and Wanner IV.8), 847 lines in a numba-compilable subset -- scalars, arrays, loops,
  no object model, `rhs`/`jacobian` as arguments. That subset ports to Rust almost directly.

So the missing piece is not the front end, the flattener, the emitter, or the method. It is a
`rustgen` (cgen's structure in Rust) and a port of one file.

## Inferred

Inferred, not measured: Rust buys what neither C nor TypeScript gives at once.

- **A self-contained native binary.** No Python, no Pyodide, no server, no runtime library:
  `rustc` on the emitted source is the whole toolchain, and the artifact is one file that
  runs anywhere the target does. This is the "fast, independent of one machine, no external
  libraries" goal in one shot.
- **The browser story the TypeScript idea wanted, without TypeScript.** Rust compiles to
  WebAssembly natively (`--target wasm32-unknown-unknown`), so the same emitted source that
  makes a native binary also makes a module a page can run. The TS idea's whole reason to
  exist (a model in the browser) falls out of the Rust one, and the artifact is still source
  a reader can inspect.

And the honest cost, the same one the TypeScript idea named: **the linear algebra is what a
port cannot borrow.** `radau.py` factorises two matrices per step -- one real `n`-by-`n` and
one complex `n`-by-`n` (the method's one real eigenvalue and its complex conjugate pair). The
real LU is already written out here as `md_solve`; the **complex LU is the piece still to
write**. For the corpus's sizes (states in single digits) that is a hundred lines and no
dependency; for the method-of-lines sizes it is a real decision and probably the boundary of
what this backend is for.

## What would turn this into a story

A spike, in this order, each answerable on its own -- the same order the TypeScript idea set,
with the method named as radau5 rather than left open:

1. **The kernel.** A `rustgen` that emits `rhs` for one corpus model, compared against the
   interpreted tier at the model's start point (as `tests/test_cgen.py` does for C). Start
   with the straight-line subset (single state set, assignment and linear blocks, no events)
   and grow it to Newton, the mode dispatcher, and the row-at-a-time tangent, refusing what is
   not yet written with a named `EmitError` the way `cgen` does. Example-driven: take a real
   corpus model, emit it, run it, and let what breaks say what to build next.
2. **The integrator.** Port `radau.py` to Rust with a dense real LU (`md_solve`, done) and a
   dense complex LU (to write) in place of the factorisations SciPy cannot lend a compiled
   caller, and reproduce one corpus model's reference points from its `.toml` in `node`.
3. **The event loop.** `indicators`, root location and `transition` -- this is where a port
   stops being mechanical, because the standstill counter and the event iteration are
   `solve.py`, not `radau.py`. A model with events is refused until this lands, exactly as
   `radau.py` refuses one today.

## Deliberately not done

- **No linking of Fortran.** RADAU5/DASSL/ODEPACK exist and are battle-tested, and linking one
  would undo all three goals at once: a Fortran toolchain per platform (not machine
  independent), an external dependency (not dependency-free), and an integrator that calls the
  right-hand side through an opaque function pointer -- the exact thing `docs/ideas/backends.md`
  builds the compiler to avoid. The Fortran is the specification; the port is validated against
  the interpreter and against the `.toml` reference points.
- **No method-of-lines sizes yet.** The dense complex LU is written for the single-digit
  states the corpus has. A stencil unrolled to hundreds of states is a separate decision, the
  same boundary the TypeScript idea drew.
- **No FMU/WASM packaging yet.** Step 1 emits source and a `main` that runs it; a build step,
  a crate, or a `wasm-bindgen` surface is only worth asking about once the integrator and the
  event loop land.
