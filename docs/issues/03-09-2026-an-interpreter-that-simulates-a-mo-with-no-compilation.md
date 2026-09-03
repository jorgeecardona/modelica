---
title: An interpreter that simulates a .mo (or its flat IR) with no compilation and no artifact
status: open
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

Two backend strategies exist for taking a model to a trajectory, and they differ in what gets
compiled.

- **Specialise-and-compile.** `modelica.sim.cgen` and `modelica.sim.rustgen` emit a model's
  ``rhs``/``jacobian`` as source, and a compiler turns *that model* into machine code. Fast per
  model, but simulating a new model needs a compile of that model. The Rust path reaches a full
  integration this way (`docs/issues/03-09-2026-a-rust-backend-is-a-third-dialect-plus-a-port-of-radau.md`).
- **Interpret the IR.** `modelica.ir.evaluate` walks the flat model's expression IR as data:
  no compile, any model, and it is the reference tier every backend is checked against. It is
  slow because it is a tree-walk in Python.

## Inferred

What the owner wants is the *second* strategy made fast, in several languages: a system that
receives a `.mo` (or its flat IR) and simulates it with **no compilation and no artifact** --
an interpreter of models, not a compiler of them. The model is data the interpreter reads.

- The value is distribution: one binary, built once in CI, simulates any model handed to it.
  Nothing is compiled per model, no `rustc`/`cc` is shipped or invoked at simulate time, and the
  same artifact serves every model in a corpus.
- The right split for a first cut is Python flattens, the fast tier interprets. This package
  already owns the parser, flattener, index reduction and the symbolic Jacobian; those are not
  performance-critical and need not be re-implemented. It serialises the flat model -- variables,
  the block-ordered ``rhs`` computation, and the Jacobian -- to a portable **bytecode** (a flat
  array of numbers: opcodes and operands), and the interpreter executes that bytecode over a
  value stack. Bytecode rather than JSON keeps the loader dependency-free: no ``serde``, no crate,
  just numbers.
- The integrator is done. `modelica.sim.rust_radau` ports Radau IIA to Rust and integrates a
  model to its `.toml` reference points. Only the *source* of ``rhs``/``jacobian`` changes: from
  compiled functions to bytecode the VM runs. The loop is unchanged.
- Backends become **interpreters** over the one flat IR: Python (`modelica.ir.evaluate`, exists),
  then Rust, then C, then TypeScript/JS. Each is a VM, each checked against the interpreter, each
  built once and fed models as data.

## What would turn this into a story

A spike, in this order:

1. **The bytecode.** A serialiser (in this package) that lowers a flat model's ``rhs`` to a flat
   numeric bytecode, and a Rust stack VM that executes it, checked against `modelica.ir.evaluate`
   at the start point -- the same bar `rustgen` meets, but the model is data, not code.
2. **The Jacobian, and drive it.** Lower the symbolic Jacobian to bytecode too, feed both to
   `rust_radau`'s integrator in place of the emitted functions, and reproduce a corpus model's
   `.toml` reference points with one un-recompiled VM.
3. **The other tiers.** The same VM in C, then TS/JS -- each a small interpreter over the same
   bytecode, each checked against the reference tier. Then `.mo` read directly in Rust (boundary
   b), if the front end is worth porting.

## Deliberately not done

- **rustgen is not dropped.** The specialise-and-compile path stays as the "ship one model as a
  standalone fast artifact / FMU" option. This idea is the general simulator beside it, not a
  replacement.
- **No `.mo` parsing in Rust yet.** Boundary (a) -- Python flattens, Rust interprets the flat IR
  -- lands first; parsing `.mo` in Rust (boundary b) is a separate, larger decision.
