---
title: Parameters are always an argument, and burning them in as literals is the other half
status: open
area: sim
owner: null
filed: 2026-08-03
kind: story
scope: S
verify: probe
---

## Measured

- The emitted functions all take `p` and read parameters out of it:
  `T_amb = p[0]`, `m_green = p[1]`, and so on. There is one mode and the caller cannot choose
  it.
- `docs/ideas/backends.md:66` describes two: "**Specialisation is a choice, not a policy.**
  Burn the parameters in as literals and fold them, or leave them as an argument so that one
  compiled artefact serves a whole sweep. Two compile modes, and the caller picks. vLLM makes
  the same decision about what to bake into a CUDA graph and what to keep as a runtime input."
- The folding machinery exists and is used elsewhere: `modelica.ir.derive.simplify` folds
  through the arithmetic constructors, and `substitute` is already used by `causalize` and
  `rewrite`. Emitting with parameters substituted is those two passes applied before the
  emitter, not new code.

## Inferred

The win is inferred and it is not the obvious one. Constant-folding a parameter does not save
an array read — it lets *everything downstream* of that parameter fold too: a branch on a
parameter disappears, `k * 0.0` disappears, a division by a constant becomes a
multiplication. On `RoastingMachine`, whose gas schedule is a chain of comparisons against
parameters, that plausibly removes whole branches from the compiled code. Plausibly. Nothing
here has measured it, and the honest first step is to emit both and read the two sources.

The relationship to the ensemble driver is worth stating so the two do not collide: **the
ensemble needs the argument mode**, which is what exists today, so nothing there is blocked by
this. This issue adds the *other* mode, for the case where one model is run many times with
one set of parameters — a notebook, a long simulation, an exported FMU with its parameters
fixed at export time.

## Done

- `emit` (and `build` through it) takes the choice: parameters as an argument, which is the
  default and what exists today, or substituted as literals and folded.
- The two agree. Every corpus model, both modes, same trajectory to the solver's own
  tolerance — the differential check the emitter's other claims already use.
- The difference is recorded on at least one model: source size and node count in each mode,
  and a timing on a named machine. If the folded source is not smaller, that is the finding
  and this closes as "measured, not worth it".

## Deliberately not done

- No automatic choice. The caller picks; a compiler that guesses which mode you wanted is
  a compiler whose output you cannot predict.
- Nothing about *structural* parameters (a parameter that sets an array size). Those are not
  optional to fold and they arrive with the arrays work.
