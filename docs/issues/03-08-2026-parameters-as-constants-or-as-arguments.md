---
title: Parameters are always an argument, and burning them in as literals is the other half
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: 78c1c82
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

## What landed

`emit(system, fold=True)` and `build(model, fold=True)`. `tests/test_fold.py`, 45 tests.
1099 pass.

Each Done bullet:

- **The choice.** One keyword, default unchanged. Folding evaluates each parameter in the
  order `Emitter.ordered` already computes for the emitted `parameters` function, substitutes
  the literal into every expression `Emitter.expr` prints, and lets `simplify` fold whatever
  that exposes. It is the two passes the entry predicted and no new machinery.
- **The two agree.** Every corpus model, both modes, 21 sample points over the model's own
  stop time: **exactly zero difference**, bit for bit, on every variable. The test asserts
  zero rather than a tolerance, because folding replaces a name with the number that name
  held and the arithmetic after it is the same operations in the same order. A tolerance
  there would hide the day that stops being true.
- **The difference, measured.** Below.

### Source size, all 21 models

| | argument | folded | change |
| --- | --- | --- | --- |
| whole corpus | 158,681 B | 145,124 B | **-8.5 %** |
| `BumpStop` | 10,918 | 8,431 | -22.8 % |
| `ProbeRC` | 6,348 | 5,358 | -15.6 % |
| `ProbeLC` | 6,768 | 5,768 | -14.8 % |
| `Pendulum` | 13,122 | 11,258 | -14.2 % |
| `PIControl` | 5,626 | 4,890 | -13.1 % |
| `VanDerPol` | 1,581 | 1,613 | **+2.0 %** |
| `RoastingMachine` | 13,581 | 13,594 | **+0.1 %** |

Sixteen of 21 shrink. The five that grow are not a failure of the fold: a literal is longer
than the name it replaced — `2260000.0` is nine characters where `L` is one — and a model
whose parameters are read once each and compared against nothing has nothing to gain and
those characters to lose.

### Runtime, Intel i9-9880H, CPython 3.12.7, best of 5 x 50,000 calls

| model | `rhs` argument | `rhs` folded | change |
| --- | --- | --- | --- |
| `BumpStop` | 2.395 µs | 1.334 µs | **-44.3 %** |
| `RCCircuit` | 0.950 µs | 0.564 µs | -40.6 % |
| `Pendulum` | 4.625 µs | 2.993 µs | -35.3 % |
| `RoastingMachine` | 3.703 µs | 2.588 µs | -30.1 % |
| `VanDerPol` | 0.838 µs | 0.768 µs | -8.3 % |
| `Robertson` | 1.121 µs | 1.145 µs | +2.2 % |

Whole-simulation wall clock moves much less — between -12 % and +7 % — because the solver,
not the right-hand side, is what those runs spend their time in.

### The entry's inference about `RoastingMachine` was wrong

It predicted that folding "plausibly removes whole branches" from the gas schedule, since it
is "a chain of comparisons against parameters". It is not: the schedule compares `time`
against numeric literals — `if time < 66.7 then 20 elseif ...` — so there is no parameter in
those conditions and nothing to fold. `RoastingMachine` comes out 13 bytes *larger*. Its
`rhs` is still 30 % faster, and that comes from the arithmetic, not from any branch.

## Deliberately not done, found while doing it

- **A latent bug in CSE placement, which folding exposed.** `_define` put a shared name in
  front of the first line *containing* it, tested with `in`. `__s2` is a substring of
  `__s23`, so a short name landed in front of a longer one's first use — which can be before
  the assignment it reads. `AtwoodPendulum` folded produced `__s2 = 0.35 * der_w1` twenty-eight
  lines above `der_w1 = _x[0]` and the generated Jacobian raised `UnboundLocalError`. Fixed
  with a word-boundary match; there is a test that reads the generated text and asserts no
  shared name is read before it is written. The old docstring's argument — that the prefix
  keeps generated names away from Modelica symbols — was true and did not cover the generated
  names colliding with *each other*.
- **`p` is still filled, and still exists.** `Code.parameters` is unchanged, `parameters(p)`
  is still emitted, and `Kernel.start` still calls it, because the workspace layout is shared
  between the modes and that is what makes the differential check meaningful. What a folded
  unit does not do is *read* it: there is a test over every model asserting no `= p[...]`
  line survives. A caller who edits `p` in folded mode changes nothing, and `observe` reports
  the value that was compiled in rather than the one in the array.
- **No C-side measurement.** `modelica.sim.cgen` takes the same `Code`, so folding reaches it
  for free, and what `-O2` already folds for itself was not measured. The claim above is
  about the Python tier.
- **No automatic choice**, as the entry required.
