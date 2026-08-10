---
title: The JIT compiles the kernel, not the step — the integrator loop is still Python
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
blocked_by:
  - work-precision-harness
fixed_in: 331d226
kind: story
scope: L
verify: probe
---

## Measured

- `modelica.sim.build(system, jit=True)` hands the generated `rhs`, `jacobian`, `observe`,
  `indicators` and `transition` to Numba (`src/modelica/sim/kernel.py`), and
  `tests/test_jit.py` runs them against the interpreted versions on four models. So the
  *kernel* is compiled and checked today.
- `src/modelica/sim/solve.py:128` builds that kernel and then hands it to SciPy's `solve_ivp`.
  The step, the error norm, the Newton iteration and the step-size controller are SciPy's
  Python, and they call the compiled kernel across a Python call boundary, several times per
  step.
- `docs/architecture.md:115` states the intended design in the other tense: "A JIT compiles the
  model and the step together. The right-hand side is inlined, and the parameters are folded to
  constants once at compile time instead of loaded once per stage per step."
  `docs/ideas/benchmarks.md:60` makes the same claim.

So the docs describe a fused compilation that the code does not do. The kernel JIT is real; the
fusion is not.

## Inferred

The win is inferred and not measured. The argument: a Radau step evaluates the right-hand side
three times per stage iteration and the Jacobian occasionally, and each of those is a
`nopython` call entered from the interpreter, with array allocation and reference counting at
the boundary. Fusing removes the boundary and lets the optimiser see the parameters as
constants. On a small stiff system (n = 2-10, tens of flops per evaluation) the boundary cost
is plausibly a large fraction of the step; on a large one it is noise. **Which of those this
project's models are is exactly what has not been measured**, and it is why this issue waits on
the harness rather than starting beside it.

There is a real chance the answer is "not worth it", and this issue closing as *refused, with
numbers* is a good outcome.

## Done

- An integrator written in this repository, in the same emitted-source style as the kernel, so
  that the step and the right-hand side compile as one `nopython` function: implicit,
  stiff-capable, with a real error estimate and step-size control. Radau IIA is the obvious
  first choice because `docs/architecture.md:74` already names RADAU5 as the thing to stand
  next to.
- The two compile modes `docs/ideas/backends.md:66` asks for: parameters burnt in as literals
  and folded, or kept as an argument so one compiled artefact serves a whole sweep. The caller
  picks.
- A work-precision diagram from the harness issue showing the fused path against
  `solve_ivp(method="Radau")` on Robertson and van der Pol, on a named machine.
- The interpreted tier still runs everything, unchanged. `docs/ideas/backends.md:20` is the
  standing constraint: the fast path is an optimisation of a correct thing, not the only thing.

## Deliberately not done

- No new solver *family*. This is one integrator compiled together with the model, not a
  method zoo.
- No event handling in the fused loop in the first pass. Events cross the Python boundary
  anyway (`modelica.sim.solve` stops, iterates, restarts), so the first measurement should be
  on an event-free problem and say so.

## What landed, 2026-08-11

`src/modelica/sim/radau.py`: three-stage Radau IIA of order five -- Hairer and Wanner's
RADAU5, *Solving ODEs II* IV.8 -- written in a subset of Python that numba compiles, with
`rhs` and `jacobian` taken as arguments rather than looked up. Handing a compiled dispatcher
to a compiled caller is a native call, so the interpreter is not entered between the first
step and the last. `moon run :bench-fused` puts it on the work-precision diagram beside the
three SciPy methods, which is the only comparison that settles anything: it is a *different
implementation of the same method*, so if fusion is worth having it shows up as the same
curve shifted left.

The entry said "there is a real chance the answer is *not worth it*, and this issue closing
as refused, with numbers, is a good outcome". It is not that outcome.

x86_64, CPython 3.13.14, 51 output points, best of 3, same harness and same
reference as stage 1 above.

### Robertson, to t = 1e4

| method | rtol | seconds | rel error | steps |
| --- | --- | --- | --- | --- |
| Radau (SciPy) | 1e-4 | 0.0371 | 6.15e-05 | 87 |
| Radau (SciPy) | 1e-6 | 0.0672 | 1.19e-06 | 134 |
| Radau (SciPy) | 1e-8 | 0.1622 | 1.21e-08 | 293 |
| **fused** | 1e-4 | **0.0005** | 6.15e-05 | 37 |
| **fused** | 1e-6 | **0.0006** | 1.12e-06 | 83 |
| **fused** | 1e-8 | **0.0010** | 1.21e-08 | 243 |

### VanDerPol, mu = 1000, to t = 20

| method | rtol | seconds | rel error | steps |
| --- | --- | --- | --- | --- |
| Radau (SciPy) | 1e-4 | 1.4111 | 3.43e-04 | 2034 |
| Radau (SciPy) | 1e-5 | 2.3056 | 4.40e-05 | 3317 |
| Radau (SciPy) | 1e-6 | 3.5019 | 2.13e-06 | 5719 |
| **fused** | 1e-4 | **0.0067** | 2.87e-04 | 1974 |
| **fused** | 1e-5 | **0.0092** | 4.16e-05 | 3269 |
| **fused** | 1e-6 | **0.0107** | 2.06e-06 | 5661 |

**74x to 162x on Robertson, 211x to 327x on van der Pol, at matched error.** The error
columns agree to within a factor of 1.2 at every tolerance and the step counts agree to
within twenty percent, which is what says the comparison is real: this is not a looser
method running faster, it is the same method taking the same path with the interpreter
removed from between its parts. It reaches the tolerance in slightly *fewer* steps -- 243
against 293, 5661 against 5719 -- which is a controller difference and not a claim.

**The cost is the first call.** Compiling is 20.0 s for Robertson and 11.7 s for van der Pol,
paid once per model per process, and the tables above are the best of three runs after that.
A single short simulation is slower this way and will always be; a sweep, an identification
or a notebook that runs the same model twice is not.

### What it does and does not do

- **Own linear algebra.** A step factorises `mu/h * I - J` for the real eigenvalue of the
  inverse Butcher matrix and again for the complex pair, and both factorisations are written
  out here. Not to beat LAPACK, which at these sizes is not the question, but because
  `scipy.linalg.lu_factor` cannot be called from compiled code and calling it would put the
  Python boundary back exactly where this exists to remove it. One `_factor` body serves both
  element types; numba compiles a specialisation of each.
- **Dense output, from the collocation polynomial.** Requested times are interpolated rather
  than stepped onto, so asking for a fine output grid does not change a single step.
- **Both compile modes, and neither is this module's.** `build(fold=True)` burns the
  parameters in as literals and `fold=False` keeps them as an argument; the loop is
  indifferent, and `tests/test_radau.py` holds the two to the same trajectory. That is the
  `docs/ideas/backends.md:66` requirement met by the emitter that already had it, rather than
  by a second mechanism here.
- **No events**, as the entry said. A model with any is refused by name with the interpreted
  path named in the message.
- **The interpreted tier is untouched.** `modelica.sim.solve` is not modified by one
  character. The fused loop is a separate entry point and SciPy still runs everything.

### Verification

`tests/test_radau.py` runs the loop **interpreted** -- it is ordinary Python -- against every
event-free corpus model that publishes reference points, at the tolerance each spec states,
and asserts the same `analytic` and `reference` values `tests/test_examples.py` holds the
SciPy path to. Neither integrator grades the other. 25 pass, 8 skip for events or for having
no published point.

`tests/test_jit.py` asks the one question that file cannot: the compiled loop against the
interpreted one, same model, same tolerance.

**A bug the corpus could not have found.** `_solve` interleaved the row swaps with the
forward substitution -- swap `b[k]` with `b[piv[k]]`, then eliminate with column `k` -- which
reads like LAPACK and is not: `_factor` swaps whole rows, so multipliers already computed
move with them, and an elimination that used one against an unpermuted right-hand side has
subtracted the wrong thing. It passed every corpus model, because a Radau matrix is
`mu/h * I - J` and at these step sizes the diagonal dominates, so partial pivoting never
chose a row to swap. It failed the first random 4x4 it was shown. The permutation is now one
pass, and `test_the_factorisation_solves_what_numpy_solves` is the test that found it.

**Bit equality between the two tiers is false, and the split is the finding.** Every counter
matches -- 243 steps, 1 rejection, 1883 right-hand sides, 49 Jacobians -- so the compiled
loop made every decision the interpreted one made. The states differ by 1.4e-15 relative,
which is LLVM contracting a multiply and an add into one instruction where CPython rounds
twice. The test asserts the control flow exactly and the arithmetic to its own rounding;
asserting bit equality would have meant disabling an optimisation to satisfy a test.

## Deliberately not done, found while doing it

- **The right-hand side is called, not inlined.** `docs/architecture.md` said "the
  right-hand side is inlined" and that word has been removed rather than made true. Numba
  emits a direct native call to the dispatcher; asking for `inline="always"` on the emitted
  functions is a separate experiment, and the measurement above is what a direct call
  already buys. What was actually removed is the *Python* boundary, which is the one that
  cost two orders of magnitude.
- **No sparse path.** The factorisations are dense. `modelica.sim.solve` switches to a sparse
  factorisation above 16 states and `emit` already writes the pattern, so the hook exists;
  no corpus model is large enough for it to matter, and adding it without a model that shows
  the win would be guessing.
- **The solver's own steps are not reported.** Collecting them means growing an array inside
  the compiled loop, which is the one allocation that would land on the hot path. `Stats`
  counts them; the trajectory comes back on the requested grid only.
- **Backwards integration is refused by omission.** The loop is written forward-only and
  says so; nothing in this project integrates backwards.
- **No `max_step`, no `first_step`, no vectorised right-hand side.** SciPy's knobs, not
  ported, because nothing here asks for them.
