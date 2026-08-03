---
title: The JIT compiles the kernel, not the step — the integrator loop is still Python
status: open
area: sim
owner: null
filed: 2026-08-03
kind: story
scope: L
verify: probe
blocked_by:
  - work-precision-harness
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
