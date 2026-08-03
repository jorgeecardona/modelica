---
title: No work-precision harness, so every speed claim in the docs is unmeasured
status: open
area: sim
owner: null
filed: 2026-08-03
kind: story
scope: L
verify: probe
---

## Measured

- `moon.yml` has `lint`, `format-check`, `typecheck`, `test`, `test-python-floor`, `test-jit`,
  `docs`, `build`, `version-guard`, `ci`, `publish`, `release`. There is no benchmark task.
- `grep -rn "solve_ivp" src/modelica/sim/solve.py` shows the SciPy backend is what runs today;
  nothing times it against anything.
- The docs make timing claims in four places: `docs/architecture.md:110` ("Matching the Fortran
  codes is the floor"), `docs/architecture.md:130` ("the target is to be *faster* on the classic
  problems"), `docs/ideas/backends.md:60`, and all of `docs/ideas/benchmarks.md`.
- One number exists, from `logs/2026-08-01.md`: `RoastingMachine/Radau`, 494 steps against 494,
  0.578 s against 0.541 s. That is a single point, on one machine, on a model that is not in any
  published test set. `docs/ideas/benchmarks.md:96` calls that an anecdote, in those words.

## Inferred

Nothing here is inferred about the *gap*. What is inferred is the ordering consequence: until
this exists, no optimisation in `modelica.sim` can be accepted or rejected on evidence, which
is why the fused-step JIT issue names this one in `blocked_by`. An optimisation landed without
a harness is a change of unknown sign.

## Done

`moon run :bench` produces a work-precision diagram — achieved error against CPU time, swept
over tolerances, one curve per method — and the numbers are reproducible from a clean checkout.

Staged as `docs/ideas/benchmarks.md:84` sets it out:

1. **Robertson and van der Pol.** Both are in the Bari IVP test set with high-precision
   reference solutions, which is what makes an error axis possible at all. Against SciPy's
   `Radau`, `BDF` and `LSODA` on the same problem.
2. **HIRES, E5, Ring Modulator.** Enough dimension and stiffness that the Jacobian strategy and
   the linear solve start to show.
3. **ScalableTestSuite.** Compile time enters, and what is under test is the symbolic pipeline
   rather than the integrator. OpenModelica publishes per-branch results against it, so there
   is a public baseline instead of a number from a paper.

Every published figure names four things, per `docs/ideas/benchmarks.md:94`: the problem, the
tolerance sweep, the reference solution the error axis uses, and the machine.

`verify: probe` and not `test`, deliberately. A green test would only say the harness ran. What
closes this is the measurement itself, on a machine that is named.

## Deliberately not done

- Not a CI gate. Timings on shared runners are noise, and a flaky performance gate teaches
  people to ignore the pipeline. This is a task somebody runs and records.
- No FLOPS number, ever. `docs/ideas/benchmarks.md:8` gives the argument: on Robertson an
  explicit method takes 10^7-10^8 steps where a stiff one takes a few hundred, and no
  throughput figure shows that.
