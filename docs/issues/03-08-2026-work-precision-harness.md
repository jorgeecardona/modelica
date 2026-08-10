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

## Partial, 2026-08-10: the harness exists and stage 1 is half measured

`scripts/bench.py` and `moon run :bench`. Not closed: stage 1 has one of its two problems, and
stages 2 and 3 are untouched. What follows is what was measured, including the two numbers
that stopped it.

### Robertson, and it is a real work-precision diagram

x86_64, CPython 3.12.7, to t = 1e4, 51 output points, best of 2. Reference: `Radau` and
`LSODA` at rtol = atol = 1e-12, **computed here, not the Bari published solution**.

| method | rtol | seconds | rel error | steps |
| --- | --- | --- | --- | --- |
| Radau | 1e-4 | 0.0167 | 6.15e-05 | 87 |
| Radau | 1e-6 | 0.0305 | 1.19e-06 | 134 |
| Radau | 1e-8 | 0.0755 | 1.21e-08 | 293 |
| Radau | 1e-10 | 0.2144 | 2.00e-10 | 797 |
| BDF | 1e-4 | 0.0179 | 1.12e-03 | 135 |
| BDF | 1e-6 | 0.0362 | 3.10e-05 | 226 |
| BDF | 1e-8 | 0.0693 | 4.73e-07 | 396 |
| BDF | 1e-10 | 0.1275 | 1.27e-08 | 709 |
| LSODA | 1e-4 | 0.0035 | 1.57e-03 | 164 |
| LSODA | 1e-6 | 0.0064 | 1.70e-04 | 285 |
| LSODA | 1e-8 | 0.0122 | 6.61e-06 | 506 |
| LSODA | 1e-10 | 0.0225 | 1.44e-06 | 906 |

**And this is why the diagram has two axes.** Compared at the same *requested tolerance*,
`LSODA` looks 10x faster than `Radau`. Compared at the same *achieved error* -- about 1.3e-6 --
`LSODA` takes 0.0225 s and `Radau` takes 0.0305 s, so it is 1.35x. The ten is an artefact of
`LSODA` delivering four orders less accuracy than it was asked for; the 1.35 is the answer.
Every speed claim made without an error axis is that first number.

### `VanDerPol` stopped it, twice, and both are findings

**It is too slow to sweep at its own horizon.** At mu = 1000 the period is about 1600, so the
`.toml`'s stop of 3000 is two relaxation cycles. Measured with `Radau`:

    stop 3000   rtol 1e-04   118.3 s
    stop  300   rtol 1e-04    11.7 s
    stop  300   rtol 1e-10   265.4 s
    stop   20   rtol 1e-10    16.5 s

A sweep of three methods over four tolerances at stop = 3000 is hours. The harness runs it to
20, which is the first sharp corner, and says so where the number is set.

**A self-computed reference is not a reference for it.** At stop = 20 the harness refused to
print a curve at all:

    the two reference methods disagree by 7.04e-08, which is more than the loosest
    tolerance under test (1e-10) -- there is no error axis

That is the guard doing its job. `Radau` and `LSODA` at rtol = atol = 1e-12 do not agree on
this problem to better than 7e-8, so an error measured against either is measuring the
reference. **`VanDerPol` needs the Bari published solution**, and that is now a concrete
prerequisite rather than a preference -- which is the opposite of what the entry assumed when
it listed the two problems together as equally ready.

### What is left

- **Stage 1, second half.** `VanDerPol` with the Bari reference values, which have to be
  obtained rather than computed.
- **Stage 2.** `HIRES`, `E5`, `Ring Modulator`. None is in the corpus; each is a model to
  write and validate first.
- **Stage 3.** `ScalableTestSuite`, which is an external Modelica library.
- **No diagram is drawn.** `moon run :bench` prints a table and writes JSON. A plot is a
  reader's convenience and the numbers are the claim.

## Deliberately not done, found while doing it

- **Not a CI gate**, as the entry required. `runInCI: false`, `cache: false`, and the reason
  is at the task.
- **`_SPARSE_STATES` is 16 and every problem here is under it**, so these numbers are the
  dense path. A sparse work-precision curve is a different run and the harness does not force
  either.
- **The reference bar may be too generous.** The guard compares the two witnesses against the
  *loosest* tolerance under test; a reference should be far tighter than the *tightest*. It
  fired anyway on `VanDerPol`, so the weaker bar was enough this time.
