---
title: No work-precision harness, so every speed claim in the docs is unmeasured
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: 56a7b62
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

## Stage 1, 2026-08-10

`scripts/bench.py` and `moon run :bench`. Stage 1 is measured; stages 2 and 3 are not, and are
split off below. `verify: probe` -- the numbers and the machine follow.

x86_64, CPython 3.12.7, 51 output points, best of 2. Reference: `Radau` and `LSODA` at
rtol = atol = 1e-12, **computed here, not the Bari published solution**.

### Robertson, to t = 1e4

| method | rtol | seconds | rel error | steps |
| --- | --- | --- | --- | --- |
| Radau | 1e-4 | 0.0165 | 6.15e-05 | 87 |
| Radau | 1e-6 | 0.0303 | 1.19e-06 | 134 |
| Radau | 1e-8 | 0.0749 | 1.21e-08 | 293 |
| BDF | 1e-4 | 0.0177 | 1.12e-03 | 135 |
| BDF | 1e-6 | 0.0356 | 3.10e-05 | 226 |
| BDF | 1e-8 | 0.0676 | 4.73e-07 | 396 |
| LSODA | 1e-4 | 0.0034 | 1.57e-03 | 164 |
| LSODA | 1e-6 | 0.0065 | 1.70e-04 | 285 |
| LSODA | 1e-8 | 0.0123 | 6.61e-06 | 506 |

### VanDerPol, mu = 1000, to t = 20

| method | rtol | seconds | rel error | steps |
| --- | --- | --- | --- | --- |
| Radau | 1e-4 | 0.7475 | 3.43e-04 | 2034 |
| Radau | 1e-5 | 1.1472 | 4.40e-05 | 3317 |
| Radau | 1e-6 | 1.8747 | 2.13e-06 | 5719 |
| BDF | 1e-4 | 0.7471 | **4.33e-01** | 3513 |
| BDF | 1e-5 | 1.0672 | 1.97e-02 | 5112 |
| BDF | 1e-6 | 1.5668 | 4.27e-03 | 7714 |
| LSODA | 1e-4 | 0.1107 | **9.60e-01** | 4442 |
| LSODA | 1e-5 | 0.1400 | 3.97e-02 | 5659 |
| LSODA | 1e-6 | 0.1728 | 2.30e-03 | 7086 |

### What the two axes say, which one axis cannot

**On Robertson the tolerance-matched comparison is a lie by a factor of seven.** `LSODA` at
1e-6 looks 4.7x faster than `Radau` at 1e-6. It is also delivering 1.70e-04 against
1.19e-06 -- two orders worse than asked. At *matched error* near 1e-6, `Radau` takes 0.0303 s
and `LSODA` needs better than its 1e-8 point, so the honest ratio is about 2x and possibly
under 1. Every speed claim made without an error axis is the 4.7.

**On VanDerPol it is worse than a lie, it is a different answer.** `LSODA` is 6.7x faster than
`Radau` at every tolerance and its error at 1e-4 is **0.96 relative** -- it is not solving the
problem. `BDF` at 1e-4 is 0.43. Only `Radau` tracks its own tolerance across the sweep, and it
is the slowest of the three. A benchmark that reported seconds at a fixed tolerance would have
ranked them exactly backwards.

That is the entry's thesis, measured: `docs/architecture.md:130` says "the target is to be
*faster* on the classic problems", and on the harder of these two the fast solver is the one
that is wrong.

### How deep a sweep can go is measured, not chosen

The reference has to be `MARGIN` = 10x better than the tightest point it grades. At
rtol = atol = 1e-12 the two witnesses agree to **4.7e-11** relative on Robertson and **1.6e-8**
on VanDerPol, so the sweeps stop at 1e-8 and 1e-6. Deeper than that on either needs the Bari
values.

**This corrected a table published earlier the same day.** The first version of the guard
compared an *absolute* difference against the *loosest* tolerance in the sweep -- wrong twice,
since a reference must beat the best point it grades rather than the worst, and 7e-8 means
nothing until you know the state is order 1 rather than order 1e6. Under it, Robertson was
swept to 1e-10 against a reference only 2x better. Those two rows were withdrawn.

## Deliberately not done

- Not a CI gate, as the entry required: `runInCI: false`, `cache: false`, reason at the task.
- **No diagram is drawn.** `moon run :bench` prints the table and writes JSON. A plot is a
  reader's convenience; the numbers are the claim.
- **`VanDerPol` runs to 20 and not to its `.toml`'s 3000.** At mu = 1000 the period is about
  1600, so 3000 is two relaxation cycles and costs 118 s at rtol 1e-4 alone; a tenth of one
  cycle at 1e-10 costs 265 s. Twenty is the first sharp corner and it is a sweep somebody will
  run. Stated where the number is set.
- **Stages 2 and 3 are split off** rather than left implied here: `HIRES`, `E5` and
  `Ring Modulator` are three models to write and validate before anything can be timed, and
  `ScalableTestSuite` is an external library. Both are their own entries now.
- **These are the dense-Jacobian path.** Every problem here is under `_SPARSE_STATES`, so none
  of them exercises the sparse solve that landed the same day.
