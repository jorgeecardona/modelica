---
title: The README says nothing in Python reaches a trajectory, and something does
status: open
area: docs
owner: null
filed: 2026-08-03
kind: bug
scope: S
verify: inspect
---

## Measured

`README.md:32` states the gap this project exists to fill:

> So two things are missing. Nothing in Python takes Modelica source all the way to a
> trajectory. And nothing treats the model itself — the flat equations, the incidence
> structure, the index — as an ordinary Python value that you can read and change.

The first sentence is false as written, and it has been for years.

- [pymoca](https://github.com/pymoca/pymoca) parses Modelica with ANTLR, flattens it, and
  emits CasADi. [RTC-Tools](https://github.com/rtc-tools/rtc-tools), by Deltares, integrates
  the result: `rtctools.simulation.simulation_problem.SimulationProblem`. That is `.mo` to a
  trajectory, in Python, in industrial use for water-system control.
- `docs/index.md` and the PyPI long description carry the same sentence.

## What is actually true

Read RTC-Tools' integrator before rewriting the sentence, because the narrower claim is the
better one. From `simulation_problem.py`:

- the derivative is a **backward difference**,
  `derivative_state - (X[index] - X_prev[index]) / dt`, at a **fixed** step;
- the residual, the derivative approximations and the delay equations go into one
  root-finding problem per step, solved by CasADi's `rootfinder` (IPOPT by default);
- initialization is a separate least-squares solve over the residual;
- **there is no event handling.** Conditionals compile to smooth CasADi expressions, so the
  integrator walks through a discontinuity rather than stopping at it.

So the true statement is about *how* rather than *whether*: nothing in Python takes Modelica
source to a trajectory with **adaptive stiff integration, index reduction and events**. Each
of those three is load-bearing and each is already in this repository, so the narrower claim
is both defensible and a better description of the work.

Inferred, and worth stating as inference: pymoca appears not to implement events or index
reduction at all — no test model in `test/models` names a `when`, a high-index system or a
state selection, and RTC-Tools' fixed-step smooth-residual approach is what a front end
*without* those forces on its consumer. That is read off a file listing, not off the source.
Anybody rewriting the README paragraph should confirm it before putting it in print.

## Done

- The sentence in `README.md`, in `docs/index.md`, and anywhere the PyPI description repeats
  it, says the narrower thing, and names pymoca and RTC-Tools rather than talking around
  them. A project that misstates its neighbours reads as though it has not looked at them.
- `docs/ideas/landscape.md` is linked from wherever the claim is made, since that is where
  the evidence now lives.
- `verify: inspect` — the artifact is the evidence. There is nothing to test.

## Deliberately not done

No comparison table in the README. The landscape page carries the detail; the front page
needs one accurate sentence and nothing more.
