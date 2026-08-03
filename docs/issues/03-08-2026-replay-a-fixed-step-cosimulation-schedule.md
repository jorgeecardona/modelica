---
title: A fixed-step co-simulation schedule never branches, so it could be recorded once
status: open
area: ssp
owner: null
filed: 2026-08-03
kind: idea
scope: M
verify: probe
blocked_by:
  - ssp-read-the-archive-and-run-the-master
---

## Measured

- `docs/ideas/backends.md:76` names this as the fifth thing owning the compiler buys:
  "**Fixed-step co-simulation is a static dispatch sequence.** Nothing branches between
  communication points, so the whole schedule can be recorded once and replayed. That is CUDA
  graphs, or `cml_replay` on the Vulkan side. Only a compiler that knows it is compiling a
  simulation can see that optimisation."
- The claim about branching is checkable against the protocol that already exists.
  `modelica.ssp.master` says what moves a master may make, and between two communication
  points a conforming run is: set inputs, step every component, get outputs, exchange. The
  order does not depend on any value.
- The exception is in the same module, and it is the whole reason the protocol was written
  before the master: **`fmi3DoStep` may return `fmi3Discard`**, and then everybody rolls back
  (`docs/ideas/verification.md:226`). That is a branch, and it is not rare enough to ignore.

## Inferred

So the shape is inferred to be: a recorded fast path, plus a live path for the step that
discards. Record the dispatch sequence once, replay it per communication point, and fall out
of the replay into the general master when a component refuses. That is the same structure as
a CUDA graph with a fallback, and it is why this is filed as an idea rather than a story —
the fallback is where the correctness lives, and nobody has written the master yet to know
what it costs.

The saving is also inferred: per communication point, the Python overhead of deciding what to
call next, which matters when the components are small and the steps are many. On a system of
two components with real FMU calls behind them it is probably nothing. That measurement is
what would turn this into a story.

## What would turn this into a story

The master, first — this waits on it deliberately. Then: run a system of several components
for many communication points, with and without a recorded schedule, and report the
difference on a named machine. If the master's own overhead is under a few percent of the
step, close this as measured and not worth it.

## Deliberately not done

- No GPU. `cml_replay` is named in `backends.md` as the precedent for the *idea*, not as a
  dependency; recording a schedule is a CPU-side structure first.
- No change to the protocol in `modelica.ssp.master`. A replay that can do something the
  checked protocol does not allow is a replay of the wrong thing.
