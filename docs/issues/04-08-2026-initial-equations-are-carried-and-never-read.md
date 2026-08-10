---
title: An `initial equation` is parsed, flattened, carried on System — and read by nothing
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-04
kind: bug
scope: M
verify: test
---

## Measured

- `initial equation` is read by the parser (`src/modelica/lang/parse.py`), expanded by
  flattening, and stored: `FlatModel.initial_equations` (`src/modelica/ir/flat.py:85`) and
  `System.initial_equations` (`src/modelica/ir/system.py:122`).
- Nothing in `modelica.sim` reads either. `grep -rn "initial_equations" src/modelica/` shows
  `build`, `lang`, `ir` — and no hit under `src/modelica/sim/`.
- The initial state comes only from the `start` attribute. `modelica.sim.emit`'s `initial`
  writes `y[slot] = <start or 0.0>` per state, so a state with no `start` begins at **zero**
  no matter what an initial equation says about it.
- Found by writing `HeatBar` with its initial condition as an equation:

      initial equation
        for i in 1:n loop
          T[i] = T_0;      // T_0 = 20
        end for;

  The run started every cell at 0, and `T[1]` at t = 2 came out 42.77 against a reference of
  54.21. Nothing warned. The model was corrected to use `start = T_0, fixed = true`, which
  is what the rest of the corpus does, and it then matched to 1e-8.

## Inferred

Two claims, and they fail differently:

- **The silence is the bug.** A wrong number from an ignored equation is the worst shape a
  simulator can fail in — the run completes, the plot looks plausible, and the only clue is
  a comparison against something else. Every other unimplemented construct in this project
  is refused by name; this one is accepted and dropped.
- **Doing it properly is a solve, not a copy.** Modelica's initialization is a *system*:
  the initial equations, the equations themselves, and the `fixed = true` start values are
  solved together for a consistent state. That is a nonlinear solve over the same blocks
  causalization already builds, and for a high-index model it is the reason
  `docs/architecture.md:190` names SUNDIALS IDA's "consistent initialization" as a feature
  worth having. Nothing here has been prototyped.

So the cheap fix and the correct fix are far apart, which is why this is filed at M with
the cheap one first.

## Done

Either of these closes it, and the first should land immediately whichever way the second
goes:

1. **Refuse what is ignored.** If a model carries initial equations and `modelica.sim`
   cannot honour them, `build` raises and names them. That turns a wrong trajectory into a
   message, which is the standing rule everywhere else in this toolchain.
2. **Or solve them.** The initial equations join the model's own equations, the states with
   `fixed = true` contribute their start values as equations, and the whole system is solved
   once for a consistent initial point before the first step. A test then writes `HeatBar`'s
   initial condition as an equation again and gets the same trajectory as the `start`
   version — the two spellings of one model agreeing is the observation that closes this.

## Deliberately not done

No change to `start`/`fixed`. They work, every corpus model uses them, and they are the
right thing for a state whose initial value is simply known.

## What landed

Both halves, and the entry said either would do. `tests/test_initial_equations.py`, 6 tests.
1291 pass.

**Honoured, for the explicit case.** An initial equation that already reads
``state = expression`` becomes an assignment in the emitted ``initial``, ordered by what each
one reads so ``x = 1; y = x + 1`` runs the right way round. `HeatBar` written the way that
found this bug now gives **exactly** the trajectory the `start = T_0` version gives:

    start attribute      T[1] at t = 2   54.21324087162906
    initial equation     T[1] at t = 2   54.21324087162906   (was 42.77)

Not close -- equal. Two spellings of one model, which is the observation this entry named as
what closes it.

**Refused, for everything else.** An initial equation that is not an explicit value for a
state, or a set of them that read each other in a cycle, raises from `build` and prints the
equation. Modelica's real initialization is a *system* -- the initial equations, the model's
own equations and every ``fixed = true`` start value solved together for a consistent point --
and that is not implemented. What is not implemented now says so, which is the standing rule
everywhere else in this toolchain and the one thing this construct did not follow.

### The value goes to two places, and one of them is easy to miss

`y[slot]` is what the solver integrates from. `w[slot]` is what `observe` reports at t = 0.
Setting only the first leaves the sample at t = 0 holding the `start` attribute -- zero, here
-- while the trajectory begins at 20, so a run disagrees with its own first point. There is a
test for it.

## Deliberately not done, found while doing it

- **No consistent-initialization solve.** The correct fix is a nonlinear solve over the blocks
  causalization already builds, and `docs/architecture.md:190` names SUNDIALS IDA's for a
  reason. The explicit subset covers what a modeller writes when they mean "this state starts
  here"; the rest is refused. Whoever takes the solve has the refusal messages as its
  specification.
- **No change to `start`/`fixed`**, as the entry required. Every corpus model still uses them
  and there is a test that a model without initial equations emits byte-identical source.
- **An initial equation on a non-state is still refused**, not silently applied to an
  algebraic variable. That case is part of the same solve.
