---
title: An `initial equation` is parsed, flattened, carried on System — and read by nothing
status: open
area: sim
owner: null
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
