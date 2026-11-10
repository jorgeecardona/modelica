---
title: A shared subexpression defined inside a Newton loop is read stale after the loop
status: done
kind: bug
scope: S
owner: jorgeecardona
filed: 2026-11-08
area: sim
verify: test
blocked_by: null
fixed_in: 0.0.221
---

## Measured

`Emitter.sharing` (`src/modelica/sim/emit.py`) defines each shared name just before its first
use. When the first use is inside a Newton `for` loop, the name is defined inside the loop, and
lines after the loop read it.

Pendulum `rhs_0`: `__s0 = 2.0 * x` is set inside the loop, then `vx = ... / __s0` after it. The
loop updates `x` after computing `__s0`, so `vx` uses the `x` from before the last step.

- At convergence the error is the last step, below `TOLERANCE` (1e-12). The start-point tests
  do not see it.
- When Newton does not converge, the error is large. Pendulum mode 0 at
  `y = (-1.249, -0.314)`: emitted `dy[1] = -1.447`, the VM tape (`modelica.sim.vm`) gives
  `1.078`. 9 of 30 random points differ.

Found by the review of the VM milestone 1 (commit 56ecbba). The tape does not have the bug: a
Newton body's values do not leak out of the loop.

## Fix

A name first used inside a loop must not be read after the loop. Either define it again after
the loop, or keep loop-local names out of `plan.names` for lines outside the loop.

Test: the emitted Python and `vm.evaluate` agree at random points of Pendulum mode 0.
