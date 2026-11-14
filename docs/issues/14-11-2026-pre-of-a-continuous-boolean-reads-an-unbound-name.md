---
title: "pre(x) of a Boolean that is not when-assigned reads an unbound name in the generated code"
status: open
area: sim
owner: null
filed: 2026-11-14
fixed_in: null
scope: M
verify: test
kind: bug
---

## Measured

- 2026-11-14, HEAD d2464b7: the on-off latch `on = pre(on) and x < 0.8 or x < 0.2`
  (tests/test_boolean_equations.py:134) emits, but does not simulate. The python backend raises
  `UnboundLocalError`; the engine raises `EmitError: nothing computes 'on'`. The test only checks
  `emit(...).source`.
- The MSL `PowerConverters.Examples.ACDC.Rectifier1Pulse.Thyristor1Pulse_R` fails the same way:
  `IdealThyristor` writes `off = s < 0 or pre(off) and not fire`, and the generated `latch()`
  reads `idealthyristor_off` before it is assigned.

## Cause

`pre` is in `_TRANSPARENT` (src/modelica/sim/emit.py:139), so `pre(off)` prints as `off`. For a
variable a `when` assigns, that name holds the value from the workspace. For any other variable
nothing loads it, and its workspace slot is written once, in `initial()`, and never again.

## Plan

1. Emit `pre(x)` as a read of `w[slots[x]]` when `x` is neither remembered nor a state, in the
   python emitter and the VM lowering.
2. Store `w[slots[x]] = x` at the end of `latch()` and after each event transition, never in
   `rhs`, `observe`, the Jacobians or the indicators.
3. Repeat the event iteration until those slots stop changing (`b = not pre(b)` is not
   idempotent).
4. Test on both backends: a thyristor-like latch in an algebraic loop (`fire` a time window
   0.1 to 0.2, source `sin(2*pi*t)`), and make the on-off latch test simulate.
