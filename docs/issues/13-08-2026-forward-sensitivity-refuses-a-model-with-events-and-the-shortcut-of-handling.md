---
title: Forward sensitivity refuses a model with events, and the shortcut of handling only time-triggered ones buys exactly one corpus model
status: open
area: sim
owner: null
filed: 2026-08-13
kind: story
scope: L
verify: test
---

## Measured

`modelica.sim.sensitivity` refuses any kernel with a watched relation, and the refusal is
load-bearing rather than cautious: integrated through the augmented system with no event
handling, `RoastingMachine`'s **own trajectory** -- before any sensitivity is read -- comes out
8.8e-2 from `simulate`'s, because the latches are set once and never refreshed.

That refusal is now the single named blocker on two entries:

- `09-08-2026-nothing-fits-a-parameter-to-data` reserves its `verify: probe` for
  `RoastingMachine` against a Cropster roast log. `RoastingMachine` has nine events.
- The identifiability report and the fit both integrate sensitivities, so both inherit it.

**The obvious shortcut does not pay.** A relation over ``time`` alone has no state in it, so
the crossing time does not move when a parameter does, and the sensitivity is *continuous*
across it -- nothing is needed but re-latching, which `modelica.sim.solve` already does.
Handling that case alone would need no new emitted function. Census of every corpus model with
events, taken 2026-08-13:

| model | events | watched, time-only | watched, state | `when` triggers |
| --- | --- | --- | --- | --- |
| `Setback` | 1 | **1** | 0 | 0 |
| `RoastingMachine` | 9 | 8 | **1** | 0 |
| `PhaseChange` | 6 | 2 | **4** | 0 |
| `BumpStop` | 1 | 0 | **1** | 0 |
| `DrainingTank` | 1 | 0 | **1** | 0 |
| `BouncingBall` | 1 | 0 | 0 | **1** |
| `Pendulum` | 2 | 0 | 0 | **2** |
| `Thermostat` | 2 | 0 | 0 | **2** |
| `TwoInertias` | 2 | 0 | 0 | **2** |

One model of nine is purely time-triggered. `RoastingMachine` -- the one the probe needs -- is
eight-ninths of the way there and stopped by a single ``T_bean > T_dry``.

## Inferred

The mathematics, from the standard result rather than from anything measured here. At a
crossing ``t*`` where ``g(t*, y) = 0`` and the state jumps ``y+ = h(y-)``:

    S+ = (dh/dy) S- + (f+ - (dh/dy) f-) * dt*/dp
    dt*/dp = -(g_y S-) / (g_t + g_y f-)

For this engine most events are latch flips rather than state jumps, so ``h`` is the identity
and it reduces to ``S+ = S- + (f+ - f-) dt*/dp``. A `when` that reinitialises a state is the
case where ``dh/dy`` is not the identity, and `Kernel.transition` is the function that would
have to be differentiated.

**What is missing is ``g_y``.** `indicators` emits ``g`` and nothing emits ``dg/dy``. It is the
same forward-mode walk `Emitter.tangent` already does for every other Jacobian, seeded on the
states and read off the indicator expressions instead of the state derivatives -- so it is a
sixth entry point of the same shape rather than a new technique. ``g_t`` falls out of the same
walk with ``time`` seeded.

Also inferred: ``dt*/dp`` is where an unidentifiable model shows a second face. A crossing time
that does not move with a parameter contributes nothing, which is the discrete analogue of the
invisible direction `identifiability` already reports.

## Done

- `sensitivity` integrates through an event and the refusal is deleted, not narrowed.
- Checked the way the continuous case is: against a central difference of two runs at
  perturbed parameters, on a model whose crossing time genuinely moves. `DrainingTank` is the
  smallest -- one state, one relation, and the moment it runs dry depends on the parameters.
- `RoastingMachine` produces an identifiability report, which is what
  `nothing-fits-a-parameter-to-data`'s probe has been waiting for.
- The `when` case is either handled or refused *separately* from the latch case, with the
  message saying which. Four corpus models are `when`-only and they are a different problem
  from the five that latch.

## Deliberately not done

- **Not the time-only shortcut.** It is one corpus model, and the census above is the reason
  to say so rather than discover it after building it.
- No adjoint method. Forward sensitivity costs one extra ODE per parameter and this project
  fits a handful; the adjoint pays off in the other regime and would be a second implementation
  to keep agreeing with the first.
