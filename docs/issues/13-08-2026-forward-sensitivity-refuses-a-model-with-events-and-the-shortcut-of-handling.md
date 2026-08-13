---
title: Forward sensitivity refuses a model with events, and the shortcut of handling only time-triggered ones buys exactly one corpus model
status: in-progress
area: sim
owner: quiet-tars-187f2
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

## `g_y` is emitted, 2026-08-13

The Inferred section named the missing piece -- "`indicators` emits ``g`` and nothing emits
``dg/dy``" -- and said it would be the same forward-mode walk read off the indicator
expressions. It is, and `Emitter.differentiate_indicators` is eleven lines.

`differentiate` reads the carried tangent rows at the state derivatives; this reads them at
the indicator expressions, which are not variables, so it goes through `combination` the way
`chain` does for an explicit assignment. `Kernel.indicator_jacobian` is the seventh entry
point and fills an ``(events, n)`` array.

### Checked the way the other two Jacobians are

`tests/test_examples.py::test_the_indicator_jacobian_agrees_with_a_difference_quotient`, the
third of that shape, over the nine corpus models with events:

| worst disagreement with a central difference | |
| --- | --- |
| `Pendulum` | 1.8e-9 |
| `PhaseChange` | 2.5e-10 |
| `BumpStop` | 1.4e-10 |
| `RoastingMachine` | 6.4e-11 |
| `Thermostat` | 5.6e-11 |
| `BouncingBall`, `DrainingTank` | 2.7e-11 |
| `Setback`, `TwoInertias` | **0** |

### The zero rows reproduce this entry's own census, from the other direction

A crossing over ``time`` alone has no state in it, so ``g_y`` is identically zero and the
crossing time does not move when a parameter does. Counting the identically-zero rows of the
emitted Jacobian:

| model | zero rows | of |
| --- | --- | --- |
| `RoastingMachine` | **8** | 9 |
| `PhaseChange` | 2 | 6 |
| `TwoInertias` | 2 | 2 |
| `Setback` | 1 | 1 |
| `Pendulum` | 1 | 2 |
| the other four | 0 | 1 each |

That is the same table the Measured section took by reading the IR for names, arrived at here
from the emitted arithmetic instead. Two independent routes to "eight of `RoastingMachine`'s
nine are schedules", which is the fact the whole entry turns on.

### What is left

The integrator. ``S+ = S- + (f+ - f-) dt*/dp`` at a latch flip, with
``dt*/dp = -(g_y S-)/(g_t + g_y f-)``, and the `when` case where ``dh/dy`` is not the
identity. ``g_t`` is still not emitted -- the same walk with ``time`` seeded -- and for the
eight `RoastingMachine` schedules it is the *only* term that is non-zero, since ``g_y`` is
zero there and ``dt*/dp`` is then ``0/g_t``, which is zero. So the schedules need no jump at
all, and the one crossing that does is ``T_bean > T_dry``.

### The other term is not a substitution, measured 2026-08-13

The note above says ``g_t`` is "the same walk with ``time`` seeded", which is right, and it is
worth recording why the cheaper route does not exist.

``dt*/dp = -(g_y S) / (g_t + g_y f)``, and the denominator is exactly ``dg/dt`` **along the
trajectory** -- one quantity rather than two. So the obvious move is to emit it directly:
`modelica.ir.derive.d_dt` of the indicator, with ``der(v)`` mapped back to whatever
causalization named that derivative. Tried:

- **`d_dt` of every corpus indicator renders**, with nothing missing from the emitter's scope.
  `RoastingMachine`'s schedules come out as constants, which is right for ``time < 66.7``.
- **The mapping does not exist for every model.** `Pendulum` raises ``KeyError: 'x'`` --
  ``d_dt`` introduces ``der(x)``, and ``x`` is not a state of the chosen mode. It is solved
  from the constraint, so the schedule computes ``x`` and never computes its derivative.

So a total time derivative of an indicator can reach a variable whose derivative the model
does not carry, and recovering it means the chain rule through the blocks -- which is
`Emitter.tangent` again, seeded on ``time`` instead of on the states. That is a **sixth
seeding**, the same shape as the five that exist, and not a rename of an existing function.

Which is a smaller job than it sounds and a bigger one than "substitute `der(x)`": the walk
cache is already keyed on ``(mode, seeds)``, so the seeding is the whole of the change, but
``time`` is not in `Emitter.tangents` and would have to be given a row.

**None of this is needed for a schedule.** Where ``g_y`` is zero -- eight of
`RoastingMachine`'s nine crossings, and `Setback`'s only one -- ``dt*/dp`` is ``0`` over
whatever the denominator is, so the crossing does not move and the sensitivity is continuous
across it. The denominator is only needed for a crossing that ``g_y`` says *does* move, and
`indicator_jacobian` now names exactly which those are.

## Both terms of `dt*/dp` are emitted, 2026-08-13

`Kernel.indicator_slope` is `dg/dt` **along the trajectory** -- the denominator, as one
quantity rather than as ``g_t`` and ``g_y f`` separately, which would be two walks for a
number that is one.

It is the walk's **third seeding** and the only one whose rows are scalars: time moves at one
and every state moves at its own derivative, which the schedule has already computed. So the
seed is an expression rather than a row of an identity, and everything downstream of it --
`chain`, `implicit`, `combination` -- is the same code the other five seedings use.
`Emitter.tangent(mode, along=True)`, cached on the same `(mode, seeding)` key.

### Two things it cost, both found by running it

**`time` had no tangent row.** It is not an observed variable, so `Emitter.tangents` had no
entry for it. One line in `__post_init__`, put there rather than in the walk so that every row
this emitter can name is named in one place.

**The seeds are not constants, so the schedule has to run first.** `DrainingTank` failed with
``UnboundLocalError: der_level``: a state's ``d/dt`` *is* its derivative variable, which a
block computes, and the walk read it before that block had run. The blocks are emitted up
front for this seeding and the walk skips its own `solve` -- so it costs no more source than
the other five rather than twice as much.

### Checked

`indicator_slope` against a difference quotient taken *along the trajectory* --
``g(t+h, y+hf)`` against ``g(t-h, y-hf)``, because a partial in ``t`` alone is the wrong thing
and would agree with a wrong implementation on every model whose crossings are schedules:

| worst | |
| --- | --- |
| `PhaseChange`, `RoastingMachine` | 1.9e-8 |
| `Pendulum` | 9.5e-10 |
| `Thermostat` | 1.3e-10 |
| `Setback`, `BouncingBall`, `BumpStop`, `DrainingTank` | 9e-11 or better |
| `TwoInertias` | **0** |

The two loose ones have constant slopes, so the number is the quotient's own error.

### The contrast is the point, and is a test

`RoastingMachine`'s eight schedules have ``g_y`` identically zero and ``dg/dt = -1``. A caller
reading one for the other would divide by zero on every schedule in the corpus. Both are
asserted.

### What is left

Only the integrator now. ``S+ = S- + (f+ - f-) dt*/dp`` at a latch flip, and the `when` case
where ``dh/dy`` is not the identity and `Kernel.transition` is what has to be differentiated.
Both terms of ``dt*/dp`` are in hand.
