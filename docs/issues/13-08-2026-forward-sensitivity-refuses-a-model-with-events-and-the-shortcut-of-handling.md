---
title: Forward sensitivity refuses a model with events, and the shortcut of handling only time-triggered ones buys exactly one corpus model
status: done
area: sim
owner: quiet-tars-187f2
filed: 2026-08-13
fixed_in: 65e3026
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

## The integrator landed, and `RoastingMachine` goes through

`modelica.sim.sensitivity` integrates segment by segment, stopping at each crossing to move
``S`` across it:

    S+ = S- + (f- - f+) dt*/dp,   dt*/dp = -(g_y S-) / (dg/dt)

**The order ``f- - f+`` is not a detail and it is how this was caught.** Written the other way
round the sensitivity still moves in a plausible direction and stays continuous where it
should; on the closed-form model in `tests/test_sensitivity_forward.py` it gives ``-0.75``
against the true ``-0.25``, and integrating with *no* jump gives ``-0.5``. Three answers, none
of them obviously wrong on inspection.

The derivation is that just after the crossing ``x(t) = x(t*) + f+ (t - t*)``, so
differentiating at fixed ``t`` gives ``S+ = [S- + f- dt*/dp] - f+ dt*/dp``.

### Measured against a central difference

| model | parameter | worst |
| --- | --- | --- |
| **`RoastingMachine`** | `UA` | **1.6e-8** |
| `BumpStop` | `stop.k` | 1.8e-9 |
| `PhaseChange` | `UA_air` | 2.2e-6 |
| `Setback` | `T_day` | 5.2e-6 |
| the closed-form `Crossing` | `k` | exact to 1e-7 |

`RoastingMachine` is the one this entry exists for: nine crossings, eight of them schedules
whose ``g_y`` is zero and which therefore move ``S`` not at all, and one -- ``T_bean > T_dry``
-- that genuinely moves. `09-08-2026-nothing-fits-a-parameter-to-data`'s probe is no longer
blocked on this.

### What is still refused, and what broke

**A `when` branch.** A latch flip leaves ``y`` continuous, so the jump above is the whole
story; a `when` moves ``y`` itself and needs ``d(transition)/dy``, which nothing emits. Four
corpus models are `when`-only and are refused with that reason.

**`DrainingTank` leaves the model's domain.** Past the moment the tank runs dry its ``sqrt``
is of a negative level, and the augmented system reaches further into that region than the
state alone does -- the Jacobian is evaluated there too. It came out as a raw
``ValueError: array must not contain infs or NaNs``; it is now a `SimulationError` naming the
model, the same guard `modelica.sim.solve._segment` already carries. Making it *work* is a
question about the model's domain rather than about sensitivity.

### Left

The `when` case, which is `d(transition)/dy` -- an eighth entry point, and the same
forward-mode walk over a transition body instead of over the blocks.

## The refusal was broader than the mathematics, 2026-08-13

`Thermostat` was being turned away and should not have been. What the jump formula needs is
that ``h`` is the identity -- that ``y`` is continuous across the crossing -- and that is
narrower than "has no `when`".

Measured over the corpus, every `when` body that survives causalization:

| model | body | ``h`` |
| --- | --- | --- |
| `Thermostat` | ``heating = ...``, ``cycles = pre(cycles) + 1`` | **identity** |
| `BouncingBall` | ``reinit(v, -e * v)`` | not the identity |

Two models, and they are the two cases. `Thermostat`'s assignments change what ``f`` *is*
without moving where ``y`` is -- which is exactly what the ``f- - f+`` term already accounts
for. Checked against a central difference once the refusal was narrowed: **1.2e-8**.

`Trigger.reinitialises` is what tells them apart, and it had to go on the public record
because `Trigger.latch` cannot: both kinds of `when` have none. Same argument as `latch`'s own
docstring gives -- it is a fact about the model's discrete structure, and re-deriving it from
the bodies outside the emitter would be a second copy.

Two things the narrowing exposed, both in code written earlier the same day:

- **The crossing watcher assumed every trigger had a latch.** It did while only latch flips
  got through. The assertion added an hour earlier fired immediately, which is what an
  assertion in unreachable code is for.
- **A `when` is watched rising-only.** `Trigger.direction` already carried that and SciPy
  spells it the same way, so the watcher passes it through rather than deciding again.

So the remaining refusal is one model and one condition: a crossing that reinitialises a
state, needing ``d(transition)/dy``.

## A reinitialised state goes through too, and the entry closes

Two more emitted terms, and the jump is complete:

    S+ = H S- + dh/dp + (H f- - f+) dt*/dp

`Kernel.transition_jacobian` is ``H = d(transition)/dy`` -- the identity except in the rows a
``reinit`` writes -- and `Kernel.transition_jacobian_parameters` is ``dh/dp``. Both are the
same forward-mode walk, gathered at the ``reinit`` expressions instead of at the state
derivatives.

`BouncingBall` needs every term and is why both exist. Its ``reinit(v, -e * pre(v))`` makes
``H`` something other than the identity *and* puts ``e`` **only** in the reset. With ``dh/dp``
missing, the sensitivity to ``g`` was already right at 5.5e-10 while the sensitivity to ``e``
was wrong by 1.0 -- the whole of itself.

    stop   bounces   dh/de exact      differenced
    0.60      1      +0.657668        +0.657668
    1.20      2      -2.144789        -2.144789
    1.80      3      -3.808208        -3.808208
    3.00      6      +1.163888        +1.163888

Every digit, through six bounces whose times all move with the parameter.

### Two things worth keeping

**``pre(x)`` is ``x`` for this purpose, and only the walk needs telling.** `BouncingBall`
writes ``-e * pre(v)``, and inside a ``when`` body ``pre(v)`` *is* the pre-event value, which
is exactly what ``d(transition)/dy`` differentiates against. At runtime the name is already
bound to it, so `Emitter.statement` needs no equivalent -- it is only the tangent walk, which
matches on node identity, that has to be told. Without it ``H``'s ``v`` row came out zero.

**Reading ``dh/dp`` one line too late cost a factor of exactly ``-e``.** `transition` returns
the post-event state, and ``-pre(v)`` evaluated at the reflected velocity rather than the
incoming one is ``-(-e v)`` instead of ``-v``. A *constant* factor across every sample and
every horizon is what made it findable -- 0.526/0.658, 1.716/2.145, 3.047/3.808, all -0.8 --
where a varying error would have looked like ordinary numerical disagreement.

### What is left refused

A mode switch, and it is not a jump at all: `Mode.derivative` differs either side, so ``S`` is
not the same shape. The state vector changes meaning rather than value, which is a different
piece of work and a different entry if anybody wants it.

## Closed: `RoastingMachine` produces a report

The Done list's third item, which is what the whole entry was for:

    RoastingMachine: 2 parameters from 1 measurements, condition 21.7 -- separable
    RoastingMachine: 2 parameters from 1 measurements, condition 20.3 -- separable
    RoastingMachine: 2 parameters from 2 measurements, condition 21.7 -- separable

``UA`` against ``C_mach`` and ``UA`` against ``m_green``, measured through ``T_probe``, and
the first pair with ``T_bean`` beside it. All separable, around 21 -- which is the answer a
thermal model measured through a lagged probe should give, and it is an answer rather than a
refusal for the first time.

`09-08-2026-nothing-fits-a-parameter-to-data` is unblocked: its `verify: probe` is this model
against a roast log, and nothing in this toolchain stands in the way of it any more. What is
left there is the log.

Every item on this entry's Done list is met except that the refusal is *narrowed* rather than
deleted -- a mode switch is still refused, and the entry above argues that one is not a jump
at all but a change in what the state vector means. That is a different piece of work and
deserves its own entry if anybody wants it.
