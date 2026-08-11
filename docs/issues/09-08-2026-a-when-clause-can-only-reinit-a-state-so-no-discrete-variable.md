---
title: A when-clause can only reinit a state, so no discrete variable in this engine remembers anything
status: done
area: sim
owner: quiet-tars-187f2
filed: 2026-08-09
fixed_in: 3d68521
scope: L
verify: test
kind: story
---

## Measured

`modelica.sim.emit.Emitter.statement` (`src/modelica/sim/emit.py:1160`) matches exactly two
cases: `reinit` of a state, and `reinit` of something that is not a state, which is an error.
Everything else is `no translation for {statement!r} inside a when-clause`.

So a `when` body may reinitialise a continuous state and do nothing else. Measured, by
building three models that are legal Modelica and asking for a kernel:

| written | outcome |
| --- | --- |
| `Boolean hot; ... hot = x > 0.5;` | `DerivativeError: a '>' expression has no derivative` |
| `discrete Real charged; when x <= 0 then charged = 1.0; end when;` | `EmitError: no translation for Assign(...) inside a when-clause` |
| `discrete Real n; when x <= 0 then n = pre(n) + 1; end when;` | `EmitError: no translation for Assign(...) inside a when-clause` |

The second and third *causalize* — `modelica.ir.causalize` accepts them and produces a
`System`. They fail at emission. The first fails earlier and for an unrelated reason: a
`Boolean` determined by a relation is handed to the differentiator as though it were
continuous.

The only discrete memory an emitted kernel has is `Trigger.latch`, and that is not memory.
Measured over every corpus model that has events, starting the event iteration from every
latch configuration at a fixed state (`modelica.verify.check_iteration`):

| model | latches | starting configurations | settled configurations |
| --- | --- | --- | --- |
| `RoastingMachine` | 9 | 512 | 1 |
| `Pendulum`, `TwoInertias` | 2 | 1 | 1 |
| `BouncingBall`, `BumpStop`, `DrainingTank`, `Setback` | 1 | 1–2 | 1 |

One settled configuration from all 512 starts. A latch is the cached value of a relation the
model can evaluate, and the event iteration repairs whatever you set it to.

## Inferred

Inferred, not measured: this is the reason a model cannot express a *regime* — a fact about
the world that persists because nothing has changed it, rather than because a relation
currently holds. Modelica has the construct (§8.3.5: a discrete variable holds its value
between events) and this engine has the parser and the causalizer for it. It has no emitter.

Also inferred: the `Boolean` failure is a separate defect from the `when` one and probably a
smaller fix — a relation-valued variable should be latched like any other watched relation
rather than differentiated. The two are filed together because a modeller hits them one after
the other while trying to write the same thing.

## Done

- A `when` body may assign a `discrete Real` or a `Boolean`, and the value persists until the
  next `when` that assigns it. `pre()` of such a variable reads the value before this event.
- The variable is state the kernel carries: a slot in `w`, written by `transition`, read by
  `rhs` and `observe`, and part of what `modelica.verify` enumerates. It is *not* a
  continuous state and no derivative is taken of it.
- A `Boolean` determined by a plain relation equation compiles, and is latched rather than
  differentiated.
- `modelica.verify.check_iteration` reports more than one settled configuration for a model
  with a memoried flag, because that is what memory means. A test asserts it, and the
  measurement above — 512 starts, 1 settled — becomes the *contrast* rather than the rule.
- One corpus model uses it. A two-regime model is the honest one, since that is what the
  construct is for.

## Deliberately not done

- No `pre()` of a continuous variable. That is a different feature with different semantics
  and no caller asking for it here.
- No state-machine syntax (`Modelica.StateGraph`, `state machine` clauses). This is the
  primitive those are built on.

## What landed

`_remembered`, three lines of emitter, and `examples/models/Thermostat`.
`tests/test_discrete.py`, 8 tests. 1445 pass.

The engine was one step from having this. Causalization already left a discrete variable
alone -- not a state, not a block unknown -- and it already had a workspace slot, because
`w` begins with one slot per observed variable. What was missing was a definition and three
places to use it:

- **`_remembered`** reads the names off the `when` bodies rather than off the `discrete`
  prefix. The specification says the assignment is what makes a variable discrete-time, and
  a variable declared `discrete` and never assigned has nothing to remember.
- **`unpack`** binds each one from its slot, so `rhs` and `observe` read the value instead
  of computing it.
- **`statement`** writes the slot for an assignment.

**`pre()` needed no code at all**, which was the one genuine surprise. `transition` runs
against the *pre-event* workspace, so by the time a body reads the name it is already bound
to the old value: `n = pre(n) + 1` emits `w[k] = n + 1`. There is a test whose whole job is
to say this is deliberate rather than lucky.

Each Done bullet:

- **A `when` body may assign a discrete variable, and it persists.** A counter over five
  events reads 0, 1, 2, 3, 4. A variable never assigned keeps its `start`.
- **It is state the kernel carries** -- a slot in `w`, written by `transition`, read by
  `rhs` and `observe`, and no derivative taken of it. A test drives the continuous part
  from one.
- **One corpus model uses it, and it is a two-regime model**: `Thermostat`, a heater with
  hysteresis, where "should the heater be on" is not a function of the current temperature.
  Two `when` clauses that cannot see each other, and `pre()` for the cycle count.

  Its reference is not another integration. Each phase is `T_inf + (T0 - T_inf) exp(-k t)`
  and each switching time is one logarithm, so the spec carries **all 33 switching times of
  a minute** plus four sampled points, from a recurrence with no integrator in it. The
  sharpest assertion is `heating * (1 - heating) = 0` at `atol = 0`: a regime that was
  interpolated, or recomputed from a relation each step, could not hold that across 33
  events.

- **Two refusals gained a remedy.** Assigning a continuous state inside a `when` now says
  to use `reinit(x, ...)`; assigning an ordinary variable says to declare it `discrete`.

## Deliberately not done, and this entry does not close them

Two Done bullets are **not** met, and they are separable enough to be their own entries
rather than reasons to keep this one open.

- **The `Boolean` half.** `Boolean hot; hot = x > 0.5;` still fails with
  `DerivativeError: a '>' expression has no derivative`. The entry itself calls it "a
  separate defect ... and probably a smaller fix", and it is: a relation-valued variable
  should be latched like any other watched relation rather than handed to the
  differentiator. That is a change in causalization, not in the emitter, and nothing in
  this change touched it.
- **`modelica.verify` does not enumerate discrete variables.** The entry asks for
  `check_iteration` to report more than one settled configuration for a model with a
  memoried flag -- which is the right test of the distinction, and it needs the checker's
  state to include the remembered slots as well as the latches. `modelica.verify` builds
  its state from `Trigger.latch` alone today.

Both should be filed against this landing rather than assumed.
