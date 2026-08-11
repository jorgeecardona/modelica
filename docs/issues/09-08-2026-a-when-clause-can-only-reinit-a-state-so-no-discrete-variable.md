---
title: A when-clause can only reinit a state, so no discrete variable in this engine remembers anything
status: in-progress
area: sim
owner: quiet-tars-187f2
filed: 2026-08-09
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
