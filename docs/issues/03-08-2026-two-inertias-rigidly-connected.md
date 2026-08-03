---
title: No corpus model reduces the index across a rotational constraint
status: in-progress
area: ir
owner: quiet-tars-187f2
filed: 2026-08-03
kind: story
scope: M
verify: test
---

## Measured

- `logs/2026-08-02.md`, open items: "Two inertias rigidly connected (index reduction across a
  rotational constraint) is the obvious next example — **deliberately avoided in DCMotor to keep
  it index 1**." So the gap is a choice already made once, with a reason.
- The corpus's high-index models are all point-mass mechanics in Cartesian coordinates:
  `Pendulum`, `AtwoodPendulum`. `examples/models/DCMotor.mo` is the only rotational
  multi-domain model and it is index 1 by construction.
- `modelica.ir.causalize` enumerates every admissible state set onto `System.candidates`
  (`src/modelica/ir/system.py:142`), and the runtime dispatches between modes — `Pendulum` is
  the mode dispatcher exercised by `tests/test_jit.py:44`. So the machinery this model would
  test is present and is currently tested by exactly one shape of problem.

## Inferred

Why this particular model, inferred from what makes it different rather than from a failure:
two inertias joined by a rigid connection give `phi1 = phi2`, a constraint between two *states*
of the same kind. Pantelides differentiates it twice, and the dummy-derivative choice is a
choice between two variables that are structurally identical — unlike a pendulum, where the
candidate sets `(x, vx)` and `(y, vy)` are distinguishable by the geometry.

The inferred risk is a tie: a symmetric model may make the candidate enumeration produce two
sets with no reason to prefer either, and the selection code has never been asked what it does
then. That is the interesting part, and it is why this is worth an example rather than a unit
test.

## Done

- `examples/models/` gains the model, written twice as every example is (`.mo` and `.py`) with a
  `.toml` carrying reference points from an independent integration.
- Its `.toml` records what it is *for*: the index, the candidate state sets, and the set
  chosen — so a change in the selection shows up as a diff in a spec file rather than as a
  silent change of behaviour.
- It simulates, and its energy or momentum invariant holds over the run at the tolerance the
  spec sets, the way `AtwoodPendulum`'s energy drift is recorded (3.6e-13 over 30 s at
  rtol 1e-12, `logs/2026-08-02.md`).
- If the symmetric tie above turns out to be real, the tie-breaking rule is written down in
  `modelica.ir.causalize` where the choice is made, not left to dictionary order.

## What landed, and the tie was real

`examples/models/TwoInertias.{mo,py,toml}`. Causalization answers:

    index:      3
    states:     ('big.phi', 'big.w')
    candidates: (('big.phi', 'big.w'), ('small.phi', 'small.w'))

The inference above was that a symmetric model would produce two candidate sets with no
reason to prefer either, and that the selection code had never been asked what it does then.
That is what happened, and the answer is that **nothing weighs them**: `_states` takes
whatever the matching left free, which follows `work.order`, which is declaration order. So
`big` wins by being written first. Deterministic, and not a judgement.

That rule is now written into `_states`' docstring, where the choice is made, instead of
being available only to somebody who reads the matching. And `candidates` is a new key in
the spec format — ordered, chosen set first — so it pins both which sets are admissible and
which one was taken. `test_selects_the_specified_states` asserts it, and
`test_flatten_section_is_well_formed` checks that a spec giving both `candidates` and
`states` does not contradict itself.

The numbers are a closed form rather than another integration: one rigid body of J = 2.5
against a viscous brake gives `w(t) = 10 exp(-t/10)`, and `2.5 w + 0.25 phi` is conserved
exactly (`d/dt (J w + d phi) = -d w + d w = 0`), so it is asserted flat at 25 over the whole
run. The constraint and both of its derivatives are asserted separately, and the second one
exists nowhere in the model text because Pantelides wrote it.

## Deliberately not done

The tie-break was **documented, not changed**. Writing a rule that prefers the larger inertia,
or the alphabetically first name, would be inventing a preference the model does not express,
and it would move `Pendulum` too. What was missing was not a better rule but a written one.

`DCMotor` stays index 1. It is the two-domain example and the first whose component holds
components; making it high-index as well would give one model two jobs and make a failure
ambiguous.
