---
title: The parameter derivative has a name reserved for it in the emitter and no code behind it
status: done
area: sim
owner: null
filed: 2026-08-09
fixed_in: c1f22f3
kind: story
scope: S
verify: test
---

## Measured

- `modelica.sim.emit` writes `rhs`, `observe`, `jacobian`, `indicators`, `transition`, and since
  `4686cd1` also `observe_jacobian` (`src/modelica/sim/emit.py:48`). Every one of them already
  takes the parameters as an argument: the signature is `observe_jacobian(t, y, p, w, out)`.
- **The emitter states this gap in its own docstring**, where it explains why `observe_jacobian`
  did not take the shorter name (`src/modelica/sim/emit.py:869`):

  > Emitted as ``observe_jacobian`` and not as ``sensitivity``. The word *sensitivity* has an
  > owner in this field: it is the derivative with respect to a **parameter**, which is what
  > identification and uncertainty propagation ask for. That is the same transformation again
  > with a different seed, it will want the name, and taking it here for a different derivative
  > would cost more than the shorter word is worth.

- The transformation it refers to is already split for reuse. `emit.differentiate` became
  `emit.tangent`, which propagates tangents through the blocks and returns what it reached;
  `jacobian` reads it off at the state derivatives, `observe_jacobian` reads it off at the
  variables (`4686cd1`). One walk, two readings.
- **Public evidence that this derivative is what practice asks for**, from a post by Anton
  Evdokimov on welding heat-source calibration
  (<https://www.linkedin.com/posts/dr-anton-evdokimov_heat-source-calibration-looks-simple-but-share-7485439140239691776-hL40/>):
  a heat source has a thermal efficiency and a spatial radius, both of which move the
  temperature near the weld, so many pairs fit the same data. His procedure is ordered by which
  measurement is sensitive to which parameter — efficiency from thermocouples far from the weld
  line, where the shape barely matters, then the radius against the fusion boundary. That
  ordering is read off d(observed)/d(parameter) and off nothing else.

## Inferred

Inferred, not measured: the seed is the only difference. `tangent` currently seeds a unit vector
in the states; seeding it in a parameter propagates through the same blocks, and the same two
readings apply — at the state derivatives it is d(rhs)/dp, at the variables it is
d(observed)/dp. Neither has been written, so neither cost is known.

An ordering constraint worth stating before either lands: `parameters-as-constants-or-as-arguments`
proposes burning parameters in as literals, and a literal has no derivative. The two are not in
conflict — the constant-folded kernel is the fast path and the sensitivity kernel is a second
build — but whichever lands second has to say so.

## Done

- `emit` emits the two parameter partials, d(rhs)/dp and d(observed)/dp, by the same forward-mode
  walk as `jacobian` and `observe_jacobian`, with the same workspace conventions.
- The caller names which parameters to differentiate against; it is not all of them. A corpus
  model declares dozens of parameters and an identification asks about three.
- `Kernel` exposes them beside the other derivatives, and `build(jit=True)` compiles them.
- Tested the way the other two derivatives are: against central differences on every corpus
  model, at the tolerance `tests/test_examples.py` already uses for the exact-versus-differenced
  comparison of `jacobian`.

## Deliberately not done

- **No trajectory sensitivity.** dy(t)/dp is *not* the partial — y itself depends on p, so the
  useful quantity for a fit comes from integrating the forward-sensitivity equations alongside
  the state. That is a solver change on top of this one, and it belongs to whatever fits
  parameters. This issue delivers the partials that change needs.
- No fit, no optimiser, no uncertainty propagation.
- No d(rhs)/dp for the *discrete* state. A latched relation is frozen between events by
  construction; what a parameter does to the instant an event fires is a different question.

## The shape, read 2026-08-10 before starting

`Emitter.tangent` is **already generic over what it seeds**. Two lines in it mention states:

    width = len(mode.states)
    for slot, state in enumerate(mode.states):
        head.append(f"{self.tangents[state]} = _seed[{slot}]")

Everything after that -- the walk over `mode.blocks`, `chain` for an assignment, `implicit`
for a solved block, the `sharing` pass -- is written in terms of `carried`, which is a
mapping from a variable to the name of its tangent row. It does not know or care that the
rows were seeded from states.

So the change is `tangent(mode, seeds=mode.states)`: seed the chosen *parameters* instead,
size the rows to `len(seeds)`, and every existing caller keeps its default. `differentiate`
becomes one of two callers rather than the only one.

Three things that then need doing and are not free:

- **A second seed matrix.** `_seed = np.eye(width)` is sized to the states. Parameter rows are
  a different width and need their own, which lands next to the note in
  `the-jacobian-is-emitted-dense` about that matrix being `n` by `n` and eagerly allocated.
- **Which parameters, at emit time.** The set has to reach `emit` the way `fold` does, because
  it decides the width of every tangent row in the function. A model with forty parameters and
  an identification asking about three should emit three columns, not forty.
- **`Code` and `Kernel` carry the names**, or a caller cannot tell which column is which.

Verification is settled by precedent: `tests/test_examples.py` already compares `jacobian` and
`observe_jacobian` against central differences on every corpus model, and the parameter
partials go in beside them at the same tolerance. Nothing new has to be invented to check it.

## What landed

The reading above held. `Emitter.tangent(mode, seeds=None)` takes the seed set as an argument;
`seeds is None` means the states and every existing caller keeps its default. Two lines change
inside it — the width and which matrix the rows are sliced from — and the block walk, `chain`,
`implicit` and the `sharing` pass are untouched. `differentiate` is now one of four callers.

- **`emit(system, *, sensitivity=("k", "T_amb"))`** emits `jacobian_parameters(t, y, p, w, out)`
  and `observe_jacobian_parameters(t, y, p, w, out)` beside the other five functions. A name the
  system does not declare as a parameter raises `EmitError` at emit time, not at call time.
- **Two more preamble constants**, `_pseed = np.eye(k)` and `_pnull = np.zeros(k)`, sized to the
  chosen parameters and not to the states. A variable the seed cannot reach gets the shared
  `_pnull` row, the same way the state jacobian gets `_null`.
- **`Code.sensitivity` and `Kernel.sensitivity`** carry the names in column order, so a caller
  can say which column is which. `build(system, sensitivity=(...))` threads through the cache
  key, so a plain build and a sensitivity build are two entries and neither shadows the other.
- Asking for nothing is the default and costs nothing: the two functions are still emitted, with
  zero columns, and no model that does not ask for sensitivities changes by one character.

**Derived parameters chain.** A parameter can be written in terms of others — `RoastingMachine`
has `m_dry = m_green * (1 - moisture)` — and the equations read the derived one. The block walk
begins after `parameters` has already run, so seeding only the named parameters leaves the chain
broken at the binding and d/d(m_green) comes out exactly zero where the difference quotient says
-0.0757. `tangent` now walks `ordered()` once before the blocks and differentiates each binding
it can reach, which repairs it. Confirmed load-bearing by removing it: 1 failed, 23 passed.

`tests/test_sensitivity.py`, 24 passed and 1 skipped. The central-difference comparison runs
over every corpus model, against up to three parameters per model that the equations actually
read, at the tolerance `test_examples.py` uses for `jacobian`. Checked by hand against algebra
on `NewtonCooling`: `d(der)/dp = [[-65.0, 0.5]]` against `-(T - T_amb) = -65.0` and `k = 0.5`.

## Deliberately not done, found while doing it

- **The difference quotient has to hold the state fixed, and that is a fact about the models,
  not about the emitter.** Several corpus models start a state *from* a parameter —
  `RoastingMachine` has `m_w` beginning at `m_green * moisture`. Calling `start(0.0, p)` on both
  sides of the difference therefore moves the state as well as the parameter, and differences a
  quantity that is not the partial. The first RoastingMachine failure was this, in the test, and
  the emitter was right. `dy(t)/dp` — the derivative that *does* include that motion — is the
  trajectory sensitivity this issue already declined.
- **No sensitivity for a folded kernel.** `fold=True` burns parameters in as literals and a
  literal has no derivative; the two options compose only in the sense that the constant-folded
  kernel is a different build. Passing both is not refused today and probably should be, but
  refusing it is a guess about which one the caller meant.
- **No sparsity for the parameter partials.** `jacobian` has `differentiate_sparse` and a
  `_pattern`; these two are dense. A parameter column is dense far more often than a state
  column — a parameter usually reaches most of the model — so the pattern would buy less.
