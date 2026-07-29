# Verifying the discrete half

A simulation engine is two machines glued together. One is continuous — a stiff integrator
with an error norm, a step-size controller and a convergence theorem behind it. The other is
discrete — latches flipping, `when` branches firing, a state set being reselected, an FMU
walking its calling sequence. Only the first has a literature of error control. The second is
where the ugly bugs live, and it is usually tested the way ordinary code is tested: run it,
and hope the run was representative.

It does not have to be. The discrete half is *finite*, so it can be enumerated instead of
sampled.

!!! info "Short version"
    Freeze simulated time and the discrete layer of this engine is a **finite** state machine.
    [musil][musil] enumerates those exhaustively — every reachable state, every interleaving —
    and returns the shortest sequence of steps that breaks a rule, or proves none can. Four
    places that pays: event iteration, dynamic state selection, the FMI 3.0 mode machine, and
    the SSP co-simulation master. All four are implemented in `modelica.verify`, behind the
    `verify` extra; nothing here shows up in a plain `pip install modelica`.

!!! note "Written as a plan, kept as a record"
    This page was written before any of it existed, and has been corrected where building it
    disagreed with the plan. The largest correction: the plan was to *abstract* the continuous
    state to the signs of the event indicators, and accept spurious counterexamples as the
    price. That turned out to be unnecessary — see [Target 1](#target-1-event-iteration).

## The finite part is already there, it just is not named

`modelica.sim.emit` compiles a model into a workspace array `w`. The slots after the observed
variables hold the **latched relations**: one `1.0`/`0.0` per relation that appears inside an
equation, frozen between events because Modelica says a relation does not change value during
continuous integration. `indicators` watches the crossings, `transition` flips the slot or runs
the `when` branch, and `code.events` records each crossing with the direction it fires in.

That is a finite automaton stored in an array of floats. Give it its own name and the split is
clean:

| | carrier | size | who decides it |
| --- | --- | --- | --- |
| continuous | `y`, the states chosen by causalization | ℝⁿ | the integrator |
| discrete | the latch slots of `w`, plus `pre`-valued variables when they land | 2ᵏ, finite | `transition` |

What makes the discrete half checkable is not an abstraction but an observation: **at one
instant of simulated time, `y` does not move.** Events happen at a point, and the integrator is
stopped while they do. So at a fixed `(t, y)` the indicators are an exact function of the latch
bits, and the whole event iteration is a finite machine over `2ᵏ` states whose transition
relation is computed by calling the emitted `indicators` and `transition` for real.

```python
@dataclass(frozen=True, slots=True)
class Instant:
    latches: tuple[bool, ...]   # what the model believes each watched relation says
    state: tuple[float, ...]    # y — constant unless a `when` branch reinitialises it
    fired: frozenset[int]       # events that have already acted here
```

Nothing is approximated, so a counterexample is a genuine execution rather than something an
over-approximation admitted. The one conservative step left is a `when` branch: the engine fires
it on a rising edge, and the model lets it fire whenever its condition merely holds — which can
report a sequence the integrator would not produce, but never miss one it would.

## What musil is

A dependency-free explicit-state model checker, in-process, in the test suite. States are
frozen dataclasses, actions are guarded pure transitions, invariants are predicates:

* `check(model)` — sweeps every reachable state for the first broken invariant or **deadlock**
  (a state with no enabled action that was not declared terminal), and returns the shortest
  trace to it. Breadth-first, so "shortest" is not a heuristic.
* `check_liveness(model, goal=..., fair=[...])` — proves that every run *eventually* reaches a
  goal, under weak or strong fairness, with a **lasso** counterexample (a path into a loop the
  system can repeat forever) when it does not.
* `compose(...)`, `channel_actions` — several components and an unreliable link between them,
  with every interleaving and every message reordering covered.
* `check_open(model, *envs)` — the system checked against an external component that misbehaves
  in every way its contract allows, with the assumptions you are still trusting on faith listed
  in the result rather than silently dropped.

The relevant property for this project is that it is a *library*: no spec language, no external
binary, no second description of the model to keep in sync.

## Target 1 — event iteration

`modelica.sim.solve` used to apply **one** event per segment and hand control straight back to
`solve_ivp`. Modelica requires an *event iteration*: applying a transition changes the blocks
that read the latch it flipped, which can leave a second relation on the wrong side of its own
latch, and the loop has to run to a fixed point before the integrator restarts.

Writing this page turned that from a missing feature into a **bug**. The integrator can never
catch that second relation: on the next segment its indicator does not *cross* zero, it starts
out already past it, so no event is reported and the stale value is carried for the rest of the
run. The iteration now runs in `_settle`, and the old stand-in — 100 events at one time point
and `simulate` raises "the model is chattering" — is back to being the backstop it should be.

What the checker states, in `modelica.verify`:

| property | how it is spelled |
| --- | --- |
| the iteration terminates | `check_liveness(goal=settled)` |
| no discrete deadlock | `check(model)` reports it for free |
| the order events are applied in does not change where it lands | one settled state per configuration |
| every latch configuration, not only the reachable one | `2ᵏ` initial states |

That last row is the one a simulation cannot do for you: a run visits the configurations its own
trajectory reaches, and the bug is in the other one. `check_run` goes further and re-checks at
every instant a real run stopped at — the bounce, the switch, the moment the tank runs dry.

An event acts at most once per instant, which is what makes the iteration terminate *today*.
When discrete variables and `pre()` land that guarantee goes away, a condition can be driven back
and forth within one instant, and the liveness check stops being a formality.

## Target 2 — dynamic state selection

The open `xfail` in the suite is `Pendulum`: the dummy-derivative choice is fixed at
causalization time and goes singular at t ≈ 0.592, so the solver dies trying to take a step it
cannot take. The fix is dynamic state selection — several admissible state sets, and a switch
when the current one degenerates.

The modes come first, and they are a property of the model rather than of the runtime:
`causalize` now enumerates every admissible state set onto `System.candidates`. A set is
admissible when holding it back still leaves an equation for every remaining unknown, so the test
is the matching itself, run once per candidate with that set forbidden. The pendulum answers
`(y, vy)`, `(x, vx)` and three structurally valid oddities — and the two the literature names are
*two exchanges* apart, which is why this enumerates rather than perturbing the chosen set one
variable at a time.

The switching is then modelled with the linear algebra as the **adversary**, because which set is
usable is not the engine's to choose and the engine only ever learns a set has gone singular by
failing to factor it. Two assumptions come out of that, and `check_open` keeps them visible
rather than letting them be swallowed:

* *index reduction leaves a usable set* — at every point, at least one candidate is regular.
  Structural analysis cannot establish this; it is a statement about numbers.
* *the Jacobians hold still during a search* — they change as the step advances, not while the
  engine is part-way through trying candidates.

The second is load-bearing and is checked as such: `check_selection(system, quiescent=False)`
drops it and the counterexample is concrete — a set recovers *after* the engine ruled it out, the
engine exhausts every candidate at a point where one of them was usable, and no step is ever
accepted again. Note it is **liveness** that fails, not safety: nothing illegal happens, the
system simply never gets anywhere. Guessing which of the two would break was wrong the first time.

## Target 3 — the FMI 3.0 mode machine

FMI 3.0 does not describe its interface as a list of functions; it describes a **state machine**
and says which `fmi3*` calls are legal in each state. Verified against the specification
sources:

| state | some of what is legal there |
| --- | --- |
| Instantiated | `fmi3EnterConfigurationMode`, `fmi3EnterInitializationMode`, `fmi3Get{VariableType}` |
| Configuration Mode | `fmi3Set{VariableType}`, `fmi3ExitConfigurationMode` |
| Initialization Mode | `fmi3Set/Get{VariableType}`, `fmi3ExitInitializationMode` |
| Event Mode | `fmi3UpdateDiscreteStates`, `fmi3EnterContinuousTimeMode` (ME), `fmi3EnterStepMode` (CS) |
| Continuous-Time Mode | `fmi3SetTime`, `fmi3SetContinuousStates`, `fmi3GetContinuousStateDerivatives`, `fmi3CompletedIntegratorStep` |
| Step Mode | `fmi3DoStep`, `fmi3EnterEventModeCS`, `fmi3EnterConfigurationMode` |
| Reconfiguration Mode | `fmi3Set{VariableType}`, `fmi3ExitConfigurationMode` |
| Intermediate Update Mode | a restricted get/set, entered by the FMU's own callback |
| Terminated | getters only |

with `fmi3ExitInitializationMode` landing in Event Mode for Model Exchange and in Step Mode for
Co-Simulation, a "FMU State Settable" super-state carrying `fmi3GetFMUState` / `fmi3SetFMUState`
across all of them, and `fmi3Terminate` or an error return reaching Terminated from anywhere.

Note what Event Mode contains: `fmi3UpdateDiscreteStates`, called repeatedly until it reports
that the discrete states no longer need updating. **FMI's own API mandates the event iteration
of target 1.** The two targets are the same obligation seen from two sides, which is the
strongest possible argument for doing target 1 first and building FMI export on top of it
rather than beside it.

The table is written **before** the FMI layer, in `modelica.fmi.modes`, and read three times:

1. the exported FMU's runtime guard — `Lifecycle.call` refuses an illegal call, which is what
   becomes `fmi3Error` rather than a corrupted instance;
2. the importer's driver — it can only emit sequences the table allows;
3. `modelica.verify.check_modes` — every reachable call sequence at once.

One table, three readers, no drift. This is the class of bug that is worst to find by testing,
because nothing is wrong with any individual call — only with the order.

Two corrections came out of checking it, and both are the reason to bother:

* **"every mode is reachable" is a real check.** Event Mode was in the Scheduled Execution
  table. Scheduled Execution does not have one — it has Clock Activation Mode, and Clock Update
  Mode reached through `fmi3ClockUpdateCallback`. Reading the table would not have shown that;
  asking which modes anything can enter did, immediately.
* **"an instance can always be freed" is reachability, not liveness.** The first version asked
  for it as liveness and got a counterexample describing an importer that enters and exits
  Configuration Mode forever — a legal caller, doing nothing wrong, that simply never frees
  anything. What the standard owes is that `fmi3FreeInstance` is *available* from wherever you
  are. That is a question about the graph, and it is now answered on the graph.

## Target 4 — the SSP co-simulation master

An SSP system is several FMUs stepped in lockstep, and that is a distributed system with all
the usual hazards: a component returns `fmi3Discard` with an earlier `lastSuccessfulTime` and
everyone must roll back; two components discard alternately; an event in one propagates to
another mid-step; an algebraic loop across the connection graph does not converge.

The protocol lives in `modelica.ssp.master` as pure transitions — what phase the master is in,
what moves are legal, what each does — written before the master so that the master has
something to be checked against. Time is counted in communication points rather than seconds:
the questions are about who is ahead of whom, and integers answer those exactly while keeping
the space finite. A refusal is one of the moves, not a separate fault model, because discarding
a step is a legal answer to `fmi3DoStep` and a master that only works when nobody discards does
not work.

The finding is the one worth writing the module for. **`canGetAndSetFMUState` is optional**, and
FMI says so. With every component able to roll back, the master is fine. Make one of them a
perfectly conforming FMU that declines to save its state and the check returns this, four moves
from the start:

```
→ save a          → begin         → step b        → a discards
INVARIANT VIOLATED: a-rollback-can-put-everyone-back
```

`b` has taken its step, `a` refuses, and there is no move that puts `b` back. No test that
happens to use FMUs which all support state saving will ever see it.

One invariant had to be corrected on the way, which is worth recording because the mistake is
easy: the first version demanded that every snapshot be current, and that fails one move after
`commit` on a perfectly healthy master. A stale snapshot is harmless — `restore` will not use
it. What has to hold is narrower and is the actual property: *when a rollback is needed,
everyone who moved can be put back*.

## What this cannot do

The limits matter as much as the claims, and three of them are hard:

* **One instant at a time.** Freezing `y` is what buys exactness, and it is also the boundary:
  each event acts at most once, so this says nothing about chattering *across* instants. That
  remains a runtime question, watched by the standstill counter in `modelica.sim.solve`.
  `check_run` narrows the gap by re-checking at every instant a real trajectory stopped at, but
  narrowing is not closing.
* **It says nothing about the continuous solve.** Accuracy, stiffness, step-size control,
  Jacobian quality: out of scope, and pretending otherwise would be worse than not checking at
  all. Those are the job of the work-precision measurements in [benchmarks](benchmarks.md).
* **It is the wrong tool for causalization.** Matching, BLT sorting and Pantelides are graph
  algorithms over unbounded inputs, not finite automata. Their properties ("the matching is
  perfect", "index reduction terminates", "differentiating preserves the solution set") want
  property-based testing over generated models. Model checking earns its place where the state
  is small and the *orderings* are the problem — which is precisely the four targets above and
  nothing else.

State explosion is the fourth limit, but a soft one: these models are small. If composing five
FMUs does blow up, `canonicalize=` (symmetry reduction over interchangeable components,
validated by `symmetry_reduction_sound`) is the first lever.

## What this project gives back to musil

The traffic lights and dining philosophers in musil's examples are a control-plane view of the
world. A simulation engine turned out to be a different client in two ways that mattered and one
that did not:

* **A bug, found by being an unusual client.** `Mode` is a `StrEnum`, so an FMI mode is a `str`,
  so musil's "several initial states" heuristic — anything iterable that is not a dataclass —
  took a single mode apart into its characters. It did not raise. It checked the model whose
  states are `I`, `n`, `s`, `t` … and reported OK, which is the worst way for a checker to be
  wrong. `str` and `bytes` are now always one state.
* **`protocol_actions`.** `transition_actions` covers `{state: {states it may become}}`. A
  published protocol has edges labelled by the *call*, and most calls are legal without moving:
  `{state: {operation: where it leaves you}}`, with `None` for "legal, stays here". That is what
  a guard needs anyway, so one table now drives the guard and the checker. FMI 3.0 is the
  motivating case, and a real industrial specification checked in a page is a better argument
  for the tool than another mutex.
* **A predicate-abstraction helper — not needed after all.** The plan wanted one, on the
  assumption that a continuous state must be abstracted before it can be checked. Freezing the
  instant made the model exact instead, so the helper would have been built for a client that
  turned out not to have the problem. Recorded rather than quietly dropped, because "we thought
  we needed X and then did not" is worth more than a silently shortened list.

Still open: composed FMUs with rollback are the workload that would justify the partial-order
reduction musil's README lists as missing. Nothing here has been big enough to need it yet.

Neither project depends on the other at runtime. musil is an *extra* here, not a base
dependency, and the checks live behind it, because
[the core is deliberately dependency-free](../architecture.md) and a verification tool is not a
reason to break that.

[musil]: https://gitlab.com/jorgeecardona/musil
