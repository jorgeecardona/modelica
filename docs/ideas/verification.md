# Verifying the discrete half

!!! note "Tracked in the register"
    What is left after the four targets is tracked in [`docs/issues`](https://gitlab.com/cardona-oss/modelica/-/tree/main/docs/issues): property-based testing for causalization, partial-order reduction, and chattering across instants. This page is the argument; the register is what is owed.

A simulation engine is two machines glued together.

One is continuous. It is a stiff integrator with an error norm, a step-size controller and a
convergence theorem behind it. The other is discrete. Latches flip, `when` branches fire, a
state set is selected again, and an FMU walks its calling sequence. Only the first has a
literature of error control. The second is where the ugly bugs live, and it is usually
tested the way ordinary code is tested: run it, and hope the run was representative.

It does not have to be. The discrete half is *finite*, so you can enumerate it instead of
sampling it.

!!! info "Short version"
    Freeze simulated time, and the discrete layer of this engine is a **finite** state
    machine. [musil][musil] enumerates one of those exhaustively: every reachable state, and
    every interleaving. It returns the shortest sequence of steps that breaks a rule, or it
    proves that none can. Four places pay for this: event iteration, dynamic state
    selection, the FMI 3.0 mode machine, and the SSP co-simulation master. All four are
    implemented in `modelica.verify`, behind the `verify` extra. None of it appears in a
    plain `pip install modelica`.

!!! note "Written as a plan, kept as a record"
    This page was written before any of it existed. It has been corrected where building
    the thing disagreed with the plan. The largest correction is this. The plan was to
    *abstract* the continuous state down to the signs of the event indicators, and to accept
    spurious counterexamples as the price. That turned out to be unnecessary. See
    [Target 1](#target-1-event-iteration).

## The finite part is already there. It has no name yet

`modelica.sim.emit` compiles a model into a workspace array `w`. The slots after the
observed variables hold the **latched relations**: one `1.0` or `0.0` for every relation
that appears inside an equation. They are frozen between events, because Modelica says a
relation does not change value during continuous integration. `indicators` watches the
crossings. `transition` flips the slot, or runs the `when` branch. `code.events` records
each crossing with the direction it fires in.

That is a finite automaton stored in an array of floats. Give it a name and the split is
clean.

| | Carrier | Size | Who decides it |
| --- | --- | --- | --- |
| continuous | `y`, the states chosen by causalization | ℝⁿ | the integrator |
| discrete | the latch slots of `w`, plus `pre`-valued variables when they land | 2ᵏ, finite | `transition` |

One observation makes the discrete half checkable, and it is not an abstraction: **at one
instant of simulated time, `y` does not move.** Events happen at a point, and the integrator
is stopped while they happen. So at a fixed `(t, y)` the indicators are an exact function of
the latch bits. The whole event iteration is then a finite machine over `2ᵏ` states, and its
transition relation is computed by calling the emitted `indicators` and `transition` for
real.

```python
@dataclass(frozen=True, slots=True)
class Instant:
    latches: tuple[bool, ...]  # what the model believes each watched relation says
    state: tuple[float, ...]  # y — constant unless a `when` branch reinitialises it
    fired: frozenset[int]  # events that have already acted here
```

Nothing is approximated. So a counterexample is a genuine execution, and not something that
an over-approximation admitted. One conservative step is left. The engine fires a `when`
branch on a rising edge, and the model lets it fire whenever its condition merely holds. The
model can therefore report a sequence that the integrator would not produce. It can never
miss one that the integrator would.

## What musil is

An explicit-state model checker with no dependencies. It runs in process, in the test suite.
States are frozen dataclasses, actions are guarded pure transitions, and invariants are
predicates.

* `check(model)` sweeps every reachable state for the first broken invariant, or for a
  **deadlock**: a state with no enabled action that was not declared terminal. It returns
  the shortest trace to it. The search is breadth-first, so "shortest" is not a heuristic.
* `check_liveness(model, goal=..., fair=[...])` proves that every run *eventually* reaches a
  goal, under weak or strong fairness. When it does not, the counterexample is a **lasso**:
  a path into a loop that the system can repeat forever.
* `compose(...)` and `channel_actions` cover several components with an unreliable link
  between them, over every interleaving and every reordering of messages.
* `check_open(model, *envs)` checks the system against an external component that misbehaves
  in every way its contract allows. The assumptions you are still trusting on faith are
  listed in the result instead of dropped in silence.

One property matters most for this project: musil is a *library*. There is no specification
language, no external binary, and no second description of the model to keep in step.

## Target 1 — event iteration

`modelica.sim.solve` used to apply **one** event per segment, and then hand control straight
back to `solve_ivp`. Modelica requires an *event iteration*. Applying a transition changes
the blocks that read the latch it flipped. That can leave a second relation on the wrong
side of its own latch, and the loop has to run to a fixed point before the integrator
restarts.

Writing this page turned that from a missing feature into a **bug**. The integrator can
never catch that second relation. On the next segment its indicator does not *cross* zero;
it starts out already past it. So no event is reported, and the stale value is carried for
the rest of the run. The iteration now runs in `_settle`. The old stand-in — 100 events at
one time point, and `simulate` raises "the model is chattering" — is back to being the
backstop it should be.

Here is what the checker states, in `modelica.verify`.

| Property | How it is spelled |
| --- | --- |
| the iteration terminates | `check_liveness(goal=settled)` |
| there is no discrete deadlock | `check(model)` reports it for free |
| the order events are applied in does not change where it lands | one settled state per configuration |
| every latch configuration, and not only the reachable one | `2ᵏ` initial states |

That last row is the one a simulation cannot do for you. A run visits the configurations its
own trajectory reaches, and the bug is in the other one. `check_run` goes further: it
re-checks at every instant a real run stopped at — the bounce, the switch, the moment the
tank runs dry.

An event acts at most once per instant, and that is what makes the iteration terminate
*today*. When discrete variables and `pre()` land, that guarantee goes away. A condition can
then be driven back and forth within one instant, and the liveness check stops being a
formality.

## Target 2 — dynamic state selection

The open `xfail` in the suite is `Pendulum`. The dummy-derivative choice is fixed at
causalization time, and it goes singular at t ≈ 0.592, so the solver dies trying to take a
step it cannot take. The fix is dynamic state selection: several admissible state sets, and
a switch when the current one degenerates.

The modes come first, and they are a property of the model rather than of the runtime.
`causalize` now enumerates every admissible state set onto `System.candidates`. A set is
admissible when holding it back still leaves an equation for every remaining unknown, so the
test is the matching itself, run once per candidate with that set forbidden. The pendulum
answers `(y, vy)`, `(x, vx)`, and three structurally valid oddities. The two that the
literature names are *two exchanges* apart, which is why this enumerates instead of
perturbing the chosen set one variable at a time.

The switching is then modelled with the linear algebra as the **adversary**. Which set is
usable is not the engine's to choose, and the engine learns that a set has gone singular
only by failing to factor it. Two assumptions come out of that, and `check_open` keeps them
visible instead of letting them be swallowed.

* *Index reduction leaves a usable set.* At every point, at least one candidate is regular.
  Structural analysis cannot establish this. It is a statement about numbers.
* *The Jacobians hold still during a search.* They change as the step advances, and not
  while the engine is part-way through trying candidates.

The second one carries load, and it is checked as such.
`check_selection(system, quiescent=False)` drops it, and the counterexample is concrete. A
set recovers *after* the engine ruled it out, the engine exhausts every candidate at a point
where one of them was usable, and no step is ever accepted again. Note which kind of
property fails: **liveness**, and not safety. Nothing illegal happens. The system simply
never gets anywhere. Guessing which of the two would break was wrong the first time.

## Target 3 — the FMI 3.0 mode machine

FMI 3.0 does not describe its interface as a list of functions. It describes a **state
machine**, and it says which `fmi3*` calls are legal in each state. The table below is
verified against the specification sources.

| State | Some of what is legal there |
| --- | --- |
| Instantiated | `fmi3EnterConfigurationMode`, `fmi3EnterInitializationMode`, `fmi3Get{VariableType}` |
| Configuration Mode | `fmi3Set{VariableType}`, `fmi3ExitConfigurationMode` |
| Initialization Mode | `fmi3Set/Get{VariableType}`, `fmi3ExitInitializationMode` |
| Event Mode | `fmi3UpdateDiscreteStates`, `fmi3EnterContinuousTimeMode` (ME), `fmi3EnterStepMode` (CS) |
| Continuous-Time Mode | `fmi3SetTime`, `fmi3SetContinuousStates`, `fmi3GetContinuousStateDerivatives`, `fmi3CompletedIntegratorStep` |
| Step Mode | `fmi3DoStep`, `fmi3EnterEventModeCS`, `fmi3EnterConfigurationMode` |
| Reconfiguration Mode | `fmi3Set{VariableType}`, `fmi3ExitConfigurationMode` |
| Intermediate Update Mode | a restricted get and set, entered by the FMU's own callback |
| Terminated | getters only |

Three more rules complete it. `fmi3ExitInitializationMode` lands in Event Mode for Model
Exchange, and in Step Mode for Co-Simulation. An "FMU State Settable" super-state carries
`fmi3GetFMUState` and `fmi3SetFMUState` across all of them. And `fmi3Terminate`, or an error
return, reaches Terminated from anywhere.

Note what Event Mode contains. `fmi3UpdateDiscreteStates` is called again and again, until
it reports that the discrete states no longer need updating. **The FMI API itself mandates
the event iteration of target 1.** The two targets are the same obligation seen from two
sides. That is the strongest possible argument for doing target 1 first, and for building
FMI export on top of it rather than beside it.

The table is written **before** the FMI layer, in `modelica.fmi.modes`, and three readers
use it.

1. The runtime guard of the exported FMU. `Lifecycle.call` refuses an illegal call, and that
   becomes `fmi3Error` instead of a corrupted instance.
2. The driver of the importer. It can only emit sequences that the table allows.
3. `modelica.verify.check_modes`, which takes every reachable call sequence at once.

One table, three readers, and no drift. This is the class of bug that is worst to find by
testing, because nothing is wrong with any individual call. Only the order is wrong.

Two corrections came out of checking it, and both are the reason to bother.

* **"Every mode is reachable" is a real check.** Event Mode was in the Scheduled Execution
  table. Scheduled Execution does not have one. It has Clock Activation Mode, and Clock
  Update Mode, which is reached through `fmi3ClockUpdateCallback`. Reading the table would
  not have shown that. Asking which modes anything can enter showed it immediately.
* **"An instance can always be freed" is reachability, and not liveness.** The first version
  asked for it as liveness and got a counterexample: an importer that enters and exits
  Configuration Mode forever. That is a legal caller, doing nothing wrong, that simply never
  frees anything. What the standard owes is that `fmi3FreeInstance` is *available* from
  wherever you are. That is a question about the graph, and it is now answered on the graph.

## Target 4 — the SSP co-simulation master

An SSP system is several FMUs stepped in lockstep. That is a distributed system, with all
the usual hazards. A component returns `fmi3Discard` with an earlier `lastSuccessfulTime`
and everyone must roll back. Two components discard one after the other. An event in one
propagates to another in the middle of a step. An algebraic loop across the connection graph
does not converge.

The protocol lives in `modelica.ssp.master` as pure transitions: what phase the master is
in, what moves are legal, and what each move does. It was written before the master, so that
the master had something to be checked against. Time is counted in communication points and
not in seconds. The questions are about who is ahead of whom, integers answer those exactly,
and integers keep the space finite. A refusal is one of the moves, and not a separate fault
model, because discarding a step is a legal answer to `fmi3DoStep`. A master that only works
when nobody discards does not work.

The finding is what makes the module worth writing. **`canGetAndSetFMUState` is optional**,
and FMI says so. When every component can roll back, the master is fine. Make one of them a
perfectly conforming FMU that declines to save its state, and the check returns this, four
moves from the start:

```
→ save a          → begin         → step b        → a discards
INVARIANT VIOLATED: a-rollback-can-put-everyone-back
```

`b` has taken its step, `a` refuses, and no move puts `b` back. No test that happens to use
FMUs which all support state saving will ever see it.

One invariant had to be corrected on the way, and the mistake is easy enough to be worth
recording. The first version demanded that every snapshot be current. That fails one move
after `commit` on a perfectly healthy master. A stale snapshot is harmless, because
`restore` will not use it. What has to hold is narrower, and it is the actual property:
*when a rollback is needed, everyone who moved can be put back.*

## What this cannot do

The limits matter as much as the claims, and three of them are hard.

* **One instant at a time.** Freezing `y` is what buys exactness, and it is also the
  boundary. Each event acts at most once, so this says nothing about chattering *across*
  instants. That stays a runtime question, watched by the standstill counter in
  `modelica.sim.solve`. `check_run` narrows the gap, because it re-checks at every instant a
  real trajectory stopped at. Narrowing is not closing.
* **It says nothing about the continuous solve.** Accuracy, stiffness, step size control and
  the quality of the Jacobian are all out of scope. Pretending otherwise would be worse than
  not checking at all. Those belong to the work-precision measurements in
  [benchmarks](benchmarks.md).
* **It is the wrong tool for causalization.** Matching, BLT sorting and Pantelides are graph
  algorithms over unbounded inputs, and not finite automata. Their properties — the matching
  is perfect, index reduction terminates, differentiating preserves the solution set — want
  property-based testing over generated models. Model checking earns its place where the
  state is small and the *orderings* are the problem. That is the four targets above, and
  nothing else.

State explosion is a fourth limit, but a soft one, because these models are small. If
composing five FMUs does blow up, then `canonicalize=` is the first lever. It is symmetry
reduction over interchangeable components, and `symmetry_reduction_sound` validates it.

## What this project gives back to musil

The traffic lights and the dining philosophers in musil's examples are a control-plane view
of the world. A simulation engine turned out to be a different kind of client in two ways
that mattered, and one that did not.

* **A bug, found by being an unusual client.** `Mode` is a `StrEnum`, so an FMI mode is a
  `str`. So musil's heuristic for "several initial states" — anything iterable that is not a
  dataclass — took a single mode apart into its characters. It did not raise. It checked the
  model whose states are `I`, `n`, `s`, `t` and the rest, and it reported OK. That is the
  worst way for a checker to be wrong. `str` and `bytes` are now always one state.
* **`protocol_actions`.** `transition_actions` covers `{state: {states it may become}}`. A
  published protocol has edges labelled by the *call*, and most calls are legal without
  moving. That is `{state: {operation: where it leaves you}}`, with `None` for "legal, stays
  here". A guard needs that shape anyway, so one table now drives the guard and the checker.
  FMI 3.0 is the motivating case, and a real industrial specification checked in one page is
  a better argument for the tool than another mutex.
* **A helper for predicate abstraction, which turned out to be unnecessary.** The plan wanted
  one, on the assumption that a continuous state must be abstracted before it can be checked.
  Freezing the instant made the model exact instead, so the helper would have been built for
  a client that does not have the problem. It is recorded here rather than dropped in
  silence, because "we thought we needed X and then did not" is worth more than a list that
  quietly got shorter.

One thing is still open. Composed FMUs with rollback are the workload that would justify the
partial-order reduction that musil's README lists as missing. Nothing here has been big
enough to need it yet.

Neither project depends on the other at runtime. musil is an *extra* here and not a base
dependency, and the checks live behind it. [The core is deliberately free of
dependencies](../architecture.md), and a verification tool is not a reason to break that.

[musil]: https://gitlab.com/jorgeecardona/musil
