# %% [markdown]
# # Enumerating the discrete half
#
# A simulation engine is two machines glued together. One is continuous — a stiff integrator
# with an error norm and a convergence theorem behind it. The other is discrete: latches
# flipping, `when` branches firing, a state set being reselected, an FMU walking its calling
# sequence. Only the first has a literature of error control.
#
# The second is **finite**, so it can be enumerated instead of sampled. That is what
# `modelica.verify` does, using [musil](https://gitlab.com/jorgeecardona/musil) as the
# checker. The design note is [Verification](../../ideas/verification/); this notebook runs it.

# %%
from pathlib import Path

from modelica.lang import parse
from modelica.sim import System, causalize, flatten, simulate

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "examples" / "models").is_dir())


def system(name: str, *libraries: str) -> System:
    """Parse, flatten and causalize one of the example models."""
    sources = [(root / "examples" / "library" / f"{lib}.mo").read_text() for lib in libraries]
    sources.append((root / "examples" / "models" / f"{name}.mo").read_text())
    return causalize(flatten([parse(text) for text in sources], name))


# %% [markdown]
# ## 1. Event iteration
#
# At one instant of simulated time the continuous state does not move, so the only thing that
# can change is which watched relations are currently latched true. There are two-to-the-few
# of those, which means the event iteration at an instant is an *exactly* finite machine —
# no abstraction, the emitted `indicators` and `transition` are the functions being called.
#
# Two questions get asked. **Does it stop?** A cycle of events that never settles is a hung
# simulation. **Does the order matter?** The engine applies the first stale event it finds,
# and nothing chooses that order on purpose — so if two orders reach different fixed points,
# that is a bug waiting for a model that notices.

# %%
from modelica.verify import check_iteration

tank = system("DrainingTank")
report = check_iteration(tank)
print(report)

# %% [markdown]
# The tank has one watched relation, so there are two configurations to start from — and a
# simulation only ever visits one of them. The bug this catches is the one hiding in the
# other.

# %%
from modelica.verify import iteration

for start in iteration(tank).initial_states():
    print(start)

# %% [markdown]
# Checking at `t = 0` is thin for a model whose only event is a `when`: nothing is pending at
# the start, so nothing has to settle. The instants worth asking about are the ones a run
# actually reached, so `check_run` simulates first and then checks every event instant.

# %%
from modelica.verify import check_run

ball = system("BouncingBall")
for report in check_run(ball, simulate(ball, stop=2.0, points=400)):
    print(report)

# %% [markdown]
# ## 2. Dynamic state selection
#
# The pendulum is the classic index-2 model: `x² + y² = L²` constrains the states, so index
# reduction has to differentiate it, and the selection it lands on goes **singular** partway
# through the swing. Structural analysis cannot see that coming — it is a statement about
# numbers, not about incidence.
#
# What it *can* do is enumerate the alternatives:

# %%
pendulum = system("Pendulum")
print(f"chosen:     {pendulum.states}")
for candidate in pendulum.candidates:
    print(f"  {candidate}")

# %% [markdown]
# `(x, vx)` is the one the literature names, and it is regular exactly where `(y, vy)` is
# not. Note it is *two* exchanges away from the chosen set, which is why the candidates are
# enumerated rather than found by perturbing the selection one variable at a time.
#
# Switching between them is not written yet. Designing it as a checked automaton first means
# modelling the linear algebra as an **adversary**: which set is usable is not the engine's
# to choose, and the engine only ever learns a set has gone singular by failing to factor it.

# %%
from modelica.verify import check_selection

selection = check_selection(pendulum)
print(selection)
for assumption in selection.safety.unverified_assumptions:
    print(f"\n  {assumption.name}\n    {assumption.description}")

# %% [markdown]
# Those assumptions are not decoration. The second one says the Jacobians hold still while
# the engine is part-way through trying candidates — drop it and the check fails, with the
# run that breaks it:

# %%
broken = check_selection(pendulum, quiescent=False)
print(f"safety still ok:  {broken.safety.ok}")
print(f"liveness ok:      {broken.liveness.ok}  ({broken.liveness.kind})")
print()
print(broken.liveness)

# %% [markdown]
# A set recovered *after* the engine had already ruled it out, so the engine exhausted every
# candidate at a point where one of them was usable all along.
#
# ## 3. The FMI 3.0 calling sequence
#
# FMI does not describe its interface as a list of functions. It describes a state machine,
# and says which `fmi3*` calls are legal in each state. Most of what goes wrong between two
# tools that both "support FMI" is a sequence error: every individual call is fine, and the
# order is not.
#
# The machine is written down once, before the FMI layer that has to obey it.

# %%
from modelica.fmi import Interface, Lifecycle, ModeError
from modelica.verify import check_modes

for interface in Interface:
    print(check_modes(interface))

# %% [markdown]
# "Every mode reachable" is not a formality — it caught Event Mode sitting in the Scheduled
# Execution table, where the standard has Clock Activation Mode instead.
#
# The same table drives the runtime guard. This is what an exported FMU turns into
# `fmi3Error`:

# %%
life = Lifecycle(Interface.MODEL_EXCHANGE)
life.call("fmi3EnterInitializationMode")
life.call("fmi3ExitInitializationMode")
life.call("fmi3EnterContinuousTimeMode")
print(f"mode: {life.mode}")

try:
    life.call("fmi3UpdateDiscreteStates")
except ModeError as error:
    print(f"\n{error}")

# %% [markdown]
# ## 4. The co-simulation master
#
# An SSP system is several FMUs stepped together, which makes the master a distributed
# algorithm wearing a numerical hat. Everybody has to end each communication step at the same
# time, any component may refuse the step it was given, and undoing a step someone else
# already took requires that they kept a copy of themselves first.
#
# FMI is explicit that the copy is optional: `canGetAndSetFMUState` is a capability flag, and
# an FMU is allowed to say no. With every component able to roll back, the master is fine:

# %%
from modelica.ssp import Component
from modelica.verify import check_master

print(check_master((Component("a", rollback=True), Component("b", rollback=True))))

# %% [markdown]
# Now make one of them a perfectly conforming FMU that declines to save its state. Rollback is
# a *precondition* of a rollback-based algorithm, so the master checks it at load time and
# refuses — which is the correct outcome, not a failure:

# %%
MIXED = (Component("a", rollback=True), Component("b", rollback=False))
mixed = check_master(MIXED)
print(f"ok: {mixed.ok}, declined: {mixed.declined}")
print(mixed)

# %% [markdown]
# That precondition is not decoration either. Drop it and the trap comes back:

# %%
print(check_master(MIXED, require_rollback=False).safety)

# %% [markdown]
# Four moves from the start. `b` takes its step, `a` refuses, and there is no move that puts
# `b` back — the master has nothing left to offer. No test that happens to use FMUs which all
# support state saving will ever see it, which is why the precondition is checked rather than
# assumed.
#
# ## What this cannot do
#
# The limits matter as much as the claims. None of this says anything about the continuous
# solve — accuracy, stiffness, step-size control, Jacobian quality are out of scope, and
# pretending otherwise would be worse than not checking at all. It is also the wrong tool for
# matching, BLT sorting and index reduction, which are graph algorithms over unbounded inputs
# rather than finite automata; those want property-based testing.
#
# Model checking earns its place where the state is small and the *orderings* are the
# problem. That is these four and nothing else.
