# %% [markdown]
# # Simulating a model, one stage at a time
#
# The design rule for this project is that **every stage is a value you can hold**. That is
# hard to believe from a prose description and obvious from a notebook, so this one walks a
# model from source text to trajectory and stops at each stage to look at what came out.
#
# The model is `examples/models/RLCCircuit.mo` — a resistor, an inductor and a capacitor,
# built from the component library in `examples/library/Electrical.mo`.

# %%
from pathlib import Path

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "examples" / "models").is_dir())
source = (root / "examples" / "models" / "RLCCircuit.mo").read_text()
library = (root / "examples" / "library" / "Electrical.mo").read_text()

print(source)

# %% [markdown]
# ## 1. Source → AST
#
# The parser is hand-written recursive descent. It returns `modelica.ir` values, and it
# refuses constructs it does not implement by name rather than misreading them.

# %%
from modelica.lang import parse

program = parse(source)
[c.name for c in program.classes]

# %% [markdown]
# ## 2. AST → flat model
#
# Flattening instantiates the components, resolves the connect equations into the Kirchhoff
# laws they stand for, and produces a pile of scalar equations that happen to be true. No
# direction of computation yet — Modelica's `=` is acausal, so nothing here says which
# variable an equation *solves for*.

# %%
from modelica.sim import flatten

flat = flatten([parse(library), program], "RLCCircuit")
print(f"{len(flat.variables)} variables, {len(flat.equations)} equations")
print(f"balanced: {flat.is_balanced()}")

# %% [markdown]
# ## 3. Flat model → causalized system
#
# This is the stage that does the work: alias elimination, index reduction where the model
# needs it, matching each equation to the variable it computes, and sorting the result into
# blocks that can be evaluated in order. Whatever is left unmatched becomes a **state**.

# %%
from modelica.sim import causalize

system = causalize(flat)
print(f"states:     {system.states}")
print(f"candidates: {system.candidates}")
print(f"index:      {system.index}")
print(f"blocks:     {len(system.blocks)}, largest {system.largest_block}")
print(f"explicit:   {system.is_explicit}")

# %% [markdown]
# `candidates` is every state set that *would* also have worked. Here there is only one, so
# there is nothing to choose. The pendulum is the interesting case — see the
# [verification notebook](verifying.py).
#
# ## 4. Causalized system → generated code
#
# The IR is compilable by construction, which is what lets a Python engine stand next to the
# Fortran solvers. `emit` writes the model out as straight-line arithmetic; nothing walks a
# tree of closures at evaluation time.

# %%
from modelica.sim import emit

code = emit(system)
print(code.source[: code.source.index("def rhs")])

# %% [markdown]
# Here is the right-hand side itself — the function the integrator calls a few times per
# step, and the reason the shape of the IR matters:

# %%
body = code.source[code.source.index("def rhs") :]
print(body[: body.index("\n\n\n")])

# %% [markdown]
# ## 5. Simulate
#
# `Radau` by default: a model that came out of a Modelica flattener is stiff more often than
# not, and an implicit method that is merely slower on an easy problem beats an explicit one
# that fails on a hard one.

# %%
from modelica.sim import simulate

result = simulate(system, stop=0.05, points=500)
result

# %% [markdown]
# The chart is a self-contained HTML page with no CDN behind it: every variable the model
# has is in the legend, the moving ones are ticked on to begin with, and events are drawn as
# vertical marks. It writes itself to a file with `result.plot().save("rlc.html")`.
#
# ## An event, and why the loop restarts
#
# `BouncingBall` is the one to look at for discontinuities. The integrator carries an
# implicit assumption that what it is integrating is smooth over the step it is taking; a
# bounce violates that exactly once per bounce. So the solver is stopped at the crossing, the
# discontinuity is applied, and integration starts again from the other side.

# %%
ball = causalize(
    flatten([parse((root / "examples" / "models" / "BouncingBall.mo").read_text())], "BouncingBall")
)
bounces = simulate(ball, stop=3.0, points=800)
for event in bounces.events:
    print(f"t = {event.time:.6f}  {event.cause}")

# %%
bounces.plot()
