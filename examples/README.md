# Examples — the behaviour spec

Every model here is a **specification**, not a demo. A Modelica model already says what the
system does; pairing it with a small file of expected behaviour turns the pair into an
acceptance test that no layer of the toolchain can quietly break.

```
examples/
  library/*.mo     small component packages (Electrical, Translational, Thermal)
  library/*.py     the same packages through the Python front-end
  models/*.mo      one self-contained example model each
  models/*.py      the same model through the Python front-end
  models/*.toml    what that model must do — the spec
```

`tests/test_examples.py` reads the corpus and runs it. Stages that are not implemented yet
report as skipped rather than failing, so the suite is a live progress meter: as `lang`,
`ir` and `sim` land, skips turn into passes without a single spec being edited.

## Written twice, on purpose

Every example exists in both front-ends, under the same stem: `RCCircuit.mo` and
`RCCircuit.py` are the same model, and the suite asserts they say the same thing. That pair
is the only thing keeping `modelica.build` and `modelica.lang` from drifting apart.

```python
class RCCircuit(Model):
    """A capacitor charging through a resistor from a 1 V source"""

    source = Electrical.ConstantVoltage(V=1)
    r = Electrical.Resistor(R=100)
    c = Electrical.Capacitor(C=0.01, v=mod(start=0, fixed=True))
    ground = Electrical.Ground()

    def equations(self):
        yield connect(self.source.p, self.r.p)
        yield connect(self.r.n, self.c.p)
        yield connect(self.c.n, self.source.n)
        yield connect(self.source.n, self.ground.p)
```

Conventions:

* A **library** twin defines its classes and ends with
  `PACKAGE = package("Electrical", __name__, "...")`. The name `PACKAGE` rather than
  `Electrical` avoids an `Electrical.Electrical` on import.
* A **model** twin is one class named after the file, so `getattr(module, path.stem)` finds
  it.
* The examples are not an installed package. A model that uses a library adds the sibling
  directory to `sys.path` at the top of the file, which keeps `import Electrical` reading
  exactly like the Modelica `import`.
* Running a twin directly prints its Modelica: `python examples/models/RCCircuit.py`.

The comparison is on normalised text today — blank lines dropped, numeric literals folded
through `repr(float(...))` so `1e4` and `10000.0` agree. Once `modelica.lang.parse` exists
it becomes a comparison of flat models as values, which is the check actually worth having.

## The spec format

TOML, one file per model, named after it. **An absent key is not an assertion** — write
only what you are willing to commit to.

```toml
summary  = "One line: what the model is."
teaches  = ["connect", "flow variables"]   # what a reader should learn from it
requires = ["Electrical"]                  # library packages to load first ([] if none)

[parse]
classes = ["RCCircuit"]        # top-level classes the file must yield

[flatten]
model     = "RCCircuit"        # the class to instantiate
states    = ["c.v"]            # continuous states, after index reduction
index     = 1                  # differentiation index of the flat DAE
balanced  = true               # #equations == #unknowns
equations = 14                 # optional, only where the count is unambiguous

[simulate]
stop      = 5.0
tolerance = 1e-8
method    = "LSODA"            # optional; any scipy `solve_ivp` method, default "Radau"

# A value at a point in time. `source` records where the number came from:
#   "analytic"  — closed form, written out in a comment
#   "reference" — scipy at rtol/atol 1e-12, i.e. a golden trajectory
#   "measured"  — read off real hardware; see `RoastingMachine`
[[simulate.point]]
t      = 1.0
values = { "c.v" = 0.632120558829 }
rtol   = 1e-6
source = "analytic"

# Something that must hold at every output point: an expression, and either a value it
# equals or bounds it stays within. `after` restricts it to t >= after.
[[simulate.invariant]]
expr = "y1 + y2 + y3"
value = 1.0
atol = 1e-8

# Expected state events (zero crossings), in order.
[simulate.events]
times = [0.451523641437, 0.813742554587]
atol  = 1e-6
```

## Drawing one

A model of a *mechanism* has a picture that a line plot cannot show, and for those the spec
carries an optional `[animate]` section: bodies at their coordinates, rods and springs
between them, the path a bob sweeps, and a panel underneath for the one scalar worth
watching. Coordinates are Modelica expressions over the model's own variables — the same
`parse_expression` + `emit_expr` road the invariants take — so `x2` means the model's `x2`.

```toml
[animate]
stop  = 40.0                  # optional; defaults to [simulate].stop
fps   = 30
speed = 1.0                   # seconds of model time per second of video
title = "..."
width = 6.6                   # inches; the height follows the plane, never given

[[animate.part]]              # kind = rod | spring | mass | pivot | track
kind  = "spring"
from  = ["0", "0"]            # `at` instead of `from`/`to` for mass and pivot
to    = ["xp", "yp"]
coils = 14

[[animate.part]]
kind  = "mass"
at    = ["x2", "y2"]
trace = 4.0                   # seconds of path kept behind it
color = 1                     # index into the palette; 0 if omitted

[[animate.plot]]
expr  = "energy"
label = "total energy (J)"
ylim  = [-35.5, -33.5]        # worth fixing for a conserved quantity — see below
```

`python examples/animate.py <Model>` writes `<Model>.mp4`; `--out x.gif` writes a GIF
instead, and `--still 24.0` writes one frame at that instant, which is what you want while
a scene is being written. It needs `matplotlib` (the `docs` extra) and, for `.mp4`, ffmpeg
— neither of which the library itself will ever depend on.

Two things that are easy to get wrong, both now defaulted the right way. Leave `xlim`/`ylim`
out unless you have a reason: the frame is computed from everything the run actually
visits, and a guess clips a chaotic path. And fix `ylim` on a panel showing a *conserved*
quantity: left to itself the axis zooms into the integrator's last digit, and a line flat
to one part in 10⁹ is drawn as violent noise — the exact opposite of the claim.

## The models

Ordered the way the toolchain gets built: each one needs everything the ones above it
needed, plus exactly one new capability.

| Model | New capability it forces | Reference |
| --- | --- | --- |
| `NewtonCooling` | `der()`, parameters, one state, one equation | analytic |
| `RCCircuit` | `connect()`, `flow`, `extends`, a component library | analytic |
| `TwoBodyHeat` | connectors in a second domain; a conservation invariant | analytic |
| `RLCCircuit` | two states, a coupled linear system | scipy |
| `MassSpringDamper` | the same physics through a different domain's connectors | scipy |
| `DrainingTank` | `if` in an equation → a state event; `sqrt` | analytic |
| `BouncingBall` | `when` / `reinit` / `pre` — repeated events, state reinitialisation | analytic |
| `RoastingMachine` | `time` in an equation → time events *and* a state event; a parameter bound to other parameters; modelling the sensor | **a real machine** |
| `Pendulum` | a constraint equation → index 3 → Pantelides + dummy derivatives | scipy |
| `AtwoodPendulum` | five coupled degrees of freedom → a 4×4 implicit block to solve | scipy + energy |
| `Robertson` | stiffness: a stiff solver is mandatory, explicit RK dies | scipy |
| `VanDerPol` | a relaxation limit cycle; step-size control across 3 orders of magnitude | scipy |

`Robertson` and `VanDerPol` are not physical-modelling examples — they are the classic
Hairer–Wanner stiff test problems, here to hold the solver honest.

`AtwoodPendulum` is the one whose reference nobody can eyeball: the trajectory is chaotic,
so a golden point five seconds in says something about this toolchain's arithmetic and
nothing about the physics. What checks the *model* is its own total energy, written out as
a variable and asserted flat. The equations came out of a computer algebra system and no
one can proofread them by reading; the invariant is what proofreads them, and it caught a
singular mass matrix in the first draft.

`RoastingMachine` is the one example whose reference is neither a closed form nor a
solver. It is a Loring S15 that roasted five batches of coffee on 2026-07-30, logged over
Modbus at 1.5 s; the model carries the first batch's own gas trace, and the spec checks it
against what the machine's bean probe actually read — 1.67 °C rms over the whole batch,
1.24 °C after the first two minutes. The machine parameters were fitted against all five
batches at once, each keeping its own gas, its own probe reading at charge, and its own
machine temperature at charge.

It is also the example that models its own instrument. One of the four states is the
thermocouple, which has heat capacity and therefore lags: at charge it has been hanging in
217 °C recirculating air and is then buried in 13.8 kg of 20 °C coffee, and the dive that
follows — the *turning point* every roaster reads — is a property of the probe as much as of
the beans. Adding that state is what lets the spec check the roast from `t = 0` rather than
skipping the first ninety seconds, and it prints a number nobody has otherwise: the coffee
runs about 8 K ahead of the display two minutes in.

The batch was also weighed, which is what makes the drying worth anything, and the weighing
overturned the physics rather than merely tuning it. Gating evaporation at boiling — water
leaves when water boils — cannot reproduce the measured 15 % mass loss at any rate: forced
to remove the right mass, the curve comes apart. Sweeping the onset downward improves the
fit and the mass balance together, monotonically, all the way to ambient, because a coffee
bean is porous and gives up bound water by diffusion the whole time it is warm. The spec
carries that table, and the reason it was nearly missed: with the machine parameters held
fixed no drying law works, since those parameters had already absorbed the missing heat.

## Adding one

1. Write the `.mo`. Keep it short and physically real; no synthetic equations.
2. Write the `.py` twin, same stem.
3. Work out the reference: closed form if one exists, otherwise scipy at `rtol=1e-12`.
   Say which in `source`. If the model is of something that exists and you can get a log
   off it, `measured` beats both — but carry a `reference` point too, because degrees of
   slack cannot catch a regression.
4. Write the `.toml`. Assert only what you are sure of.
5. `make test` — the corpus guard checks the set is well-formed, and the twin check runs
   immediately, even while the stages that would simulate it are still skipped.
