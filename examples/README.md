# Examples — the behaviour spec

Every model here is a **specification**, and not a demonstration. A Modelica model already
says what the system does. Put a small file of expected behaviour beside it, and the pair
becomes an acceptance test that no layer of the toolchain can quietly break.

```
examples/
  library/*.mo     small component packages: Electrical, Translational, Thermal,
                   Rotational, Machines, Blocks
  library/*.py     the same packages through the Python front end
  models/*.mo      one self-contained example model each
  models/*.py      the same model through the Python front end
  models/*.toml    what that model must do — the spec
```

`tests/test_examples.py` reads the corpus and runs it. A stage that is not implemented yet
reports as skipped and not as failed. The suite is therefore a live progress meter: as
`lang`, `ir` and `sim` land, skips turn into passes, and no spec is edited.

## Written twice, on purpose

Every example exists in both front ends, under the same stem. `RCCircuit.mo` and
`RCCircuit.py` are the same model, and the suite asserts that they say the same thing. That
pair is the only thing that keeps `modelica.build` and `modelica.lang` from drifting apart.

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

Four conventions apply.

* A **library** twin defines its classes and ends with
  `PACKAGE = package("Electrical", __name__, "...")`. The name is `PACKAGE` and not
  `Electrical`, which avoids an `Electrical.Electrical` on import.
* A **model** twin is one class named after the file, so `getattr(module, path.stem)` finds
  it.
* The examples are not an installed package. A model that uses a library adds the sibling
  directory to `sys.path` at the top of the file. `import Electrical` then reads exactly
  like the Modelica `import`.
* Run a twin directly to print its Modelica: `python examples/models/RCCircuit.py`.

Today the comparison is on normalised text. Blank lines are dropped, and numeric literals
go through `repr(float(...))` so that `1e4` and `10000.0` agree. Once the comparison of
flat models as values replaces it, the check becomes the one actually worth having.

## The spec format

TOML, one file per model, named after it. **An absent key asserts nothing.** Write only
what you are willing to commit to.

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

# A value at a point in time. `source` says where the number came from:
#   "analytic"  — a closed form, written out in a comment
#   "reference" — scipy at rtol/atol 1e-12, which is a golden trajectory
#   "measured"  — read off real hardware; see `RoastingMachine`
[[simulate.point]]
t      = 1.0
values = { "c.v" = 0.632120558829 }
rtol   = 1e-6
source = "analytic"

# Something that must hold at every output point: an expression, and either a value it
# equals or bounds it stays within. `after` restricts the check to t >= after.
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

A model of a *mechanism* has a picture that a line plot cannot show. For those the spec
carries an optional `[animate]` section: bodies at their coordinates, rods and springs
between them, the path a bob sweeps, and a panel underneath for the one scalar worth
watching. The coordinates are Modelica expressions over the model's own variables, on the
same `parse_expression` and `emit_expr` road the invariants take, so `x2` means the model's
`x2`.

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

`python examples/animate.py <Model>` writes `<Model>.mp4`. `--out x.gif` writes a GIF
instead. `--still 24.0` writes one frame at that instant, which is what you want while you
write a scene. It needs `matplotlib`, which is in the `docs` extra, and for `.mp4` it needs
ffmpeg. The library itself will never depend on either.

Two things are easy to get wrong, and both now default the right way. Leave `xlim` and
`ylim` out unless you have a reason: the frame is computed from everything the run visits,
and a guess clips a chaotic path. And fix `ylim` on a panel that shows a *conserved*
quantity: left alone, the axis zooms into the last digit of the integrator, and a line that
is flat to one part in 10⁹ is drawn as violent noise. That is the opposite of the claim.

## Seeing one

Every model in the corpus draws. A picture is often the fastest way to check that a model
says what you meant.

```bash
python -m modelica.diagram examples/models/RLCCircuit.mo examples/library/Electrical.mo > rlc.svg
python -m modelica.diagram examples/models/DCMotor.mo examples/library/Electrical.mo \
    examples/library/Rotational.mo examples/library/Machines.mo --depth 2 > motor.svg
```

`--view incidence` draws the equations against the variables they contain, and
`--view blocks` draws the order the solver runs in. `tests/test_diagram.py` draws every
model in the corpus on every run, and it asserts that no box names a component the flat
model does not have.

## The models

They are ordered the way the toolchain was built. Each one needs everything that the ones
above it needed, plus exactly one new capability.

| Model | The new capability it forces | Reference |
| --- | --- | --- |
| `NewtonCooling` | `der()`, parameters, one state, one equation | analytic |
| `RCCircuit` | `connect()`, `flow`, `extends`, a component library | analytic |
| `TwoBodyHeat` | connectors in a second domain, and a conservation invariant | analytic |
| `RLCCircuit` | two states, and a coupled linear system | scipy |
| `MassSpringDamper` | the same physics through the connectors of another domain | scipy |
| `DrainingTank` | `if` in an equation, which makes a state event; `sqrt` | analytic |
| `BouncingBall` | `when`, `reinit` and `pre`: repeated events, and a state set again | analytic |
| `RoastingMachine` | `time` in an equation, which makes time events *and* a state event; a parameter bound to other parameters; a model of the sensor | **a real machine** |
| `Pendulum` | a constraint equation, which makes index 3: Pantelides and dummy derivatives | scipy |
| `AtwoodPendulum` | five coupled degrees of freedom, which leave a 4×4 implicit block | scipy and energy |
| `Robertson` | stiffness: a stiff solver is mandatory, and explicit Runge-Kutta dies | scipy |
| `VanDerPol` | a relaxation limit cycle: step size control over three orders of magnitude | scipy |
| `PIControl` | `input` and `output` connectors: a signal has a direction, and a loop | scipy |
| `DCMotor` | two domains at once, and a component that holds components | scipy |

`Robertson` and `VanDerPol` are not physical-modelling examples. They are the classic
Hairer-Wanner stiff test problems, and they are here to hold the solver honest.

`AtwoodPendulum` is the one whose reference nobody can read by eye. The trajectory is
chaotic, so a golden point five seconds in says something about the arithmetic of this
toolchain and nothing about the physics. What checks the *model* is its own total energy,
written out as a variable and asserted flat. A computer algebra system produced the
equations and nobody can proofread them by reading. The invariant is what proofreads them,
and it caught a singular mass matrix in the first draft.

`PIControl` and `DCMotor` are the two newest. Each one was added to make the toolchain do
something nobody had asked it to do.

`PIControl` is a block diagram and not a physical network. Its connectors carry an `input
Real` or an `output Real` instead of a potential and a flow, so a connection is directed:
it says which end computes the number. Three things came out of adding it. The flattener
compared connectors by class name, which made `connect(source.y, error.u1)` illegal;
Modelica compares them by the variables they carry, and an output joined to an input is the
ordinary case. The flattener gained the opposite check at the same time, because two
outputs on one wire really is an error and the message should say which two. And the corpus
harness turned out to read the `after` key on an invariant, exactly as this page documents
it, and then ignore it. "The loop settles on the setpoint" was quietly being asserted of
the transient as well.

`DCMotor` is the first example where a component holds components. `Machines.Motor` has a
winding, a coil, the converter and the rotor inside it, and three connectors on its
boundary that are wired from inside and from outside. It is also the first model that lives
in two domains at once. `EMF` joins them: it turns current into torque and speed into
voltage, so it carries one connector of each kind. It found a bug that nothing before it
could have found. A modification written on an inner component, `winding(R = Rw)`, was not
given the path of the enclosing instance, so the binding named a variable that did not
exist. At the top level that prefix is empty and nothing moves, which is exactly why every
earlier example passed. Adding it also moved `ref()` into `modelica.build`, where the
second caller proved that it belonged.

`RoastingMachine` is the one example whose reference is neither a closed form nor a solver.
It is a Loring S15 that roasted five batches of coffee on 2026-07-30, logged over Modbus
every 1.5 s. The model carries the gas trace of the first batch, and the spec checks it
against what the bean probe of the machine actually read: 1.67 °C rms over the whole batch,
and 1.24 °C after the first two minutes. The machine parameters were fitted against all
five batches at once. Each batch kept its own gas, its own probe reading at charge, and its
own machine temperature at charge.

It is also the example that models its own instrument. One of the four states is the
thermocouple. The thermocouple has heat capacity, so it lags. At charge it has been hanging
in 217 °C recirculating air, and it is then buried in 13.8 kg of coffee at 20 °C. The dive
that follows is the *turning point* that every roaster reads, and it is a property of the
probe as much as of the beans. That state is what lets the spec check the roast from `t =
0` instead of skipping the first ninety seconds. It also prints a number that nobody has
otherwise: two minutes in, the coffee runs about 8 K ahead of the display.

The batch was also weighed, which is what makes the drying model worth anything. The
weighing overturned the physics rather than tuning it. Gate the evaporation at boiling —
water leaves when water boils — and no rate reproduces the measured 15 % mass loss: force
it to remove the right mass, and the curve comes apart. Sweep the onset downward and the
fit and the mass balance improve together, monotonically, all the way to ambient. A coffee
bean is porous, and it gives up bound water by diffusion for as long as it is warm. The
spec carries that table, and it carries the reason the finding was nearly missed: with the
machine parameters held fixed, no drying law works, because those parameters had already
absorbed the missing heat.

## Adding one

1. Write the `.mo`. Keep it short and physically real. Write no synthetic equations.
2. Write the `.py` twin, with the same stem.
3. Work out the reference. Use a closed form if one exists, and otherwise scipy at
   `rtol=1e-12`. Say which one in `source`. If the model is of something that exists, and
   you can get a log off it, then `measured` beats both. Carry a `reference` point as well,
   because degrees of slack cannot catch a regression.
4. Write the `.toml`. Assert only what you are sure of.
5. Run `make test`. The corpus guard checks that the set is well formed, and the twin check
   runs at once, even while the stages that would simulate it are still skipped.
6. Draw it: `python -m modelica.diagram examples/models/<Model>.mo <the libraries it uses>`.
   A wrong connection is easier to see than to read.
