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

# A value at a point in time. `source` records where the number came from:
#   "analytic"  — closed form, written out in a comment
#   "reference" — scipy at rtol/atol 1e-12, i.e. a golden trajectory
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
| `Pendulum` | a constraint equation → index 3 → Pantelides + dummy derivatives | scipy |
| `Robertson` | stiffness: a stiff solver is mandatory, explicit RK dies | scipy |
| `VanDerPol` | a relaxation limit cycle; step-size control across 3 orders of magnitude | scipy |

`Robertson` and `VanDerPol` are not physical-modelling examples — they are the classic
Hairer–Wanner stiff test problems, here to hold the solver honest.

## Adding one

1. Write the `.mo`. Keep it short and physically real; no synthetic equations.
2. Write the `.py` twin, same stem.
3. Work out the reference: closed form if one exists, otherwise scipy at `rtol=1e-12`.
   Say which in `source`.
4. Write the `.toml`. Assert only what you are sure of.
5. `make test` — the corpus guard checks the set is well-formed, and the twin check runs
   immediately, even while the stages that would simulate it are still skipped.
