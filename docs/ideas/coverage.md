# Reachability and agreement

Every number the example corpus produces is a measurement of the corpus more than of the
toolchain: fifteen models, each written by the people who wrote the compiler, each chosen
to force exactly one capability. `moon run :coverage` produces the other kind of number —
over every example model in the [Modelica Standard Library][msl] at a pinned version, how
far does each one get through this toolchain, and why does it stop where it stops.

Two claims are measured separately, because they fail differently and a single "it works"
hides both:

* **Reachability** — the model parses, flattens, causalizes and simulates. This is a
  language-coverage question.
* **Agreement** — it simulates *and produces the same trajectory as a tool people trust*.
  This is a semantics question, and it is the one that can be quietly wrong.

There is a published precedent for saying this plainly. [Rumoca][rumoca] reports that
**28 % of MSL examples reach end-to-end simulation** under it, and that **about a third of
the models that simulate disagree with OpenModelica**. Those numbers are unflattering and
checkable, which is what makes them worth having — and worth copying as a habit. The
[landscape survey](landscape.md) has the longer argument.

## The rules

The denominator is the half of a fraction that is easiest to fudge, so its rule is stated
here: an *example model* is a non-`package.mo` file under the `Modelica` package whose
text carries an `experiment` annotation, named by its path —
`Modelica/Blocks/Examples/PID_Controller.mo` is `Modelica.Blocks.Examples.PID_Controller`.

* The MSL is fetched at the pinned version into `.cache/msl/`, never vendored.
* Every refusal is attributed: a model that does not reach the end records which stage
  refused it and why, and the report orders the refusals by how many models each one
  blocks. That work list is the real product — "arrays block N models" is a decision,
  a percentage alone is a headline.
* The reference for agreement is OpenModelica (`omc` on `PATH`), never a second run of
  this toolchain. When `omc` is absent the agreement number is reported as unmeasured,
  with the command that would measure it.
* Not a CI gate, deliberately. The number moves as language coverage moves, and gating on
  it would make a green pipeline a matter of taste.
* Measuring and improving are different jobs. The run changes nothing; `coverage.json`
  holds the per-model detail for whoever picks up the work list.

## The measurements, newest first

### 2026-09-10, after a boolean latch that reads its own pre value is isolated

An on-off controller reads its previous state: `on = pre(on) and u < hi or u < lo`. `pre(on)` is the
value at the last event, a known in the current phase, not a reference to the current `on` -- so `on`
stands alone on the left and its boolean right-hand side is the answer, read off without a derivative
(a boolean `or` has none). The check for whether the unknown stood alone counted the `pre(on)` as
`on`, so it took the linear-solve path instead and causalization fell over. It ignores a `pre` now, so
the isolated shape catches the equation, as it already does for an ordinary `hot = x > 0.5`.

Causalize climbs from 22 to 23 of 430 example models: MSL's `HeatTransfer...ControlledTemperature`,
whose `OnOffController` is this latch, gets past the fall-over (it meets an initial-equation wall
after). Reachable simulation holds at 21.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     50 | 11.6 % |
| causalized |     23 |  5.3 % |
| simulated  |     21 |  4.9 % |

The leaders are unchanged: a composite component with a value (70), the `Clock` builtin (62), an array
shape not known at flatten (52), a subscript on an inner component and a type alias to an array type in
parse (32 each), and a condition not known at flatten (31). Agreement is still unmeasured (no `omc`).

### 2026-09-10, after a package constant in a function body resolves

`Modelica.Units.Conversions.to_degC` reads `Modelica.Constants.T_zero` in its body and is inlined
wherever a model converts a temperature. The inlined body was prefixed but never took the rewrite that
turns a package constant into its value -- the rewrite an equation gets -- so the name survived as a
reference that flattening then could not resolve, and every model using the conversion failed with
"unknown name 'Modelica.Constants.T_zero'". The body is now resolved in the function's own scope, so a
package constant or an enumeration literal it reads becomes its value before prefixing sees it.

Flattening climbs from 47 to 50 of 430 example models: the electrical and thermal examples that convert
a temperature clear the wall. Reachable simulation holds at 21 -- the newly flattening models meet a
later wall, a derivative of a boolean `or` among them -- so causalize edges to 22 while simulate stays.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     50 | 11.6 % |
| causalized |     22 |  5.1 % |
| simulated  |     21 |  4.9 % |

The leaders are unchanged: a composite component with a value (70), the `Clock` builtin (62), an array
shape not known at flatten (52), a subscript on an inner component and a type alias to an array type in
parse (32 each), and a condition not known at flatten (31). Agreement is still unmeasured (no `omc`).

### 2026-09-10, after a homotopy written with named arguments resolves

`homotopy(actual = ..., simplified = ...)` is how the standard library's ideal op-amp writes the
operator, and every consumer of the call read the actual off the first positional argument -- so with
the arguments named and none positional, the differentiator indexed an empty tuple and the flatten
stage fell over with a bare "tuple index out of range", an internal error rather than a refusal. The
names are resolved to the positional `homotopy(actual, simplified)` at flatten now, once, so the
derivative, the emitter and the evaluator each read it the way they always did.

Reachable simulation climbs from 19 to 21 of 430 example models: `OpAmps.Multivibrator` and
`SignalGenerator`, which build on the ideal op-amp, run end to end. Causalize and simulate stay level
with each other at 21, every model that causalizes also simulates.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     47 | 10.9 % |
| causalized |     21 |  4.9 % |
| simulated  |     21 |  4.9 % |

The leaders are unchanged: a composite component with a value (70), the `Clock` builtin (62), an array
shape not known at flatten (52), a subscript on an inner component and a type alias to an array type in
parse (32 each), and a condition not known at flatten (31). Agreement is still unmeasured (no `omc`).

### 2026-09-10, after the multiple-import list parses

`import Modelica.Blocks.Types.{ExternalCombiTable1D, Extrapolation, Smoothness}` brings several names
from one package in a single clause, each the same as writing the qualified import three times. The
standard library's rotational components -- `Clutch`, `LossyGear` and their neighbours -- open with two
such clauses, and the dotted-name reader consumed the dot before the brace and then choked on it, so
those files did not parse and every model that depends on them stopped there. The clause is read a
segment at a time now, and a `{...}` list becomes one import per name, so nothing below the parser
meets a fourth form.

The refusal leaves the work list -- eleven models carried it -- but the stage counts hold at 47
flattened and 19 simulated: the rotational examples that now get past their dependencies meet the next
wall, an array whose shape is not known at flatten (`mu_pos`, `lossTable`), so the frontier moves
without the tally.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     47 | 10.9 % |
| causalized |     19 |  4.4 % |
| simulated  |     19 |  4.4 % |

The leaders are unchanged: a composite component with a value (70), the `Clock` builtin (62), an array
shape not known at flatten (52), a subscript on an inner component and a type alias to an array type in
parse (32 each), and a condition not known at flatten (31). Agreement is still unmeasured (no `omc`).

### 2026-09-10, after a qualified call to a builtin reads as the builtin

`Modelica.Math.sin(x)` is the builtin sin under its library path, and the standard library reaches the
maths functions this way where a call is not inlined. Both the ends of the toolchain that consume a
call knew `sin` but not the qualified name: the emitter refused it, "no translation for the function
'Modelica.Math.sin'", and the differentiator had "no derivative rule" for it. Each now reads a
qualified call as the builtin -- the last segment, and only when it is a name that stage already has,
so an unknown call still fails by its full name.

Reachable simulation climbs from 14 to 19 of 430 example models: the rotational and other examples
that call a maths function qualified were causalizing and then failing at emit, so this closes the gap
between causalize and simulate -- both now stand at 19, every model that causalizes also simulates.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     47 | 10.9 % |
| causalized |     19 |  4.4 % |
| simulated  |     19 |  4.4 % |

The leaders are unchanged, all before causalize: a composite component with a value (70), the `Clock`
builtin (62), an array shape not known at flatten (52), a subscript on an inner component and a type
alias to an array type in parse (32 each), and a condition not known at flatten (31). Agreement is
still unmeasured (no `omc`).

### 2026-09-10, after the same element inherited through two paths is merged

The other half of the "twice" refusal: a name that reaches a class through two `extends` paths.
MSL's battery `CellData` extends both a base record and an icon record, and each declares an identical
`constant String CellType`, so the name arrives twice -- the same element, which the tools merge, but
flattening refused it, "CellData inherits 'CellType' twice". The inherited-member loop in `effective`
now compares the two the same way the re-statement fix compares a re-declaration: identical (but for a
description or a placement) is the one element, kept once; a difference is still a clash. A
modification the second `extends` writes on the name still applies, on top of the member already
placed.

That empties the "twice" refusal from the work list -- 12 two measurements ago, 3 after the
re-statement fix, 0 now. The stage counts hold at 47 flattened and 14 simulated: the battery examples
that clear this wall meet the next one, an array whose shape is not known at flatten (`OCV_SOC[:,2]`),
so the frontier moves without the tally.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     47 | 10.9 % |
| causalized |     19 |  4.4 % |
| simulated  |     14 |  3.3 % |

The leaders are unchanged: a composite component with a value (70), the `Clock` builtin (62), an array
shape not known at flatten (52), a subscript on an inner component and a type alias to an array type in
parse (32 each), and a condition not known at flatten (31). Agreement is still unmeasured (no `omc`).

### 2026-09-10, after an identical re-statement of an inherited component is accepted

A derived class may declare a component its base already declares, identically: MSL's
`Electrical.QuasiStatic...Interfaces.TwoPin` declares `SI.AngularVelocity omega`, which its base
`TwoPinElementary` already has, and the power-converter `HBridge` does the same with `fire_p`. That is
the same variable written twice, not a name clash, and the tools accept it. Flattening refused it,
"declares 'omega' twice", because a component whose name matched an inherited member was rejected
outright. It now compares the two declarations: identical but for a description or a diagram placement
is the same element, kept once; a declaration that actually differs (a different type) is still a
clash.

The "declares twice" refusals fall from 12 to 3 -- the 3 that remain are the diamond case, a name
inherited through two `extends` paths (`inherits 'CellType' twice`), which is a separate merge. The
stage counts hold at 47 flattened and 14 simulated: the models that clear this wall (the quasi-static
resonance examples, the H-bridge drives) meet the next one right after -- a `connect` to a signal with
two sources, a subscript on an inner component in `Complex`, a condition not known at flatten -- so the
frontier moves without the tally moving.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     47 | 10.9 % |
| causalized |     19 |  4.4 % |
| simulated  |     14 |  3.3 % |

The leaders are unchanged: a composite component with a value (70), the `Clock` builtin (62), an array
shape not known at flatten (52), a subscript on an inner component and a type alias to an array type in
parse (32 each), and a condition not known at flatten (31). Agreement is still unmeasured (no `omc`).

### 2026-09-09, after an enumeration literal in an if-equation condition resolves

The condition of a structural `if`-equation was prefixed but not put through the rewrite that turns an
enumeration literal into its number. The op-amp and limiter blocks the analog library builds on write
`if homotopyType == Modelica.Blocks.Types.LimiterHomotopy.NoHomotopy then`, a kept structural `if`
(its guard `strict` defaults to true), and the literal on the right stayed a name that flattening then
could not resolve, so the model failed with "unknown name". A top-level equation and a `when`
condition already took the full rewrite; the if-equation condition was the one place that took only a
bare prefix. The rewrite is now one method, shared by all three, so the literal becomes its ordinal
before the condition is evaluated.

Flattening climbs from 31 to 47 of 430 example models, and reachable simulation from eleven to
fourteen: the op-amp examples (`OpAmps.Adder`, `Comparator`, and their neighbours) and the models that
instantiate a limiter clear the wall together. The "unknown name" line, which led the flatten work
list two measurements ago, is off the top of it entirely.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     47 | 10.9 % |
| causalized |     19 |  4.4 % |
| simulated  |     14 |  3.3 % |

The leaders are now a composite component with a value (70), the `Clock` builtin (62), an array shape
not known at flatten (52), a subscript on an inner component and a type alias to an array type in
parse (32 each), and a condition not known at flatten (31). Agreement is still unmeasured (no `omc`).

### 2026-09-09, after a package constant resolves unqualified and across packages

The `Modelica.Constants` values the library leans on stopped a run at flatten. Two forms were
missing. A **bare** constant, brought into a model by `import Modelica.Constants.pi;` or
`import Modelica.Constants.*;` or declared in the enclosing package, was read as a component that did
not exist and refused with "unknown name" (commit for 0.0.176). A **cross-package** constant, one
whose own value names a constant in another package -- `Modelica.Constants.eps =
ModelicaServices.Machine.eps` is the case that mattered, and it accounted for 36 of the models on the
"unknown name" line -- resolved its qualified reference no further than the package it named, so eps
never became its number (this change, 0.0.177). The evaluation now follows a qualified constant
reference into the package it names, bounded by a hop count so a library whose packages name each
other in a ring fails cleanly instead of looping; a same-package chain is still bounded by name.

Flattening climbs from 12 to 31 across the two fixes (17 after the bare form, 31 after the
cross-package one); reachable simulation moves from eight to eleven. Most of the fourteen models the
cross-package fix newly flattens stop a stage later -- an electrical example that reads `eps` also
carries a `connect` to a component with an unknown array shape -- so causalize and simulate move by
one each while flatten moves by fourteen. The "unknown name" line falls from 55 to 24; what remains
there is led by an enumeration literal of a library type used as a default (`LimiterHomotopy`,
sixteen models), which is a separate lookup.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     31 |  7.2 % |
| causalized |     16 |  3.7 % |
| simulated  |     11 |  2.6 % |

The work list's leaders are now all flatten walls that are not about names: a composite component
with a value (70), the `Clock` builtin (62), an array shape not known at flatten (52), and a
condition not known at flatten (31). Agreement is still unmeasured (no `omc` on this machine).

### 2026-09-09, after an `if` inside a `when` body parses

A `when` body is a run of statements and may hold an `if` or a `for`; `Blocks.Math`'s extremum
blocks latch `if u < pre(y_min) then t_min = time; else ...` inside a `when`, and the standard
library's samplers do the same. The when-body parser read only simple assignments, so the
compound form was refused at parse. It reads the algorithm statement forms now, and a plain
assignment body is unchanged.

The stage counts do not move (parsed holds 415 / 430): the files that carry this construct meet
another one right after (`Blocks.Math` an array-valued modification, the Clocked samplers the
`Clock` builtin), and flattening refuses an `if` in a `when` by name -- a conditional in the
discrete-event pipeline is separate work. What moved is the parse frontier: the `if`-in-`when`
refusal is gone from the work list, and `IfStatement inside a 'when' is not supported yet` is now
the honest message, at flatten, for the models that reach it.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     12 |  2.8 % |
| causalized |     10 |  2.3 % |
| simulated  |      8 |  1.9 % |

The largest blockers are unchanged: a composite component with a value (66), an array shape not
known at flatten (52), unqualified constants (50), the `Clock` builtin (49), and, in parse, a type
alias to an array type and a subscript on an inner component. Agreement is still unmeasured.

### 2026-09-09, after quoted identifiers and the bare operator class

Two constructs the operator-overloading half of the library needs, both on `Complex.mo` and the
files that import it. A **quoted identifier**, `'+'` or `'a b'`, is a name spelled with an operator
or a space; the lexer reads it now, keeping the quotes so it is its own name and round-trips. A
**bare operator class**, `operator '+' ... end '+'`, is an overload group named by the operator it
overloads; it reads as a package, a namespace it is never instantiated from, so the file loads. The
overloading itself is still refused where an operator is applied.

The `unexpected character '` refusal is gone from the work list, and one more example model parses
end to end (parsed 414 -> 415). The `Complex.mo` chain does not clear yet: it meets its third
construct, a subscript on an inner component (`c1[i].re`), which the 18 `ComplexBlocks` models now
share as their blocker.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    415 | 96.5 % |
| flattened  |     12 |  2.8 % |
| causalized |     10 |  2.3 % |
| simulated  |      8 |  1.9 % |

`a subscript on an inner component` (30) has become the largest parse blocker, ahead of a type
alias to an array type (`type QuaternionBase = Real[4]`, part of the 32 `[` refusals). The flatten
walls are unchanged: a composite component with a value (66), an array shape not known at flatten
(52), unqualified constants (50). Agreement is still unmeasured (no `omc`).

### 2026-09-09, after an array dimension on the type

`FixedHeatFlow[n] heat` puts the array dimension on the type rather than the name, which is the
same declaration as `FixedHeatFlow heat[n]` and the Fluid examples write it that way. The clause
reads a dimension after the type name now, shared by every name in it; a name's own dimensions
come first, so `Real[3] x[2]` is `x[2, 3]`, per the specification. The flattener already expands an
array of components, so nothing below the parser changed.

`Modelica.Fluid.Examples.BranchingDynamicPipes` and its neighbours parse now (parsed 412 -> 414).
Reachability holds at eight. Two of the three `[` refusals remain and are separate constructs: a
type alias to an array type (`type QuaternionBase = Real[4]`) and a subscripted assignment target
in an algorithm (`a[1] := 1`).

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    414 | 96.3 % |
| flattened  |     12 |  2.8 % |
| causalized |     10 |  2.3 % |
| simulated  |      8 |  1.9 % |

The flatten walls that lead the work list are unchanged: a composite component that carries a value
(66), an array shape not known at flatten time (52), and unqualified constants in an equation
(`pi`, `eps`, 50). Agreement is still unmeasured (no `omc`).

### 2026-09-09, after `pure`/`impure` and external functions parse

Two function-declaration constructs, both on the utility libraries, both cleared from the work
list. `pure function` and `impure function` carry a purity hint the tool does not act on, so the
prefix is accepted and dropped. An `external "C" [ref =] call [annotation];` body is the whole of
a function implemented in another language; it cannot run here, so it is read and dropped, and a
function that reaches the simulator this way is refused where a call would inline it (it has no
algorithm), not at parse.

`Modelica.Utilities` (Streams, System, Internal), `Modelica.Blocks.Tables` and
`Modelica.Math.Random` are written this way and imported widely, so refusing them blocked a slice
of the library. Six more files parse (2526 files, 105 → 99 refused), and one more example model
parses end to end (parsed 411 → 412). The models behind the two constructs advance to the same
flatten walls the electrical and machine libraries already lead with, so reachability holds at
eight:

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    412 | 95.8 % |
| flattened  |     12 |  2.8 % |
| causalized |     10 |  2.3 % |
| simulated  |      8 |  1.9 % |

The work list is now flatten work almost top to bottom: a composite component that carries a value
(66), an array shape not known at flatten time (52), an unqualified name in an equation (`pi`,
`eps`, from the Constants the utility files pull in, 50 + rarer), and the `Clock` builtin (49).
`Strings.mo`, `Random.mo` and `Files.mo` still trip on a `[` construct (an array-valued
modification) and so did not clear. Agreement is still unmeasured (no `omc`).

### 2026-09-09, after a `when` tuple equation reads `=`

**Eight models simulate now, up from four.** A `when` in an equation section takes equations,
and a multi-output call is an equation written `(a, b) = f(x)`. The parser reads a `when` body as
statements and its tuple case required `:=`, so the equation spelling was refused. The one file
that trips on it is `Modelica.Blocks.Interfaces` (its `PartialNoise` draws a random pair that
way), and that file is imported across the library, so refusing it blocked well over a hundred
example models at the point they looked it up. Either operator is accepted now.

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    411 | 95.6 % |
| flattened  |     12 |  2.8 % |
| causalized |     10 |  2.3 % |
| simulated  |      8 |  1.9 % |

The four new reachable models are Mechanics again -- `Rotational.Examples.ElasticBearing` and
`Translational.Examples.{Accelerate, SignConvention, WhyArrows}` -- which use Blocks sources and
sensors and so needed that file to load. The parse frontier has moved off the Blocks `:=`/`=`;
what leads the work list now is genuine flatten work: a composite component that carries a value
(66) and array shapes not known at flatten time (52). Agreement is still unmeasured (no `omc`).

### 2026-09-09, after `StateSelect` -- the first models reach simulation

**Reachability moved off zero.** Four example models parse, flatten, causalize and simulate
end to end, the first that have:

* `Modelica.Mechanics.Rotational.Examples.RollingWheel`
* `Modelica.Mechanics.Translational.Examples.Damper`
* `Modelica.Mechanics.Translational.Examples.ElastoGap`
* `Modelica.Mechanics.Translational.Examples.InitialConditions`

`StateSelect` is a predefined enumeration the language provides but no file declares, a hint on
which variables a tool should prefer as states. The standard library carries it two ways, and
both blocked models: as an attribute, `Real x(stateSelect = StateSelect.prefer)`, and as a
parameter type, `parameter StateSelect ss = StateSelect.default`. The attribute joins the
accepted set (the hint has no effect here, state selection is not implemented), and the
enumeration is injected into the flattener as a prelude, so its name resolves and its members
map to 1..5 like any other enumeration's. That was the last thing between these four Mechanics
models and a trajectory: they set `stateSelect` on their flange states.

| models | stage | refusal |
| -----: | ----- | ------- |
|    139 | parse | `:=` versus `=` inside a `when` statement (the Clocked library) |
|     59 | flatten | a composite component cannot carry a value |
|     50 | flatten | an array shape not known at flatten time |
|     32 | parse | `expected ';', found '['` (an array-valued modification) |
|     19 | parse | `impure` on a function |
|     17 | parse | a single-quoted identifier (`'+'`, operator function names) |
|    ~110 | parse/flatten | the rest; `coverage.json` has every one |

Agreement is still unmeasured (no `omc` on this machine), so what these four produce has not yet
been checked against a reference. Reaching a trajectory is the first claim; agreeing with a tool
people trust is the second, and it is the one worth measuring next. The parse frontier is now
the Clocked library's `:=`/`=`, and the flatten frontier is genuine work: composite-component
bindings and array equations.

### 2026-09-09, after the `quantity` attribute is accepted

`quantity` is a descriptive type attribute, like `unit`: `type Voltage = Real(quantity=
"ElectricPotential", unit="V")`. The whole `Modelica.Units.SI` type set carries it, so the
flattener refusing it as unknown blocked the electrical and machine libraries wholesale. It is
accepted and ignored now, the way `unit` and `displayUnit` already were: one name added to the
attribute set.

This was the dominant flatten wall the operator-record change had just exposed (~230 models on
one attribute). It is gone from the work list, and the models behind it advanced deep into
flattening, so the frontier is no longer one wall but many smaller ones, spread across the
libraries that now load:

| models | stage | refusal |
| -----: | ----- | ------- |
|    119 | parse | `:=` versus `=` inside a `when` statement (the Clocked library) |
|     52 | flatten | a composite component cannot carry a value |
|     48 | flatten | an array shape that is not known at flatten time |
|     31 | parse | `expected ';', found '['` (an array-valued modification) |
|     29 | flatten | the `StateSelect` enumeration (`nothing named 'StateSelect'`) |
|     19 | parse | `impure` on a function |
|     14 | flatten | the `stateSelect` attribute (the sibling of `quantity`, next) |
|   ~100 | parse/flatten | the rest; `coverage.json` has every one |

Reachability is still 0 %, because the models now reach real semantic gaps rather than a single
descriptive attribute. The next cheap win is the pair the electrical library keeps hitting:
`stateSelect`, another accept-and-ignore attribute, and the `StateSelect` enumeration behind its
value. After that the walls are genuine work: array equations and composite-component bindings.

### 2026-09-09, after the `operator` class prefix parses

`operator record` and `operator function` parse now, each as its plain kind (a record, a
function), carried through and refused where the overloading would be applied. This was the
top of the previous work list: `Modelica.Units` defines dozens of `operator record` aliases of
`Complex`, its file would not parse, and every model that imports `Modelica.Units.SI` (nearly
all of them) needs that file. The file parses now, and one fewer file refuses across the
library (107 to 106).

The stage line did not move (parsed 411 / 430, reachability 0 %), because those 311 models were
already counted as parsed: they reached parse on their own files and then died at flatten
looking up a name in the unparseable `Units.mo`, attributed back to it. What moved is the work
list, which is the half that matters. **Operator overloading is gone from it.** The 311 now
flatten `Modelica.Units.SI` and get much further, to a new and now dominant wall: the
`quantity` attribute.

| models | stage | refusal |
| -----: | ----- | ------- |
|   ~230 | flatten | `unknown attribute(s) ['quantity']` on a `Real` subtype (summed over its many names: `Vrms`, `VNominal`, `f`, ...) |
|     48 | parse | `:=` versus `=` inside a `when` statement |
|     19 | parse | `impure` on a function |
|     10 | flatten | the `Clock` builtin (`nothing named 'Clock'`) |
|      7 | flatten | a medium's composite component cannot carry a value |
|      6 | parse | a subscript on an `inner` component |
|    ~40 | parse/flatten | the rest; `coverage.json` has every one |

The `quantity` attribute is descriptive, like `unit`: `type Voltage = Real(quantity=
"ElectricPotential", unit="V")`. The flattener rejects the unknown attribute rather than
carrying it. Accepting (and ignoring) `quantity` the way `unit` is already ignored is the next
construct worth building, and it is now what stands between the bulk of the electrical and
machine libraries and their first flatten.

### 2026-09-09, first run on macOS, after the array and builtin work

The harness ran on this platform for the first time. `RLIMIT_AS` is a Linux guard that macOS
refuses (`current limit exceeds maximum limit`), and the bare `setrlimit` was aborting the
whole run before it started. The request is clamped to the current hard limit now, and a
platform that refuses it outright runs without the guard rather than not running at all, with
a note saying so.

The measurement, against MSL v4.1.0 with `modelica` 0.0.163: parsed **411 / 430 (95.6 %)**,
one past the 2026-09-01 row. Reachability is still 0 %, and the work list keeps its shape:
one construct stands between the library and its first flatten, and it is the same one.

| models | stage | refusal |
| -----: | ----- | ------- |
|    311 | parse | operator overloading (`operator record`, reached through `import Modelica.Units.SI`) |
|     48 | parse | `:=` versus `=` inside a `when` statement |
|     19 | parse | `impure` on a function |
|     10 | flatten | the `Clock` builtin (`nothing named 'Clock'`) |
|      7 | flatten | a medium's composite component cannot carry a value |
|      6 | parse | a subscript on an `inner` component |
|     29 | parse/flatten | twelve rarer refusals; `coverage.json` has every one |

Operator overloading is three-quarters of the library on one construct, and it is a
prerequisite, not a leaf: `Modelica.Units.SI` defines its unit types with `operator record`,
and almost every model imports it. Until that parses, the 311 cannot advance to show what
they would fail on next, so reachability cannot move off 0 % no matter what else lands. That
is the next construct worth building for this number: the array, reduction and builtin work
since 2026-09-01, which the fifteen-model example corpus exercises, does not touch it.

### 2026-09-01, after the base prefix on short classes

`connector RealInput = input Real` — the MSL's most-used connector. The causality is
carried on the class (it can only come from a short definition, so the class round-trips
through the short form), and flattening applies it to every instance whose declaration
says nothing itself. The 58-model row cleared; ten Clocked models now fail in *flatten*,
resolving the `Clock` builtin — the first models to get that far. Parsed holds at
410 / 430 (95.3 %); a `:=`-versus-`=` parse point (48) surfaced behind it.

### 2026-09-01, after bare call statements

`isValidTable(table);` — the statement twin of the bare call equation, with the same
empty output list. Parsed 409 → 410 (95.3 %). The work list is three rows now: operator
overloading (311), `input` on a short class definition (58 — the
`connector RealInput = input Real` family), and `impure` (19).

### 2026-09-01, after comprehensions, in both spellings

`{x[i] for i in 1:n}` and the reduction call `sum(x[i] for i in 1:n)` are one
`Comprehension` node; the printer always writes the braces, which parses back to the same
value. Carried like `Array`, refused beside it at the point of use. **Parsed 406 → 409
(95.1 %).** The tail rows keep trading places; operator overloading (311) is now
three-quarters of everything left.

### 2026-09-01, after ranges become expressions

`parameter Integer columns[:] = 2:size(table, 2)` — a range as a binding value. The range
tail now reads in `expression()` itself, where the specification's `simple_expression`
puts it, and the `for` header stopped being special. Parsed 405 → 406 (94.4 %); the
Tables/Sources chain advanced to array comprehensions (37, next).

### 2026-09-01, after `when` in a statement position

The when-statement reuses the when-equation's node — branches of statements either way —
and the *section* decides how an assignment inside it prints, extending the existing
`:=`/`=` note. Bodies are full algorithm statements. The fold refuses it by name. The
Sources/Tables chain advanced again, to `expected ';', found ':'` (32). Parsed holds at
405 / 430 (94.2 %).

### 2026-09-01, after the statement twins

`assert(...)` as a statement and `(a, b) := f(x)` — the statement twins of the equation
forms that already existed — parse, round-trip, and are refused by name where a model's
section folds into equations. The Blocks.Sources chain advanced one more construct, to
`when` inside an algorithm section (31 models, next). Parsed holds at 405 / 430 (94.2 %).

### 2026-09-01, after `initial algorithm` folds

`initial algorithm` is to `initial equation` what `algorithm` is to `equation`, and it
now parses into `Class.initial_algorithms` and folds into the initial equations during
flattening. The 32 models blocked on the section header advanced one construct deeper
(their chain now meets `expected '=', found '('`, 34 models — next). Parsed holds at
405 / 430 (94.2 %): these were dependency cascades, and the frontier inside them moved.

### 2026-09-01, after slices read where the grammar expects them

`x[1:3]` and `A[:, 1]` parse: a range was already readable — but only in the header of a
`for`, and a subscript is the other place the grammar expects one. The bare `:` gets its
own node (`Colon`), because it can only ever be the whole subscript. Both are carried the
way `Array` is and refused beside it at the point of use. **Parsed rose 359 → 405
(94.2 %).** The remaining list is operator overloading (305 — now 71 % of the library on
one construct) plus the function-declaration rows (`algorithm` 32, `input` 25, `impure`
19) that are its own prerequisites, and tails of a few models each.

### 2026-09-01, after `final` on a component clause

`final parameter Real THD1 = V3/V1` — the element prefix, accepted and discarded the way
the parser's four other `final` sites already do: finality forbids later modification,
and nothing downstream modifies further yet. One branch in the prefix loop; **parsed
345 → 359 (83.5 %)**. The remaining work list now reads as one story: the function
declaration chain (`algorithm` sections 32, `input` 25, `impure` 19) feeding into
operator overloading (258), plus array slices (51 + 9).

### 2026-09-01, after connector-array elements wire

`connect(plug.pin[2], x)` is implemented, not merely parsed: the subscript is structural
the way a dimension is, evaluated during flattening, and the wire lands on the element
the array expansion already declared. Connecting a whole array at once, a subscript
anywhere but the last name, and a `connect` inside a `for` loop stay refused by name.
**Parsed rose 314 → 345 (80.2 %).** The frontier is now two constructs: operator
overloading (240) and whatever `expected name, found keyword 'final'` turns out to be
(90, suddenly visible now that `Modelica.Blocks.Tables` loads this far).

| models | stage | refusal |
| -----: | ----- | ------- |
|    240 | parse | operator overloading |
|     90 | parse | `expected name, found keyword 'final'` |
|     47 | parse | array slices (`[:]`) |
|     10 | parse | `expected name, found keyword 'input'` |
|      7 | flatten | a composite component cannot have a value (media bindings) |
|      7 | parse | `expected an expression, found operator ':'` |
|     29 | parse/flatten | rarer refusals; `coverage.json` has every model |

### 2026-09-01, after bare call equations parse

`Connections.root(a.b);` is a function call as an equation — specification 8.3.1 with an
empty output list. It now parses into `CallEquation` with no targets, and flattening
refuses the form by name; the connection-graph semantics behind `Connections.root` and
`Connections.branch` are a separate job. The QuasiStatic interface files load, and
**parsed jumped 277 → 314 (73.0 %)**. Reachability still 0 %; operator overloading now
blocks over half the library on its own.

| models | stage | refusal |
| -----: | ----- | ------- |
|    226 | parse | operator overloading |
|     83 | parse | arrays of connectors |
|     44 | parse | array slices (`[:]`) |
|     35 | parse | `expected name, found keyword 'final'` |
|      7 | flatten | a composite component cannot have a value (media bindings) |
|     35 | parse | four rarer refusals; `coverage.json` has every model |

### 2026-09-01, after `assert` equations carry

`assert(condition, message);` in an equation section parses and flattens into a new
`FlatModel.asserts` — prefixed like every equation, never counted by causalization,
because an assert binds no unknown. The simulator does not evaluate asserts yet; the node
records that instead of the model losing its diagnostics. `Modelica.Electrical.Analog.
Basic.Resistor` and its neighbours load now; the `expected '=', found ';'` row shrank
48 → 44, and what remains under that message is a different construct
(`Modelica.Electrical.QuasiStatic` interfaces — likely the overconstrained-connector
operators `Connections.branch/root`). Parsed holds at 277 / 430; the cleared models
redistribute (operator overloading 190 → 193).

### 2026-09-01, after matrix expressions parse

`[1, 2; 3, 4]` becomes an `ir.Matrix`, following `ir.Array`'s design exactly: it survives
parsing and flattening so the file that holds it loads, and evaluate, derive and emit
refuse it by name at the point of use. One matrix in one file had been keeping whole
packages from loading; with those files in, **parsed rose 262 → 277 (64.4 %)**.
Reachability still 0 %.

| models | stage | refusal |
| -----: | ----- | ------- |
|    190 | parse | operator overloading |
|     80 | parse | arrays of connectors |
|     48 | parse | `expected '=', found operator ';'` |
|     44 | parse | array slices (`[:]`) |
|     32 | parse | `expected name, found keyword 'final'` |
|      8 | flatten | a composite component cannot have a value (media bindings) |
|     28 | parse | six rarer refusals; `coverage.json` has every model |

### 2026-09-01, after conditional components work

`Sensor s if useSensor;` is implemented, not merely accepted: the condition is evaluated
during flattening — structural the way a dimension is — and a false one removes the
component together with every `connect()` touching it. The 53 models this construct
blocked moved to their next refusals; one now parses end-to-end (262 / 430, 60.9 %).
Reachability still 0 %.

| models | stage | refusal |
| -----: | ----- | ------- |
|    183 | parse | operator overloading (`operator record`, mostly via `Modelica.Units`) |
|     72 | parse | arrays of connectors |
|     66 | parse | matrix expressions |
|     41 | parse | array slices (`[:]`) |
|     32 | parse | `expected name, found keyword 'final'` |
|      8 | flatten | a composite component cannot have a value (media bindings) |
|     28 | parse | ten rarer refusals; `coverage.json` has every model |

### 2026-08-31, after `expandable connector` parses

Same day, one construct later: the parser now reads `expandable connector` (instantiating
one is still refused in flattening, where it would mean something). Reachability did not
move — that is the expected shape of this curve for a while — but the *work list* did,
which is the half that matters. The 258 models that all stopped at `Modelica/Icons.mo:629`
spread out to their next blockers:

| models | stage | refusal |
| -----: | ----- | ------- |
|    173 | parse | operator overloading — `operator record`, mostly reached through `import Modelica.Units.SI` (`Units.mo:1090`) |
|     72 | parse | arrays of connectors |
|     53 | parse | `expected ';', found keyword 'if'` — conditional components |
|     40 | parse | `expected ']', found operator ':'` — array slices |
|     32 | parse | `expected name, found keyword 'final'` |
|     24 | parse | matrix expressions |
|      8 | flatten | a composite component cannot have a value (media bindings) |
|     28 | parse | eleven rarer refusals; `coverage.json` has every model |

The stage table is unchanged from the first measurement below (parsed counts a model's
own file and ancestor packages, and those did not move). The harness also learned to
trace one more failure shape back to its cause: an `import` that does not resolve
because the imported file refused.

### 2026-08-31, first measurement

Against **MSL v4.1.0**, with `modelica` 0.0.96. The library is
2157 `.mo` files; 484 of them refuse to parse. Of the **430 example models**:

| stage      | models | of 430 |
| ---------- | -----: | -----: |
| parsed     |    261 | 60.7 % |
| flattened  |      0 |  0.0 % |
| causalized |      0 |  0.0 % |
| simulated  |      0 |  0.0 % |

**Reachability is 0 %** (Rumoca reports 28 % on the same library), and agreement is
therefore vacuous: nothing reached the reference. "Parsed" counts a model whose own file
and ancestor packages parse; every one of the 261 then dies looking up a dependency whose
file refused, which the harness traces back to the construct that refused it.

The work list, first refusal per model — kept as taken; a measurement keeps its date:

| models | stage | refusal |
| -----: | ----- | ------- |
|    258 | parse | `expected name, found keyword 'expandable'` — one `expandable connector` at `Modelica/Icons.mo:629`, imported by nearly everything |
|     44 | parse | `expected ';', found keyword 'if'` — conditional components (`... if condition`) |
|     40 | parse | `expected ']', found operator ':'` — array slices |
|     34 | parse | arrays of connectors |
|     24 | parse | matrix expressions |
|     17 | parse | `expected name, found keyword 'final'` |
|     13 | parse | five rarer refusals; `coverage.json` has every model |

One caution against optimism: the list shows each model's *first* refusal only. Parsing
`expandable connector` would not make 258 models simulate — it would re-run the
measurement and let each of them advance to its next blocker. That is what the number is
for: rerun after every language-coverage change, and publish the new table with its date.

[msl]: https://github.com/modelica/ModelicaStandardLibrary
[rumoca]: https://arxiv.org/abs/2606.14998
