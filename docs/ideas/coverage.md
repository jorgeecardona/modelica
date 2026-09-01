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
