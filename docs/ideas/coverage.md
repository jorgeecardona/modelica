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

## The measurement

Taken **2026-08-31**, against **MSL v4.1.0**, with `modelica` 0.0.96. The library is
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

The work list, first refusal per model:

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
