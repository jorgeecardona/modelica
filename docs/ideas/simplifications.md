# Simplifying Modelica

!!! note "A living list"
    These are candidate simplifications of the Modelica language, collected here because this
    project is an unusually good place to have the argument. It is a frontend written from
    scratch, with a [coverage harness](coverage.md) that drives every example model in the
    Modelica Standard Library through parse, flatten, causalize and simulate and attributes
    every refusal. So the cost of a construct is not a matter of taste here: it is a count of
    how many real models it blocks and how much of the parser it took to accept. An idea that
    earns a place on this page comes with a number or a concrete pain. An idea that turns
    actionable graduates to [`docs/issues`](https://gitlab.com/cardona-oss/modelica/-/tree/main/docs/issues).

## How a change to Modelica actually happens

Worth stating first, so the list points somewhere. The language is governed by the Modelica
Association, and the specification by its MAP-Lang design group. A change is a **Modelica
Change Proposal (MCP)**: an issue or pull request on
[`modelica/ModelicaSpecification`](https://github.com/modelica/ModelicaSpecification), or a
full numbered MCP that moves through stages (under discussion, in preparation, under
evaluation, accepted). A serious MCP carries a champion, a written rationale, a
backward-compatibility analysis, and usually a prototype in a real tool.

The honest constraint is that the Association adds far more than it removes, and backward
compatibility is close to sacred. A pure simplification rarely lands as a removal. It lands
as one of three shapes, and picking the shape is half the work:

* **Optional sugar.** The common case gets a shorter spelling; the old spelling still parses.
  Nothing is taken away, so nothing breaks.
* **A Flat Modelica cleanup.** Flat Modelica is the MAP effort to standardize the intermediate
  form a tool produces after flattening. A construct that only ever exists to be flattened
  away is a candidate to be simpler there without touching the surface language.
* **A specification clarification.** Two tools disagree because the spec is ambiguous; pinning
  it down removes complexity without adding syntax.

## The candidates

Ordered by the evidence behind them, strongest first.

### 1. Make `Complex` built in, and retire general operator overloading

**Evidence.** Operator overloading is the single largest blocker in the coverage run: on
2026-09-09 it stopped **311 of 430** example models at parse, three-quarters of the library on
one construct. Almost all of it exists to give one type, `Complex`, ordinary arithmetic. The
machinery is a whole sublanguage: `operator record`, `operator function`, and
quoted-identifier operator names (`'+'`, `'*'`, `'constructor'`). `Modelica.Units` alone
defines dozens of `operator record ComplexCurrent = Complex(...)` aliases, and importing
`Modelica.Units.SI` (which nearly every model does) drags the whole file in.

**The simplification.** Make `Complex` a built-in type, the way `Real` and `Integer` are, with
its arithmetic defined by the language rather than by a user-space operator record. Most of the
reason general operator overloading exists in the standard library then goes away.

**The tension.** Some users overload operators for their own quantity types. A built-in
`Complex` does not serve them. The proposal is stronger if it either keeps a narrow, clearly
bounded overloading facility or shows those uses are rare enough to live without it.

### 2. One way to build an array, not five

**Evidence.** The library builds arrays with comprehensions (`{x[i] for i in 1:n}`), ranges
(`1:n`), constructor functions (`fill`, `zeros`, `ones`, `identity`, `linspace`), matrix
literals (`[1, 2; 3, 4]`), and brace array literals (`{1, 2, 3}`). Each is a separate path in
the parser, and this project added them one at a time, each unblocking a slice of models. They
overlap heavily in what they express.

**The simplification.** A smaller orthogonal core (comprehensions plus literals) with the rest
as ordinary library functions returning arrays. Fewer grammar productions, one mental model.

**The tension.** `[A, B; C, D]` for a block matrix is genuinely readable and hard to give up.
A proposal probably keeps matrix literals and prunes the rest.

### 3. One assignment token

**Evidence.** `expected ':=', found '='` blocks **48** models. Modelica writes an equation with
`=` and an algorithm assignment with `:=`, but the section a statement sits in already says
which it is. The token is redundant with context, and the redundancy is a frequent parse
divergence.

**The simplification.** Allow `=` inside an algorithm section; the section disambiguates.

**The tension.** `x = x + 1` reads as a false equation to an eye trained on the distinction, and
some see the two-token rule as a guardrail rather than a cost. This is the most arguable entry
on the list, which is why it is here rather than higher.

### 4. Make `inner`/`outer` coupling explicit

**Evidence.** `inner`/`outer` is action at a distance: a deeply nested component reaches an
enclosing `inner` declaration by a name-and-type match, with nothing at the use site saying so.
It complicates both flattening (the match is resolved across the whole instance tree) and
reading (the dependency is invisible locally).

**The simplification.** Pass the shared object down explicitly, or require an explicit binding at
the point of use, so the coupling is visible where it happens.

**The tension.** `world` in MultiBody is the canonical `inner`, and the whole point is not wiring
it through every joint. A proposal has to make the explicit form cheap enough that the implicit
one is not missed.

### 5. Fewer class restrictions

**Evidence.** A class is one of nine kinds: `model`, `block`, `connector`, `record`, `type`,
`function`, `package`, `operator`, and `expandable connector`. Several distinctions are
mechanical: a `block` is a model whose connectors are all causal, a `type` is often a record of
one field, and the tool can see both from the contents.

**The simplification.** Infer the restriction where the contents already determine it, and keep
the keyword only where it changes meaning rather than merely labels it.

**The tension.** A restriction is also an early error check: writing `connector` and then putting
an equation in it is caught because the kind was declared. Inference trades that check for
brevity, and not everyone wants the trade.

## Add to this

The list is meant to grow. The bar is the same for every entry: a construct, the evidence that
it costs more than it earns (a coverage number, a parser path, a real model that reads worse for
it), a concrete simplification, and the honest tension against it. When an entry is sharp enough
to act on, it becomes an issue in the register, and if it is ever worth taking upstream, it
becomes the rationale half of an MCP.
