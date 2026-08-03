---
title: There is no honest number for how much Modelica this reads, because the corpus is ours
status: open
area: ci
owner: null
filed: 2026-08-03
kind: story
scope: L
verify: probe
---

## Measured

- The corpus is fifteen models in `examples/models/`, every one written for this project by
  the people building it. It is excellent at what it is for — each model forces exactly one
  new capability, and `examples/README.md` orders them that way — and it says nothing about
  coverage. A corpus is dense where its authors' intuition is dense.
- `modelica.lang.parse` refuses eleven constructs by name (`src/modelica/lang/parse.py:85`).
  So the *reachable* fraction of any real library is bounded by that list, and nobody has
  measured what the bound is.
- Rumoca publishes the two numbers this project does not have
  ([arXiv 2606.14998](https://arxiv.org/abs/2606.14998)): **28 % of MSL examples reach
  end-to-end simulation**, and of the models that do simulate, **about 33 % disagree with
  OpenModelica**. Those are the reachability gap and the agreement gap, named as such.

## Inferred

Two claims are worth separating, because they fail differently and a single "it works"
hides both:

- **Reachability** — the file parses, flattens, causalizes and runs. This is mostly a
  language-coverage question, and today it is answered by the refusal list.
- **Agreement** — it runs *and produces the same trajectory as a tool people trust*. This is
  a semantics question, and it is the one that can be quietly wrong. Rumoca's 33 % is the
  useful warning: a third of what simulated was not right, and nothing about running says so.

Inferred consequence for this project: the corpus checks agreement well (analytic solutions,
independent references, invariants) on models chosen to be checkable, and checks reachability
not at all. A number against somebody else's library would be the first statement about this
toolchain that its authors did not get to choose the inputs for.

## Why it is worth the L, added 2026-08-03

There is a published precedent for how to say this, recorded in `docs/ideas/landscape.md`:
Rumoca reports 28 % reachability and 33 % disagreement against OpenModelica. Those numbers
are unflattering and checkable, which is what makes them worth having. The corpus here is
fifteen models written by the same people who wrote the compiler, and every one passes — a
measurement of the corpus more than of the toolchain.

## Done

`moon run :coverage` reports two numbers over a named external library, at a named version:

    reachable   parsed / flattened / causalized / simulated, as a percentage
    agreeing    of those that simulated, how many match the reference within tolerance

- The library is the Modelica Standard Library, or a stated subset of it, so the number can
  be compared with Rumoca's and with anybody else's.
- The reference is OpenModelica, driven through OMPython or through an exported FMU read by
  FMPy — not a second run of this toolchain.
- **Every refusal is attributed.** A model that does not reach the end reports *which* stage
  refused it and why, so the output is a work list ordered by how many models each missing
  construct blocks. That is the real product: "arrays block 340 models" is a decision, and
  "28 %" alone is a headline.
- The number is published in the docs next to the date and the MSL version it was taken
  against, the way `docs/ideas/benchmarks.md:94` already requires for timings.

## Deliberately not done

- Not a CI gate. This is a measurement that moves as the language coverage moves, and gating
  on it would make a green pipeline a matter of taste. `verify: probe` for the same reason —
  a passing test would only prove the harness ran.
- No vendoring of the MSL. Fetch it at a pinned version, or skip with a message saying how to
  get it.
- No attempt to make the number go up as part of this issue. Measuring and improving are
  different jobs, and doing them in one change means never knowing which did what.
