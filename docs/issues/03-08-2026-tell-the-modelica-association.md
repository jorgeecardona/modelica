---
title: The Association has not been told this exists, and the trademark note assumes they will be
status: open
area: docs
owner: null
filed: 2026-08-03
kind: idea
scope: XS
verify: inspect
---

## Measured

- `README.md:118` carries a trademark section: "Modelica® is a registered trademark of the
  Modelica Association. FMI and SSP are also its trademarks. This page uses those names to
  identify the language this software reads and the standards it implements."
- The distribution on PyPI is named `modelica` (`README.md:7`), which is the trademark itself as
  a package name. The repository keeps the older name `pymodelica` because PyPI holds that one
  for an abandoned Modelon/JModelica.org upload.
- Emailing `board@modelica.org` was raised on 2026-07-28 and **deferred by Jorge**, in his words:
  "lets make it great before emailing them" (`logs/2026-07-28.md`, open items). It has not been
  raised since.

## Inferred

The risk is inferred and it is asymmetric. Nominative use — using a trademark to say what your
software reads — is normally fine, and the README already frames it that way. A *package name*
is a different question, because it is how people install the thing and it is what appears in
their dependency list. If the Association objects, the fix is a rename, and the cost of a rename
grows with every user who has written `modelica` in a `pyproject.toml`.

So the cost of asking is a paragraph, and the cost of not asking rises monotonically with
adoption. That is the whole argument, and it is why this sits in the register rather than in a
log: it is a decision that gets more expensive to defer, held by exactly one person.

Not inferred: Jorge's reason for waiting is a good one. A first contact that comes with a working
toolchain, a documentation site and an example corpus reads differently from one that comes with
a plan.

## What "done" looks like

Either:

- the mail is sent to `board@modelica.org` — what this is, that the name is used nominatively,
  the PyPI situation, and a direct question about the package name — and the answer is recorded
  in `docs/` beside the trademark note; **or**
- the deferral is recorded here with the condition attached ("after X ships"), so it stops being
  something that has to be remembered.

`verify: inspect` — the artifact is the evidence, and there is nothing to test.

## Deliberately not done

No rename, no pre-emptive change of the PyPI name, and no legal opinion. This issue is one
email and one decision, and the decision is Jorge's alone.
