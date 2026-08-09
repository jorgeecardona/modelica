---
title: "`make docs` serves a cached page and calls itself "same as CI", and the cache outlived the code"
status: open
area: ci
owner: null
filed: 2026-08-09
kind: bug
scope: S
verify: inspect
---

## Measured

Measured on 2026-08-09, while looking for why a page showed a diagram the code no longer
draws.

- `Makefile:30` says `docs: ## build the mkdocs site (strict, same as CI)`.
- `mkdocs.yml:76` sets `execute: true` with `allow_errors: false`, and the comment above it
  states the intent: the notebooks "are committed with their outputs stripped, so they diff
  like the percent files and **cannot render results from an older version of the code**".
  `docs/notebooks/drawing.ipynb` does carry zero stored outputs — that half is true.
- **`mkdocs-jupyter` caches the executed notebook**, in `.cache/mkdocs-jupyter/`, which is
  gitignored (`.gitignore:16`) and 7.2 MB here. With that cache warm, `make docs` finishes in
  **3.5 seconds** and the page it writes is whatever was executed the last time the cache
  missed.
- What that produced, concretely: after `f52f711` renamed `RLCCircuit`'s components to
  `E.Resistor` via an import alias, and after `7780e61` and `b54ce31` changed how the model
  is drawn entirely, a freshly built `site/notebooks/drawing/index.html` still contained
  `Electrical.Resistor`, five `class="box"` rectangles and no `component icon` group. Three
  commits of change, none of it on the page.
- Deleting `.cache/` and rebuilding takes **9 minutes 20 seconds** and produces the current
  picture.
- CI does not have this problem: `.gitlab-ci.yml:47` caches `.venv` keyed on `uv.lock` and
  nothing else, so every pipeline executes the notebooks cold.

## Inferred

Inferred, not measured: this is a trap specifically for the person checking their own work.
CI is right, so nothing fails; the local page is wrong, so the contributor concludes the code
is broken and goes looking for a bug that is not there. That is what happened here — an
afternoon's worth of diagram changes appeared to have had no effect.

The Makefile's claim is the part that makes it a bug rather than a quirk. "Same as CI" is
what tells a reader the local page can be trusted, and it is exactly what is false: CI
executes and the local build usually does not.

## Done

- `make docs` produces the same page CI produces, or says plainly that it did not. Either
  the cache is off for that target, or the target clears it, or the help text stops claiming
  parity and names the flag that gets it.
- A second target for the fast path, if the 3.5-second build is worth keeping — it is, for
  prose edits — under a name that does not promise the notebooks ran.
- The `mkdocs.yml` comment is corrected. Stripping stored outputs does not stop a page
  "rendering results from an older version of the code" while a cache does exactly that.

## Deliberately not done

- No attempt to make the cache correct by keying it on the source tree. `mkdocs-jupyter`
  owns that key; the fix here is to stop trusting it, not to reimplement it.
- No change to CI, which is already right for the reason that matters: it caches the
  environment and not the answers.
