---
title: "`make docs` serves a cached page and calls itself "same as CI", and the cache outlived the code"
status: done
area: ci
owner: cute-replicant-6e76d
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

## What landed

Two targets and a corrected comment. `verify: inspect` -- the artifacts are the `Makefile`
and `mkdocs.yml`.

    docs          build the site the way CI does -- notebooks re-executed, cache dropped first
    docs-prose    rebuild fast for a prose edit -- reuses cached notebook output, which may be stale

`make docs` now drops `.cache/mkdocs-jupyter` before building, so it re-executes the
notebooks the way a clean CI runner does. `make docs-prose` is the 3.5-second path, kept
because it is the right tool for a prose edit, under a name that does not promise the
notebooks ran.

`.cache/mkdocs-jupyter` and not `.cache`, which holds nothing else today -- naming the
directory the plugin owns is the difference between a target and a `rm -rf` waiting for
somebody to put something else there.

### A second discrepancy, and it was worse than the cache

`make docs` ran:

    uv run --extra docs mkdocs build --strict

and `moon run :docs` runs:

    uv run --extra docs --extra sim --extra verify mkdocs build --strict

So the help text said "same as CI" while building without the numerics *or* the model
checker. The notebooks under `docs/notebooks/` import both. That is not a stale page, it is a
different build, and it was in the same line that claimed parity. Both targets now carry all
three extras.

### The comment in `mkdocs.yml` said the wrong thing

It claimed the committed notebooks "cannot render results from an older version of the code"
because their outputs are stripped. Stripping the *stored* outputs does not stop that; the
cache does exactly that. The comment now says so, with the measurement beside it -- 3.5 s warm
and stale against 9 m 20 s cold and correct -- and names which target drops the cache.

### The cost of being correct fell by fifteen times while this was open

The entry recorded 3.5 s warm against **9 m 20 s** cold. Measured again on 2026-08-10, with
the same two targets: **2.9 s warm against 36 s cold**.

The old number is not withdrawn -- it was true when it was taken -- but it is no longer the
argument for keeping a fast path. `compiled-kernels-are-not-cached` closed in between and
`modelica.sim.kernel.build` is now cached on the content of the model, so a notebook that
builds the same model in six cells compiles it once. A nine-minute penalty is a reason to
tolerate a stale page; thirty-six seconds is not, and if `docs-prose` stops being used that
is the right outcome rather than a regression.

## Deliberately not done, found while doing it

- **No cache setting in `mkdocs.yml`.** The plugin's caching is right for the common case and
  the fix belongs in the target that promises freshness, not in a global switch that makes
  every prose edit cost nine minutes.
- **CI is unchanged.** A clean runner has no `.cache` to begin with, so it was never the one
  serving stale pages -- the divergence was always local, which is exactly why the help text
  claiming parity mattered.
