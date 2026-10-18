# Contributing

Thanks for looking at `modelica`. This is a pre-alpha rewrite of a 2010 experiment; the
core is dependency-free and Python 3.12+.

## Contributing with an AI agent

Most changes here will be written with a coding agent — that's expected and welcome. To get a good
result from yours:

- **Point it at [`AGENTS.md`](AGENTS.md) first** — the machine-facing contract for this repo: build
  and test commands, the typed-API and style rules, and the repo-specific gotchas. Agents read it
  automatically when started from the repo root.
- **Ship a green `make check`** (lint + typecheck + test). Have the agent run it and fix what it
  finds *before* opening the MR, not leave it for review.
- **Don't let it bypass the hooks.** The patch version auto-bumps on `src/` changes and CI's
  version-guard enforces it; a `--no-verify` commit fails the pipeline.
- **The distribution is `modelica`, not `pymodelica`** (see AGENTS.md); keep the diff small and
  typed, add a `CHANGELOG.md` entry, and don't reformat unrelated files.

## Setup

The toolchain is [uv](https://docs.astral.sh/uv/) and [moon](https://moonrepo.dev/),
pinned in `.prototools`. `proto install` brings up both. If you already have `uv`, that
alone is enough for everything below.

```bash
uv sync --extra sim --extra verify   # dev environment
make install-hooks                   # pre-commit hook: bumps the patch version, runs ruff
```

## The loop

```bash
make check    # lint (ruff) + typecheck (pyright) + test (pytest). This is what CI runs.
make format   # auto-format and autofix lint
make docs     # build the docs site strictly, re-executing every notebook
```

CI runs one command, `moon run :ci`, so a green `make check` locally is the same gate.

## Tests

`pytest` under `tests/`. Some tests compile and run generated C (the `sim` C dialect); they
are skipped automatically when no C compiler is on `PATH`. New behaviour needs a test that
would fail without it.

## Versioning and releases

The `version` in `pyproject.toml` is the single source of truth. There is no manual release
step.

- The pre-commit hook increments the patch version when a file under `src/**/*.py` changes.
- `scripts/ci-version-guard` fails the pipeline if shipped source moved and the version did
  not, so the hook cannot be bypassed.
- On `main`, `moon run :release` publishes to PyPI via GitLab OIDC Trusted Publishing, but
  only for a version that is not already there. No API token is stored anywhere.

## Style

- Code: `ruff` and `pyright` in strict mode, both enforced by `make check`.
- Prose (docs, comments, commit messages): no em-dashes; claim first, then evidence;
  concrete names and paths over qualitative words. The docs follow ASD-STE100 Simplified
  Technical English, described on the [style page](https://jorgeecardona.gitlab.io/pymodelica/style/).

## Merge requests

Open the MR against `main`. Confirm `make check` and `make docs` both pass. Keep the commit
subject to one line that states what changed.
