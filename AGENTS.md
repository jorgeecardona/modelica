# AGENTS.md

Guidance for AI coding agents working in this repo. Humans: read README.md and CHANGELOG.md.

## Overview

Pure-Python Modelica toolchain: parse, flatten, simulate, and exchange via FMI 3.0 and SSP 2.0.

The distribution and import package are named `modelica`, not `pymodelica`. The git repo keeps
the historical name; PyPI's `pymodelica` is an abandoned Modelon/JModelica.org upload.

## Build

- `uv sync` installs dependencies. Toolchain (moon, uv) is pinned in `.prototools`; `proto install` brings it up.
- `make install-hooks` once, to enable the tracked git hooks (pre-commit + pre-push).
- `make build` builds the sdist + wheel into `dist/`.
- The core install is dependency-free. Numerics live in extras: `sim` (numpy + scipy), `sundials`, `verify` (musil).

## Test

- `make test` runs the suite in parallel (pytest-parallex, one process per core). Drop the flag with `uv run pytest -q` to debug a single failure serially.
- `make check` runs lint + format-check + typecheck + test. Run it before you commit.
- Full CI is `moon run :ci`; it adds the 3.12 python-floor run, the jit test, build, docs, and the version guards on top of `make check`.

## Style

- Format and autofix: `make format` (ruff, line length 100).
- Lint: `make lint`. Types: `make typecheck` (pyright strict on `src` and `tests`).
- ruff runs with `select = ["ALL"]`; every opt-out is justified in `pyproject.toml`. Keep the public API typed (`py.typed`).

## Gotchas

- Version lives only in `pyproject.toml`. The pre-commit hook auto-bumps it when `src/` changes, and CI (version-guard + version-ahead) fails a `src/` change that carries no bump. Do not edit versions by hand.
- No `jit` extra on purpose: Numba pins `numpy<2.5` while this project runs 2.5, and uv locks one resolution across all extras. Numba runs only in throwaway envs via `moon run :test-jit` and `moon run :bench-fused`. Do not add numba to `pyproject.toml`.
- Python floor is 3.12 (numpy 2.4+ requires it, and its stubs need `collections.abc.Buffer`). No 3.11 back-compat shims.
- `bench` and `coverage` are not CI gates: timings and coverage on a shared runner are noise. Someone runs them and records the number with the machine and MSL version beside it.
- Docs notebooks under `docs/notebooks` are executed at build time (`make docs`); a broken page fails `mkdocs build --strict`. Narrative pages are `.py` jupytext; runnable ones are `.ipynb` committed with outputs stripped.
- Record follow-ups in `docs/issues/`, not as scattered TODO comments.
