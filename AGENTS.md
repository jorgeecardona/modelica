# AGENTS.md

Guidance for AI coding agents working in this repo. Humans: read README.md and CHANGELOG.md.

## Overview

`modelica` (distribution name) is a pure-Python Modelica toolchain: parse `.mo` source files,
flatten them to a DAE, simulate, and interoperate with FMI 3.0 (single components) and SSP 2.0
(systems of components). The git repo is named `pymodelica`; the import package and PyPI
distribution are both `modelica`. Pre-alpha. Python 3.12+. No runtime dependencies at the core
(`pip install modelica` brings nothing); numerics arrive as the `sim` extra (numpy + scipy).

Layers under `src/modelica/` — each usable without the ones above it:

| Module | Purpose |
|---|---|
| `lang` | Hand-written lexer + parser: Modelica source → AST with source positions |
| `build` | Python builder API → same AST values as `lang` |
| `ir` | AST → flat model (variables, equations); matching, BLT, index reduction (Pantelides) |
| `sim` | Flat model → ODE/DAE problem; solver backends (Radau, LSODA, SUNDIALS); events |
| `diagram` | Any layer → SVG/Mermaid/DOT picture; also the `__main__` CLI entry point |
| `fmi` | FMI 3.0 import (run an FMU) and export (emit a flat model as an FMU) |
| `ssp` | SSP 2.0: read `.ssd`/`.ssv`/`.ssb`, run a co-simulation master |

The bundled Modelica Standard Library slice (`Modelica.*`) covers Units.SI, Electrical.Analog,
Mechanics.Translational and Rotational, Thermal.HeatTransfer, and a subset of Blocks.

## Build

- Toolchain (moon 2.2.6, uv 0.11.19) pinned in `.prototools`; run `proto install` once after
  cloning to bring up both.
- `uv sync --extra sim --extra verify` installs the dev environment.
- `make install-hooks` wires up `.githooks/` (pre-commit auto-bumps patch version + runs ruff;
  pre-push runs the full check). Run once after cloning.
- `make build` builds sdist and wheel into `dist/` (via `uv build`).
- CI: `moon run :ci` (all gates); `moon run :release` on `main` (publishes to PyPI via OIDC).

## Test

- `make test` — pytest with `--parallel=process` (one worker per CPU core via pytest-parallex).
  Drop the flag (`uv run pytest -q`) when debugging a single failure.
- `make check` — the full CI gate: `lint + format-check + typecheck + test`. Must pass before
  every commit. A green `make check` is the same gate CI runs.
- `moon run :test-python-floor` — proves the package imports and tests pass on Python 3.12
  (the `requires-python` floor) in an isolated environment.
- `moon run :test-jit` — Numba/JIT tests in a throwaway env (Numba pins numpy < 2.5 and cannot
  live in the main lock). `tests/test_jit.py` skips itself outside that env.
- Some simulation tests compile generated C; they are skipped automatically when no C compiler
  is on `PATH`.

## Style

- `make format` — auto-format and autofix (ruff, line-length 100; covers `src tests examples
  docs/notebooks`).
- `make lint` — ruff check only, no fixes (same as CI, scoped to `src tests`).
- `make format-check` — fail if anything is not formatted (CI scope: src, tests, examples,
  docs/notebooks). This is in the `make check` gate.
- `make typecheck` — pyright strict mode over `src/` and `tests/`.
- Docs and comments follow ASD-STE100 Simplified Technical English (claim first, then evidence;
  no em-dashes; concrete names over qualitative words).
- Commit subject: one line stating what changed.

## Gotchas

- **Version lives as `__version__` in `src/modelica/__init__.py`** (hatchling reads it, so
  `pyproject.toml` never disagrees). The pre-commit hook auto-bumps the patch version
  whenever a file under `src/**/*.py` is staged. Never hand-edit it; never commit with
  `--no-verify` on a src change — `scripts/ci-version-guard` will fail the pipeline anyway.
- **Distribution name vs. repo name:** the package is `modelica`, not `pymodelica`. PyPI's
  `pymodelica` is an abandoned upload; don't reference it.
- **Numba is intentionally absent from `pyproject.toml`.** It pins `numpy < 2.5`; including it
  rolls the entire lock back. The JIT backend is tested via `moon run :test-jit` (isolated env).
  Don't add a `jit` extra without reading the long comment in `pyproject.toml` first.
- **Docs notebooks are executed at build time** (`make docs` / `moon run :docs` with
  `mkdocs build --strict`). A notebook that stops working fails the build. Narrative pages are
  `.py` jupytext percent format (diffable); interactive pages are committed `.ipynb` with output
  stripped.
- **`examples/`** holds `.py` twins of `.mo` files. Tests assert both flatten to the same model.
  Per-file ruff ignores allow the Python to mirror the Modelica naming (single-letter variables
  like `l`, `==` as an equation builder, etc.).
- **`make docs` drops the mkdocs-jupyter cache first** (full re-execution). Use `make docs-prose`
  to rebuild fast when only editing prose (reuses cached notebook output — may be stale).
