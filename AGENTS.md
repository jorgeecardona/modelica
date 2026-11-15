# AGENTS.md

Guidance for AI coding agents working in this repo. Humans: read README.md and CHANGELOG.md.

## Overview

`modelica` (distribution name) is a Modelica toolchain for Python: parse `.mo` source files,
flatten them to a DAE, simulate, and draw them. FMI 3.0 and SSP 2.0 are started, not usable. The front end is Python. Simulation defaults to a precompiled Rust
engine (`engine/`, a bytecode VM shipped in platform wheels; see `docs/vm/index.md`) and falls
back to SciPy where no engine is installed. The repo, the import package and the PyPI
distribution are all `modelica`. Version 0.1 (tested, API not frozen). Python 3.12+. `pip install modelica` brings NumPy
and SciPy and simulates; the `sim` extra is kept empty so old install commands still work.

Layers under `src/modelica/` — each usable without the ones above it:

| Module | Purpose |
|---|---|
| `lang` | Hand-written lexer + parser: Modelica source → AST with source positions |
| `build` | Python builder API → same AST values as `lang` |
| `ir` | AST → flat model (variables, equations); matching, BLT, index reduction (Pantelides) |
| `sim` | System → simulation on the engine (`vm`, `engine`) or SciPy (`emit`, `kernel`, `solve`); events |
| `diagram` | Any layer → SVG/Mermaid/DOT picture; also the `__main__` CLI entry point |
| `fmi` | FMI 3.0: `modelDescription.xml` and the calling sequence; no FMU import or export yet |
| `ssp` | SSP 2.0: the co-simulation master protocol; no archive reading or stepping yet |

The bundled Modelica Standard Library slice (`Modelica.*`) covers Units.SI, Electrical.Analog,
Mechanics.Translational and Rotational, Thermal.HeatTransfer, and a subset of Blocks.

## Build

- Toolchain (moon, uv) pinned in `.prototools`; run `proto install` once after cloning to bring up both.
- `uv sync --extra sim --extra verify` installs the dev environment.
- `make install-hooks` wires up `.githooks/` (pre-commit auto-bumps patch version + runs ruff;
  pre-push runs the full check). Run once after cloning.
- `make build` builds sdist and wheel into `dist/` (via `uv build`).
- CI: `moon run :ci` (all gates); `moon run :release` on `main` (publishes to PyPI via OIDC).

## Test

- `make test` — pytest with `--parallel=process` (one worker per CPU core via pytest-parallex).
  Drop the flag (`uv run pytest -q`) when debugging a single failure.
- `make check` — `lint + format-check + typecheck + test`. Must pass before every commit.
  `:ci` runs more: test-python-floor, build, docs, version-guard, and version-ahead
  (`__version__` must be greater than `origin/main` when `src/` changed).
- `moon run :test-python-floor` — proves the package imports and tests pass on Python 3.12
  (the `requires-python` floor) in an isolated environment.
- Engine tests build `engine/src` with `rustc` once when no engine wheel is installed; they are
  skipped when neither is available.

## Style

- `make format` — auto-format and autofix (ruff, line-length 100; covers `src tests examples
  docs/notebooks`).
- `make lint` — ruff check only, no fixes (same as CI, scoped to `src tests examples docs/notebooks`; `select = ["ALL"]` with per-file ignores).
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
- **The name is `modelica` everywhere** (repo, package, distribution). PyPI's `modelica` is
  an unrelated abandoned Modelon/JModelica.org upload; don't reference it.
- **Docs notebooks are executed at build time** (`make docs` / `moon run :docs` with
  `mkdocs build --strict`). A notebook that stops working fails the build. Narrative pages are
  `.py` jupytext percent format (diffable); interactive pages are committed `.ipynb` with output
  stripped.
- **`examples/`** holds `.py` twins of `.mo` files. Tests assert both flatten to the same model.
  Per-file ruff ignores allow the Python to mirror the Modelica naming (single-letter variables
  like `l`, `==` as an equation builder, etc.).
- **`make docs` drops the mkdocs-jupyter cache first** (full re-execution). Use `make docs-prose`
  to rebuild fast when only editing prose (reuses cached notebook output — may be stale).
