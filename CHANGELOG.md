# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org/); the version in `pyproject.toml` is the single
source of truth and is bumped automatically when shipped source changes.

## 0.0.2

- Trademark notice: Modelica® / FMI / SSP belong to the Modelica Association; this is an
  independent implementation, not affiliated or endorsed. Stated in the README (so it is
  on the PyPI page), the docs, and the package docstring.

## 0.0.1

Project restart. The 2010 Python 2 / pyparsing prototype is retired (it remains in git
history); nothing from it is carried forward.

- Claim the `modelica` distribution name on PyPI.
- Repository moved from GitHub to GitLab (`gitlab.com/jorgeecardona/pymodelica`).
- Toolchain: uv + moon pinned via proto; ruff (`select = ["ALL"]`) and pyright strict.
- Single-job GitLab CI driven by `moon run :ci`; PyPI releases via OIDC Trusted
  Publishing; docs on GitLab Pages.
- Python 3.11+ only.
