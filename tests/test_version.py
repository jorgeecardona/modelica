"""The version in ``modelica.__version__`` mirrors pyproject's.

scripts/bump-version rewrites both, and this is what catches it when only one of them moves.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import modelica


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert modelica.__version__ == data["project"]["version"]


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", modelica.__version__)
