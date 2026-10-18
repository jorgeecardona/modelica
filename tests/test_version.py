"""``modelica.__version__`` is the single source of truth; pyproject reads it dynamically.

scripts/bump-version rewrites only the literal in src/modelica/__init__.py; hatchling reads it
back via ``[tool.hatch.version]``, so there is one source and nothing to drift against.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

import modelica


def test_version_matches_metadata() -> None:
    # hatchling's dynamic version reads the literal in __init__.py, so the installed metadata must
    # equal __version__. If they drift, the [tool.hatch.version] wiring is broken.
    assert modelica.__version__ == importlib.metadata.version("modelica")


def test_pyproject_version_is_dynamic() -> None:
    # Guard the single source: pyproject declares the version dynamically and carries no static
    # `version = "..."` that could diverge from the literal in __init__.py.
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert "version" not in data["project"]
    assert "version" in data["project"]["dynamic"]


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", modelica.__version__)


def test_the_readme_has_no_relative_links() -> None:
    """The README is the PyPI long description, and PyPI does not rewrite links.

    A relative link resolves against ``pypi.org`` there, so ``[notes](docs/architecture.md)``
    becomes a 404 on the page most people arrive at first. Every link in this file has to be
    absolute, and the documentation ones point at the site the pipeline publishes.
    """
    readme = Path(__file__).resolve().parent.parent / "README.md"
    links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", readme.read_text(encoding="utf-8"))
    relative = [link for link in links if not link.startswith(("http://", "https://", "#"))]
    assert not relative, f"relative links break on PyPI: {relative}"


def test_the_readme_points_at_the_published_documentation() -> None:
    """The pipeline builds a site. Somebody arriving from PyPI should be told where."""
    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    site = data["project"]["urls"]["Documentation"]
    assert site in (root / "README.md").read_text(encoding="utf-8")
