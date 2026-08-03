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
