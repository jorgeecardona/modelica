"""Read a Modelica library stored as a directory tree (``package.mo`` per package).

This is the layout of the Modelica Standard Library and of most user libraries: a
directory per package, its ``package.mo`` declaring the package, and one ``.mo`` file or
sub-directory per member class.

`closure` reads only what a model can reach. It indexes the tree without parsing, then
follows each dotted name the model writes the way Modelica looks names up: from the scope
it is written in, outward to the top. A file is read only when a name resolves into it. The
full MSL takes about 43 s to parse; a model that uses a few components reads a few files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from modelica.ir import nodes as ir
from modelica.lang.parse import parse

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ["closure", "load", "modelicapath"]

type _Path = tuple[str, ...]

_NAME = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*")
_NOT_CODE = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', re.DOTALL)
"""String literals and comments: a name inside them refers to no class."""


@dataclass
class _Node:
    """One package directory or member file of a library tree, not yet parsed."""

    path: _Path
    """The class path this node declares, ``("Modelica", "Electrical")``."""
    file: Path
    """The ``package.mo`` of a directory, or the member's ``.mo`` file."""
    children: dict[str, _Node] = field(default_factory=dict[str, "_Node"])

    @property
    def is_package_directory(self) -> bool:
        return self.file.name == "package.mo"


def _index(directory: Path, parent: _Path = ()) -> _Node:
    """The tree of package directories and member files under ``directory``, with no parsing."""
    package_file = directory / "package.mo"
    if not package_file.exists():
        message = f"{directory} is not a Modelica package: it has no package.mo"
        raise ValueError(message)
    path = (*parent, directory.name)
    node = _Node(path, package_file)
    for child in sorted(directory.iterdir()):
        if child.is_dir() and (child / "package.mo").exists():
            node.children[child.name] = _index(child, path)
        elif child.suffix == ".mo" and child.name != "package.mo":
            node.children[child.stem] = _Node((*path, child.stem), child)
    return node


def _references(text: str) -> list[_Path]:
    """The dotted names written in ``text``'s code, outside strings and comments."""
    code = _NOT_CODE.sub(" ", text)
    return [tuple(part.strip() for part in match.split(".")) for match in _NAME.findall(code)]


@dataclass
class _Reach:
    """The fixed point of `closure`: which files are read, and why each one was."""

    roots: dict[str, _Node]
    texts: dict[Path, str] = field(default_factory=dict[Path, str])
    why: dict[Path, str] = field(default_factory=dict[Path, str])
    queue: list[tuple[str, _Path, str]] = field(default_factory=list[tuple[str, _Path, str]])
    """Texts still to scan: the text, the class path it is written in, and where it is from."""

    def node(self, path: _Path) -> _Node | None:
        found = self.roots.get(path[0]) if path else None
        for part in path[1:]:
            if found is None:
                return None
            found = found.children.get(part)
        return found

    def read(self, node: _Node, reason: str) -> None:
        """Read ``node`` and every enclosing package, and queue their text for scanning."""
        for depth in range(1, len(node.path) + 1):
            each = self.node(node.path[:depth])
            if each is None or each.file in self.texts:
                continue
            text = each.file.read_text(encoding="utf-8-sig")
            self.texts[each.file] = text
            self.why[each.file] = reason
            self.queue.append((text, each.path, str(each.file)))

    def resolve(self, name: _Path, scope: _Path, where: str) -> None:
        """Read the files ``name``, written in class ``scope``, can refer to (lexical lookup)."""
        for depth in range(len(scope), -1, -1):
            found = self.node((*scope[:depth], name[0]))
            if found is None:
                continue
            for part in name[1:]:
                inner = found.children.get(part)
                if inner is None:
                    break
                found = inner
            self.read(found, f"{'.'.join(name)!r}, named in {where}")
            return


def modelicapath(value: str | None = None) -> list[Path]:
    """The library directories ``MODELICAPATH`` lists: every package directory in each entry.

    ``value`` is the variable's text; ``None`` reads the environment. Entries are separated by
    the platform's path separator. Nothing reads this unless asked: an OpenModelica install
    puts a full Modelica Standard Library there, which would replace the bundled slice.
    """
    import os  # noqa: PLC0415 - one function reads the environment

    text = os.environ.get("MODELICAPATH", "") if value is None else value
    found: list[Path] = []
    for entry in filter(None, text.split(os.pathsep)):
        base = Path(entry)
        if (base / "package.mo").exists():
            found.append(base)
        elif base.is_dir():
            found.extend(d for d in sorted(base.iterdir()) if (d / "package.mo").exists())
    return found


def closure(roots: Sequence[Path], sources: Iterable[str]) -> ir.Program:
    """The parts of the libraries at ``roots`` that ``sources`` can reach, as one program.

    A name the index does not hold (a component, a loop index, a builtin, or a class
    inherited through ``extends`` of a package) reads nothing.

    Raises:
        SyntaxError: a file the model reaches does not parse; the message says what named it.
        ValueError: a root has no ``package.mo``.
    """
    reach = _Reach({tree.path[0]: tree for tree in (_index(root) for root in roots)})
    for text in sources:
        reach.queue.append((text, (), "the model"))
    while reach.queue:
        text, scope, where = reach.queue.pop()
        for name in _references(text):
            reach.resolve(name, scope, where)
    try:
        classes = [
            _build(tree, reach.texts) for tree in reach.roots.values() if tree.file in reach.texts
        ]
    except SyntaxError as error:
        chain = [reach.why.get(Path(error.filename or ""), "")]
        message = f"{error.msg}; read because {chain[0]}"
        raise SyntaxError(
            message, (error.filename, error.lineno, error.offset, error.text)
        ) from error
    return ir.Program(tuple(classes))


def _build(node: _Node, texts: dict[Path, str]) -> ir.Class:
    """The class of a read package directory, with its read members inside it."""
    declared = parse(texts[node.file], str(node.file)).classes
    if len(declared) != 1:
        message = f"{node.file} declares {len(declared)} classes; a package.mo declares one"
        raise ValueError(message)
    members: list[ir.Class] = []
    for child in node.children.values():
        if child.file not in texts:
            continue
        if child.is_package_directory:
            members.append(_build(child, texts))
        else:
            members.extend(parse(texts[child.file], str(child.file)).classes)
    root = declared[0]
    return replace(root, classes=root.classes + tuple(members))


def load(directory: Path, skipped: dict[Path, str] | None = None) -> ir.Class:
    """The whole package ``directory`` holds, every member file and sub-package inside it.

    For measuring a library (``scripts/coverage.py``). With ``skipped``, a member file that
    does not parse is left out and recorded there with the parser's message; without it the
    first such file raises.

    Raises:
        SyntaxError: a file does not parse and ``skipped`` is ``None``, or a ``package.mo``
            does not parse.
        ValueError: ``directory`` has no ``package.mo``, or it declares more than one class.
    """
    package_file = directory / "package.mo"
    if not package_file.exists():
        message = f"{directory} is not a Modelica package: it has no package.mo"
        raise ValueError(message)
    declared = parse(package_file.read_text(encoding="utf-8-sig"), str(package_file)).classes
    if len(declared) != 1:
        message = f"{package_file} declares {len(declared)} classes; a package.mo declares one"
        raise ValueError(message)
    members: list[ir.Class] = []
    for child in sorted(directory.iterdir()):
        try:
            if child.is_dir() and (child / "package.mo").exists():
                members.append(load(child, skipped))
            elif child.suffix == ".mo" and child.name != "package.mo":
                members.extend(parse(child.read_text(encoding="utf-8-sig"), str(child)).classes)
        except SyntaxError as error:
            if skipped is None:
                raise
            skipped[child] = str(error)
    root = declared[0]
    return replace(root, classes=root.classes + tuple(members))
