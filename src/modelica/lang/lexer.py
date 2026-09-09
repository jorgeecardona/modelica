"""Modelica source text -> a flat list of tokens.

Kept apart from the parser because the two have genuinely different jobs. This module
knows what a number, a name and a string *look like* and nothing at all about what a model
is; comments and whitespace are dropped here, which is why the parser never mentions them.

Every token carries its line and column, so an error raised three stages later can still
point at the character that caused it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = ["KEYWORDS", "Kind", "ModelicaSyntaxError", "Token", "tokenize"]


class ModelicaSyntaxError(SyntaxError):
    """Malformed Modelica source.

    Subclasses the built-in :class:`SyntaxError` so that ``except SyntaxError`` catches it
    and so that CPython renders the offending line with a caret under it for free.
    """


class Kind(StrEnum):
    """What a token *is*. The values read well inside an error message."""

    NAME = "name"
    KEYWORD = "keyword"
    INT = "integer literal"
    REAL = "real literal"
    STRING = "string literal"
    OP = "operator"
    END = "end of input"


#: Reserved words, per the Modelica specification. `der` and `initial` are on this list
#: even though they are used like functions; the parser handles that where it matters.
KEYWORDS: Final = frozenset(
    {
        "algorithm", "and", "annotation", "block", "break", "class", "connect",
        "connector", "constant", "constrainedby", "der", "discrete", "each", "else",
        "elseif", "elsewhen", "encapsulated", "end", "enumeration", "equation",
        "expandable", "extends", "external", "false", "final", "flow", "for", "function",
        "if", "import", "impure", "in", "initial", "inner", "input", "loop", "model",
        "not", "operator", "or", "outer", "output", "package", "parameter", "partial",
        "protected", "public", "pure", "record", "redeclare", "replaceable", "return",
        "stream", "then", "true", "type", "when", "while", "within",
    },
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Token:
    """One lexeme.

    For a string literal ``text`` holds the *decoded* contents -- quotes stripped, escapes
    resolved -- because every consumer wants the value rather than the spelling.
    """

    kind: Kind
    text: str
    line: int
    column: int

    def describe(self) -> str:
        """How the token is named in an error message."""
        return "end of input" if self.kind is Kind.END else f"{self.kind} {self.text!r}"


_ESCAPES: Final[dict[str, str]] = {
    "\\": "\\",
    '"': '"',
    "'": "'",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "a": "\a",
    "v": "\v",
    "?": "?",
}

_STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)

# A quoted identifier, `'+'` or `'a b'`: any name, including one spelled with an operator or a
# keyword. The standard library uses it for overloaded operators (`operator function '+'`). The
# quotes are kept in the token, so it is a distinct name from the bare form and round-trips.
_QIDENT = re.compile(r"'(?:[^'\\]|\\.)*'")

# Order matters: a real literal must be tried before an integer, and the two-character
# operators before the one-character ones.
_TOKEN = re.compile(
    r"""
      (?P<space>[ \t\r\n]+)
    | (?P<comment>//[^\n]*)
    | (?P<real>\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\.\d+(?:[eE][-+]?\d+)?)
    | (?P<int>\d+)
    | (?P<name>[A-Za-z_]\w*)
    | (?P<op><>|<=|>=|==|:=|\.[-+*/^]|[-+*/^()\[\]{},;:.=<>])
    """,
    re.VERBOSE,
)

_KIND_OF: Final[dict[str, Kind]] = {"real": Kind.REAL, "int": Kind.INT, "op": Kind.OP}
_SKIPPED: Final = frozenset({"space", "comment"})


def error(source: str, filename: str, line: int, column: int, message: str) -> ModelicaSyntaxError:
    """Build a syntax error that knows which line to show."""
    lines = source.splitlines()
    text = lines[line - 1] if 1 <= line <= len(lines) else None
    return ModelicaSyntaxError(message, (filename, line, column, text))


def _unescape(body: str) -> str:
    if "\\" not in body:
        return body
    out: list[str] = []
    index = 0
    while index < len(body):
        if body[index] == "\\" and index + 1 < len(body):
            out.append(_ESCAPES.get(body[index + 1], body[index + 1]))
            index += 2
        else:
            out.append(body[index])
            index += 1
    return "".join(out)


def _advance(chunk: str, line: int, line_start: int, position: int) -> tuple[int, int]:
    """Update the line counter after consuming ``chunk``, which ends at ``position``."""
    newlines = chunk.count("\n")
    if not newlines:
        return line, line_start
    trailing = len(chunk) - chunk.rfind("\n") - 1
    return line + newlines, position - trailing


def tokenize(source: str, filename: str = "<modelica>") -> list[Token]:  # noqa: C901 -- one branch
    # per lexeme kind (block comment, string, quoted identifier, the rest); the branches are the
    # grammar of a token, not tangled control flow.
    """Split Modelica source into tokens, dropping whitespace and comments."""
    tokens: list[Token] = []
    position, line, line_start = 0, 1, 0

    while position < len(source):
        column = position - line_start + 1

        if source.startswith("/*", position):
            closed = source.find("*/", position + 2)
            if closed < 0:
                raise error(source, filename, line, column, "unterminated block comment")
            chunk, position = source[position : closed + 2], closed + 2
            line, line_start = _advance(chunk, line, line_start, position)
            continue

        if source[position] == '"':
            quoted = _STRING.match(source, position)
            if quoted is None:
                raise error(source, filename, line, column, "unterminated string")
            chunk, position = quoted.group(), quoted.end()
            tokens.append(Token(Kind.STRING, _unescape(chunk[1:-1]), line, column))
            line, line_start = _advance(chunk, line, line_start, position)
            continue

        if source[position] == "'":
            named = _QIDENT.match(source, position)
            if named is None:
                raise error(source, filename, line, column, "unterminated quoted identifier")
            chunk, position = named.group(), named.end()
            tokens.append(Token(Kind.NAME, chunk, line, column))  # quotes kept: `'+'` is a name
            line, line_start = _advance(chunk, line, line_start, position)
            continue

        found = _TOKEN.match(source, position)
        if found is None:
            message = f"unexpected character {source[position]!r}"
            raise error(source, filename, line, column, message)

        group, chunk, position = found.lastgroup, found.group(), found.end()
        if group in _SKIPPED:
            line, line_start = _advance(chunk, line, line_start, position)
            continue
        if group == "name":
            kind = Kind.KEYWORD if chunk in KEYWORDS else Kind.NAME
        else:
            kind = _KIND_OF[group] if group is not None else Kind.OP
        tokens.append(Token(kind, chunk, line, column))

    tokens.append(Token(Kind.END, "", line, position - line_start + 1))
    return tokens
