"""Modelica source text -> :mod:`modelica.ir`.

Recursive descent, one method per grammar rule, no parser generator. The grammar is small
enough that hand-writing it costs less than a build step, and error messages written by
hand are worth more here than anywhere else: a modelling mistake should read like a
sentence, not like a shift/reduce conflict.

The target is not "whatever Modelica allows" but **exactly what**
:mod:`modelica.lang.unparse` **can emit**. That is a checkable specification -- every tree
the printer can produce must parse back to itself -- and it keeps the surface honest.
Constructs outside it (``import``, ``enumeration``, ``inner``/``outer``, ``external``)
raise a
"not supported yet" error rather than being quietly misread; a parser that shrugs at syntax
it does not implement produces a model that silently means something else.

Reading a construct is not the same as running it. A ``redeclare`` of a *class* parses here
and :mod:`modelica.ir.flatten` refuses it by name, the same two-step the arrays and the
algorithm sections went through: the language first, so the shape exists to be tested, then
the semantics.

Two things are deliberately accepted and discarded: a ``within`` clause, because the IR is
a flat list of classes and nothing downstream asks which package a file came from, and
``annotation(...)``, which carries diagram geometry and tool hints rather than semantics.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from modelica import ir
from modelica.lang.lexer import Kind, ModelicaSyntaxError, Token, error, tokenize
from modelica.lang.precedence import BINARY, IF, NON_ASSOCIATIVE, UNARY_SIGN

__all__ = ["ModelicaSyntaxError", "parse", "parse_expression"]


# `class` is Modelica's unrestricted form; the IR has no separate node for it, and for
# every purpose downstream it behaves as a model.
_RESTRICTIONS: Final[dict[str, ir.Restriction]] = {
    "model": "model",
    "connector": "connector",
    "package": "package",
    "record": "record",
    "block": "block",
    "type": "type",
    "function": "function",
    "class": "model",
}

_VARIABILITY: Final[dict[str, ir.Variability]] = {
    "parameter": "parameter",
    "constant": "constant",
    "discrete": "discrete",
}
_CAUSALITY: Final[dict[str, ir.Causality]] = {"input": "input", "output": "output"}
_CONNECTOR: Final[dict[str, ir.ConnectorKind]] = {"flow": "flow", "stream": "stream"}
_SCOPING: Final[dict[str, ir.Scoping]] = {"inner": "inner", "outer": "outer"}

# Infix operators, spelled the way the IR spells them. The levels come from the shared
# table; this map exists only so the operator reaches `ir.Binary` as a literal type.
_INFIX: Final[dict[str, ir.BinaryOp]] = {
    "or": "or",
    "and": "and",
    "==": "==",
    "<>": "<>",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "+": "+",
    "-": "-",
    # `.+`/`.-` are identical to `+`/`-` here -- an array `+`/`-` is already elementwise -- so
    # they fold to the plain operator. `.*`/`./`/`.^` differ from `*`/`/`/`^` for arrays (a
    # contraction versus an elementwise product, a refusal versus an elementwise divide/power),
    # so they keep their own op and the flattener reads them elementwise.
    ".+": "+",
    ".-": "-",
    "*": "*",
    "/": "/",
    ".*": ".*",
    "./": "./",
    "^": "^",
    ".^": ".^",
}
_SIGN: Final[dict[str, ir.UnaryOp]] = {"+": "+", "-": "-"}

if set(_INFIX) | {"not"} != set(BINARY):  # pragma: no cover - guards against drift
    raise AssertionError("_INFIX and precedence.BINARY disagree about which operators exist")

# Keywords that end a list of equations: the class itself, or the next section.
_SECTION_END: Final = frozenset(
    {"end", "equation", "initial", "algorithm", "public", "protected", "annotation"},
)
# `expandable` is here for the lookahead's sake: it can only sit directly before
# `connector`, and `class_definition` enforces that; the lookahead only needs to know a
# class is coming.
_ELEMENT_PREFIX: Final = frozenset({"encapsulated", "partial", "final", "expandable"})
_OPEN: Final = frozenset("([{")
_CLOSE: Final = frozenset(")]}")

# Said plainly and early, rather than misparsed and blamed on the solver later.
_UNSUPPORTED: Final[dict[str, str]] = {
    "external": "external functions are not supported yet",
    "operator": "operator overloading is not supported yet",
}

# Keywords that end a list of statements. The same set as `_SECTION_END` plus the words
# that close a compound statement, because a statement list is always inside something.
_STATEMENT_END: Final = _SECTION_END | {"elseif", "else", "elsewhen"}


def _merge(first: ir.Modification, second: ir.Modification) -> ir.Modification:
    """Two annotations on one class are one annotation, the later entry winning."""
    if not first:
        return second
    if not second:
        return first
    return ir.Modification(
        value=second.value if second.value is not None else first.value,
        entries={**first.entries, **second.entries},
    )


class _Parser:
    """One pass over one source string.

    Holds the token list and a cursor into it. Every method either consumes a complete
    grammar production and returns an IR value, or raises.
    """

    def __init__(self, source: str, filename: str) -> None:
        self.source = source
        self.filename = filename
        self.tokens = tokenize(source, filename)
        self.at = 0

    # -- cursor ---------------------------------------------------------------------

    @property
    def token(self) -> Token:
        """The token under the cursor. Sticks at end-of-input rather than running off."""
        return self.tokens[min(self.at, len(self.tokens) - 1)]

    def _ahead(self, distance: int) -> Token:
        return self.tokens[min(self.at + distance, len(self.tokens) - 1)]

    def _error(self, message: str, where: Token | None = None) -> ModelicaSyntaxError:
        """Build a syntax error pointing at ``where``, or at the cursor. Callers raise it."""
        found = where or self.token
        return error(self.source, self.filename, found.line, found.column, message)

    def _check(self, kind: Kind, text: str | None = None) -> bool:
        token = self.token
        return token.kind is kind and (text is None or token.text == text)

    def _accept(self, kind: Kind, text: str | None = None) -> Token | None:
        if not self._check(kind, text):
            return None
        token = self.token
        self.at += 1
        return token

    def _expect(self, kind: Kind, text: str | None = None) -> Token:
        token = self._accept(kind, text)
        if token is None:
            wanted = repr(text) if text is not None else str(kind)
            raise self._error(f"expected {wanted}, found {self.token.describe()}")
        return token

    def _at(self, token: Token) -> ir.Span:
        """Where ``token`` was written, as the IR spells a position."""
        return ir.Span(self.filename, token.line, token.column)

    def _keyword(self, *texts: str) -> bool:
        token = self.token
        return token.kind is Kind.KEYWORD and token.text in texts

    def _refuse_unsupported(self) -> None:
        """Turn a recognised-but-unimplemented keyword into a clear error."""
        token = self.token
        if token.kind is Kind.KEYWORD and token.text in _UNSUPPORTED:
            raise self._error(_UNSUPPORTED[token.text])

    # -- names, comments, annotations -----------------------------------------------

    def _dotted(self) -> tuple[str, ...]:
        self._accept(Kind.OP, ".")  # a leading dot means global scope; nothing to record
        parts = [self._expect(Kind.NAME).text]
        while self._accept(Kind.OP, "."):
            parts.append(self._expect(Kind.NAME).text)
        return tuple(parts)

    def _type_name(self) -> str:
        return ".".join(self._dotted())

    def _reference(self) -> ir.Ref:
        # Taken before the name is read, so it points at the first character of `c.v` rather
        # than at whatever follows it.
        where = self._at(self.token)
        return ir.Ref(self._dotted(), where)

    def _string_comment(self) -> str:
        token = self._accept(Kind.STRING)
        return token.text if token is not None else ""

    def _annotation(self) -> ir.Modification:
        """``annotation(Icon(graphics = {...}))`` -- kept, because it is the diagram.

        An annotation is a class-modification and nothing else, so the modification parser
        reads it unchanged. It used to be counted over and thrown away, on the reasoning
        that geometry is not semantics. That is true of the equations and false of the
        model: where a component sits, and what it looks like, is written here and nowhere
        else, and a tool that discards it can only ever guess at a diagram.
        """
        if not self._check(Kind.KEYWORD, "annotation"):
            return ir.EMPTY
        self.at += 1
        if not self._check(Kind.OP, "("):
            raise self._error("expected '(' after 'annotation'")
        return self._modification()

    def _terminate(self) -> ir.Modification:
        """The tail every declaration and equation shares: comment, annotation, ``;``."""
        self._string_comment()
        annotation = self._annotation()
        self._expect(Kind.OP, ";")
        return annotation

    # -- expressions ----------------------------------------------------------------

    def expression(self) -> ir.Expr:
        """Parse one expression, range tail included: ``a:b``, and ``a:s:b``.

        The range reads at this one level rather than through the precedence table --
        ``:`` is not an operator every expression must parenthesise against; it is a tail
        the grammar allows exactly here (specification: simple_expression). The parts are
        read below the tail, so a branch of an ``if`` may be a range but a range part may
        not be a bare ``if``, which is what the specification says.
        """
        if self._keyword("if"):
            return self._if_expression()
        first = self._binary(IF + 1)
        if not self._accept(Kind.OP, ":"):
            return first
        second = self._binary(IF + 1)
        if self._accept(Kind.OP, ":"):
            return ir.Range(first, self._binary(IF + 1), second)
        return ir.Range(first, second)

    def whole_expression(self) -> ir.Expr:
        """Parse one expression and insist it is the entire input."""
        node = self.expression()
        self._expect(Kind.END)
        return node

    def _if_expression(self) -> ir.IfExpr:
        self.at += 1  # `if`, or the `elseif` of the chain this call continues
        condition = self.expression()
        self._expect(Kind.KEYWORD, "then")
        then = self.expression()
        if self._keyword("elseif"):
            return ir.IfExpr(condition, then, self._if_expression())
        self._expect(Kind.KEYWORD, "else")
        return ir.IfExpr(condition, then, self.expression())

    def _infix(self) -> ir.BinaryOp | None:
        """The infix operator under the cursor, if there is one. Does not consume it."""
        token = self.token
        if token.kind in (Kind.OP, Kind.KEYWORD):
            return _INFIX.get(token.text)
        return None

    def _binary(self, minimum: int) -> ir.Expr:
        """Precedence climbing over :data:`modelica.lang.precedence.BINARY`.

        Sharing the table with the printer is what makes ``parse(unparse(x)) == x`` a
        structural property rather than a coincidence: the level that decides whether the
        printer emits a bracket is the same level that decides whether the parser stops.
        """
        node = self._prefix(minimum)
        while (op := self._infix()) is not None and BINARY[op] >= minimum:
            level = BINARY[op]
            self.at += 1
            node = ir.Binary(op, node, self._binary(level + 1))
            following = self._infix()
            if op in NON_ASSOCIATIVE and following is not None and BINARY[following] == level:
                raise self._error(f"{op!r} does not chain with {following!r}; add parentheses")
        return node

    def _prefix(self, minimum: int) -> ir.Expr:
        """``not`` and the unary sign, each admitted only where it binds tightly enough."""
        token = self.token
        if token.kind is Kind.KEYWORD and token.text == "not" and minimum <= BINARY["not"]:
            self.at += 1
            return ir.Unary("not", self._binary(BINARY["not"]))
        if token.kind is Kind.OP and token.text in _SIGN and minimum <= UNARY_SIGN:
            self.at += 1
            return ir.Unary(_SIGN[token.text], self._binary(UNARY_SIGN))
        return self._primary()

    def _primary(self) -> ir.Expr:  # noqa: C901, PLR0911, PLR0912 -- one case per token
        token = self.token
        match token.kind:
            case Kind.INT:
                self.at += 1
                return ir.IntLiteral(int(token.text))
            case Kind.REAL:
                self.at += 1
                return ir.RealLiteral(float(token.text))
            case Kind.STRING:
                self.at += 1
                return ir.StringLiteral(token.text)
            case Kind.OP if token.text == "(":
                self.at += 1
                inner = self.expression()
                if self._check(Kind.OP, ","):
                    # Reachable only where an output list cannot be: inside an expression.
                    # `Parser.equation` reads the list itself before it gets here.
                    raise self._error("an output list belongs at the head of an equation")
                self._expect(Kind.OP, ")")
                return inner
            case Kind.OP if token.text == "{":
                return self._array()
            case Kind.OP if token.text == "[":
                return self._matrix()
            case Kind.KEYWORD if token.text in ("true", "false"):
                self.at += 1
                return ir.BoolLiteral(token.text == "true")
            case Kind.KEYWORD if token.text in ("der", "initial"):
                self.at += 1
                args, named = self._arguments()
                return ir.Call(token.text, args, named)
            case Kind.NAME:
                where = self._at(token)
                parts = self._dotted()
                if self._check(Kind.OP, "("):
                    args, named = self._arguments()
                    return ir.Call(".".join(parts), args, named)
                indices = self._bracketed()
                if not indices:
                    return ir.Ref(parts, where)
                if self._check(Kind.OP, "."):
                    raise self._error("a subscript on an inner component is not supported yet")
                return ir.Subscript(ir.Ref(parts, where), indices)
            case _:
                self._refuse_unsupported()
        raise self._error(f"expected an expression, found {token.describe()}")

    def _matrix(self) -> ir.Matrix:
        """``[1, 2; 3, 4]`` -- at least one row, each row at least one expression."""
        self._expect(Kind.OP, "[")
        rows: list[tuple[ir.Expr, ...]] = []
        while True:
            row = [self.expression()]
            while self._accept(Kind.OP, ","):
                row.append(self.expression())
            rows.append(tuple(row))
            if not self._accept(Kind.OP, ";"):
                break
        self._expect(Kind.OP, "]")
        return ir.Matrix(tuple(rows))

    def _array(self) -> ir.Array | ir.Comprehension:
        """``{a, b, c}`` nested as deep as it goes -- or ``{f(i) for i in 1:n}``."""
        self._expect(Kind.OP, "{")
        if self._accept(Kind.OP, "}"):
            return ir.Array(())
        elements: list[ir.Expr] = [self.expression()]
        if self._keyword("for"):
            body = ir.Comprehension(elements[0], self._iterators())
            self._expect(Kind.OP, "}")
            return body
        while self._accept(Kind.OP, ","):
            elements.append(self.expression())
        self._expect(Kind.OP, "}")
        return ir.Array(tuple(elements))

    def _iterators(self) -> tuple[tuple[str, ir.Expr | None], ...]:
        """``for i in 1:n, j in 1:m`` -- the range may be omitted, and is ``None`` then."""
        self._expect(Kind.KEYWORD, "for")
        found: list[tuple[str, ir.Expr | None]] = []
        while True:
            name = self._expect(Kind.NAME).text
            over = self.expression() if self._accept(Kind.KEYWORD, "in") else None
            found.append((name, over))
            if not self._accept(Kind.OP, ","):
                break
        return tuple(found)

    def _bracketed(self) -> tuple[ir.Expr, ...]:
        """``[n]`` or ``[i, j]`` -- a declared shape after a name, or a subscript after one.

        The same syntax in both places, which is why it is one function. Empty when the next
        token is not a bracket, so a caller can use the result to decide which node to build.
        """
        if not self._accept(Kind.OP, "["):
            return ()
        found = [self._subscript()]
        while self._accept(Kind.OP, ","):
            found.append(self._subscript())
        self._expect(Kind.OP, "]")
        return tuple(found)

    def _subscript(self) -> ir.Expr:
        """One index: ``2``, ``i + 1``, the slice ``1:3``, or the bare ``:``.

        The bare colon means every index of its dimension, and it can only be the whole
        subscript -- ``[:2]`` is not Modelica -- so one token decides.
        """
        if self._check(Kind.OP, ":"):
            self.at += 1
            return ir.Colon()
        return self.expression()

    def _arguments(self) -> tuple[tuple[ir.Expr, ...], tuple[tuple[str, ir.Expr], ...]]:
        """The arguments of a call, positional first and then the ones given by name.

        Modelica requires every named argument to follow the positional ones, and every
        graphical record is written entirely by name: ``Line(points = ..., color = ...)``.
        """
        self._expect(Kind.OP, "(")
        if self._accept(Kind.OP, ")"):
            return (), ()
        args: list[ir.Expr] = []
        named: list[tuple[str, ir.Expr]] = []
        while True:
            if (
                self._check(Kind.NAME)
                and self._ahead(1).kind is Kind.OP
                and self._ahead(1).text == "="
            ):
                name = self._expect(Kind.NAME).text
                self._expect(Kind.OP, "=")
                named.append((name, self.expression()))
            elif named:
                raise self._error("a positional argument cannot follow a named one")
            else:
                args.append(self.expression())
            if len(args) == 1 and not named and self._keyword("for"):
                # `sum(x[i] for i in 1:n)` -- the brace-less spelling of a comprehension,
                # legal only as the whole argument list of a reduction call.
                args = [ir.Comprehension(args[0], self._iterators())]
                break
            if not self._accept(Kind.OP, ","):
                break
        self._expect(Kind.OP, ")")
        return tuple(args), tuple(named)

    # -- modifications --------------------------------------------------------------

    def _modification(self) -> ir.Modification:
        """``(a = 1, b(c = 2)) = 3`` -- the exact inverse of ``unparse._modification``."""
        entries: dict[str, ir.Modification] = {}
        if self._accept(Kind.OP, "(") and not self._accept(Kind.OP, ")"):
            while True:
                self._accept(Kind.KEYWORD, "each")
                self._accept(Kind.KEYWORD, "final")
                if self._check(Kind.KEYWORD, "redeclare"):
                    name, entry = self._redeclaration()
                    entries[name] = entry
                else:
                    self._refuse_unsupported()
                    name = self._type_name()
                    entries[name] = self._modification()
                    self._string_comment()
                if not self._accept(Kind.OP, ","):
                    break
            self._expect(Kind.OP, ")")
        value: ir.Expr | None = None
        if self._accept(Kind.OP, "=") or self._accept(Kind.OP, ":="):
            value = self.expression()
        return ir.Modification(value, entries)

    def _redeclared_element(self) -> ir.Component:
        """``redeclare Capacitor element;`` standing on its own in a class body.

        The same words as a redeclaration inside a modification and a different place, so
        the same reading with a semicolon at the end. It is returned as a declaration rather
        than filed with the components, because it declares nothing: it says an inherited
        component is now a different class, and `ir.Class.redeclarations` is where that
        lives.
        """
        self._expect(Kind.KEYWORD, "redeclare")
        self._accept(Kind.KEYWORD, "each")
        self._accept(Kind.KEYWORD, "final")
        again = self._accept(Kind.KEYWORD, "replaceable") is not None
        if self._looks_like_class():
            raise self._error(
                "redeclaring a class as an element is not supported yet; "
                "only a component can be replaced"
            )
        component = self._one_component(replaceable=again)
        self._annotation()
        self._expect(Kind.OP, ";")
        return component

    def _redeclaration(self) -> tuple[str, ir.Modification]:
        """``redeclare Capacitor element(C = 0.01)``, or ``redeclare package M = Water``.

        Returns the name being replaced and a modification carrying the whole replacement
        declaration. The name is the key the enclosing modification files it under, which
        is what lets flattening find it beside the ordinary entries rather than in a second
        table that would have to be kept in step with them.
        """
        self._expect(Kind.KEYWORD, "redeclare")
        self._accept(Kind.KEYWORD, "each")
        self._accept(Kind.KEYWORD, "final")
        again = self._accept(Kind.KEYWORD, "replaceable") is not None
        if self._looks_like_class():
            replacement = self.class_definition(terminated=False)
            if again:
                replacement = replace(replacement, replaceable=True)
            return replacement.name, ir.Modification(redeclare=replacement)
        component = self._one_component(replaceable=again)
        return component.name, ir.Modification(redeclare=component)

    def _one_component(self, *, replaceable: bool) -> ir.Component:
        """A single declaration without its semicolon -- what a redeclaration replaces with."""
        type_name = self._type_name()
        name = self._expect(Kind.NAME).text
        dimensions = self._bracketed()
        modification = self._modification()
        constraining = self._constraining()
        return ir.Component(
            name=name,
            type_name=type_name,
            dimensions=dimensions,
            modification=modification,
            replaceable=replaceable,
            constraining=constraining,
            comment=self._string_comment(),
        )

    def _enumeration(self, name: str, *, partial: bool, terminated: bool) -> ir.Class:
        """``type Mode = enumeration(day, night) "comment";`` -- a new type, not an alias.

        Read where a short class definition's base would be, because that is where Modelica
        writes it, and returned as a ``type`` with no ``extends``: an enumeration renames
        nothing. Each literal may carry its own description string, which is read and
        dropped -- the printer has nowhere to put it, and the round trip is the contract.
        """
        self._expect(Kind.KEYWORD, "enumeration")
        self._expect(Kind.OP, "(")
        literals: list[str] = []
        if not self._check(Kind.OP, ")"):
            while True:
                literals.append(self._expect(Kind.NAME).text)
                self._string_comment()
                if not self._accept(Kind.OP, ","):
                    break
        self._expect(Kind.OP, ")")
        if not literals:
            raise self._error(f"enumeration {name!r} has no literals")
        comment = self._string_comment()
        annotation = self._annotation()
        if terminated:
            self._expect(Kind.OP, ";")
        return ir.Class(
            name=name,
            restriction="type",
            partial=partial,
            comment=comment,
            annotation=annotation,
            literals=tuple(literals),
        )

    def _import(self) -> list[ir.Import]:
        """``import A.B.C;``, ``import D = A.B.C;``, ``import A.B.*;`` or ``import A.B.{C, D};``.

        The renaming form is told apart by looking one token past the name: ``import D =``
        can only be a rename, and ``import D.E`` can only be a path, so a single token of
        lookahead settles it without backtracking. The list form brings several names from one
        package in a single clause, each the same as ``import A.B.C``, and it becomes one import
        per name -- so everything below the parser sees only the three forms it already knew.
        """
        self._expect(Kind.KEYWORD, "import")
        alias = ""
        if self._check(Kind.NAME) and self._ahead(1).text == "=":
            alias = self._expect(Kind.NAME).text
            self._expect(Kind.OP, "=")
        # Read the path a segment at a time rather than with `_type_name`, so a `.{...}` member list
        # is caught where the dotted-name reader would consume the dot and choke on the brace.
        self._accept(Kind.OP, ".")  # a leading dot is global scope; nothing to record
        parts = [self._expect(Kind.NAME).text]
        members: list[str] = []
        while self._accept(Kind.OP, "."):
            if self._check(Kind.OP, "{"):
                members = self._import_members()
                break
            parts.append(self._expect(Kind.NAME).text)
        path = ".".join(parts)
        # `.*` is one token: the lexer reads it as Modelica's element-wise product, which
        # is what it is everywhere except here.
        wildcard = self._accept(Kind.OP, ".*") is not None
        if (wildcard or members) and alias:
            raise self._error("an 'import' cannot both rename and use a wildcard or a list")
        if wildcard and members:
            raise self._error("an 'import' cannot be both a wildcard and a list")
        self._string_comment()
        self._annotation()
        self._expect(Kind.OP, ";")
        if members:
            return [ir.Import(f"{path}.{member}") for member in members]
        return [ir.Import(path, alias, wildcard=wildcard)]

    def _import_members(self) -> list[str]:
        """The ``{C, D, E}`` of a multiple import -- one or more names between braces."""
        self._expect(Kind.OP, "{")
        names = [self._expect(Kind.NAME).text]
        while self._accept(Kind.OP, ","):
            names.append(self._expect(Kind.NAME).text)
        self._expect(Kind.OP, "}")
        return names

    def _replaceable(self) -> tuple[ir.Class | None, list[ir.Component]]:
        """The element after a ``replaceable``: a class definition, or a component clause.

        Returns whichever of the two it was, because the caller files them in different
        lists. The short class form reads its own constraining clause -- it owns the
        semicolon that follows it -- so only the long form can still have one waiting.
        """
        if not self._looks_like_class():
            return None, self._component_clause(replaceable=True)
        definition = self.class_definition()
        constraint = definition.constraining or self._constraining()
        return replace(definition, replaceable=True, constraining=constraint), []

    def _constraining(self) -> str:
        """``constrainedby TwoPin`` -- the interface a replacement has to provide.

        Its own class-modification is read and dropped. Modelica lets a constraining clause
        carry default modifications for the replacement; nothing here uses them, and reading
        them is what stops the parser mistaking the following text for something else.
        """
        if not self._accept(Kind.KEYWORD, "constrainedby"):
            return ""
        name = self._type_name()
        self._modification()
        return name

    # -- equations ------------------------------------------------------------------

    def _equations(self) -> list[ir.Equation]:
        """Everything up to the next section keyword or the end of the class."""
        found: list[ir.Equation] = []
        while not (self._check(Kind.END) or self._keyword(*_SECTION_END)):
            found.append(self.equation())
        return found

    def equation(self) -> ir.Equation:  # noqa: PLR0911 -- one return per equation form
        """Parse one equation, including its terminating semicolon."""
        if self.token.kind is Kind.KEYWORD:
            match self.token.text:
                case "connect":
                    return self._connect()
                case "when":
                    return self._when()
                case "if":
                    return self._if_equation()
                case "for":
                    return self._for_equation()
                case _:
                    self._refuse_unsupported()
        where = self._at(self.token)
        token = self.token
        ahead = self._ahead(1)
        if (
            token.kind is Kind.NAME
            and token.text == "assert"
            and ahead.kind is Kind.OP
            and ahead.text == "("
        ):
            return self._assert_equation(where)
        if self._check(Kind.OP, "("):
            outputs = self._output_list()
            if outputs is not None:
                return self._call_equation(outputs, where)
        lhs = self.expression()
        if isinstance(lhs, ir.Call) and self._check(Kind.OP, ";"):
            # `Connections.root(a.b);` or `terminate("done");` -- a function call as an
            # equation, specification 8.3.1 with an empty output list. The semicolon is
            # the tell: an `=` would have made it the left side of an `Equal`.
            self._terminate()
            return ir.CallEquation((), lhs, where)
        self._expect(Kind.OP, "=")
        rhs = self.expression()
        self._terminate()
        return ir.Equal(lhs, rhs, where)

    def _output_list(self) -> list[ir.Ref | None] | None:
        """``(a, b)`` at the head of an equation, or ``None`` if this was a bracket.

        An equation may open with a bracket for two reasons -- an output list, or an
        expression the writer parenthesised -- and one token of lookahead does not separate
        them. This reads the list and *rewinds* when what follows is not ``=``, which is the
        only honest way round: ``(a + b) = c`` is a legal equation and must not be mistaken
        for a list of one output.
        """
        mark = self.at
        self.at += 1
        outputs: list[ir.Ref | None] = []
        while True:
            if self._check(Kind.OP, ",") or self._check(Kind.OP, ")"):
                outputs.append(None)  # a skipped result: `(, b) = f(x)`
            elif self.token.kind is Kind.NAME:
                where = self._at(self.token)
                outputs.append(ir.Ref(self._dotted(), where))
            else:
                self.at = mark
                return None
            if self._accept(Kind.OP, ","):
                continue
            break
        if not self._accept(Kind.OP, ")") or not self._check(Kind.OP, "="):
            self.at = mark
            return None
        return outputs

    def _call_equation(self, outputs: list[ir.Ref | None], where: ir.Span) -> ir.CallEquation:
        """The right side has to be one call. Anything else is not this equation form."""
        self._expect(Kind.OP, "=")
        rhs = self.expression()
        if not isinstance(rhs, ir.Call):
            raise self._error("an output list is assigned from a function call, not an expression")
        self._terminate()
        return ir.CallEquation(tuple(outputs), rhs, where)

    def _for_equation(self) -> ir.ForEquation:
        """``for i in 1:n loop ... end for;`` -- read one, with one index.

        One index and one range, which is the shape a method of lines is written in. Modelica
        allows several indices in one header; that is refused by name here rather than read as
        the first of them.
        """
        self._expect(Kind.KEYWORD, "for")
        index = self._expect(Kind.NAME).text
        if self._check(Kind.OP, ","):
            raise self._error("a 'for' over several indices is not supported yet")
        self._expect(Kind.KEYWORD, "in")
        over = self.expression()
        self._expect(Kind.KEYWORD, "loop")
        body: list[ir.Equation] = []
        while not self._keyword("end"):
            if self._check(Kind.END):
                raise self._error("unterminated 'for' equation: expected 'end for;'")
            body.append(self.equation())
        self._expect(Kind.KEYWORD, "end")
        self._expect(Kind.KEYWORD, "for")
        self._terminate()
        return ir.ForEquation(index, over, tuple(body))

    def _assert_equation(self, where: ir.Span) -> ir.Assert:
        """``assert(condition, "message");`` -- the one call the equation grammar names.

        The third argument, positional or written ``level = ...``, is the assertion level.
        Anything else in the call is a mistake worth naming here rather than a shape for
        flattening to trip over.
        """
        self.at += 1
        args, named = self._arguments()
        self._terminate()
        extras = dict(named)
        level = extras.pop("level", None)
        positional = list(args)
        # Two shapes are legal: (condition, message) and (condition, message, level).
        condition_and_message = 2
        if len(positional) == condition_and_message + 1 and level is None:
            level = positional.pop()
        if len(positional) != condition_and_message or extras:
            raise self._error(
                "assert takes a condition, a message, and optionally a level",
            )
        return ir.Assert(positional[0], positional[1], level, where)

    def _connect(self) -> ir.Connect:
        self._expect(Kind.KEYWORD, "connect")
        self._expect(Kind.OP, "(")
        a = self._endpoint()
        self._expect(Kind.OP, ",")
        b = self._endpoint()
        self._expect(Kind.OP, ")")
        return ir.Connect(a, b, self._terminate())

    def _endpoint(self) -> ir.Ref | ir.Subscript:
        """One side of a ``connect``, which is a component and not an expression.

        A trailing subscript names one element of an array of connectors --
        ``connect(p[k], q)`` -- and parses to a :class:`~modelica.ir.Subscript` on the
        reference. A subscript anywhere but the end is still refused by name.
        """
        reference = self._reference()
        if self._check(Kind.OP, "["):
            indices = self._bracketed()
            if self._check(Kind.OP, "."):
                raise self._error(
                    "a subscript inside a connect endpoint must be on the last name",
                )
            return ir.Subscript(reference, indices)
        return reference

    def _when(self) -> ir.When:
        opening = self.token
        branches: list[ir.WhenBranch] = []
        keyword = "when"
        while self._accept(Kind.KEYWORD, keyword):
            condition = self.expression()
            self._expect(Kind.KEYWORD, "then")
            body: list[ir.Statement] = []
            while not self._keyword("elsewhen", "end"):
                if self._check(Kind.END):
                    raise self._error("unterminated 'when': expected 'end when;'", opening)
                # A `when` body is a run of statements, and may hold an `if` or a `for` -- the
                # `Blocks.Math` extremum blocks latch `if u < pre(...) then ...` inside `when`. The
                # algorithm-statement parser reads the compound forms; the simple ones fall through
                # it unchanged, so an ordinary assignment body is read exactly as before.
                body.append(self._algorithm_statement())
            branches.append(ir.WhenBranch(condition, tuple(body)))
            keyword = "elsewhen"
        self._expect(Kind.KEYWORD, "end")
        self._expect(Kind.KEYWORD, "when")
        self._terminate()
        return ir.When(tuple(branches))

    def _statement(self) -> ir.Statement:
        """The body of a ``when``: an assignment, or ``reinit``.

        The simple half. A ``when`` sits in an equation section, where a body of ordinary
        statements is all the specification allows; the compound forms belong to an
        ``algorithm`` section and are read by :meth:`_algorithm_statement`.
        """
        if (
            self._check(Kind.NAME, "assert")
            and self._ahead(1).kind is Kind.OP
            and self._ahead(1).text == "("
        ):
            return self._assert_equation(self._at(self.token))
        if self._check(Kind.OP, "("):
            # `(a, b) := f(x);` reads two results from one call. Inside a `when` this is an
            # equation section, where the same multi-output form is written `(a, b) = f(x)` --
            # `Modelica.Blocks.Interfaces.PartialNoise` draws its random pair that way -- so
            # either operator is accepted here, exactly as the scalar assignment below does.
            where = self._at(self.token)
            self.at += 1
            targets: list[ir.Ref | None] = []
            while True:
                if self._check(Kind.OP, ",") or self._check(Kind.OP, ")"):
                    targets.append(None)
                else:
                    targets.append(self._reference())
                if not self._accept(Kind.OP, ","):
                    break
            self._expect(Kind.OP, ")")
            self._assignment_operator()
            call = self.expression()
            if not isinstance(call, ir.Call):
                raise self._error("an output list is assigned from a function call")
            self._terminate()
            return ir.CallStatement(tuple(targets), call, where)
        if self._check(Kind.NAME, "reinit") and self._ahead(1).text == "(":
            self.at += 2
            target = self._reference()
            self._expect(Kind.OP, ",")
            value = self.expression()
            self._expect(Kind.OP, ")")
            self._terminate()
            return ir.Reinit(target, value)
        target = self._reference()
        if self._check(Kind.OP, "("):
            # `isValidTable(table);` -- a bare call as a statement, the twin of the bare
            # call equation, and the same empty output list.
            where = self._at(self.token)
            args, named = self._arguments()
            self._terminate()
            return ir.CallStatement((), ir.Call(target.name, args, named), where)
        self._assignment_operator()
        value = self.expression()
        self._terminate()
        return ir.Assign(target, value)

    def _assignment_operator(self) -> None:
        """Accept `:=` (as in an algorithm) or `=` (the equation-section `when` spelling)."""
        if self._accept(Kind.OP, ":=") is None:
            self._expect(Kind.OP, "=")

    # -- algorithm sections -----------------------------------------------------------

    def _statements(self, *, until: frozenset[str] | set[str]) -> list[ir.Statement]:
        """A run of statements, stopping before ``until`` without consuming it."""
        found: list[ir.Statement] = []
        while not (self._check(Kind.END) or self._keyword(*until)):
            found.append(self._algorithm_statement())
        return found

    def _algorithm_statement(self) -> ir.Statement:  # noqa: PLR0911 -- one per form
        """One statement of an ``algorithm`` section, compound forms included."""
        if self.token.kind is Kind.KEYWORD:
            match self.token.text:
                case "if":
                    return self._if_statement()
                case "for":
                    return self._for_statement()
                case "while":
                    return self._while_statement()
                case "when":
                    return self._when_statement()
                case "break":
                    self.at += 1
                    self._terminate()
                    return ir.Break()
                case "return":
                    self.at += 1
                    self._terminate()
                    return ir.Return()
                case _:
                    self._refuse_unsupported()
        return self._statement()

    def _when_statement(self) -> ir.When:
        """``when`` in an algorithm section: the same node, the other spelling.

        The IR reuses :class:`~modelica.ir.When` -- branches of statements either way --
        and the *section* decides how an assignment inside it prints, exactly as the
        existing note on ``:=`` versus ``=`` says. The bodies here are full algorithm
        statements, where a when-equation's body is the simple half.
        """
        opening = self.token
        branches: list[ir.WhenBranch] = []
        keyword = "when"
        while self._accept(Kind.KEYWORD, keyword):
            condition = self.expression()
            self._expect(Kind.KEYWORD, "then")
            body: list[ir.Statement] = []
            while not self._keyword("elsewhen", "end"):
                if self._check(Kind.END):
                    raise self._error("unterminated 'when': expected 'end when;'", opening)
                body.append(self._algorithm_statement())
            branches.append(ir.WhenBranch(condition, tuple(body)))
            keyword = "elsewhen"
        self._expect(Kind.KEYWORD, "end")
        self._expect(Kind.KEYWORD, "when")
        self._terminate()
        return ir.When(tuple(branches))

    def _if_statement(self) -> ir.IfStatement:
        opening = self.token
        branches: list[ir.StatementBranch] = []
        keyword = "if"
        while self._accept(Kind.KEYWORD, keyword):
            condition = self.expression()
            self._expect(Kind.KEYWORD, "then")
            branches.append(
                ir.StatementBranch(condition, tuple(self._loop_body(opening, "if"))),
            )
            keyword = "elseif"
        otherwise: tuple[ir.Statement, ...] = ()
        if self._accept(Kind.KEYWORD, "else"):
            otherwise = tuple(self._loop_body(opening, "if"))
        self._closing(opening, "if")
        return ir.IfStatement(tuple(branches), otherwise)

    def _for_statement(self) -> ir.ForStatement:
        opening = self._expect(Kind.KEYWORD, "for")
        index = self._expect(Kind.NAME).text
        if self._check(Kind.OP, ","):
            raise self._error("a 'for' over several indices is not supported yet")
        self._expect(Kind.KEYWORD, "in")
        over = self.expression()
        self._expect(Kind.KEYWORD, "loop")
        body = self._loop_body(opening, "for")
        self._closing(opening, "for")
        return ir.ForStatement(index, over, tuple(body))

    def _while_statement(self) -> ir.WhileStatement:
        opening = self._expect(Kind.KEYWORD, "while")
        condition = self.expression()
        self._expect(Kind.KEYWORD, "loop")
        body = self._loop_body(opening, "while")
        self._closing(opening, "while")
        return ir.WhileStatement(condition, tuple(body))

    def _loop_body(self, opening: Token, construct: str) -> list[ir.Statement]:
        """The inside of a compound statement, with the unterminated case named."""
        found: list[ir.Statement] = []
        while not self._keyword(*_STATEMENT_END):
            if self._check(Kind.END):
                raise self._close(opening, construct)
            found.append(self._algorithm_statement())
        return found

    def _closing(self, opening: Token, construct: str) -> None:
        """Consume ``end <construct>;``, or say which construct was left open.

        Without this the failure surfaces as "expected 'while', found name 'M'" at the
        class's own ``end``, which names the wrong line and the wrong problem. The body
        stopped at a keyword that closes *something*, and the only thing worth reporting is
        which loop it did not close.
        """
        if not (self._keyword("end") and self._ahead(1).text == construct):
            raise self._close(opening, construct)
        self.at += 2
        self._terminate()

    def _close(self, opening: Token, construct: str) -> ModelicaSyntaxError:
        return self._error(
            f"unterminated {construct!r}: expected 'end {construct};'",
            opening,
        )

    def _if_equation(self) -> ir.IfEquation:
        opening = self.token
        branches: list[ir.IfBranch] = []
        keyword = "if"
        while self._accept(Kind.KEYWORD, keyword):
            condition = self.expression()
            self._expect(Kind.KEYWORD, "then")
            branches.append(ir.IfBranch(condition, tuple(self._branch_body(opening))))
            keyword = "elseif"
        otherwise: tuple[ir.Equation, ...] = ()
        if self._accept(Kind.KEYWORD, "else"):
            otherwise = tuple(self._branch_body(opening))
        self._expect(Kind.KEYWORD, "end")
        self._expect(Kind.KEYWORD, "if")
        self._terminate()
        return ir.IfEquation(tuple(branches), otherwise)

    def _branch_body(self, opening: Token) -> list[ir.Equation]:
        found: list[ir.Equation] = []
        while not self._keyword("elseif", "else", "end"):
            if self._check(Kind.END):
                raise self._error("unterminated 'if': expected 'end if;'", opening)
            found.append(self.equation())
        return found

    # -- declarations ---------------------------------------------------------------

    def _component_clause(self, *, replaceable: bool = False) -> list[ir.Component]:
        """One type name, one set of prefixes, one or more names sharing them."""
        variability: ir.Variability = "continuous"
        causality: ir.Causality = "none"
        connector: ir.ConnectorKind = "none"
        scoping: ir.Scoping = "none"
        while (token := self.token).kind is Kind.KEYWORD:
            if token.text in _VARIABILITY:
                variability = _VARIABILITY[token.text]
            elif token.text in _CAUSALITY:
                causality = _CAUSALITY[token.text]
            elif token.text in _CONNECTOR:
                connector = _CONNECTOR[token.text]
            elif token.text in _SCOPING:
                if scoping != "none":
                    # `inner outer` declares one *and* refers to another, which is a third
                    # rule rather than the two together. Refused rather than half-read.
                    raise self._error("'inner outer' is not supported yet")
                scoping = _SCOPING[token.text]
            elif token.text == "final":
                # Accepted and discarded, as every other `final` in this parser is:
                # finality forbids later modification, and nothing downstream modifies
                # further, so there is nothing yet for the flag to forbid.
                pass
            else:
                break
            self.at += 1
        self._refuse_unsupported()

        type_name = self._type_name()
        # An array dimension may sit on the type, shared by every name in the clause:
        # `FixedHeatFlow[n] heat` is `FixedHeatFlow heat[n]`. The specification puts a name's own
        # dimensions before the type's (`Real[3] x[2]` is `x[2, 3]`), so they concatenate that way.
        type_dimensions = self._bracketed()
        declared: list[ir.Component] = []
        constraining = ""
        while True:
            named = self._expect(Kind.NAME)
            name = named.text
            dimensions = self._bracketed() + type_dimensions
            modification = self._modification()
            # `Sensor s if useSensor;` -- in a declaration the keyword can only start the
            # condition-attribute, so there is nothing to disambiguate.
            condition = self.expression() if self._accept(Kind.KEYWORD, "if") else None
            # The constraining clause belongs to the *clause*, after the last name, so a
            # declaration reads it and the loop keeps whichever one it found.
            constraining = self._constraining() or constraining
            declared.append(
                ir.Component(
                    name=name,
                    type_name=type_name,
                    span=self._at(named),
                    dimensions=dimensions,
                    modification=modification,
                    condition=condition,
                    variability=variability,
                    causality=causality,
                    connector=connector,
                    scoping=scoping,
                    replaceable=replaceable,
                    constraining=constraining,
                    comment=self._string_comment(),
                ),
            )
            if not self._accept(Kind.OP, ","):
                break
        declared = [replace(c, constraining=constraining) for c in declared]
        annotation = self._annotation()
        self._expect(Kind.OP, ";")
        # One annotation, written once, for a clause that may declare several names.
        return [replace(component, annotation=annotation) for component in declared]

    # -- classes --------------------------------------------------------------------

    def _looks_like_class(self) -> bool:
        offset = 0
        while (token := self._ahead(offset)).kind is Kind.KEYWORD and token.text in _ELEMENT_PREFIX:
            offset += 1
        # `pure`/`impure` sit in front of the kind (`pure function`), so the class keyword is a
        # token further on. `operator` is a class start in its own right -- `operator record`,
        # `operator function`, or a bare `operator '+'` overload group -- so seeing it is enough.
        if self._ahead(offset).text in ("pure", "impure"):
            offset += 1
        if self._ahead(offset).text == "operator":
            return True
        token = self._ahead(offset)
        return token.kind is Kind.KEYWORD and token.text in _RESTRICTIONS

    def class_definition(self, *, terminated: bool = True) -> ir.Class:  # noqa: C901 -- one flat
        # run of optional prefixes (encapsulated, partial, pure/impure, operator, ...) then a
        # short or long body; the branches are the grammar, not tangled logic.
        """Parse one class, short form or long form.

        ``terminated`` is false inside a class-modification, where a redeclaration is an
        element of a comma-separated list and the semicolon that ends a declaration in a
        class body has no place.
        """
        self._accept(Kind.KEYWORD, "encapsulated")
        self._accept(Kind.KEYWORD, "final")
        partial = self._accept(Kind.KEYWORD, "partial") is not None
        expandable = self._accept(Kind.KEYWORD, "expandable") is not None
        # `pure function` / `impure function`: purity is an optimisation hint (a function with
        # side effects may not be lifted out of a loop), so it is accepted and dropped -- the
        # standard library's `Modelica.Utilities` and `Modelica.Math.Random` are written with it,
        # and those files are imported widely, so refusing it blocked a slice of the library.
        if self._accept(Kind.KEYWORD, "pure") is None:
            self._accept(Kind.KEYWORD, "impure")
        # The `operator` classes so a file that defines them loads: `operator record`/`operator
        # function` parse as their plain kind, and a bare `operator '+'` -- an overload group named
        # by the operator it overloads -- reads as a package, a namespace it is never instantiated
        # from. `Modelica.Units` defines dozens of `operator record` aliases behind `SI`, and
        # `Complex.mo` the bare form; the overloading itself is refused where an operator is applied.
        operator = self._accept(Kind.KEYWORD, "operator") is not None
        keyword = self.token
        if expandable and not (keyword.kind is Kind.KEYWORD and keyword.text == "connector"):
            raise self._error("`expandable` qualifies only a connector")
        if keyword.kind is Kind.KEYWORD and keyword.text in _RESTRICTIONS:
            self.at += 1
            restriction = _RESTRICTIONS[keyword.text]
        elif operator:
            restriction = "package"  # a bare operator class: the name follows, no kind keyword
        else:
            raise self._error(f"expected a class definition, found {keyword.describe()}")
        name = self._expect(Kind.NAME).text
        if self._accept(Kind.OP, "="):
            if self._check(Kind.KEYWORD, "enumeration"):
                return self._enumeration(name, partial=partial, terminated=terminated)
            for word in ("flow", "stream", "discrete", "parameter", "constant"):
                if self._keyword(word):
                    raise self._error(
                        f"'{word}' on a short class definition is not supported yet",
                    )
            causality: ir.Causality = "none"
            if self._accept(Kind.KEYWORD, "input"):
                causality = "input"
            elif self._accept(Kind.KEYWORD, "output"):
                causality = "output"
            base = ir.Extends(self._type_name(), self._modification())
            # Read here rather than by the caller, because the short form owns its own
            # semicolon and the constraining clause sits in front of it.
            constraining = self._constraining()
            comment = self._string_comment()
            annotation = self._annotation()
            if terminated:
                self._expect(Kind.OP, ";")
            return ir.Class(
                name=name,
                restriction=restriction,
                partial=partial,
                expandable=expandable,
                causality=causality,
                constraining=constraining,
                comment=comment,
                annotation=annotation,
                extends=(base,),
            )
        return self._long_definition(name, restriction, partial=partial, expandable=expandable)

    def _external_clause(self) -> None:
        """Read and drop a function's external body: ``external "C" [ref =] call [annotation];``.

        The call is in another language and cannot run here, so a function that reaches the
        simulator this way is refused where a call would inline it (it has no algorithm section),
        not at parse. Reading it lets the standard library's `Modelica.Utilities`, and the table
        and random functions -- all external, all imported widely -- load.
        """
        self._accept(Kind.STRING)  # the language, "C" or "FORTRAN 77" -- optional
        if not self._check(Kind.OP, ";") and not self._check(Kind.KEYWORD, "annotation"):
            self.expression()  # `ref` (the output) or the call itself
            if self._accept(Kind.OP, "="):
                self.expression()  # the external call, when `ref =` preceded it
        self._annotation()
        self._terminate()

    def _long_definition(  # noqa: C901, PLR0912, PLR0915 -- one flat dispatch, one arm per
        # element kind. Modelica has eleven of them and they share nothing but the loop that
        # reads them; splitting the arms across helpers to satisfy three counters would put
        # the grammar in one place and the reading of it in another.
        self,
        name: str,
        restriction: ir.Restriction,
        *,
        partial: bool,
        expandable: bool = False,
    ) -> ir.Class:
        comment = self._string_comment()
        nested: list[ir.Class] = []
        imports: list[ir.Import] = []
        extends: list[ir.Extends] = []
        redeclarations: list[ir.Component] = []
        components: list[ir.Component] = []
        equations: list[ir.Equation] = []
        initial_equations: list[ir.Equation] = []
        algorithms: list[tuple[ir.Statement, ...]] = []
        initial_algorithms: list[tuple[ir.Statement, ...]] = []
        annotation = ir.EMPTY

        while not self._keyword("end"):
            if self._check(Kind.END):
                raise self._error(f"unterminated class {name!r}: expected 'end {name};'")
            if self._accept(Kind.OP, ";"):
                continue
            if self._accept(Kind.KEYWORD, "public") or self._accept(Kind.KEYWORD, "protected"):
                continue
            if self._check(Kind.KEYWORD, "annotation"):
                # A class may write more than one, and they mean one merged annotation:
                # `Icon(...)` in the first and `Documentation(...)` in the second is the
                # same class as both in one.
                annotation = _merge(annotation, self._annotation())
                self._expect(Kind.OP, ";")
                continue
            if self._accept(Kind.KEYWORD, "initial"):
                if self._accept(Kind.KEYWORD, "algorithm"):
                    initial_algorithms.append(tuple(self._statements(until=_SECTION_END)))
                    continue
                self._expect(Kind.KEYWORD, "equation")
                initial_equations += self._equations()
                continue
            if self._accept(Kind.KEYWORD, "equation"):
                equations += self._equations()
                continue
            if self._accept(Kind.KEYWORD, "algorithm"):
                # Appended as its own section, never merged with the one before it: see
                # `ir.Class.algorithms` for why the boundary carries information.
                algorithms.append(tuple(self._statements(until=_SECTION_END)))
                continue
            if self._check(Kind.KEYWORD, "import"):
                imports.extend(self._import())
                continue
            if self._accept(Kind.KEYWORD, "extends"):
                extends.append(ir.Extends(self._type_name(), self._modification()))
                self._terminate()
                continue
            if self._check(Kind.KEYWORD, "redeclare"):
                redeclarations.append(self._redeclared_element())
                continue
            if self._accept(Kind.KEYWORD, "replaceable"):
                definition, declared = self._replaceable()
                if definition is not None:
                    nested.append(definition)
                components += declared
                continue
            if self._accept(Kind.KEYWORD, "external"):
                self._external_clause()
                continue
            if self._looks_like_class():
                nested.append(self.class_definition())
                continue
            self._refuse_unsupported()
            components += self._component_clause()

        self._expect(Kind.KEYWORD, "end")
        closing = self._expect(Kind.NAME)
        if closing.text != name:
            raise self._error(f"class {name!r} is closed by 'end {closing.text}'", closing)
        self._expect(Kind.OP, ";")
        return ir.Class(
            name=name,
            restriction=restriction,
            partial=partial,
            expandable=expandable,
            comment=comment,
            imports=tuple(imports),
            extends=tuple(extends),
            redeclarations=tuple(redeclarations),
            components=tuple(components),
            equations=tuple(equations),
            initial_equations=tuple(initial_equations),
            initial_algorithms=tuple(initial_algorithms),
            algorithms=tuple(algorithms),
            classes=tuple(nested),
            annotation=annotation,
        )

    def program(self) -> ir.Program:
        """Parse a whole file: an optional ``within`` clause and the classes after it."""
        if self._accept(Kind.KEYWORD, "within"):
            if not self._check(Kind.OP, ";"):
                self._dotted()
            self._expect(Kind.OP, ";")
        classes: list[ir.Class] = []
        while not self._check(Kind.END):
            classes.append(self.class_definition())
        return ir.Program(tuple(classes))


def parse(source: str, filename: str = "<modelica>") -> ir.Program:
    """Parse Modelica source text into an :class:`~modelica.ir.Program`.

    ``filename`` appears in error messages; it is otherwise unused, so a string parsed
    from memory needs no file to exist.
    """
    return _Parser(source, filename).program()


def parse_expression(source: str, filename: str = "<modelica>") -> ir.Expr:
    """Parse a single expression -- the unit the round-trip tests exercise hardest."""
    return _Parser(source, filename).whole_expression()
