"""Modelica source text -> :mod:`modelica.ir`.

Recursive descent, one method per grammar rule, no parser generator. The grammar is small
enough that hand-writing it costs less than a build step, and error messages written by
hand are worth more here than anywhere else: a modelling mistake should read like a
sentence, not like a shift/reduce conflict.

The target is not "whatever Modelica allows" but **exactly what**
:mod:`modelica.lang.unparse` **can emit**. That is a checkable specification -- every tree
the printer can produce must parse back to itself -- and it keeps the surface honest.
Constructs outside it (arrays, ``for``, ``algorithm``, function bodies, ``redeclare``)
raise a "not supported yet" error rather than being quietly misread; a parser that shrugs
at syntax it does not implement produces a model that silently means something else.

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
    "*": "*",
    "/": "/",
    "^": "^",
}
_SIGN: Final[dict[str, ir.UnaryOp]] = {"+": "+", "-": "-"}

if set(_INFIX) | {"not"} != set(BINARY):  # pragma: no cover - guards against drift
    raise AssertionError("_INFIX and precedence.BINARY disagree about which operators exist")

# Keywords that end a list of equations: the class itself, or the next section.
_SECTION_END: Final = frozenset(
    {"end", "equation", "initial", "algorithm", "public", "protected", "annotation"},
)
_ELEMENT_PREFIX: Final = frozenset({"encapsulated", "partial", "final"})
_OPEN: Final = frozenset("([{")
_CLOSE: Final = frozenset(")]}")

# Said plainly and early, rather than misparsed and blamed on the solver later.
_UNSUPPORTED: Final[dict[str, str]] = {
    "algorithm": "algorithm sections are not supported yet",
    "break": "'break' is not supported yet",
    "enumeration": "enumerations are not supported yet",
    "external": "external functions are not supported yet",
    "for": "'for' loops and array equations are not supported yet",
    "import": "'import' is not supported yet; refer to classes by their full name",
    "inner": "'inner'/'outer' elements are not supported yet",
    "operator": "operator overloading is not supported yet",
    "outer": "'inner'/'outer' elements are not supported yet",
    "redeclare": "'redeclare' is not supported yet",
    "replaceable": "'replaceable' elements are not supported yet",
    "return": "'return' is not supported yet",
    "while": "'while' loops are not supported yet",
}


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
        return ir.Ref(self._dotted())

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
        """Parse one expression."""
        if self._keyword("if"):
            return self._if_expression()
        return self._binary(IF + 1)

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

    def _primary(self) -> ir.Expr:  # noqa: C901, PLR0911 -- one case per token
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
                    raise self._error("output expression lists are not supported yet")
                self._expect(Kind.OP, ")")
                return inner
            case Kind.OP if token.text == "{":
                return self._array()
            case Kind.OP if token.text == "[":
                raise self._error("matrix expressions are not supported yet")
            case Kind.KEYWORD if token.text in ("true", "false"):
                self.at += 1
                return ir.BoolLiteral(token.text == "true")
            case Kind.KEYWORD if token.text in ("der", "initial"):
                self.at += 1
                args, named = self._arguments()
                return ir.Call(token.text, args, named)
            case Kind.NAME:
                parts = self._dotted()
                if self._check(Kind.OP, "("):
                    args, named = self._arguments()
                    return ir.Call(".".join(parts), args, named)
                return ir.Ref(parts)
            case _:
                self._refuse_unsupported()
        raise self._error(f"expected an expression, found {token.describe()}")

    def _array(self) -> ir.Array:
        """``{a, b, c}``, nested as deep as it goes: a point is one, an extent is two."""
        self._expect(Kind.OP, "{")
        if self._accept(Kind.OP, "}"):
            return ir.Array(())
        elements: list[ir.Expr] = [self.expression()]
        while self._accept(Kind.OP, ","):
            elements.append(self.expression())
        self._expect(Kind.OP, "}")
        return ir.Array(tuple(elements))

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

    # -- equations ------------------------------------------------------------------

    def _equations(self) -> list[ir.Equation]:
        """Everything up to the next section keyword or the end of the class."""
        found: list[ir.Equation] = []
        while not (self._check(Kind.END) or self._keyword(*_SECTION_END)):
            found.append(self.equation())
        return found

    def equation(self) -> ir.Equation:
        """Parse one equation, including its terminating semicolon."""
        if self.token.kind is Kind.KEYWORD:
            match self.token.text:
                case "connect":
                    return self._connect()
                case "when":
                    return self._when()
                case "if":
                    return self._if_equation()
                case _:
                    self._refuse_unsupported()
        lhs = self.expression()
        self._expect(Kind.OP, "=")
        rhs = self.expression()
        self._terminate()
        return ir.Equal(lhs, rhs)

    def _connect(self) -> ir.Connect:
        self._expect(Kind.KEYWORD, "connect")
        self._expect(Kind.OP, "(")
        a = self._reference()
        self._expect(Kind.OP, ",")
        b = self._reference()
        self._expect(Kind.OP, ")")
        return ir.Connect(a, b, self._terminate())

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
                body.append(self._statement())
            branches.append(ir.WhenBranch(condition, tuple(body)))
            keyword = "elsewhen"
        self._expect(Kind.KEYWORD, "end")
        self._expect(Kind.KEYWORD, "when")
        self._terminate()
        return ir.When(tuple(branches))

    def _statement(self) -> ir.Statement:
        """The body of a ``when``: an assignment, or ``reinit``."""
        if self._check(Kind.NAME, "reinit") and self._ahead(1).text == "(":
            self.at += 2
            target = self._reference()
            self._expect(Kind.OP, ",")
            value = self.expression()
            self._expect(Kind.OP, ")")
            self._terminate()
            return ir.Reinit(target, value)
        target = self._reference()
        if self._accept(Kind.OP, ":=") is None:
            self._expect(Kind.OP, "=")
        value = self.expression()
        self._terminate()
        return ir.Assign(target, value)

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

    def _component_clause(self) -> list[ir.Component]:
        """One type name, one set of prefixes, one or more names sharing them."""
        variability: ir.Variability = "continuous"
        causality: ir.Causality = "none"
        connector: ir.ConnectorKind = "none"
        while (token := self.token).kind is Kind.KEYWORD:
            if token.text in _VARIABILITY:
                variability = _VARIABILITY[token.text]
            elif token.text in _CAUSALITY:
                causality = _CAUSALITY[token.text]
            elif token.text in _CONNECTOR:
                connector = _CONNECTOR[token.text]
            else:
                break
            self.at += 1
        self._refuse_unsupported()

        type_name = self._type_name()
        declared: list[ir.Component] = []
        while True:
            name = self._expect(Kind.NAME).text
            if self._check(Kind.OP, "["):
                raise self._error("arrays are not supported yet")
            declared.append(
                ir.Component(
                    name=name,
                    type_name=type_name,
                    modification=self._modification(),
                    variability=variability,
                    causality=causality,
                    connector=connector,
                    comment=self._string_comment(),
                ),
            )
            if not self._accept(Kind.OP, ","):
                break
        annotation = self._annotation()
        self._expect(Kind.OP, ";")
        # One annotation, written once, for a clause that may declare several names.
        return [replace(component, annotation=annotation) for component in declared]

    # -- classes --------------------------------------------------------------------

    def _looks_like_class(self) -> bool:
        offset = 0
        while (token := self._ahead(offset)).kind is Kind.KEYWORD and token.text in _ELEMENT_PREFIX:
            offset += 1
        token = self._ahead(offset)
        return token.kind is Kind.KEYWORD and token.text in _RESTRICTIONS

    def class_definition(self) -> ir.Class:
        """Parse one class, short form or long form."""
        self._accept(Kind.KEYWORD, "encapsulated")
        self._accept(Kind.KEYWORD, "final")
        partial = self._accept(Kind.KEYWORD, "partial") is not None
        keyword = self.token
        if keyword.kind is not Kind.KEYWORD or keyword.text not in _RESTRICTIONS:
            raise self._error(f"expected a class definition, found {keyword.describe()}")
        self.at += 1
        restriction = _RESTRICTIONS[keyword.text]
        name = self._expect(Kind.NAME).text
        if self._accept(Kind.OP, "="):
            base = ir.Extends(self._type_name(), self._modification())
            comment = self._string_comment()
            annotation = self._annotation()
            self._expect(Kind.OP, ";")
            return ir.Class(
                name=name,
                restriction=restriction,
                partial=partial,
                comment=comment,
                annotation=annotation,
                extends=(base,),
            )
        return self._long_definition(name, restriction, partial=partial)

    def _long_definition(  # noqa: C901 -- a flat dispatch over element kinds
        self,
        name: str,
        restriction: ir.Restriction,
        *,
        partial: bool,
    ) -> ir.Class:
        comment = self._string_comment()
        nested: list[ir.Class] = []
        extends: list[ir.Extends] = []
        components: list[ir.Component] = []
        equations: list[ir.Equation] = []
        initial_equations: list[ir.Equation] = []
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
                self._expect(Kind.KEYWORD, "equation")
                initial_equations += self._equations()
                continue
            if self._accept(Kind.KEYWORD, "equation"):
                equations += self._equations()
                continue
            if self._accept(Kind.KEYWORD, "extends"):
                extends.append(ir.Extends(self._type_name(), self._modification()))
                self._terminate()
                continue
            self._refuse_unsupported()
            if self._looks_like_class():
                nested.append(self.class_definition())
                continue
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
            comment=comment,
            extends=tuple(extends),
            components=tuple(components),
            equations=tuple(equations),
            initial_equations=tuple(initial_equations),
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
