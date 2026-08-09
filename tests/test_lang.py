"""The two directions of :mod:`modelica.lang`, checked against each other.

The parser and the printer share one precedence table, which is what makes
``parse(unparse(x)) == x`` a structural property rather than a coincidence: the level that
decides whether the printer emits a bracket is the level that decides where the parser
stops. These tests are the evidence for that claim.

Three kinds of check live here. **Round trips** over the example corpus and over a table of
expressions -- the property itself. **Precedence** cases pinned as trees, so the one
deliberate deviation from the specification (see :mod:`modelica.lang.precedence`) is
written down where it can fail rather than only in a docstring. And **refusals**: every
construct the printer cannot emit must be rejected by name, because a parser that quietly
misreads ``for`` is worse than one that has never heard of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelica import ir
from modelica.lang import (
    ModelicaSyntaxError,
    parse,
    parse_expression,
    tokenize,
    unparse,
    unparse_expr,
)
from modelica.lang.lexer import Kind

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
CORPUS = sorted(EXAMPLES.glob("*/*.mo"))


def _ids(paths: list[Path]) -> list[str]:
    return [p.stem for p in paths]


# --------------------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------------------


def test_corpus_is_not_empty() -> None:
    assert CORPUS, f"no .mo files found under {EXAMPLES}"


@pytest.mark.parametrize("source", CORPUS, ids=_ids(CORPUS))
def test_corpus_round_trips_through_text(source: Path) -> None:
    """Print a parsed model and read it back: the same value, not merely a similar one."""
    once = parse(source.read_text(encoding="utf-8"), source.name)
    twice = parse(unparse(once), source.name)
    assert twice == once


@pytest.mark.parametrize("source", CORPUS, ids=_ids(CORPUS))
def test_printing_is_a_fixed_point(source: Path) -> None:
    """The printer's output is already normalised -- printing it again changes nothing."""
    printed = unparse(parse(source.read_text(encoding="utf-8"), source.name))
    assert unparse(parse(printed, source.name)) == printed


#: Expressions in the exact spelling the printer produces. Parsing then printing must be
#: the identity on every one of them, which pins the bracketing rules from both sides.
CANONICAL = [
    "a",
    "a.b.c",
    "1",
    "1.5",
    "true",
    "false",
    '"a string"',
    "der(h)",
    "sqrt(2.0 * g * h)",
    "f()",
    "-a",
    "not b",
    "a + b",
    "a - b - c",
    "a - (b - c)",
    "a * b + c",
    "a * (b + c)",
    "(a + b) * c",
    "a / b / c",
    "a / (b / c)",
    "-a * b",
    "-(a * b)",
    "-a ^ b",
    "(-a) ^ b",
    "a ^ (b ^ c)",
    "(a ^ b) ^ c",
    "a < b and c >= d",
    "not (a and b)",
    "a or b and c",
    "(a or b) and c",
    "if a > 0.0 then b else c",
    "1.0 + (if a then 2.0 else 3.0)",
    "if a then b elseif c then d else e",
    "if a then b elseif c then d elseif e then f else g",
]


@pytest.mark.parametrize("text", CANONICAL)
def test_expression_round_trips_through_text(text: str) -> None:
    assert unparse_expr(parse_expression(text)) == text


@pytest.mark.parametrize("text", CANONICAL)
def test_expression_round_trips_through_the_tree(text: str) -> None:
    node = parse_expression(text)
    assert parse_expression(unparse_expr(node)) == node


def test_elseif_is_a_spelling_and_the_one_the_printer_picks() -> None:
    """``elseif`` is not a node: both spellings parse to the same nested tree.

    Which leaves the printer free to choose, and it chooses the chain. The nested spelling
    is what the tree literally is, but a conditional whose else-branch is another
    conditional is one decision taken in stages, and stacking parentheses to say so costs
    a reader one bracket per stage for no information. A nine-level gas schedule is the
    case that settled it.
    """
    for text in ("if a then b elseif c then d else e", "if a then b else (if c then d else e)"):
        assert parse_expression(text) == ir.IfExpr(
            A, B, ir.IfExpr(C, ir.Ref(("d",)), ir.Ref(("e",)))
        )
        assert unparse_expr(parse_expression(text)) == "if a then b elseif c then d else e"


def test_a_conditional_in_a_then_branch_still_gets_its_brackets() -> None:
    """Only the else-branch chains.

    A conditional nested in a ``then`` is a different tree and has to keep the parentheses
    that say so, or it would print as a chain and parse back as something else entirely.
    """
    node = ir.IfExpr(A, ir.IfExpr(B, C, ir.Ref(("d",))), ir.Ref(("e",)))
    printed = unparse_expr(node)
    assert printed == "if a then (if b then c else d) else e"
    assert parse_expression(printed) == node


# --------------------------------------------------------------------------------------
# Precedence, pinned as trees
# --------------------------------------------------------------------------------------

A, B, C = ir.Ref(("a",)), ir.Ref(("b",)), ir.Ref(("c",))

TREES = [
    ("a + b * c", ir.Binary("+", A, ir.Binary("*", B, C))),
    ("a * b + c", ir.Binary("+", ir.Binary("*", A, B), C)),
    ("a - b - c", ir.Binary("-", ir.Binary("-", A, B), C)),
    ("a / b * c", ir.Binary("*", ir.Binary("/", A, B), C)),
    ("a ^ b * c", ir.Binary("*", ir.Binary("^", A, B), C)),
    ("not a and b", ir.Binary("and", ir.Unary("not", A), B)),
    ("a and b or c", ir.Binary("or", ir.Binary("and", A, B), C)),
    # The deliberate deviation: the sign binds tighter than `*`, so this is `(-a) * b`
    # rather than the specification's `-(a * b)`. The two are equal in IEEE arithmetic,
    # and the tighter binding is what makes the round trip exact.
    ("-a * b", ir.Binary("*", ir.Unary("-", A), B)),
    # ...but it stays looser than `^`, which the specification does require.
    ("-a ^ b", ir.Unary("-", ir.Binary("^", A, B))),
]


@pytest.mark.parametrize(("text", "expected"), TREES, ids=[t for t, _ in TREES])
def test_precedence(text: str, expected: ir.Expr) -> None:
    assert parse_expression(text) == expected


@pytest.mark.parametrize("text", ["a ^ b ^ c", "a < b < c", "a == b == c", "a >= b >= c"])
def test_non_associative_operators_refuse_to_chain(text: str) -> None:
    """``a ^ b ^ c`` has no reading the printer could reproduce, so it is an error."""
    with pytest.raises(ModelicaSyntaxError):
        parse_expression(text)


def test_a_signed_exponent_needs_brackets() -> None:
    """``factor: primary [ "^" primary ]`` -- the specification forbids ``a ^ -b``."""
    with pytest.raises(ModelicaSyntaxError):
        parse_expression("a ^ -b")
    assert parse_expression("a ^ (-b)") == ir.Binary("^", A, ir.Unary("-", B))


# --------------------------------------------------------------------------------------
# What a model parses to
# --------------------------------------------------------------------------------------

MODEL = """
within Some.Package;

model Ball "A bouncing ball"
  parameter Real g = 9.81 "Gravity";
  parameter Real e = 0.8;
  Real h(start = 1.0, fixed = true);
  Real v(start = 0.0, fixed = true);
equation
  der(h) = v;
  der(v) = -g;
  when h <= 0.0 then
    reinit(v, -e * pre(v));
  end when;
  annotation(experiment(StopTime = 5.0));
end Ball;
"""


def test_a_model_parses_to_the_expected_value() -> None:
    ball = parse(MODEL).find("Ball")
    assert ball is not None
    assert ball.restriction == "model"
    assert ball.comment == "A bouncing ball"
    assert [c.name for c in ball.components] == ["g", "e", "h", "v"]
    assert [c.variability for c in ball.components] == 2 * ["parameter"] + 2 * ["continuous"]
    assert ball.components[0].modification.value == ir.RealLiteral(9.81)
    assert ball.components[0].comment == "Gravity"
    assert ball.components[2].modification.entries["fixed"].value == ir.BoolLiteral(value=True)
    assert len(ball.equations) == 3
    assert ball.equations[0] == ir.Equal(ir.Call("der", (ir.Ref(("h",)),)), ir.Ref(("v",)))
    when = ball.equations[2]
    assert isinstance(when, ir.When)
    assert isinstance(when.branches[0].body[0], ir.Reinit)


def test_within_is_accepted_and_dropped() -> None:
    """A `within` clause names the package a file belongs to. Nothing downstream reads it."""
    assert "within" not in unparse(parse(MODEL))


def test_an_annotation_survives_into_the_ir() -> None:
    """It used to be counted over and thrown away. It is the diagram, so it is kept.

    Where a component sits, and what a class looks like, is written in an annotation and
    nowhere else. A tool that discards it can only ever guess at a picture of the model.
    """
    source = (
        "model M\n"
        "  Real x annotation(Placement(transformation(extent = {{-10, -10}, {10, 10}})));\n"
        "equation\n"
        "  annotation(Icon(graphics = {Line(points = {{-90, 0}, {-70, 0}})}));\n"
        "end M;\n"
    )
    model = parse(source).classes[0]
    assert set(model.annotation.entries) == {"Icon"}
    assert set(model.components[0].annotation.entries) == {"Placement"}
    assert parse(unparse(parse(source))) == parse(source)


def test_a_connect_keeps_the_line_it_was_drawn_as() -> None:
    source = (
        "model M\n  Real a;\n  Real b;\nequation\n"
        "  connect(a, b) annotation(Line(points = {{0, 0}, {10, 0}}, color = {0, 0, 255}));\n"
        "end M;\n"
    )
    connect = parse(source).classes[0].equations[0]
    assert isinstance(connect, ir.Connect)
    assert set(connect.annotation.entries) == {"Line"}


CONNECTOR = """
type Voltage = Real(unit = "V");

connector Pin
  Voltage v;
  flow Real i;
end Pin;
"""


def test_a_connector_and_a_short_type_definition() -> None:
    voltage, pin = parse(CONNECTOR).classes
    assert voltage.restriction == "type"
    assert voltage.extends[0].type_name == "Real"
    assert voltage.extends[0].modification.entries["unit"].value == ir.StringLiteral("V")
    assert pin.restriction == "connector"
    assert [c.connector for c in pin.components] == ["none", "flow"]


CIRCUIT = """
partial model TwoPin
  Pin p;
  Pin n;
end TwoPin;

model Resistor
  extends TwoPin;
  parameter Real R = 1.0;
equation
  connect(p, n);
end Resistor;
"""


def test_extends_and_connect() -> None:
    two_pin, resistor = parse(CIRCUIT).classes
    assert two_pin.partial
    assert resistor.extends == (ir.Extends("TwoPin"),)
    assert resistor.equations == (ir.Connect(ir.Ref(("p",)), ir.Ref(("n",))),)


# --------------------------------------------------------------------------------------
# Refusals: unsupported constructs are named, not misread
# --------------------------------------------------------------------------------------

UNSUPPORTED = [
    ("external", "function f\nexternal;\nend f;\n"),
]
# `algorithm`, `while`, `break` and `return` used to be on this list, and so did
# `replaceable`, `redeclare`, `import`, `enumeration` and `inner`/`outer`. They are read
# now, and `tests/test_algorithm.py`, `tests/test_redeclare.py`, `tests/test_import.py`,
# `tests/test_enumeration.py` and `tests/test_scoping.py` hold them to what they mean.


@pytest.mark.parametrize(("keyword", "source"), UNSUPPORTED, ids=[k for k, _ in UNSUPPORTED])
def test_unsupported_constructs_are_refused_by_name(keyword: str, source: str) -> None:
    with pytest.raises(ModelicaSyntaxError, match=keyword):
        parse(source)


def test_an_array_literal_is_read() -> None:
    """`{1, 2}` is what every point, extent and colour in an annotation is made of."""
    assert parse_expression("{1, 2, 3}") == ir.Array(
        (ir.IntLiteral(1), ir.IntLiteral(2), ir.IntLiteral(3))
    )
    nested = parse_expression("{{-100, -100}, {100, 100}}")
    assert isinstance(nested, ir.Array)
    assert len(nested.elements) == 2


def test_a_declared_shape_is_read() -> None:
    """`Real T[n]` carries its shape as an expression, because Modelica writes it as one."""
    (cls,) = parse("model M\n  parameter Integer n = 3;\n  Real T[n];\nend M;\n").classes
    shaped = next(c for c in cls.components if c.name == "T")
    assert shaped.dimensions == (ir.Ref(("n",)),)
    scalar = next(c for c in cls.components if c.name == "n")
    assert scalar.dimensions == ()


def test_a_subscript_is_a_reference_with_indices() -> None:
    assert parse_expression("T[i]") == ir.Subscript(ir.Ref(("T",)), (ir.Ref(("i",)),))
    assert parse_expression("A[i, j + 1]") == ir.Subscript(
        ir.Ref(("A",)),
        (ir.Ref(("i",)), ir.Binary("+", ir.Ref(("j",)), ir.IntLiteral(1))),
    )


def test_a_for_equation_carries_its_index_and_its_range() -> None:
    source = (
        "model M\n"
        "  parameter Integer n = 3;\n"
        "  Real T[n];\n"
        "equation\n"
        "  for i in 1:n loop\n"
        "    T[i] = 0;\n"
        "  end for;\n"
        "end M;\n"
    )
    (cls,) = parse(source).classes
    (equation,) = cls.equations
    assert isinstance(equation, ir.ForEquation)
    assert equation.index == "i"
    assert equation.over == ir.Range(ir.IntLiteral(1), ir.Ref(("n",)))
    assert len(equation.body) == 1


def test_the_step_of_a_range_is_the_middle_term() -> None:
    """`1:2:n` steps by 2 up to n. Reading it as `start:stop:step` is the classic error."""
    source = "model M\nequation\n  for i in 1:2:n loop\n    x = i;\n  end for;\nend M;\n"
    (cls,) = parse(source).classes
    (equation,) = cls.equations
    assert isinstance(equation, ir.ForEquation)
    assert equation.over == ir.Range(ir.IntLiteral(1), ir.Ref(("n",)), ir.IntLiteral(2))


ARRAY_REFUSALS = [
    ("comprehension", "{i for i in 1:3}"),
    ("several indices", "model M\nequation\n  for i, j in 1:3 loop\n  end for;\nend M;\n"),
    ("inner subscript", "a[1].b"),
]


@pytest.mark.parametrize(("what", "source"), ARRAY_REFUSALS, ids=[w for w, _ in ARRAY_REFUSALS])
def test_the_array_syntax_not_implemented_is_refused_by_name(what: str, source: str) -> None:
    """What arrived with arrays, and what did not. Silence here would be a misread model."""
    reader = parse if source.startswith("model") else parse_expression
    with pytest.raises(ModelicaSyntaxError, match="not supported yet"):
        reader(source)


def test_named_function_arguments_are_read() -> None:
    assert parse_expression("f(x = 1)") == ir.Call("f", (), (("x", ir.IntLiteral(1)),))
    mixed = parse_expression("f(1, y = 2)")
    assert mixed == ir.Call("f", (ir.IntLiteral(1),), (("y", ir.IntLiteral(2)),))


def test_a_positional_argument_cannot_follow_a_named_one() -> None:
    with pytest.raises(ModelicaSyntaxError, match="positional"):
        parse_expression("f(x = 1, 2)")


def test_a_mismatched_end_name_is_an_error() -> None:
    with pytest.raises(ModelicaSyntaxError, match="Other"):
        parse("model M\nend Other;\n")


def test_an_error_points_at_the_offending_line() -> None:
    with pytest.raises(ModelicaSyntaxError) as caught:
        parse("model M\n  Real x\nend M;\n")
    assert caught.value.lineno == 3
    assert caught.value.filename == "<modelica>"


def test_the_filename_reaches_the_error() -> None:
    with pytest.raises(ModelicaSyntaxError) as caught:
        parse("model M\n  Real x\nend M;\n", "Ball.mo")
    assert caught.value.filename == "Ball.mo"


# --------------------------------------------------------------------------------------
# Lexer
# --------------------------------------------------------------------------------------


def test_comments_and_whitespace_never_reach_the_parser() -> None:
    tokens = tokenize("a /* block\ncomment */ + // line comment\nb")
    assert [t.text for t in tokens] == ["a", "+", "b", ""]
    assert tokens[-1].kind is Kind.END


def test_a_token_knows_where_it_came_from() -> None:
    (_first, second, *_rest) = tokenize("model\n  Real x;\n")
    assert (second.text, second.line, second.column) == ("Real", 2, 3)


def test_keywords_are_distinguished_from_names() -> None:
    kinds = {t.text: t.kind for t in tokenize("model Model der")}
    assert kinds == {"model": Kind.KEYWORD, "Model": Kind.NAME, "der": Kind.KEYWORD, "": Kind.END}


def test_a_string_token_carries_the_decoded_value() -> None:
    (token, _end) = tokenize(r'"a \"quoted\" line\nbreak"')
    assert token.kind is Kind.STRING
    assert token.text == 'a "quoted" line\nbreak'


@pytest.mark.parametrize(
    ("text", "kind"),
    [("1", Kind.INT), ("1.", Kind.REAL), ("1.5", Kind.REAL), ("1e3", Kind.REAL), (".5", Kind.REAL)],
)
def test_numbers_are_classified(text: str, kind: Kind) -> None:
    assert tokenize(text)[0].kind is kind


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('"unterminated', "unterminated string"),
        ("/* unterminated", "unterminated block comment"),
        ("a @ b", "unexpected character"),
    ],
)
def test_lexer_errors(source: str, message: str) -> None:
    with pytest.raises(ModelicaSyntaxError, match=message):
        tokenize(source)
