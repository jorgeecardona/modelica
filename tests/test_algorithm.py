"""``algorithm`` sections, from the text down to the equations they mean.

Three claims, in the order the file makes them. A section parses, prints and parses back
to the same value, and the Python front end says the same things the ``.mo`` front end
says. A ``function`` -- a class with ``input``/``output`` components and one section --
is *inlined* at a call, so what reaches the solver is an expression. And a section in a
model becomes one equation per variable it assigns, by the same symbolic execution.

Everything here therefore ends in ordinary equations, which is the point: no stage below
:mod:`modelica.ir.flatten` learns that statements exist. The corpus model that simulates
one is ``BumpStop``; what this file holds is the conversion itself and its edges.

The two spellings of an assignment are the subtlety worth a test of their own. The IR has
one :class:`~modelica.ir.Assign`, and Modelica writes it as ``:=`` in an algorithm section
and as ``=`` inside a ``when`` in an equation section. Getting that wrong produces a file
that round-trips through this toolchain and is rejected by every other one.
"""

from __future__ import annotations

import pytest

from modelica import build, ir
from modelica.ir.causalize import causalize
from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse, unparse_equation

CLIP = """\
function clip "Hold a signal between two values"
  input Real u;
  input Real lo;
  input Real hi;
  output Real y;
algorithm
  y := u;
  if y < lo then
    y := lo;
  elseif y > hi then
    y := hi;
  end if;
end clip;
"""

LOOPS = """\
model Loops "Every statement form in one place"
  Real x(start = 0, fixed = true);
  Real y;
equation
  der(x) = 1.0;
algorithm
  y := 0.0;
  for i in 1:5 loop
    y := y + i;
    if y > 6.0 then
      break;
    end if;
  end for;
algorithm
  while y < 100.0 loop
    y := y * 2.0;
  end while;
  return;
end Loops;
"""


def test_a_function_is_a_class_with_a_section() -> None:
    clip = parse(CLIP).classes[0]
    assert clip.restriction == "function"
    assert [c.causality for c in clip.components] == ["input", "input", "input", "output"]
    assert not clip.equations
    (section,) = clip.algorithms
    assert section[0] == ir.Assign(ir.Ref(("y",)), ir.Ref(("u",)))


def test_an_elseif_chain_is_one_statement_with_two_branches() -> None:
    (section,) = parse(CLIP).classes[0].algorithms
    branching = section[1]
    assert isinstance(branching, ir.IfStatement)
    # `elseif` is a second branch, not a nested statement in the `else`.
    assert len(branching.branches) == 2
    assert branching.otherwise == ()


def test_two_sections_stay_two() -> None:
    """A boundary between sections is information; merging them would lose it."""
    loops = parse(LOOPS).classes[0]
    assert len(loops.algorithms) == 2
    assert len(loops.equations) == 1


def test_every_statement_form_survives_a_round_trip() -> None:
    for source in (CLIP, LOOPS):
        program = parse(source)
        assert parse(unparse(program)) == program


def test_the_corpus_spelling_is_reproduced_exactly() -> None:
    """Not merely round-trip-equal: the printer emits the text that was read."""
    for source in (CLIP, LOOPS):
        assert unparse(parse(source)) == source


def test_the_statement_forms_are_the_ones_expected() -> None:
    first, second = parse(LOOPS).classes[0].algorithms
    assert isinstance(first[1], ir.ForStatement)
    assert isinstance(first[1].body[1], ir.IfStatement)
    assert first[1].body[1].branches[0].body == (ir.Break(),)
    assert isinstance(second[0], ir.WhileStatement)
    assert second[1] == ir.Return()


def test_an_assignment_is_written_with_walrus_in_an_algorithm() -> None:
    assert "y := u;" in CLIP
    assert unparse(parse(CLIP)).count(":=") == CLIP.count(":=")


def test_an_assignment_inside_a_when_is_still_written_with_equals() -> None:
    """The same IR node, the other spelling, because a ``when`` is an equation."""
    source = "model M\n  discrete Real d;\nequation\n  when time > 1.0 then\n    d = 1.0;\n  end when;\nend M;\n"
    assert unparse(parse(source)) == source


WHEN_WITH_IF = """\
model M
  discrete Real d;
equation
  when time > 1.0 then
    if time < 2.0 then
      d = 1.0;
    else
      d = 2.0;
    end if;
  end when;
end M;
"""


def test_an_if_inside_a_when_body_parses_and_round_trips() -> None:
    """`Blocks.Math` latches `if u < pre(...) then ...` inside a `when`; it reads and prints back."""
    program = parse(WHEN_WITH_IF)
    (when,) = program.find("M").equations  # type: ignore[union-attr]
    assert isinstance(when, ir.When)
    assert isinstance(when.branches[0].body[0], ir.IfStatement)
    assert parse(unparse(program)) == program


def test_flattening_an_if_inside_a_when_is_refused_by_name() -> None:
    with pytest.raises(FlattenError, match=r"IfStatement inside a 'when' is not supported yet"):
        flatten(parse(WHEN_WITH_IF), "M")


# --------------------------------------------------------------------------------------
# The Python front end says the same things
# --------------------------------------------------------------------------------------


# Named and documented to match the `.mo` twin above, which is the whole assertion: a
# Modelica function is lower-case, and its description string carries no full stop.
class clip(build.Function):  # noqa: N801
    """Hold a signal between two values"""  # noqa: D400

    u = build.input_(build.Real())
    lo = build.input_(build.Real())
    hi = build.input_(build.Real())
    y = build.output(build.Real())

    def algorithm(self):  # noqa: ANN201 - the front end reads the yielded statements
        yield build.assign(self.y, self.u)
        yield build.if_statement(
            self.y < self.lo,
            build.assign(self.y, self.lo),
            elseif=[(self.y > self.hi, [build.assign(self.y, self.hi)])],
        )


def test_the_two_front_ends_build_the_same_function() -> None:
    assert build.to_ir(clip) == parse(CLIP).classes[0]


def test_the_python_front_end_prints_the_same_source() -> None:
    assert build.to_modelica(clip) == CLIP


# --------------------------------------------------------------------------------------
# What is refused, and by what name
# --------------------------------------------------------------------------------------


def test_initial_algorithm_parses_now() -> None:
    """This test used to pin the refusal; the construct moved with the frontier."""
    program = parse("model M\n  Real x;\ninitial algorithm\n  x := 1.0;\nend M;\n")
    m = program.find("M")
    assert m is not None
    assert len(m.initial_algorithms) == 1


def test_an_external_body_parses_as_a_bodyless_function() -> None:
    """An external body reads now (dropped), rather than being refused at parse."""
    program = parse(
        'function f\n  input Real x;\n  output Real y;\nexternal "C" y = c(x);\nend f;\n'
    )
    f = program.find("f")
    assert f is not None
    assert f.algorithms == ()


def test_an_unterminated_loop_names_the_construct() -> None:
    with pytest.raises(ModelicaSyntaxError, match="end while"):
        parse("model M\nalgorithm\n  while true loop\n    x := 1.0;\nend M;\n")


def test_a_section_of_loops_names_the_form_that_stopped_it() -> None:
    """`Loops` uses every statement form, and the ones with no expression are refused."""
    with pytest.raises(FlattenError, match=r"the algorithm section of 'Loops' uses a 'for' loop"):
        flatten(parse(LOOPS), "Loops")


# --------------------------------------------------------------------------------------
# A section in a model is the equations it is equivalent to
# --------------------------------------------------------------------------------------

SECTION = """\
model Section "A clipped signal, written as statements instead of as an equation"
  parameter Real lo = 0.0;
  parameter Real hi = 1.0;
  Real x(start = -0.5, fixed = true);
  Real u;
  Real y;
equation
  der(x) = 1.0;
  u = 2.0 * x;
algorithm
  y := u;
  if y < lo then
    y := lo;
  elseif y > hi then
    y := hi;
  end if;
end Section;
"""


def test_a_section_becomes_one_equation_per_variable_it_assigns() -> None:
    """Three statements assigning one variable are one equation, not three."""
    assert _equation_text(SECTION, "Section") == [
        "der(x) = 1.0;",
        "u = 2.0 * x;",
        "y = if u < lo then lo elseif u > hi then hi else u;",
    ]


def test_the_model_is_balanced_after_the_conversion() -> None:
    """The count is what the balance check reads, so the conversion has to get it right."""
    flat = flatten(parse(SECTION), "Section")
    assert flat.is_balanced()


def test_what_the_section_produced_causalizes_like_any_other_equation() -> None:
    """The claim the conversion exists to make: nothing below flattening sees a statement."""
    system = causalize(flatten(parse(SECTION), "Section"))
    assert list(system.states) == ["x"]


def test_a_section_may_assign_a_component_variable() -> None:
    """A statement target can be dotted, and it is the component's variable it defines."""
    source = (
        "model Inner\n  Real v;\nend Inner;\n\n"
        "model Outer\n  Inner c;\n  Real w;\nequation\n  w = 3.0;\n"
        "algorithm\n  c.v := 2.0 * w;\nend Outer;\n"
    )
    assert _equation_text(source, "Outer") == ["w = 3.0;", "c.v = 2.0 * w;"]


def test_two_sections_each_contribute_their_own_equations() -> None:
    source = (
        "model Two\n  Real a;\n  Real b;\n  Real c;\nequation\n  a = 1.0;\n"
        "algorithm\n  b := a + 1.0;\n"
        "algorithm\n  c := b + 1.0;\nend Two;\n"
    )
    assert _equation_text(source, "Two") == ["a = 1.0;", "b = a + 1.0;", "c = b + 1.0;"]


def test_a_section_may_call_a_function() -> None:
    """Inlining runs on what the section produced, so the two features compose."""
    source = (
        "function twice\n  input Real a;\n  output Real b;\nalgorithm\n  b := 2.0 * a;\nend twice;\n\n"
        "model Uses\n  Real p;\n  Real q;\nequation\n  p = 1.5;\nalgorithm\n  q := twice(p);\nend Uses;\n"
    )
    assert _equation_text(source, "Uses") == ["p = 1.5;", "q = 2.0 * p;"]


def test_reading_a_variable_the_section_also_assigns_is_refused() -> None:
    """``y := y + 1`` has no meaning here: a continuous variable has no earlier value."""
    source = "model Bad\n  Real y;\nalgorithm\n  y := y + 1.0;\nend Bad;\n"
    with pytest.raises(FlattenError, match="reads 'y' before assigning it"):
        flatten(parse(source), "Bad")


# --------------------------------------------------------------------------------------
# Calls: a function is inlined into the expression it computes
# --------------------------------------------------------------------------------------

CALLER = """\
function clip "Hold a signal between two values"
  input Real u;
  input Real lo = 0.0;
  input Real hi;
  output Real y;
algorithm
  y := u;
  if y < lo then
    y := lo;
  elseif y > hi then
    y := hi;
  end if;
end clip;

function square
  input Real a;
  output Real b;
algorithm
  b := a * a;
end square;

model Clipped "A ramp squared and then clipped"
  Real x(start = 0, fixed = true);
  Real y;
equation
  der(x) = 1.0;
  y = clip(square(x), hi = 4.0);
end Clipped;
"""


def _equation_text(source: str, model: str) -> list[str]:
    return [line for e in flatten(parse(source), model).equations for line in unparse_equation(e)]


def test_a_call_becomes_the_expression_the_body_computes() -> None:
    """Inlined, not called: what comes out is differentiable by the existing pass."""
    assert _equation_text(CALLER, "Clipped") == [
        "der(x) = 1.0;",
        "y = if x * x < 0.0 then 0.0 elseif x * x > 4.0 then 4.0 else x * x;",
    ]
    # The `if` is a merge over the branch conditions, so `lo` reached it as its declared
    # default and `hi` as a named argument, and `square(x)` was inlined before either.


def _function(body: str, *, header: str = "  input Real u;\n  output Real y;\n") -> str:
    return f"function f\n{header}algorithm\n{body}end f;\n\nmodel M\n  Real z;\nequation\n  z = f(1.0);\nend M;\n"


REFUSALS = [
    ("a 'while' loop", "  y := u;\n  while y < 2.0 loop\n    y := y * 2.0;\n  end while;\n"),
    ("a 'for' loop", "  y := u;\n  for i in 1:3 loop\n    y := y + i;\n  end for;\n"),
    ("'break'", "  y := u;\n  if y > 0.0 then\n    break;\n  else\n    y := 1.0;\n  end if;\n"),
    ("never assigns its output", "  u := 1.0;\n"),
]


@pytest.mark.parametrize(
    ("what", "body"), REFUSALS, ids=[w.strip("'").split()[-1] for w, _ in REFUSALS]
)
def test_a_body_that_cannot_be_inlined_says_which_form_stopped_it(what: str, body: str) -> None:
    with pytest.raises(FlattenError, match=what):
        flatten(parse(_function(body)), "M")


def test_a_variable_assigned_on_only_one_path_is_refused() -> None:
    """Without an ``else`` there is no value to merge against, and no honest guess."""
    with pytest.raises(FlattenError, match="on some paths and not on others"):
        flatten(parse(_function("  if u > 0.0 then\n    y := 1.0;\n  end if;\n")), "M")


def test_a_two_output_function_in_an_expression_yields_its_first_output() -> None:
    """An expression wants one value, so a call to a two-output function reads the first output.

    The second is computed by the same run of the body and dropped (the tuple form
    ``(a, b) = f(x)`` reads both instead). The outputs are made distinct so the assertion pins that
    it is the first, ``y := u``, that survives and not the second, ``w := 2 * u``.
    """
    header = "  input Real u;\n  output Real y;\n  output Real w;\n"
    source = _function("  y := u;\n  w := 2.0 * u;\n", header=header)
    assert _equation_text(source, "M") == ["z = 1.0;"]


def test_an_input_with_no_value_and_no_default_is_refused() -> None:
    header = "  input Real u;\n  input Real v;\n  output Real y;\n"
    with pytest.raises(FlattenError, match="no value for input 'v'"):
        flatten(parse(_function("  y := u + v;\n", header=header)), "M")


def test_an_unknown_named_argument_is_refused() -> None:
    source = _function("  y := u;\n").replace("z = f(1.0);", "z = f(u = 1.0, q = 2.0);")
    with pytest.raises(FlattenError, match="no input named 'q'"):
        flatten(parse(source), "M")


def test_a_recursive_function_is_stopped_rather_than_hanging() -> None:
    source = _function("  y := f(u);\n")
    with pytest.raises(FlattenError, match="recursive"):
        flatten(parse(source), "M")


def test_a_builtin_call_is_left_alone() -> None:
    """``sqrt`` names no class, and asking the scope about it must not become an error."""
    source = "model M\n  Real z;\nequation\n  z = sqrt(2.0) + der(time);\nend M;\n"
    assert _equation_text(source, "M") == ["z = sqrt(2.0) + der(time);"]


def test_an_initial_algorithm_folds_into_the_initial_equations() -> None:
    """The same statements as an ``initial equation`` section, holding at initialization.

    MSL's Pulse source (Blocks/Sources.mo) computes its period count this way, and 32
    example models' first refusal was this section header.
    """
    source = "model M\n  Real x;\ninitial algorithm\n  x := 2;\nequation\n  der(x) = -x;\nend M;\n"
    program = parse(source)
    assert parse(unparse(program)) == program
    flat = flatten([program], "M")
    assert len(flat.initial_equations) == 1
    assert len(flat.equations) == 1


def test_an_assert_statement_is_read_and_round_trips() -> None:
    """``assert(...)`` in an algorithm section, met at Blocks/Sources.mo:1468."""
    source = 'model M\n  Real x;\nalgorithm\n  assert(x > 0, "gone");\nend M;\n'
    program = parse(source)
    m = program.find("M")
    assert m is not None
    assert isinstance(m.algorithms[0][0], ir.Assert)
    assert parse(unparse(program)) == program


def test_an_output_list_assignment_is_read_and_round_trips() -> None:
    """``(a, b) := f(x);`` -- the statement twin of the call equation."""
    source = "model M\n  Real a;\n  Real b;\nalgorithm\n  (a, b) := f(1);\nend M;\n"
    program = parse(source)
    m = program.find("M")
    assert m is not None
    statement = m.algorithms[0][0]
    assert isinstance(statement, ir.CallStatement)
    assert statement.targets == (ir.Ref(("a",)), ir.Ref(("b",)))
    assert parse(unparse(program)) == program


def test_both_are_refused_by_name_when_a_section_folds() -> None:
    """A model's algorithm section becomes equations, and these two cannot yet."""
    source = 'model M\n  Real x;\nalgorithm\n  assert(x > 0, "gone");\nend M;\n'
    with pytest.raises(FlattenError, match="uses 'assert'"):
        flatten([parse(source)], "M")


WHEN_SECTION = """\
model M
  discrete Real y;
  Real x;
equation
  der(x) = 1.0;
algorithm
  when x > 1.0 then
    y := 2.0 * x;
  elsewhen x > 2.0 then
    y := 0.0;
  end when;
end M;
"""


def test_a_when_statement_is_read_and_round_trips_with_its_own_spelling() -> None:
    """``when`` in an algorithm: the same node as the when-equation, printed with ``:=``.

    Blocks/Sources.mo interpolates its tables inside one, and 31 example models' first
    refusal was this keyword in a statement position.
    """
    program = parse(WHEN_SECTION)
    m = program.find("M")
    assert m is not None
    statement = m.algorithms[0][0]
    assert isinstance(statement, ir.When)
    assert len(statement.branches) == 2
    assert parse(unparse(program)) == program
    assert unparse(program) == WHEN_SECTION


def test_a_when_statement_is_refused_by_name_when_the_section_folds() -> None:
    with pytest.raises(FlattenError, match="uses a 'when'"):
        flatten(parse(WHEN_SECTION), "M")


def test_a_bare_call_statement_is_read_and_round_trips() -> None:
    """``isValidTable(table);`` -- Blocks/Sources.mo checks its table this way."""
    source = "model M\n  Real x;\nalgorithm\n  check(x);\nend M;\n"
    program = parse(source)
    m = program.find("M")
    assert m is not None
    statement = m.algorithms[0][0]
    assert statement == ir.CallStatement((), ir.Call("check", (ir.Ref(("x",)),), ()))
    assert parse(unparse(program)) == program
    assert unparse(program) == source


def test_an_assignment_to_an_array_element_parses_and_round_trips() -> None:
    """``buf[iTick] := u`` -- the standard library's FFT and routing blocks write one element.

    The subscript rides on the target the way it does on the last name of any reference, so the
    target is a :class:`~modelica.ir.Subscript`; a plain name stays a bare :class:`~modelica.ir.Ref`.
    Reading this is what lets ``Blocks/Math.mo`` and ``Blocks/Routing.mo`` load at all.
    """
    source = "model M\n  Real b[3];\n  Integer i;\nalgorithm\n  b[i] := 5.0;\nend M;\n"
    program = parse(source)
    m = program.find("M")
    assert m is not None
    statement = m.algorithms[0][0]
    assert isinstance(statement, ir.Assign)
    assert statement.target == ir.Subscript(ir.Ref(("b",)), (ir.Ref(("i",)),))
    assert parse(unparse(program)) == program


def test_a_when_equation_may_assign_an_array_element() -> None:
    """``when c then k[i] = 0; end when;`` -- Routing writes its selection vector this way."""
    source = (
        "model M\n  Real k[3];\n  Boolean c;\nequation\n  when c then\n    k[2] = 0.0;\n"
        "  end when;\nend M;\n"
    )
    program = parse(source)
    assert parse(unparse(program)) == program


def test_an_assignment_to_an_array_element_is_refused_when_it_is_flattened() -> None:
    """Parsing it is one thing; writing a single scalarised element is not carried through yet."""
    source = (
        "model M\n  Real b[3](each start = 0);\n  Boolean c;\n"
        "equation\n  der(b[1]) = 0;\n  der(b[2]) = 0;\n  der(b[3]) = 0;\n"
        "  when c then\n    b[1] = 5.0;\n  end when;\n  c = time > 1;\nend M;\n"
    )
    with pytest.raises(FlattenError, match="assignment to the array element"):
        flatten(parse(source), "M")
