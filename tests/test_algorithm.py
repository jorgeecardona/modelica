"""``algorithm`` sections: the language half.

What this pins is a *shape*, not a semantics. An algorithm section parses, prints, and
parses back to the same value; a ``function`` is a class with ``input``/``output``
components and one such section; and the Python front end says the same things the ``.mo``
front end says. Nothing here simulates, because nothing yet can:
:mod:`modelica.ir.flatten` refuses a class that carries a section, and the last test in
this file is what holds it to refusing *by name* rather than dropping it quietly.

The two spellings of an assignment are the subtlety worth a test of their own. The IR has
one :class:`~modelica.ir.Assign`, and Modelica writes it as ``:=`` in an algorithm section
and as ``=`` inside a ``when`` in an equation section. Getting that wrong produces a file
that round-trips through this toolchain and is rejected by every other one.
"""

from __future__ import annotations

import pytest

from modelica import build, ir
from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse

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


def test_initial_algorithm_is_refused_by_name() -> None:
    with pytest.raises(ModelicaSyntaxError, match="initial algorithm"):
        parse("model M\ninitial algorithm\n  x := 1.0;\nend M;\n")


def test_an_external_body_is_still_refused() -> None:
    """The one refusal kept: an external function is an FFI question, not a language one."""
    with pytest.raises(ModelicaSyntaxError, match="external"):
        parse("function f\nexternal;\nend f;\n")


def test_an_unterminated_loop_names_the_construct() -> None:
    with pytest.raises(ModelicaSyntaxError, match="end while"):
        parse("model M\nalgorithm\n  while true loop\n    x := 1.0;\nend M;\n")


def test_flattening_refuses_a_section_rather_than_dropping_it() -> None:
    """The whole point of the split: parsed is not the same as understood."""
    program = parse(LOOPS)
    with pytest.raises(FlattenError, match="algorithm section"):
        flatten(program, "Loops")
