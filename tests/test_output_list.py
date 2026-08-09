"""``(a, b) = f(x)`` -- the only way Modelica lets a function return two things.

Specification §8.3.1. Until this landed, `modelica.lang.parse` refused the form outright and
a function in this toolchain could return exactly one value.

What it is *not*: a tuple. There is no tuple value in this IR and adding one would reach
every expression walker in the project to serve a form that appears only at the head of an
equation. `ir.CallEquation` is its own node, and flattening turns one of them into one
ordinary `ir.Equal` per wanted output -- so causalization, the emitter and every backend go
on seeing one equation determining one unknown, and none of them learns that a call can
return a pair.

The load-bearing test is `test_the_body_runs_once_for_all_of_its_outputs`. Two outputs of one
call are two views of *one* computation; a version that inlined the body per output would
compute shared work twice and, for a body with a branch, could read two different arms.
"""

from __future__ import annotations

import pytest

from modelica.ir import nodes as ir
from modelica.ir.causalize import causalize
from modelica.ir.flatten import FlattenError
from modelica.lang import ModelicaSyntaxError, parse, unparse
from modelica.lang.unparse import unparse_equation

SPLIT = """
package P
  function split "Two results from one computation"
    input Real u;
    output Real half;
    output Real twice;
  algorithm
    half := 0.5 * u;
    twice := 2.0 * u;
  end split;

  model T
    Real a;
    Real b;
    Real x(start = 3.0, fixed = true);
  equation
    der(x) = -x;
    (a, b) = split(x);
  end T;
end P;
"""


def _flat(source: str, model: str = "P.T"):  # noqa: ANN202 - the test only reads it
    flatten = pytest.importorskip("modelica.sim").flatten
    return flatten([parse(source)], model)


def _equations(source: str, model: str = "P.T") -> list[str]:
    return [unparse_equation(e)[0].strip() for e in _flat(source, model).equations]


def test_two_results_become_two_equations() -> None:
    """One `CallEquation` in, one `Equal` per target out, in declaration order."""
    assert _equations(SPLIT) == ["der(x) = -x;", "a = 0.5 * x;", "b = 2.0 * x;"]


def test_the_two_results_are_not_the_same_value() -> None:
    """The obvious way to get this wrong, and it would pass every structural check.

    A flattener that read the first output twice, or that mapped both targets to the last
    output written, produces a model that flattens, causalizes, compiles and simulates. Only
    the numbers say otherwise.
    """
    numpy = pytest.importorskip("numpy")
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    result = simulate(causalize(_flat(SPLIT)), stop=1.0, points=5)
    numpy.testing.assert_allclose(result["a"], 0.5 * result["x"], rtol=1e-12)
    numpy.testing.assert_allclose(result["b"], 2.0 * result["x"], rtol=1e-12)
    assert result["b"][0] == pytest.approx(4.0 * result["a"][0])


def test_the_body_runs_once_for_all_of_its_outputs() -> None:
    """Two outputs of one call are two views of one computation, not two computations.

    The body below branches on its input and writes both outputs in each arm. Running it
    once per output is not merely wasteful -- for a body whose branch reads something that
    moves, the two runs can take different arms and the pair stops being consistent. Checked
    on the *text*: both equations must carry the same condition, and the shared subexpression
    must be the same node in both.
    """
    source = """
    package P
      function poles
        input Real u;
        output Real lo;
        output Real hi;
      protected
        Real scaled;
      algorithm
        scaled := 3.0 * u;
        if u > 0 then
          lo := scaled;
          hi := scaled + 1.0;
        else
          lo := -scaled;
          hi := -scaled + 1.0;
        end if;
      end poles;

      model T
        Real lo;
        Real hi;
        Real x(start = 1.0, fixed = true);
      equation
        der(x) = -x;
        (lo, hi) = poles(x);
      end T;
    end P;
    """
    written = _equations(source)
    assert len(written) == 3
    assert written[1].count("x > 0") == 1
    assert written[2].count("x > 0") == 1


def test_a_result_can_be_skipped() -> None:
    """``(, b) = f(x)`` is legal and means the first result is not wanted."""
    source = SPLIT.replace("(a, b) = split(x);", "(, b) = split(x);\n    a = 0;")
    assert _equations(source) == ["der(x) = -x;", "b = 2.0 * x;", "a = 0;"]


def test_a_parenthesised_left_side_is_still_an_ordinary_equation() -> None:
    """``(a + b) = c`` is a legal equation and must not be read as a list of one output.

    One token of lookahead does not separate the two, so the parser reads the list and
    rewinds when what follows the bracket is not ``=``. This is the case that made the rewind
    necessary rather than optional.
    """
    source = """
    model T
      Real a; Real b; Real c;
    equation
      (a + b) = c;
      a = 1;
      b = 2;
    end T;
    """
    assert isinstance(parse(source).classes[0].equations[0], ir.Equal)


def test_one_target_in_a_list_is_still_a_call_equation() -> None:
    """``(a) = f(x)`` -- legal, and it is the form, not a bracketed name."""
    source = """
    model T
      Real a; Real x;
    equation
      (a) = f(x);
      x = 1;
    end T;
    """
    assert isinstance(parse(source).classes[0].equations[0], ir.CallEquation)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("(a, b) = c + 1;", "assigned from a function call"),
        ("(a, b) = wobble(x);", "not a function this model can see"),
    ],
)
def test_what_is_refused_says_why(source: str, message: str) -> None:
    text = f"model T\n  Real a; Real b; Real c; Real x;\nequation\n  {source}\n  x = 1;\nend T;"
    with pytest.raises((ModelicaSyntaxError, FlattenError), match=message):
        _flat(text, "T")


def test_asking_for_more_results_than_the_function_has_is_refused() -> None:
    source = SPLIT.replace("(a, b) = split(x);", "(a, b, x) = split(x);")
    with pytest.raises(FlattenError, match="returns 2 value"):
        _flat(source)


def test_it_round_trips_through_the_unparser() -> None:
    """``parse(unparse(x)) == parse(x)`` -- what every other node in `modelica.lang` holds to."""
    for form in ("(a, b) = f(x, 1.0);", "(, b) = f(x);", "(a, ) = f(x);"):
        text = f"model T\n  Real a;\n  Real b;\n  Real x;\nequation\n  {form}\nend T;\n"
        assert parse(unparse(parse(text))) == parse(text)


def test_the_python_front_end_says_it_too() -> None:
    """Every feature of this corpus is written twice, and this is the second way.

    ``==`` builds an equation everywhere else in `modelica.build`; here it builds the other
    kind of equation. What it must not do is build one silently from something that is not a
    call, because the parser refuses that and the two front ends have to agree on what is
    legal.
    """
    build = pytest.importorskip("modelica.build")
    equation = build.results(build.ref("a"), build.ref("b")) == build.call("split", build.ref("x"))
    assert unparse_equation(equation)[0].strip() == "(a, b) = split(x);"
    assert (
        unparse_equation(
            build.results(build.ref("a"), None) == build.call("split", build.ref("x"))
        )[0].strip()
        == "(a, ) = split(x);"
    )
    with pytest.raises(TypeError, match="not an expression"):
        _ = build.results(build.ref("a")) == build.ref("x") + 1
