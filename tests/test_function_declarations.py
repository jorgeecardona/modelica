"""Function declaration prefixes and bodies the standard library needs to load.

Two constructs, both on functions, both blocking a slice of the coverage run because the files
that carry them are imported widely:

* **purity** -- `pure function` and `impure function`. Purity is an optimisation hint (an impure
  function, a random draw say, may not be lifted out of a loop). The tool does not act on it, so
  the prefix is accepted and dropped, and the function reads as a plain one.
* **an external body** -- `external "C" [ref =] call [annotation];`, the whole body of a function
  implemented in another language. It cannot run here, so it is read and dropped; a function that
  reaches the simulator this way is refused where a call would inline it (it has no algorithm),
  not at parse. `Modelica.Utilities`, `Modelica.Math.Random` and the table functions are written
  this way, and importing them is what unblocked their dependents.
"""

from __future__ import annotations

from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.unparse import unparse, unparse_equation

PURE = """\
package P
  pure function f
    input Real x;
    output Real y;
  algorithm
    y := x;
  end f;
  impure function g
    input Real x;
    output Real y;
  algorithm
    y := x;
  end g;
end P;
"""

EXTERNAL = """\
package P
  function length
    input String s;
    output Integer n;
    external "C" n = ModelicaStrings_length(s) annotation (Library = "ModelicaExternalC");
  end length;
  function shuffle
    input Real state[3];
    output Real out[3];
    external "C" ModelicaRandom_shuffle(state, out);
  end shuffle;
end P;
"""


def test_pure_and_impure_functions_read_as_plain_functions() -> None:
    members = {c.name: c for c in parse(PURE, "<test>").classes[0].classes}
    assert members["f"].restriction == "function"
    assert members["g"].restriction == "function"
    assert parse(unparse(parse(PURE, "<test>")), "<roundtrip>")  # round-trips


def test_an_external_body_parses_in_both_forms_and_round_trips() -> None:
    """`ref = call` and a bare `call`, with and without an annotation, all read."""
    members = {c.name: c for c in parse(EXTERNAL, "<test>").classes[0].classes}
    assert members["length"].restriction == "function"
    # The external body is dropped -- there is no algorithm section to run.
    assert members["length"].algorithms == ()
    assert parse(unparse(parse(EXTERNAL, "<test>")), "<roundtrip>")


def test_a_model_that_only_imports_an_external_function_still_flattens() -> None:
    """A dependency being external does not stop a model that never calls it from flattening."""
    source = f"{EXTERNAL[:-1]}\nmodel M\n  Real z(start = 1);\nequation\n  der(z) = -z;\nend M;\n"
    flat = flatten(parse(source, "<test>"), "M")
    assert any(v.name == "z" for v in flat.variables)


def test_a_package_constant_in_a_function_body_resolves() -> None:
    """MSL's unit conversions read `Modelica.Constants.T_zero` inside `to_degC`, then inline them.

    The body of an inlined function is resolved in the function's own scope, so a package constant
    (or an enumeration literal) it reads becomes its value before prefixing would try to make it a
    variable -- which it was not doing, so a model calling the conversion failed with "unknown name".
    """
    source = (
        "package P\n"
        "  package Constants\n    constant Real T_zero = -273.15;\n  end Constants;\n"
        "  function to_degC\n    input Real k;\n    output Real c;\n"
        "  algorithm\n    c := k + P.Constants.T_zero;\n  end to_degC;\n"
        "  model M\n    Real x;\n  equation\n    x = P.to_degC(300);\n  end M;\n"
        "end P;\n"
    )
    equation = unparse_equation(flatten(parse(source), "P.M").equations[0])[0]
    assert equation == "x = 300 + -273.15;"


def test_a_multi_output_function_in_an_expression_uses_its_first_output() -> None:
    """A function with several outputs, called where one value is expected, yields the first.

    MSL's `ToSpacePhasor` returns a space phasor and a zero-sequence value, and `v_ =
    ToSpacePhasor(i)` wants the phasor; the flattener used to refuse a call to any function with more
    than one output. The rest are computed by the same run of the body and then dropped -- the tuple
    form `(a, b) = f(x)` reads every output instead, through a separate path.
    """
    source = (
        "package P\n"
        "  function divmod\n"
        "    input Real a;\n    output Real q;\n    output Real r;\n"
        "  algorithm\n    q := a / 2;\n    r := a - 2 * q;\n  end divmod;\n"
        "  model M\n    Real x;\n    Real y;\n"
        "  equation\n    der(x) = -x;\n    y = P.divmod(x) + 1;\n  end M;\n"
        "end P;\n"
    )
    equations = [unparse_equation(e)[0].strip() for e in flatten(parse(source), "P.M").equations]
    assert "y = x / 2 + 1;" in equations
