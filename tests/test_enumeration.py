"""``enumeration``: a type whose values are names, and a number underneath.

The specification is blunt about the encoding -- a literal *is* its position in the
declaration, counted from one -- and that is what this toolchain emits. So an enumeration
needs no fifth base type: it flattens to an `Integer` that has been told its range, and
`modelica.sim.emit`, which puts every variable in a float workspace, never learns that the
type existed.

Two consequences are what the tests below are about. The ordinal has to be *right*, because
it is the whole meaning of the type and nothing downstream can check it. And `Schedule.day`
has to become a number before prefixing, because it is not a component reference and
prefixing it would produce a variable that does not exist.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse, unparse_equation

MODEL = """\
model M
  type Mode = enumeration(red, green, blue);
  parameter Mode chosen = Mode.green;
  Real x;
equation
  x = if chosen == Mode.blue then 1.0 else 2.0;
end M;
"""


def test_the_literals_are_kept_in_order() -> None:
    """Order is the content: reordering them changes what every literal means."""
    mode = parse(MODEL).classes[0].classes[0]
    assert mode.literals == ("red", "green", "blue")
    assert mode.restriction == "type"
    # An enumeration defines a type; it does not rename one.
    assert mode.extends == ()


def test_it_round_trips() -> None:
    program = parse(MODEL)
    assert parse(unparse(program)) == program


def test_it_prints_as_it_was_written() -> None:
    source = 'model M\n  type E = enumeration(a, b) "Two of them";\nend M;\n'
    assert '  type E = enumeration(a, b) "Two of them";' in unparse(parse(source)).splitlines()


def test_a_literal_becomes_its_position_counted_from_one() -> None:
    flat = flatten(parse(MODEL), "M")
    assert unparse_equation(flat.equations[0])[0] == "x = if chosen == 3 then 1.0 else 2.0;"


def test_the_variable_is_an_integer_that_knows_its_range() -> None:
    chosen = next(v for v in flatten(parse(MODEL), "M").variables if v.name == "chosen")
    assert chosen.base_type == "Integer"
    assert (chosen.minimum, chosen.maximum) == (ir.IntLiteral(1), ir.IntLiteral(3))
    assert chosen.binding == ir.IntLiteral(2)


def test_a_literal_in_a_binding_is_resolved_too() -> None:
    """A binding is not an equation, so it takes the same rewrite in a second place."""
    source = "model M\n  type E = enumeration(a, b);\n  parameter E e = E.b;\nend M;\n"
    binding = next(v for v in flatten(parse(source), "M").variables if v.name == "e")
    assert binding.binding == ir.IntLiteral(2)


def test_a_component_of_that_name_wins_over_the_type() -> None:
    """A local name is a local name, whatever else it might resolve to."""
    source = (
        "model M\n"
        "  type Mode = enumeration(day);\n"
        "  Real Mode;\n"
        "  Real y;\n"
        "equation\n"
        "  Mode = 1.0;\n"
        "  y = Mode;\n"
        "end M;\n"
    )
    assert unparse_equation(flatten(parse(source), "M").equations[1])[0] == "y = Mode;"


def test_an_enumeration_with_no_literals_is_refused() -> None:
    with pytest.raises(ModelicaSyntaxError, match="has no literals"):
        parse("model M\n  type E = enumeration();\nend M;\n")


def test_a_literal_that_was_never_declared_stays_a_reference_and_fails_later() -> None:
    """Not silently zero: an unknown member of a known type has to be an error somewhere."""
    source = "model M\n  type E = enumeration(a);\n  Real x;\nequation\n  x = E.z;\nend M;\n"
    with pytest.raises(FlattenError):
        flatten(parse(source), "M")


def test_a_qualified_package_constant_resolves_to_its_value() -> None:
    """``P.Constants.pi`` is a constant declared in a package, so it becomes its evaluated value.

    Its binding may be an expression over the library's own maths -- ``2 * P.Math.asin(1.0)`` -- and
    the qualified call to a builtin is read as the builtin so the value can be evaluated here.
    """
    source = (
        "package P\n"
        "  package Math\n"
        "    function asin\n      input Real x;\n      output Real y;\n"
        '    external "builtin";\n    end asin;\n'
        "  end Math;\n"
        "  package Constants\n"
        "    constant Real pi = 2 * P.Math.asin(1.0);\n"
        "    constant Real eps = 1e-15;\n"
        "  end Constants;\n"
        "end P;\n"
        "model M\n  Real x;\nequation\n  x = P.Constants.pi + P.Constants.eps;\nend M;\n"
    )
    equation = unparse_equation(flatten(parse(source), "M").equations[0])[0]
    assert equation == "x = 3.141592653589793 + 1e-15;"


def test_an_unknown_package_constant_still_fails() -> None:
    """A constant that is not there is a typo, not a silent zero: it has to be an error."""
    source = (
        "package P\n  package Constants\n    constant Real pi = 3.14;\n  end Constants;\nend P;\n"
        "model M\n  Real x;\nequation\n  x = P.Constants.nope;\nend M;\n"
    )
    with pytest.raises(FlattenError, match="unknown name"):
        flatten(parse(source), "M")


def test_a_package_constant_defined_by_another_resolves() -> None:
    """`D2R = pi / 180` names another constant of the same package, so pi is put in first."""
    source = (
        "package P\n"
        "  package Math\n    function asin\n      input Real x;\n      output Real y;\n"
        '    external "builtin";\n    end asin;\n  end Math;\n'
        "  package Constants\n"
        "    constant Real pi = 2 * P.Math.asin(1.0);\n"
        "    constant Real deg = pi / 180;\n"
        "  end Constants;\n"
        "end P;\n"
        "model M\n  Real x;\nequation\n  x = P.Constants.deg;\nend M;\n"
    )
    equation = unparse_equation(flatten(parse(source), "M").equations[0])[0]
    assert equation == "x = 0.017453292519943295;"


def test_a_cyclic_package_constant_is_not_resolved_and_does_not_loop() -> None:
    """`a = b + 1; b = a - 1` cannot reduce; the `seen` guard stops the recursion, so it fails."""
    source = (
        "package P\n  package Constants\n"
        "    constant Real a = b + 1;\n    constant Real b = a - 1;\n"
        "  end Constants;\nend P;\n"
        "model M\n  Real x;\nequation\n  x = P.Constants.a;\nend M;\n"
    )
    with pytest.raises(FlattenError, match="unknown name"):
        flatten(parse(source), "M")


_CONSTANTS = (
    "package P\n"
    "  package Constants\n"
    "    constant Real pi = 3.141592653589793;\n"
    "    constant Real eps = 1e-15;\n"
    "  end Constants;\n"
    "end P;\n"
)


def test_a_named_import_makes_a_package_constant_a_bare_name() -> None:
    """``import P.Constants.pi;`` then a bare ``pi`` is the constant's value, as the qualified form is.

    The multi-part reference `P.Constants.pi` already reduced; this is the same constant reached by
    the simple name the import brought into scope, which is how most of the library writes it.
    """
    source = (
        _CONSTANTS + "model M\n  import P.Constants.pi;\n  Real x;\nequation\n  x = pi;\nend M;\n"
    )
    equation = unparse_equation(flatten(parse(source), "M").equations[0])[0]
    assert equation == "x = 3.141592653589793;"


def test_a_wildcard_import_makes_every_package_constant_a_bare_name() -> None:
    """``import P.Constants.*;`` puts both ``pi`` and ``eps`` in scope by their own names."""
    source = (
        _CONSTANTS
        + "model M\n  import P.Constants.*;\n  Real x;\nequation\n  x = pi + eps;\nend M;\n"
    )
    equation = unparse_equation(flatten(parse(source), "M").equations[0])[0]
    assert equation == "x = 3.141592653589793 + 1e-15;"


def test_a_local_component_wins_over_an_imported_constant() -> None:
    """A component named ``pi`` is a component, whatever an import says: the bare name stays a ref."""
    source = (
        _CONSTANTS + "model M\n  import P.Constants.*;\n"
        "  parameter Real pi = 7;\n  Real x;\nequation\n  x = pi;\nend M;\n"
    )
    equation = unparse_equation(flatten(parse(source), "M").equations[0])[0]
    assert equation == "x = pi;"


def test_a_constant_in_an_enclosing_package_resolves_by_its_simple_name() -> None:
    """A model reads a ``constant`` its enclosing package declares by the simple name, no import."""
    source = (
        "package P\n  constant Real g = 9.81;\n"
        "  model M\n    Real x;\n  equation\n    x = g;\n  end M;\nend P;\n"
    )
    equation = unparse_equation(flatten(parse(source), "P.M").equations[0])[0]
    assert equation == "x = 9.81;"


def test_a_bare_name_no_import_introduces_still_fails() -> None:
    """A bare name nothing brings into scope is a typo, not a silent zero."""
    source = (
        _CONSTANTS + "model M\n  import P.Constants.*;\n  Real x;\nequation\n  x = nope;\nend M;\n"
    )
    with pytest.raises(FlattenError, match="unknown name"):
        flatten(parse(source), "M")


def test_a_package_constant_defined_by_one_in_another_package_resolves() -> None:
    """A constant whose value names one in another package still becomes its number.

    ``Modelica.Constants.eps = ModelicaServices.Machine.eps`` reaches across packages, resolved by the
    qualified name from the top of the program. The two share the simple name ``eps``, which the
    same-package guard would read as a cycle; a hop count bounds the crossing instead.
    """
    source = (
        "package Services\n  package Machine\n"
        "    constant Real eps = 2.220446049250313e-16;\n  end Machine;\nend Services;\n"
        "package P\n  package Constants\n"
        "    constant Real eps = Services.Machine.eps;\n"
        "  end Constants;\nend P;\n"
        "model M\n  Real x;\nequation\n  x = P.Constants.eps;\nend M;\n"
    )
    equation = unparse_equation(flatten(parse(source), "M").equations[0])[0]
    assert equation == "x = 2.220446049250313e-16;"


def test_a_cross_package_constant_ring_is_not_resolved_and_does_not_loop() -> None:
    """A cross-package ring is stopped by the hop count, so it fails cleanly, not forever.

    ``A.a = B.b; B.b = A.a`` names back and forth across two packages; the hop count bounds the
    crossing, so it fails with the ordinary report rather than chasing the ring.
    """
    source = (
        "package A\n  constant Real a = B.b;\nend A;\n"
        "package B\n  constant Real b = A.a;\nend B;\n"
        "model M\n  Real x;\nequation\n  x = A.a;\nend M;\n"
    )
    with pytest.raises(FlattenError, match="unknown name"):
        flatten(parse(source), "M")
