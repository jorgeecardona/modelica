"""``operator record`` / ``operator function``: the class parses as its plain kind.

Operator overloading is the single largest blocker in the coverage run over the Modelica
Standard Library: on 2026-09-09 it stopped 311 of 430 example models at parse, three-quarters
of the library on one construct. Almost all of it exists to give one type, ``Complex``,
ordinary arithmetic, and ``Modelica.Units`` defines dozens of ``operator record`` aliases of
it. Every model that imports ``Modelica.Units.SI`` (nearly all of them) needs that file to
parse, and the file did not, because the parser refused the ``operator`` prefix.

The prefix now parses: an ``operator record`` reads as a ``record`` and an ``operator
function`` as a ``function``, so the file loads. The overloading itself is not implemented --
an ``operator function '+'`` body is written with a single-quoted identifier the lexer does not
read, and applying an overloaded operator is refused where it would be used -- but that is a
separate matter from the file parsing, which is what unblocks the 311.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse

if TYPE_CHECKING:
    from modelica import ir

SHORT = """\
package Units
  operator record ComplexCurrent =
    Complex(redeclare Real re "Real part", redeclare Real im "Imaginary part")
    "Complex electric current";
end Units;
"""

LONG = """\
package P
  operator record Phasor
    Real re;
    Real im;
  end Phasor;
end P;
"""


def _member(source: str) -> ir.Class:
    return parse(source, "<test>").classes[0].classes[0]


def test_a_short_operator_record_reads_as_a_record() -> None:
    """The `Modelica.Units` form: an alias of `Complex` with redeclared parts."""
    member = _member(SHORT)
    assert member.name == "ComplexCurrent"
    assert member.restriction == "record"
    # The base and its redeclarations survive, so the alias is a real record, not a stub.
    assert member.extends[0].type_name == "Complex"


def test_a_long_operator_record_reads_as_a_record_with_its_fields() -> None:
    member = _member(LONG)
    assert member.name == "Phasor"
    assert member.restriction == "record"
    assert [component.name for component in member.components] == ["re", "im"]


def test_the_prefix_round_trips_through_the_printer() -> None:
    """Unparse then reparse both forms: the record the prefix produced prints and reads back."""
    for source in (SHORT, LONG):
        again = parse(unparse(parse(source, "<test>")), "<roundtrip>")
        assert again.classes[0].classes[0].restriction == "record"


def test_operator_still_needs_a_kind_after_it() -> None:
    """A bare `operator` that is not a class start is still refused, not silently accepted."""
    with pytest.raises(ModelicaSyntaxError):
        parse("model M\n  Real x = operator;\nequation\n  x = 1;\nend M;\n", "<test>")
