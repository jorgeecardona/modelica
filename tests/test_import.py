"""``import``: three spellings of one idea, and where a name is looked up.

An import binds a simple name to a full path and does nothing else -- no equation, no
component, no change below :mod:`modelica.ir.flatten`. So what there is to test is the
*order* of lookup, which is where an import can quietly mean the wrong class:

* a class's own nested classes win over anything it imported;
* an import wins over what an enclosing class declares;
* the path is resolved from the top of the program, never relative to the importer.

The last one is the rule that keeps this from being ambiguous, and it is also what stops
the lookup recursing -- the root scope has no imports of its own to consult.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse

MODELS = Path(__file__).resolve().parent.parent / "examples" / "models"

LIB = """\
package Lib
  model Part
    Real v;
  equation
    v = 1.0;
  end Part;
end Lib;
"""


def _one_variable(body: str, model: str = "M") -> list[str]:
    return [v.name for v in flatten(parse(LIB + body), model).variables]


def test_a_qualified_import_binds_the_last_segment() -> None:
    assert _one_variable("model M\n  import Lib.Part;\n  Part p;\nend M;\n") == ["p.v"]


def test_a_renaming_import_binds_the_name_it_gives() -> None:
    assert _one_variable("model M\n  import L = Lib;\n  L.Part p;\nend M;\n") == ["p.v"]


def test_a_wildcard_import_binds_every_class_in_the_package() -> None:
    assert _one_variable("model M\n  import Lib.*;\n  Part p;\nend M;\n") == ["p.v"]


def test_the_three_forms_are_told_apart_in_the_ir() -> None:
    source = "model M\n  import Lib.Part;\n  import L = Lib;\n  import Lib.*;\nend M;\n"
    qualified, renamed, wild = parse(source).classes[0].imports
    assert (qualified.name, qualified.alias, qualified.wildcard) == ("Part", "", False)
    assert (renamed.name, renamed.alias, renamed.wildcard) == ("L", "L", False)
    # A wildcard introduces many names, so it introduces no single one.
    assert (wild.name, wild.path, wild.wildcard) == ("", "Lib", True)


def test_a_nested_class_wins_over_an_import_of_the_same_name() -> None:
    """Modelica's order, and the case where getting it wrong picks the wrong class."""
    source = (
        "model M\n"
        "  import Lib.*;\n"
        "  model Part\n    Real w;\n  equation\n    w = 2.0;\n  end Part;\n"
        "  Part p;\n"
        "end M;\n"
    )
    assert _one_variable(source) == ["p.w"]


def test_an_import_wins_over_a_class_declared_further_out() -> None:
    source = (
        "package Outer\n"
        "  model Part\n    Real w;\n  equation\n    w = 2.0;\n  end Part;\n\n"
        "  model M\n    import Lib.Part;\n    Part p;\n  end M;\n"
        "end Outer;\n"
    )
    assert [v.name for v in flatten(parse(LIB + source), "Outer.M").variables] == ["p.v"]


def test_an_import_that_does_not_resolve_says_so() -> None:
    wanted = re.escape("imports 'Lib.Missing', which does not resolve")
    with pytest.raises(FlattenError, match=wanted):
        flatten(parse(LIB + "model M\n  import Lib.Missing;\n  Missing p;\nend M;\n"), "M")


def test_a_wildcard_that_does_not_hold_the_name_falls_through() -> None:
    """Not an error on its own: a wildcard is a place to look, not a promise."""
    with pytest.raises(FlattenError, match="nothing named 'Absent' is in scope"):
        flatten(parse(LIB + "model M\n  import Lib.*;\n  Absent p;\nend M;\n"), "M")


@pytest.mark.parametrize(
    "source",
    [
        "model M\n  import Lib.Part;\nend M;\n",
        "model M\n  import L = Lib;\nend M;\n",
        "model M\n  import Lib.*;\nend M;\n",
    ],
    ids=["qualified", "renamed", "wildcard"],
)
def test_each_form_prints_back_as_itself(source: str) -> None:
    assert unparse(parse(source)) == source


def test_an_import_cannot_both_rename_and_wildcard() -> None:
    with pytest.raises(ModelicaSyntaxError, match="cannot both rename and use a wildcard"):
        parse("model M\n  import L = Lib.*;\nend M;\n")


def test_imports_are_written_before_everything_else() -> None:
    """They bind the names the rest of the body uses, so they read first."""
    printed = unparse(parse("model M\n  import Lib.*;\n  Real x;\nend M;\n")).splitlines()
    assert printed[1] == "  import Lib.*;"
    assert printed[2] == "  Real x;"


def test_a_list_import_binds_each_name() -> None:
    """``import Lib.{Part}`` is one clause per name, each the same as ``import Lib.Part``.

    The standard library's `Rotational.Components.Clutch` and its neighbours write their table
    functions this way, three names between braces in a single clause.
    """
    assert _one_variable("model M\n  import Lib.{Part};\n  Part p;\nend M;\n") == ["p.v"]


def test_a_list_import_becomes_one_import_per_name() -> None:
    """Each name is its own qualified import in the IR, so nothing below the parser is new."""
    imports = parse("model M\n  import P.{A, B, C};\nend M;\n").classes[0].imports
    assert [(i.path, i.name, i.wildcard) for i in imports] == [
        ("P.A", "A", False),
        ("P.B", "B", False),
        ("P.C", "C", False),
    ]


def test_a_list_import_prints_back_as_separate_imports() -> None:
    """It expands to one import per name, so it prints as the plain form it has become."""
    printed = unparse(parse("model M\n  import P.{A, B};\n  Real x;\nend M;\n")).splitlines()
    assert "  import P.A;" in printed
    assert "  import P.B;" in printed


def test_a_list_import_cannot_rename() -> None:
    with pytest.raises(ModelicaSyntaxError, match="cannot both rename"):
        parse("model M\n  import L = P.{A, B};\nend M;\n")


def test_the_corpus_uses_one() -> None:
    """``RLCCircuit`` names five classes, so an alias earns its place there.

    Its Python twin says the same thing -- ``test_examples.py`` compares the two -- which is
    the real assertion: the builder knows the class as ``Electrical.Resistor`` and has to
    write ``E.Resistor``, because that is what was imported.
    """
    circuit = parse((MODELS / "RLCCircuit.mo").read_text(encoding="utf-8")).classes[0]
    assert [(i.alias, i.path) for i in circuit.imports] == [("E", "Electrical")]
    assert {c.type_name.split(".")[0] for c in circuit.components} == {"E"}
