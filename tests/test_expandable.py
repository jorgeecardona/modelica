"""``expandable connector``: read faithfully, refused only where it would mean something.

The MSL declares one at ``Modelica/Icons.mo:629``, and before the parser accepted the
keyword that single declaration blocked 258 of the library's 430 example models — every
model whose icon chain passes through ``Modelica.Icons``. Parsing it is therefore worth
having on its own. *Instantiating* one is a different job: the members of an expandable
connector exist by being connected, and until that expansion is implemented the flattener
refuses it by name — the same accept-then-refuse two-step the array syntax uses.
"""

from __future__ import annotations

import pytest

from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import ModelicaSyntaxError, parse, unparse

BUS = """\
expandable connector Bus
  Real v;
end Bus;
"""


def test_an_expandable_connector_is_read_and_round_trips() -> None:
    program = parse(BUS)
    bus = program.classes[0]
    assert bus.restriction == "connector"
    assert bus.expandable
    assert parse(unparse(program)) == program


def test_expandable_qualifies_only_a_connector() -> None:
    with pytest.raises(ModelicaSyntaxError, match="`expandable` qualifies only a connector"):
        parse("expandable model M\nend M;\n")


def test_a_plain_connector_stays_plain() -> None:
    program = parse("connector Pin\n  Real v;\nend Pin;\n")
    assert not program.classes[0].expandable
    assert "expandable" not in unparse(program)


def test_instantiating_an_expandable_connector_is_refused_by_name() -> None:
    source = BUS + "model M\n  Bus bus;\nend M;\n"
    with pytest.raises(FlattenError, match="bus: expandable connectors are not supported yet"):
        flatten(parse(source), "M")


def test_extending_an_expandable_connector_marks_the_child() -> None:
    """The MSL pattern: a concrete bus ``extends Modelica.Icons.SignalBus``.

    The child carries its own ``expandable`` keyword in source, so nothing is inferred
    here -- this pins down that the flag on the child, not the parent, is what the
    refusal reads.
    """
    child = parse(BUS + "expandable connector ControlBus\n  extends Bus;\nend ControlBus;\n")
    control = child.find("ControlBus")
    assert control is not None
    assert control.expandable


def test_a_declared_but_uninstantiated_expandable_connector_flattens_past() -> None:
    """Declaring the class must not poison the package that holds it.

    That was the measured failure: one declaration in ``Modelica.Icons`` took 258 models
    with it.
    """
    source = BUS + "model M\n  Real x;\nequation\n  x = 1;\nend M;\n"
    assert [v.name for v in flatten(parse(source), "M").variables] == ["x"]
