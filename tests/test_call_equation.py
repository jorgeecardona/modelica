"""A function call as an equation: ``Connections.root(a.b);``, read and refused by name.

The MSL's QuasiStatic packages open their interface equations with the
overconstrained-connector operators -- ``Connections.root``, ``Connections.branch`` --
which are grammatically a call equation with an empty output list (specification 8.3.1).
44 example models' first refusal was the parser tripping on the semicolon where it
expected ``=``. The form now parses into :class:`ir.CallEquation` with no targets, and
flattening refuses it by name: the connection-graph semantics are a separate job, and
until they exist the refusal should say what was met, not where a lookup failed.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import parse, unparse

SOURCE = """\
model M
  Real x;
equation
  Connections.root(x);
  x = 1;
end M;
"""


def test_a_bare_call_is_read_as_a_call_equation_and_round_trips() -> None:
    program = parse(SOURCE)
    m = program.find("M")
    assert m is not None
    call = m.equations[0]
    assert call == ir.CallEquation((), ir.Call("Connections.root", (ir.Ref(("x",)),), ()))
    assert parse(unparse(program)) == program


def test_flattening_refuses_it_by_name() -> None:
    with pytest.raises(FlattenError, match=r"call equation is not supported yet: Connections"):
        flatten(parse(SOURCE), "M")


def test_an_expression_before_equals_is_still_an_equation() -> None:
    program = parse("model M\n  Real x;\nequation\n  sin(x) = 0.5;\nend M;\n")
    m = program.find("M")
    assert m is not None
    assert isinstance(m.equations[0], ir.Equal)
