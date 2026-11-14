"""A function ``if`` whose condition is known when it is inlined runs only the arm it picks.

``Polyphase.Functions.symmetricOrientation(m)`` assigns its output in the arm ``mod(m, 2) == 0``
selects. Every arm was run, and the one that does not assign it was refused (36 MSL examples).
"""

from __future__ import annotations

from modelica.ir.flatten import flatten
from modelica.lang import parse

SOURCE = """\
function pick
  input Integer m;
  output Real y;
algorithm
  if mod(m, 2) == 0 then
    y := 2;
  elseif m == 1 then
    y := 1;
  else
    y := 3;
  end if;
end pick;
model M
  parameter Integer m = 3;
  Real y;
equation
  y = pick(m);
end M;
"""


def test_only_the_arm_the_known_condition_picks_runs() -> None:
    flat = flatten(parse(SOURCE), "M")
    assert "3" in repr(flat.equations)


def test_an_arm_that_assigns_nothing_is_fine_when_it_is_not_picked() -> None:
    source = SOURCE.replace("  elseif m == 1 then\n    y := 1;\n", "  elseif m == 1 then\n")
    flatten(parse(source), "M")


COMPONENT = """\
function so
  input Integer m;
  output Real o[m];
algorithm
  if mod(m, 2) == 0 then
    o := fill(0, m);
  else
    o := {(k - 1) * 2 * 3.14 / m for k in 1:m};
  end if;
end so;
model C
  parameter Integer m = 3;
  parameter Real o[m] = so(m);
end C;
model M
  C c;
end M;
"""


def test_a_call_in_a_component_reads_the_component_parameter() -> None:
    """``so(m)`` inside component ``c`` reads ``c.m``: the arm and the shape follow it."""
    from modelica.ir.evaluate import evaluate  # noqa: PLC0415

    flat = flatten(parse(COMPONENT), "M")
    bindings = {v.name: v.binding for v in flat.variables}
    second = bindings["c.o[2]"]
    assert second is not None
    assert evaluate(second, {"c.m": 3.0}) == 2 * 3.14 / 3
    five = flatten(parse(COMPONENT.replace("C c;", "C c(m = 5);")), "M")
    assert "c.o[5]" in {v.name for v in five.variables}
