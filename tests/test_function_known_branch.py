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
