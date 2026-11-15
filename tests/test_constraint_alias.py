"""A ``type`` alias of ``Real`` satisfies a slot declared ``Real``.

``Magnetic.FundamentalWave.Types.SalientInductance`` redeclares ``replaceable Real d`` as
``SI.Inductance``.
That was refused: "d: 'SI.Inductance' does not satisfy 'Real'".
"""

from __future__ import annotations

import pytest

from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import parse

SOURCE = """\
type Inductance = Real(unit = "H");
record Salient
  replaceable Real d;
end Salient;
record SalientInductance = Salient(redeclare Inductance d);
model M
  parameter SalientInductance L(d = 2);
end M;
"""


def test_an_alias_of_real_satisfies_a_real_slot() -> None:
    flat = flatten(parse(SOURCE), "M")
    assert any(v.name == "L.d" for v in flat.variables)


def test_a_different_predefined_type_is_still_refused() -> None:
    source = SOURCE.replace("type Inductance = Real", "type Inductance = Integer").replace(
        '(unit = "H")', ""
    )
    with pytest.raises(FlattenError, match="does not satisfy 'Real'"):
        flatten(parse(source), "M")
