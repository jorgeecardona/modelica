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
from modelica.lang.unparse import unparse

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
