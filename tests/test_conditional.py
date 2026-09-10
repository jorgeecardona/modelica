"""Conditional components: ``Sensor s if useSensor;`` and the removal that follows.

The MSL measurement (docs/ideas/coverage.md) counts 53 example models whose first refusal
was this construct. Unlike ``expandable connector`` this one is implemented, not merely
accepted: the condition is evaluated during flattening -- it is structural the way a
dimension is, deciding which variables and equations exist -- and a component whose
condition is false disappears together with every ``connect()`` that touches it, which is
the one reference the specification allows to survive the removal.
"""

from __future__ import annotations

import pytest

from modelica import ir
from modelica.ir.flatten import FlattenError, flatten
from modelica.lang import parse, unparse, unparse_equation

LIB = """\
connector Pin
  Real v;
  flow Real i;
end Pin;
model Source
  Pin p;
equation
  p.v = 1;
end Source;
model Load
  Pin p;
equation
  p.i = p.v;
end Load;
"""


def model(use: str) -> ir.Program:
    return parse(
        LIB
        + f"""\
model M
  parameter Boolean useLoad = {use};
  Source s;
  Load l if useLoad;
equation
  connect(s.p, l.p);
end M;
""",
    )


def test_the_condition_is_read_and_round_trips() -> None:
    program = model("false")
    m = program.find("M")
    assert m is not None
    load = next(c for c in m.components if c.name == "l")
    assert load.condition == ir.Ref(("useLoad",))
    assert parse(unparse(program)) == program


def test_a_false_condition_removes_the_component_and_its_connects() -> None:
    flat = flatten(model("false"), "M")
    names = [v.name for v in flat.variables]
    assert not any(name.startswith("l.") for name in names)
    # The severed connect leaves the source's pin alone in its connection set: its flow
    # sums to zero on its own, and no equation mentions the load.
    assert not any("l." in str(equation) for equation in flat.equations)


def test_a_true_condition_keeps_the_component() -> None:
    flat = flatten(model("true"), "M")
    names = [v.name for v in flat.variables]
    assert "l.p.v" in names
    assert "l.p.i" in names


_HEATPORT = (
    "connector HP\n  Real T;\nend HP;\n"
    "partial model Cond\n  parameter Boolean useHeatPort = false;\n  HP heatPort if useHeatPort;\n"
    "end Cond;\n"
    "model NPN\n  extends Cond(useHeatPort = useOther);\n  parameter Boolean useOther = {use};\n"
    "  Real p;\nequation\n  p = 1;\nend NPN;\n"
    "model M\n  NPN d;\nend M;\n"
)


def test_a_condition_gated_by_a_parameter_set_through_an_extends_resolves() -> None:
    """MSL's transistor writes ``extends ConditionalHeatPort(useHeatPort = useTemperatureDependency)``.

    The gating parameter is set, through the ``extends``, to another parameter of the class. That one
    may be declared after it, so its value is not in hand when the parameter is instantiated; the
    parameter values are resolved among themselves before the conditions are evaluated, so the heat
    port is kept or dropped by what the sibling settles on.
    """
    off = [v.name for v in flatten(parse(_HEATPORT.format(use="false")), "M").variables]
    assert not any(name.startswith("d.heatPort") for name in off)
    on = [v.name for v in flatten(parse(_HEATPORT.format(use="true")), "M").variables]
    assert "d.heatPort.T" in on


_ENUM_GATED = """\
package P
  type Choice = enumeration(A, B);
  model Sub
    Real x(start = 1);
  equation
    der(x) = -x;
  end Sub;
  model M
    import P.Choice;
    parameter Choice mode = Choice.{value};
    Sub sub if mode == Choice.A;
    Real y(start = 1);
  equation
    der(y) = -y;
  end M;
end P;
"""


def test_a_condition_comparing_an_enumeration_literal_resolves() -> None:
    """The power converters gate a component on `refType == SingleReferenceType.Sawtooth`.

    A conditional component's condition takes the enumeration-literal rewrite an equation gets, so the
    literal becomes its ordinal before the names are resolved; otherwise the imported `Choice.A` reads
    as a variable that does not exist. The component is kept when the parameter picks the same literal
    and dropped when it does not.
    """
    kept = [v.name for v in flatten(parse(_ENUM_GATED.format(value="A")), "P.M").variables]
    assert "sub.x" in kept
    dropped = [v.name for v in flatten(parse(_ENUM_GATED.format(value="B")), "P.M").variables]
    assert not any(name.startswith("sub") for name in dropped)


def test_a_condition_nobody_can_evaluate_is_refused() -> None:
    source = LIB + "model M\n  Real x;\n  Load l if x > 0;\nequation\n  x = 1;\nend M;\n"
    with pytest.raises(FlattenError, match="must be known when the model is flattened"):
        flatten(parse(source), "M")


def test_a_condition_that_is_not_a_boolean_is_refused() -> None:
    source = LIB + "model M\n  Load l if 3;\nend M;\n"
    with pytest.raises(FlattenError, match="must be a Boolean"):
        flatten(parse(source), "M")


# Structural if-*equations* are the same idea as a conditional component: the condition is a
# parameter or a constant, evaluated during flattening, and only the chosen branch's equations
# survive. The coefficients below (2.0 / 7.0 / 9.0) are distinct so the active branch is
# unambiguous in the flattened equation list.
IF_MODEL = """\
model M
  parameter Boolean fast = {fast};
  Real x(start = 1, fixed = true);
equation
  if fast then
    der(x) = -2.0 * x;
  else
    der(x) = -7.0 * x;
  end if;
end M;
"""

ELSEIF_MODEL = """\
model M
  parameter Integer sel = {sel};
  Real x(start = 1, fixed = true);
equation
  if sel == 1 then
    der(x) = -2.0 * x;
  elseif sel == 2 then
    der(x) = -7.0 * x;
  else
    der(x) = -9.0 * x;
  end if;
end M;
"""


def _equations(source: str) -> str:
    return " ".join(str(equation) for equation in flatten(parse(source), "M").equations)


def test_a_structural_if_equation_keeps_only_the_active_branch() -> None:
    on = _equations(IF_MODEL.format(fast="true"))
    assert "2.0" in on
    assert "7.0" not in on
    off = _equations(IF_MODEL.format(fast="false"))
    assert "7.0" in off
    assert "2.0" not in off


def test_a_structural_elseif_chain_picks_its_branch() -> None:
    mid = _equations(ELSEIF_MODEL.format(sel="2"))
    assert "7.0" in mid
    assert "2.0" not in mid
    assert "9.0" not in mid
    assert "9.0" in _equations(ELSEIF_MODEL.format(sel="5"))


def test_an_if_equation_condition_resolves_an_enumeration_literal() -> None:
    """The MSL writes `if homotopyType == Types.LimiterHomotopy.NoHomotopy` in a kept structural if.

    The condition names an enumeration literal, which has to become its number before the condition
    can be evaluated at flatten -- the same rewrite a top-level equation and a `when` condition get,
    which the if-equation condition was skipping.
    """
    source = (
        "package P\n"
        "  type Mode = enumeration(off, on);\n"
        "  model M\n"
        "    parameter Mode m = Mode.on;\n"
        "    parameter Boolean strict = true;\n"
        "    Real x(start = 1, fixed = true);\n"
        "  equation\n"
        "    if strict then\n"
        "      if m == P.Mode.on then\n"
        "        der(x) = -2.0 * x;\n"
        "      else\n"
        "        der(x) = -7.0 * x;\n"
        "      end if;\n"
        "    else\n"
        "      der(x) = -9.0 * x;\n"
        "    end if;\n"
        "  end M;\n"
        "end P;\n"
    )
    eqs = " ".join(str(equation) for equation in flatten(parse(source), "P.M").equations)
    assert "2.0" in eqs  # m == on holds, so the -2.0 branch survives
    assert "7.0" not in eqs
    assert "9.0" not in eqs


def test_an_if_equation_on_a_continuous_variable_is_refused() -> None:
    source = (
        "model M\n  Real x(start = 1, fixed = true);\n  Real y;\n"
        "equation\n  if x > 0 then\n    y = 2 * x;\n  else\n    y = -x;\n  end if;\n"
        "  der(x) = -y;\nend M;\n"
    )
    with pytest.raises(FlattenError, match="condition of an if-equation must be known"):
        flatten(parse(source), "M")


DEFERRED_IF = """\
package P
  record Data
    parameter Real pref = {pref};
  end Data;
  model Comp
    parameter Real pref = 1;
    Real tau;
    Real w(start = 1);
  equation
    if pref <= 0 then
      tau = 0;
    else
      tau = -w;
    end if;
    der(w) = tau;
  end Comp;
  model M
    Comp comp(pref = data.pref);
    Data data;
  end M;
end P;
"""


def _tau_rhs(source: str) -> ir.Expr:
    """The right-hand side of the equation that defines `comp.tau`, whichever branch it came from."""
    flat = flatten(parse(source), "P.M")
    equation = next(e for e in flat.equations if e.lhs == ir.Ref(("comp", "tau")))
    return equation.rhs


def test_an_if_equation_whose_condition_a_later_sibling_supplies_is_deferred() -> None:
    """The machine friction reads `frictionParameters.PRef`, set from a data record declared later.

    The condition cannot be evaluated where the if sits, so it is held and resolved once every
    parameter has settled. Both branches are reachable, chosen by the value the sibling supplies.
    """
    # data.pref = 0, so `pref <= 0` holds and the tau = 0 branch is the one kept.
    assert _tau_rhs(DEFERRED_IF.format(pref="0")) == ir.IntLiteral(0)
    # data.pref = 5, so the else branch survives: tau = -w.
    assert _tau_rhs(DEFERRED_IF.format(pref="5")) == ir.Unary("-", ir.Ref(("comp", "w")))


# The MSL SPICE transistors write `initial equation if UIC then vcs = IC; end if;` -- a
# parameter-gated initial condition, the same structural if the equation section allows, so it
# belongs among the initial equations, not refused as "only `a = b` may appear" there.
INITIAL_IF = """\
model M
  parameter Boolean useInitial = {useInitial};
  parameter Real v0 = 3;
  Real v(start = 0);
initial equation
  if useInitial then
    v = v0;
  end if;
equation
  der(v) = -v;
end M;
"""


def test_an_if_equation_may_appear_in_an_initial_equation_section() -> None:
    """The chosen branch's `a = b` lands among the initial equations; a false condition emits none."""
    kept = flatten(parse(INITIAL_IF.format(useInitial="true")), "M")
    assert [unparse_equation(e)[0].strip() for e in kept.initial_equations] == ["v = v0;"]
    dropped = flatten(parse(INITIAL_IF.format(useInitial="false")), "M")
    assert dropped.initial_equations == ()
