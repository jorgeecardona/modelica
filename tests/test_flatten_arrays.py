"""Arrays, scalarised at flatten time.

The design this pins is that **nothing below :mod:`modelica.ir.flatten` sees an index**.
A declared shape becomes one scalar variable per element, a ``for`` equation is written out
once per value of its index, and a subscript is rewritten to the name of the element it
selects. So these tests look at names and counts rather than at a new kind of node: if an
array survived into a ``FlatModel``, every stage after it would need to learn what one is.

The other half is what stays refused. A shape or a subscript decides how many equations a
model has, so it must be a number *here*; a range whose bounds are not known when the model
is built cannot be deferred to the solver, and this file says so in each case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse_equation

if TYPE_CHECKING:
    from modelica.ir.flat import FlatModel


def flat(body: str, name: str = "M") -> FlatModel:
    """Flatten a one-class model written as its body."""
    return flatten([parse(f"model {name}\n{body}end {name};\n")], name)


def equations(model: FlatModel) -> list[str]:
    """The flat equations as text, which is how a reader checks an expansion."""
    return [unparse_equation(e)[0].strip() for e in model.equations]


def test_a_declared_shape_becomes_one_variable_per_element() -> None:
    model = flat("  parameter Integer n = 3;\n  Real T[n];\n")
    assert [v.name for v in model.variables] == ["n", "T[1]", "T[2]", "T[3]"]


def test_a_two_dimensional_shape_expands_in_row_major_order() -> None:
    model = flat("  Real A[2, 3];\n")
    assert [v.name for v in model.variables] == [
        "A[1, 1]",
        "A[1, 2]",
        "A[1, 3]",
        "A[2, 1]",
        "A[2, 2]",
        "A[2, 3]",
    ]


def test_an_element_keeps_the_attributes_of_the_declaration() -> None:
    """Every element starts at 20, and nothing about attributes was written twice.

    An element *is* the declaration: the elements are built by handing the component back
    with its shape removed, so every rule about attributes already applies to them.
    """
    model = flat('  Real T[3](start = 20, unit = "degC");\n')
    assert {v.name for v in model.variables} == {"T[1]", "T[2]", "T[3]"}
    for variable in model.variables:
        assert variable.unit == "degC"
        assert variable.start is not None


def test_a_for_equation_is_written_out_once_per_index() -> None:
    model = flat(
        "  parameter Integer n = 4;\n"
        "  Real T[n];\n"
        "equation\n"
        "  for i in 1:n loop\n"
        "    T[i] = i;\n"
        "  end for;\n"
    )
    assert equations(model) == ["T[1] = 1;", "T[2] = 2;", "T[3] = 3;", "T[4] = 4;"]


def test_a_subscript_may_be_arithmetic_on_the_index() -> None:
    """``T[i - 1]`` and ``T[i + 1]`` are what a stencil is made of.

    They are also why the index is substituted *before* the names are resolved: by the time
    resolution sees the subscript, ``i + 1`` is a number.
    """
    model = flat(
        "  Real T[4];\nequation\n  for i in 2:3 loop\n    T[i] = T[i - 1] + T[i + 1];\n  end for;\n"
    )
    assert equations(model) == ["T[2] = T[1] + T[3];", "T[3] = T[2] + T[4];"]


def test_the_step_of_a_range_is_honoured_and_may_be_negative() -> None:
    forward = flat("  Real x[9];\nequation\n  for i in 1:3:9 loop\n    x[i] = i;\n  end for;\n")
    assert equations(forward) == ["x[1] = 1;", "x[4] = 4;", "x[7] = 7;"]
    backward = flat("  Real x[3];\nequation\n  for i in 3:-1:1 loop\n    x[i] = i;\n  end for;\n")
    assert equations(backward) == ["x[3] = 3;", "x[2] = 2;", "x[1] = 1;"]


def test_an_empty_range_writes_no_equations() -> None:
    """``for i in 2:1`` is legal and means nothing happens, so silence is the answer."""
    model = flat("  Real x[3];\nequation\n  for i in 2:1 loop\n    x[i] = 0;\n  end for;\n")
    assert equations(model) == []


def test_a_for_equation_may_initialise() -> None:
    """Initialising ``n`` cells is exactly what a method-of-lines model needs.

    An initial section used to accept only ``a = b``, which left no way to write it.
    """
    model = flat(
        "  Real T[3];\ninitial equation\n  for i in 1:3 loop\n    T[i] = 20;\n  end for;\n"
    )
    assert len(model.initial_equations) == 3
    assert not model.equations


def test_a_nested_loop_shadows_the_outer_index_when_it_reuses_the_name() -> None:
    model = flat(
        "  Real A[2, 2];\n"
        "equation\n"
        "  for i in 1:2 loop\n"
        "    for j in 1:2 loop\n"
        "      A[i, j] = 0;\n"
        "    end for;\n"
        "  end for;\n"
    )
    assert equations(model) == ["A[1, 1] = 0;", "A[1, 2] = 0;", "A[2, 1] = 0;", "A[2, 2] = 0;"]


def test_a_component_array_expands_into_instances() -> None:
    """An array of components is the same rule one level up.

    ``c[2]`` is ``c[1]`` and ``c[2]``, each instantiated whole. Nothing in the element's own
    class knows it is in an array.
    """
    source = (
        "package P\n"
        "  model Cell\n"
        "    Real v;\n"
        "  equation\n"
        "    der(v) = -v;\n"
        "  end Cell;\n"
        "  model M\n"
        "    Cell c[2];\n"
        "  end M;\n"
        "end P;\n"
    )
    model = flatten([parse(source)], "P.M")
    assert [v.name for v in model.variables] == ["c[1].v", "c[2].v"]


# --------------------------------------------------------------------------------------
# What stays refused, and why each one has to be
# --------------------------------------------------------------------------------------


def test_a_shape_that_is_not_known_when_the_model_is_built_is_refused() -> None:
    """The number of equations cannot depend on something decided later.

    That is what makes a parameter used as a dimension a *structural* parameter.
    """
    with pytest.raises(FlattenError, match="must be known when the model is flattened"):
        flat("  Real x;\n  Real T[x];\n")


def test_a_fractional_shape_is_refused() -> None:
    with pytest.raises(FlattenError, match="whole number"):
        flat("  parameter Real n = 2.5;\n  Real T[n];\n")


def test_a_range_with_unknown_bounds_is_refused() -> None:
    with pytest.raises(FlattenError, match="must be known when the model is flattened"):
        flat("  Real x;\n  Real T[3];\nequation\n  for i in 1:x loop\n    T[i] = 0;\n  end for;\n")


def test_a_range_that_steps_by_zero_is_refused() -> None:
    with pytest.raises(FlattenError, match="steps by zero"):
        flat("  Real T[3];\nequation\n  for i in 1:0:3 loop\n    T[i] = 0;\n  end for;\n")


def test_an_array_of_connectors_is_refused_by_name() -> None:
    """Arrays of connectors are the next thing after this, and are refused at the parser.

    Without that, the bracket met a parser expecting a comma and the model was blamed for a
    syntax error it did not have.
    """
    source = (
        "package P\n"
        "  connector Pin\n"
        "    Real v;\n"
        "    flow Real i;\n"
        "  end Pin;\n"
        "  model M\n"
        "    Pin p[2];\n"
        "  equation\n"
        "    for k in 1:1 loop\n"
        "      connect(p[k], p[k + 1]);\n"
        "    end for;\n"
        "  end M;\n"
        "end P;\n"
    )
    with pytest.raises(ModelicaSyntaxError, match="arrays of connectors"):
        flatten([parse(source)], "P.M")
