"""``inner``/``outer``: one declaration, reached from anywhere below it.

This is the only construct in the language where a component's equations depend on a
declaration that appears nowhere in its own text. A body writes ``outer Environment
environment;`` and gets whichever environment the scene it was dropped into declared.

Two things follow, and they are what this file is about.

An ``outer`` component **declares nothing**. There is no ``high.environment`` in the flat
model, only the one ``environment`` the enclosing class declared, and every reference
resolves there. An implementation that created a variable per ``outer`` would give each
body its own gravity and the models would still run.

And the picture has to agree. :mod:`modelica.diagram` draws components; drawing an
``outer`` would put a box there for an instance that does not exist, which is the failure
``docs/architecture.md:217`` exists to prevent. The last test here is that check.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from modelica import diagram, ir
from modelica.ir.evaluate import EvaluationError, evaluate
from modelica.ir.flatten import FlattenError
from modelica.ir.source import flatten
from modelica.lang import parse
from modelica.lang.lexer import ModelicaSyntaxError
from modelica.lang.unparse import unparse, unparse_equation

if TYPE_CHECKING:
    from pathlib import Path

SCENE = """\
model Env
  parameter Real g = 9.81;
end Env;

model Body
  outer Env env;
  Real h(start = 10, fixed = true);
  Real v(start = 0, fixed = true);
equation
  der(h) = v;
  der(v) = -env.g;
end Body;

model Scene
  inner Env env;
  Body a;
  Body b(h(start = 20));
end Scene;
"""


def test_it_round_trips() -> None:
    program = parse(SCENE)
    assert parse(unparse(program)) == program
    assert unparse(program) == SCENE


def test_an_outer_component_declares_nothing() -> None:
    """One environment for two bodies, and no `a.env` at all."""
    names = [v.name for v in flatten(parse(SCENE), "Scene").variables]
    assert names == ["env.g", "a.h", "a.v", "b.h", "b.v"]


def test_both_bodies_read_the_same_declaration() -> None:
    equations = [unparse_equation(e)[0] for e in flatten(parse(SCENE), "Scene").equations]
    assert "der(a.v) = -env.g;" in equations
    assert "der(b.v) = -env.g;" in equations


def test_an_unmatched_outer_is_refused_by_name() -> None:
    """Flattening `Body` on its own: nothing above it declares the inner it asks for."""
    with pytest.raises(FlattenError, match="is `outer` and no enclosing class declares"):
        flatten(parse(SCENE), "Body")


def test_a_nearer_inner_shadows_a_further_one() -> None:
    """The rule is *nearest* enclosing, so a scene inside a scene keeps its own."""
    source = SCENE + (
        "model Nested\n"
        "  inner Env env(g = 1.0);\n"
        "  Scene inner_scene;\n"
        "  Body direct;\n"
        "end Nested;\n"
    )
    equations = [unparse_equation(e)[0] for e in flatten(parse(source), "Nested").equations]
    # The body declared here reads this class's inner...
    assert "der(direct.v) = -env.g;" in equations
    # ...and the bodies inside the nested scene read the one that scene declares.
    assert "der(inner_scene.a.v) = -inner_scene.env.g;" in equations


def test_inner_outer_together_is_refused_by_name() -> None:
    """Declaring one and referring to another is a third rule, not the two at once."""
    with pytest.raises(ModelicaSyntaxError, match="'inner outer' is not supported yet"):
        parse("model M\n  inner outer Real g;\nend M;\n")


def test_the_diagram_does_not_draw_an_outer(tmp_path: Path) -> None:
    """The check the issue asked for: the picture shows what the flattener builds.

    `Body` writes `outer Env env`, so a naive walk puts an `a.env` box in the picture --
    an instance the flat model has no variable for. Two bodies would show two
    environments where the model has one.
    """
    path = tmp_path / "scene.mo"
    path.write_text(SCENE, encoding="utf-8")
    drawn = {node.id for node in diagram.of(str(path), model="Scene", depth=2).nodes}
    assert "a.env" not in drawn
    assert {"env", "a", "b"} <= drawn


# --------------------------------------------------------------------------------------
# What a span is for: an error below the parser that points at the line
# --------------------------------------------------------------------------------------

MISTAKE = "model Motor\n  parameter Real R = 1;\n  Real i;\nequation\n  i = Rw * 2.0;\nend Motor;\n"


def test_an_unknown_name_says_where_it_was_written() -> None:
    """The message this whole span exists for, pinned so it cannot rot back.

    `CHANGELOG.md` records the real occurrence: a modification bound a name that did not
    exist, code generation died on "nothing computes 'Rw'", and it was found by reading the
    file rather than by being told where to look.
    """
    with pytest.raises(FlattenError, match=re.escape("unknown name 'Rw' in an equation at Motor.mo:5:7")):
        flatten(parse(MISTAKE, "Motor.mo"), "Motor")


def test_the_interpreter_says_where_too() -> None:
    equation = parse(MISTAKE, "Motor.mo").classes[0].equations[0]
    assert isinstance(equation, ir.Equal)
    wanted = re.escape("nothing in the environment answers 'Rw' at Motor.mo:5:7")
    with pytest.raises(EvaluationError, match=wanted):
        evaluate(equation.rhs, {})


def test_a_model_built_from_python_says_what_it_always_said() -> None:
    """No source text, no span, and the message is no worse than it was."""
    built = ir.Equal(ir.Ref(("i",)), ir.Binary("*", ir.Ref(("Rw",)), ir.RealLiteral(2.0)))
    wanted = re.escape("nothing in the environment answers 'Rw'") + "$"
    with pytest.raises(EvaluationError, match=wanted):
        evaluate(built.rhs, {})
