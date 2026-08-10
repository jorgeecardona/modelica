"""Two compile modes, one answer.

`docs/ideas/backends.md:66` asks for a choice: "Burn the parameters in as literals and fold
them, or leave them as an argument so that one compiled artefact serves a whole sweep. Two
compile modes, and the caller picks."

A choice is only a choice if both options compute the same thing, so the load-bearing test
here is the differential one: every corpus model, both modes, the same trajectory. What
folding is allowed to change is the *source*; a mode that also changed the answer would be a
second model wearing the first one's name.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim.emit import EmitError, emit

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MODELS = sorted((EXAMPLES / "models").glob("*.mo"))


def _spec(model: Path) -> Any:
    return tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))


def _system(model: Path) -> System:
    spec = _spec(model)
    flatten = pytest.importorskip("modelica.sim").flatten
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    return causalize(flatten(sources, model.stem))


@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_the_two_modes_integrate_to_the_same_trajectory(model: Path) -> None:
    """The whole claim. Every variable, on a grid, over the model's own stop time.

    Held to zero rather than to a tolerance, and that is what was measured: the two modes
    agree bit for bit on all 21 models. They should -- folding replaces a name by the number
    that name held, and the arithmetic that follows is the same operations in the same order.
    A tolerance here would hide the day that stops being true.
    """
    numpy = pytest.importorskip("numpy")
    build = pytest.importorskip("modelica.sim.kernel").build
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    system = _system(model)
    stop = float(_spec(model).get("simulate", {}).get("stop", 1.0))
    grid = numpy.linspace(0.0, stop, 21)

    loose = simulate(build(system), stop=stop, times=grid, tolerance=1e-10)
    tight = simulate(build(system, fold=True), stop=stop, times=grid, tolerance=1e-10)

    assert loose.names == tight.names
    here, there = numpy.searchsorted(loose.time, grid), numpy.searchsorted(tight.time, grid)
    for name in loose.names:
        numpy.testing.assert_array_equal(loose[name][here], tight[name][there], err_msg=name)


@pytest.mark.parametrize("model", MODELS, ids=[m.stem for m in MODELS])
def test_a_folded_unit_reads_no_parameter_out_of_the_array(model: Path) -> None:
    """``p`` is filled and never read. That is what "burned in" has to mean.

    The parameters still appear in `Code.parameters` and ``parameters(p)`` still fills the
    array, because the workspace layout is shared with the other mode and `Kernel.start`
    calls it. What must not survive is a *read*: a folded kernel whose answer still depended
    on ``p`` for some parameters and not others would be the one thing a caller choosing
    between two modes cannot check for themselves.
    """
    source = emit(_system(model), fold=True).source
    reads = [
        line
        for line in source.splitlines()
        if "p[" in line and "= p[" in line and not line.lstrip().startswith("p[")
    ]
    assert reads == [], reads


def test_the_folded_source_is_smaller_where_there_is_anything_to_fold() -> None:
    """Measured over the corpus, not asserted per model.

    Five of the 21 come out very slightly *larger*, and the reason is not a failure of the
    fold: a literal is longer than the name it replaced. ``2260000.0`` is nine characters
    where ``L`` is one, and a model whose parameters are read once each and never compared
    against anything has nothing to gain and those characters to lose. The claim is about the
    total.
    """
    total = [
        (len(emit(_system(m)).source), len(emit(_system(m), fold=True).source)) for m in MODELS
    ]
    argument, folded = sum(a for a, _ in total), sum(b for _, b in total)
    assert folded < argument
    assert sum(1 for a, b in total if b < a) > len(total) // 2


def test_a_parameter_that_cannot_be_evaluated_is_refused_by_name() -> None:
    """Folding is all or nothing, and says which one stopped it.

    A fold that quietly skipped a parameter would emit a kernel where ``p`` matters for some
    and not others -- the exact ambiguity the two modes exist to remove.
    """
    flatten = pytest.importorskip("modelica.sim").flatten
    source = """
    model Wobbly
      parameter Real k = time;
      Real x(start = 1.0, fixed = true);
    equation
      der(x) = -k * x;
    end Wobbly;
    """
    system = causalize(flatten([parse(source)], "Wobbly"))
    emit(system)  # the argument mode does not care what the binding says
    with pytest.raises(EmitError, match="cannot fold: cannot evaluate 'k'"):
        emit(system, fold=True)


def test_a_shared_name_is_defined_before_its_use_and_not_before_a_longer_name() -> None:
    """The regression the folded mode found: ``__s2`` is a substring of ``__s23``.

    `_define` placed a name in front of the first line *containing* it, so a name whose
    number was a prefix of a longer one landed in front of that longer one's first use --
    which can be before the assignment it reads. `AtwoodPendulum` folded is the model that
    does it: ``__s2 = 0.35 * der_w1`` written 28 lines above ``der_w1 = _x[0]``, and the
    generated Jacobian raised `UnboundLocalError`.

    Checked here as a property of the text rather than by running it, because running it is
    what the trajectory test above already does, and this names the shape of the fault.
    """
    import re  # noqa: PLC0415 - one test needs it

    source = emit(_system(EXAMPLES / "models" / "AtwoodPendulum.mo"), fold=True).source
    for body in source.split("\ndef ")[1:]:
        lines = body.splitlines()
        assigned: set[str] = set()
        for line in lines:
            for name in re.findall(r"\b_+s\d+\b", line.split("=", 1)[-1] if "=" in line else line):
                assert name in assigned, f"{name} read before it is written:\n{line.strip()}"
            head = line.split("=", 1)[0].strip() if "=" in line else ""
            if re.fullmatch(r"_+s\d+", head):
                assigned.add(head)
