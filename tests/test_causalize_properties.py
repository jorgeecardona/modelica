"""Causalization against generated models, rather than against models a person chose.

`tests/test_causalize.py` holds fifteen four-line models, each aimed at one pass, and
`tests/test_examples.py` runs the whole corpus end to end. Both are worth having and both
have the same blind spot: every model in them was written by somebody thinking about the
cases the implementation was thinking about. The corpus is dense where intuition is dense.

`docs/ideas/verification.md:255` says what the other instrument is, and says it as a limit on
the model checker rather than as an omission: "Matching, BLT sorting and Pantelides are graph
algorithms over unbounded inputs, and not finite automata. Their properties -- the matching is
perfect, index reduction terminates, differentiating preserves the solution set -- **want
property-based testing over generated models**."

Two generators, because a single one cannot say both things worth saying.

`solvable` builds a model that is causal *by construction* -- states first, then algebraic
variables each written over what came before -- and then shuffles the equations and flips
their sides at random. Causalization has to rediscover an order that is known to exist, so
every property below is a real assertion rather than one that passes because the model was
rejected. `arbitrary` throws structure at it with no such guarantee, and asks only the
question that has an answer for every input: a `System` or a `CausalizationError`, and
nothing else.

The properties are structural. No number is evaluated here and none needs to be: whether a
matching is perfect, whether a schedule reads only what it has already computed, and whether
the blocks partition the unknowns are all statements about incidence.

**This file found two bugs on the day it was written**, both in alias elimination, and it
keeps the model each one shrank to:

* a three-equation linear ODE that causalization never returned from, because two
  eliminations put a cycle in the derivative map and `_Work.level` walked it forever.
  Fixed; `test_a_linear_ode_system_terminates` is the regression test, and the register entry
  carries the sha.
* a triangular model refused as structurally singular, because the matching that succeeds is
  not the one whose leftovers are states. Still open, carried below as an expected failure.

The register has both. The generated properties `assume` past a refusal and past a model that
exceeds `BUDGET`, so they go on saying what they say about the models that get through rather
than being taken down with them.
"""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from modelica.ir.causalize import CausalizationError, causalize
from modelica.ir.rewrite import names
from modelica.ir.source import flatten
from modelica.lang import parse

if TYPE_CHECKING:
    from modelica.ir.system import Mode, System

# Generated models are small on purpose. The failures these algorithms have live in the
# *shape* of a model and not in its size -- a variable that is its own derivative's alias, a
# constraint over two states, a block that goes trivial after reduction -- and a five-variable
# model can hold every one of them while still being readable when hypothesis shrinks to it.
# The bug this file found shrank to three equations.
VARIABLES = 5
TERMS = 3

# Two budgets, because the two generators cost three orders of magnitude apart. Measured
# 2026-08-13 over 120 draws each: `solvable` is 1.4 ms median and 3.1 ms worst, `arbitrary`
# is 5.1 ms median but 663 ms at the 95th percentile and 933 ms worst -- an unstructured
# five-equation model can send index reduction and state-set enumeration a long way. That
# is itself filed; here it is the reason the second number is not the first.
STRUCTURED = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
UNSTRUCTURED = settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


# --------------------------------------------------------------------------------------
# Generating models
# --------------------------------------------------------------------------------------


@st.composite
def _term(draw: st.DrawFn, readable: list[str]) -> str:
    """One summand: a variable, a scaled variable, a shifted one, or a constant."""
    if not readable:
        return str(draw(st.integers(1, 9)))
    name = draw(st.sampled_from(readable))
    scale = draw(st.integers(2, 5))
    return draw(st.sampled_from([name, f"{scale} * {name}", f"({name} + 1)", name]))


@st.composite
def _sum(draw: st.DrawFn, readable: list[str]) -> str:
    return " + ".join(draw(st.lists(_term(readable), min_size=1, max_size=TERMS)))


@st.composite
def solvable(draw: st.DrawFn) -> str:
    """A model built so that a causal order exists, then hidden.

    ``s`` states get ``der(x_i) = f(states, algebraics)``, and ``k`` algebraic variables are
    written one at a time over everything already introduced -- which is a triangular system
    and therefore has a perfect matching and index one. Then the equations are shuffled and
    each has a fifty-fifty chance of being written the other way round, so nothing about the
    order they arrive in tells causalization the answer.

    Hiding the order is the whole point. A generator that emitted the triangle as written
    would pass every property below while testing nothing but the parser.
    """
    states = draw(st.integers(1, 3))
    algebraics = draw(st.integers(0, VARIABLES - states))
    state_names = [f"x{i}" for i in range(states)]
    algebraic_names = [f"a{i}" for i in range(algebraics)]

    declarations = [
        f"  Real {name}(start = {draw(st.integers(1, 4))}, fixed = true);" for name in state_names
    ]
    declarations += [f"  Real {name};" for name in algebraic_names]

    equations: list[tuple[str, str]] = []
    known = list(state_names)
    for name in algebraic_names:
        equations.append((name, draw(_sum(known))))
        known.append(name)
    equations.extend((f"der({name})", draw(_sum(known))) for name in state_names)

    lines: list[str] = []
    for index in draw(st.permutations(range(len(equations)))):
        left, right = equations[index]
        lines.append(f"  {right} = {left};" if draw(st.booleans()) else f"  {left} = {right};")
    return "model M\n" + "\n".join(declarations) + "\nequation\n" + "\n".join(lines) + "\nend M;\n"


@st.composite
def arbitrary(draw: st.DrawFn) -> str:
    """Equations over variables and their derivatives, with nothing promised.

    Most of these are structurally singular, over-determined, or index-2 by accident -- 102
    of 120 draws were refused when this was measured -- and that is the point: the only
    property asserted about them is the one that holds for every input, which is that
    causalization answers rather than falls over.
    """
    count = draw(st.integers(1, VARIABLES))
    variables = [f"x{i}" for i in range(count)]
    readable = [*variables, *(f"der({name})" for name in variables)]
    equations = draw(
        st.lists(st.tuples(_sum(readable), _sum(readable)), min_size=1, max_size=VARIABLES)
    )
    declarations = "\n".join(f"  Real {name}(start = 1, fixed = true);" for name in variables)
    body = "\n".join(f"  {left} = {right};" for left, right in equations)
    return f"model M\n{declarations}\nequation\n{body}\nend M;\n"


class BudgetError(Exception):
    """Causalization did not come back inside the budget."""


BUDGET = 5.0
"""Seconds a single generated model gets. Generous: the median draw is 1.4 ms.

A budget rather than hypothesis's `deadline`, which measures how long a call *took* and
therefore cannot interrupt one that never ends. This can, because a `while` loop in pure
Python checks for signals between bytecodes, and a draw that never ends is a thing this
generator produces: see the register entry named on `test_a_linear_ode_system_terminates`.

Unix only. `signal.setitimer` does not exist on Windows and this project's CI is Linux, so
the guard degrades to no guard rather than to a skipped test -- on a platform where a hang
would hang, which is the honest trade until someone runs the suite there.
"""

HAS_ALARM = hasattr(signal, "setitimer")


def _causalize(source: str, seconds: float = BUDGET) -> System:
    """Causalize under the budget, so a draw that never ends does not take the suite with it."""
    if not HAS_ALARM:  # pragma: no cover - Windows
        return causalize(flatten([parse(source)], "M"))

    def ring(_signal: int, _frame: object) -> None:
        raise BudgetError

    previous = signal.signal(signal.SIGALRM, ring)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return causalize(flatten([parse(source)], "M"))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _causalized(source: str) -> System:
    """Causalize, or hand this draw back to hypothesis.

    `assume` rather than a bare `return`, so that a run where *every* draw is refused fails
    the health check instead of passing on nothing. Both outcomes discarded here have a
    register entry of their own and a test below that carries the shrunk model, so nothing is
    being swallowed: the properties in this section are about the *schedule*, and a model with
    no schedule has nothing to say about one.
    """
    try:
        return _causalize(source)
    except (CausalizationError, BudgetError):
        assume(False)  # noqa: FBT003 - hypothesis's spelling for "discard this draw"
        raise


# --------------------------------------------------------------------------------------
# What the schedule has to satisfy
# --------------------------------------------------------------------------------------


def _reads(mode: Mode) -> list[set[str]]:
    """Every name each block reads, block by block, in schedule order."""
    reads: list[set[str]] = []
    for block in mode.blocks:
        seen: set[str] = set()
        for equation in block.equations:
            seen.update(names(equation.lhs))
            seen.update(names(equation.rhs))
        for _, expression in block.torn:
            seen.update(names(expression))
        if block.assignment is not None:
            seen.update(names(block.assignment))
        reads.append(seen)
    return reads


def _known(system: System, mode: Mode) -> set[str]:
    """What is available before the first block runs."""
    return {
        "time",
        *(variable.name for variable in system.parameters),
        *mode.states,
        *system.aliases,
    }


def _assert_causal(system: System, source: str) -> None:
    """Every block reads only what an earlier one computed, plus its own unknowns."""
    for mode in system.modes:
        known = _known(system, mode)
        for position, reading in enumerate(_reads(mode)):
            block = mode.blocks[position]
            unresolved = reading - known - set(block.solved)
            assert not unresolved, (
                f"block {position} reads {sorted(unresolved)} before anything computes them"
                f"\n{source}"
            )
            known.update(block.solved)


@STRUCTURED
@given(solvable())
def test_a_schedule_only_reads_what_it_has_already_computed(source: str) -> None:
    """The property BLT sorting exists to establish, checked on the sorted output.

    A block may read a parameter, the time, a state, or a variable an *earlier* block solved
    -- and its own unknowns, which is what makes a simultaneous block simultaneous. A block
    that reads a variable computed later is a sorting bug, and it is one no corpus model
    would necessarily catch: with fifteen models the order that happens to come out is
    usually the order the equations were written in.
    """
    _assert_causal(_causalized(source), source)


@STRUCTURED
@given(solvable())
def test_the_matching_is_perfect_and_no_variable_is_solved_twice(source: str) -> None:
    """Every unknown is solved exactly once, and every block is square.

    Two claims that sound like one. *Square* is a property of each block -- a strongly
    connected component has as many equations as unknowns, and a block that does not is a
    matching that lost or gained one. *Exactly once* is a property of the whole schedule, and
    it is the half a per-block assertion cannot see: two blocks both solving ``a`` is a
    matching that is not a bijection, and the second one silently wins.
    """
    system = _causalized(source)
    for mode in system.modes:
        solved: list[str] = []
        for block in mode.blocks:
            assert len(block.unknowns) == len(block.equations), f"block is not square\n{source}"
            solved.extend(block.solved)
        repeated = sorted({name for name in solved if solved.count(name) > 1})
        assert not repeated, f"solved more than once: {repeated}\n{source}"
        # Nothing is left over either: a variable that is neither a state, a parameter, an
        # alias nor solved by a block has no way of ever getting a value.
        missing = (
            {variable.name for variable in system.variables}
            - set(solved)
            - _known(system, system.modes[0])
        )
        assert not missing, f"nothing computes {sorted(missing)}\n{source}"


@STRUCTURED
@given(solvable())
def test_every_state_has_a_derivative_something_computes(source: str) -> None:
    """A state whose derivative nothing produces cannot be integrated.

    The map is `Mode.derivative`, and it is not always ``der_x``: alias elimination routinely
    finds that a variable the modeller named *is* the derivative. Either way the name it
    gives has to be a name the schedule assigns, or the right-hand side reads an array slot
    nothing wrote.
    """
    system = _causalized(source)
    for mode in system.modes:
        # A block, *or* something already known. The second half is not a loophole: this
        # property first failed on `der(x0) = x0; der(x1) = x0`, where alias elimination
        # correctly makes `x0` the derivative of `x1` -- and `x0` is a state, so it is known
        # rather than computed. The strict version of this assertion was wrong about the
        # engine, and stating it correctly is what the failure was for.
        available = {name for block in mode.blocks for name in block.solved} | _known(system, mode)
        for state in mode.states:
            assert state in mode.derivative, f"{state} is a state with no derivative\n{source}"
            assert mode.derivative[state] in available, (
                f"nothing computes {mode.derivative[state]}, the derivative of {state}\n{source}"
            )


@STRUCTURED
@given(solvable())
def test_a_model_with_a_causal_order_needs_no_index_reduction(source: str) -> None:
    """Nothing was differentiated, because there was nothing to differentiate.

    A triangular model is index one, so a reported index above one says Pantelides fired on a
    model that did not need it. That is not a crash and would not show up in any corpus test
    that only checks the numbers at the far end.
    """
    system = _causalized(source)
    assert system.index == 1, f"index {system.index} on a model with a causal order\n{source}"


@UNSTRUCTURED
@given(arbitrary())
def test_causalization_answers_or_refuses(source: str) -> None:
    """The property that holds for every input, including the malformed ones.

    A `System` or a `CausalizationError`, and nothing else. That is a lower bar than the ones
    above and it is the one worth having on unstructured input: a `KeyError` from inside the
    matching, a `RecursionError` from index reduction, or an infinite loop are all failures a
    caller cannot act on, and the first two are what an error message is for.

    Where it does answer, the schedule still has to be causal -- an accidentally-solvable
    random model deserves the same scrutiny as a deliberately-solvable one.
    """
    try:
        system = _causalize(source)
    except CausalizationError:
        return
    _assert_causal(system, source)


# --------------------------------------------------------------------------------------
# What it found
# --------------------------------------------------------------------------------------


REFUSED = """model M
  Real x0(start = 1, fixed = true);
  Real a0;
  Real a1;
equation
  der(x0) = a0;
  a0 + (x0 + 1) + x0 = a1;
  a0 = x0 + x0;
end M;
"""
"""The three-line model `solvable` shrank to, kept verbatim.

Triangular as written -- ``a0 = 2*x0``, then ``a1 = a0 + 2*x0 + 1``, then ``der(x0) = a0`` --
and refused as structurally singular. Alias elimination reads ``der(x0) = a0`` as an alias
and renames ``a0`` to be *the* derivative of ``x0``, which is the documented and usually
desirable move; the matching then reaches for ``a0`` first because a higher differentiation
level goes first, leaving equation ``a0 = 2*x0`` nothing free but the state ``x0`` and
stranding ``a1``. Both matchings are maximum. Only one of them leaves a state as the
leftover, and `_Matching`'s own docstring says why that is the whole game: "the unknowns left
over become the states, so the choice among them *is* state selection".
"""


@pytest.mark.xfail(
    reason="a-triangular-model-is-refused-as-structurally-singular-once-alias-elimination-renames",
    raises=CausalizationError,
    strict=True,
)
def test_a_model_with_a_causal_order_is_never_refused() -> None:
    """A model built triangular has a perfect matching, so causalization must find one.

    Measured over 120 `solvable` draws on 2026-08-13: 9 refused. Written out as the single
    shrunk model rather than left as a generated property, because an expected failure that
    re-searches for its own counterexample on every run is a slow way to say something a
    three-line model says exactly.
    """
    _causalize(REFUSED)


HANGS = """model M
  Real x0(start = 1, fixed = true);
  Real x1(start = 1, fixed = true);
  Real x2(start = 1, fixed = true);
equation
  der(x0) = x2;
  der(x1) = 2 * x0;
  der(x2) = x0;
end M;
"""
"""Three explicit ODEs that causalization used to not come back from.

`der(x0) = x2` is an alias, so `x2` becomes *the* derivative of `x0`; `der(x2) = x0` is then
the same shape, so `x0` becomes the derivative of `x2`. That made the `integral` map cyclic
and `_Work.level` walked it forever -- over 90 s before it was given up on. `_permits` guarded
against the one-step version and stopped one hop short of this one.

Both other lines are load-bearing, measured by changing one at a time while it still hung:
drop the scale on `der(x1) = 2 * x0` and that equation is eliminable too, so nothing is left
for the matching to walk and `level` is never called (2.0 ms); break the cycle with
`der(x2) = 3 * x0` (1.6 ms).

Fixed in `_permits`, which now walks the chain the substitution would join.
"""


@pytest.mark.skipif(not HAS_ALARM, reason="the budget needs signal.setitimer")
def test_a_linear_ode_system_terminates() -> None:
    """Three states, index one, no algebraic loop -- and it used to never return.

    `BUDGET` is what makes this a test rather than a hung suite: the assertion is that
    causalization comes back at all, and `_causalize` raises `BudgetError` if it does not.
    The answer is checked too, because terminating with the wrong states would be a worse
    outcome than hanging -- `x2` is `der(x0)` and `x0` is `der(x2)`, so all three are states
    and the model is index 1.
    """
    system = _causalize(HANGS)
    assert system.index == 1
    assert set(system.states) == {"x0", "x1", "x2"}
    _assert_causal(system, HANGS)


@pytest.mark.parametrize(
    ("flow", "constraint"),
    [("-u", "x + y = 3"), ("2 * u", "2 * x + 3 * y = 7"), ("-u", "x = y + 1")],
    ids=["shared", "scaled", "shifted"],
)
def test_a_constraint_between_two_states_is_reduced_not_refused(flow: str, constraint: str) -> None:
    """Index reduction, on the smallest models that need it.

    `solvable` cannot generate these: it builds a triangle, and a triangle is index one by
    construction. Two positions moved by one shared flow and tied by a constraint is the
    smallest linear thing that is not -- the constraint has to be differentiated before there
    is an equation for ``u``, which is what Pantelides is for.

    The first draft of this test wrote the constraint over two *independently* driven states
    and was over-determined rather than index-2, which causalization refused with "index
    reduction did not terminate". Correctly: three equations and two unknowns is not a model.
    The algebraic ``u`` is what makes it one, and that is worth writing down because the wrong
    version looks more like a DAE than the right one does.

    Written out rather than generated because there are three of them and each is a line. A
    generator that produced index-2 models *on purpose* is worth having and is a bigger piece
    of work than this file; the register carries it.
    """
    source = f"""model M
  Real x(start = 1, fixed = true);
  Real y(start = 2, fixed = true);
  Real u;
equation
  der(x) = u;
  der(y) = {flow};
  {constraint};
end M;
"""
    system = _causalize(source)
    assert system.index >= 2, f"index {system.index} on a model with a constraint"
    assert len(system.states) == 1, f"one position is free, not {len(system.states)}"
    _assert_causal(system, source)
