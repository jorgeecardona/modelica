"""The compiled path, run rather than claimed.

:func:`modelica.sim.build` takes ``jit=True`` and hands every generated function to Numba.
Until this file existed nothing executed that path -- not one test, and Numba is in no
dependency group -- so "there is a Numba backend" was a sentence in a docstring rather than
a fact about the code. An optimisation nobody runs is an optimisation that is already broken
and has not been told yet.

The check is *differential* rather than absolute. Every assertion here compares the compiled
function against the interpreted one on the same inputs, which is the property that actually
matters: `jit=True` must be an optimisation, not a second implementation with its own
opinions. It also means the test needs no reference numbers of its own and cannot drift away
from what the emitter emits.

Numba cannot be installed beside this project's numpy -- it pins `numpy<2.5` up to its
current release -- so this file skips itself in the normal environment. `moon run :test-jit`
builds a throwaway environment where numpy resolves down and runs it there. The reason it is
a separate task rather than a `dev` dependency is written out in `pyproject.toml`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

pytest.importorskip("numba", reason="the compiled path needs numba; see moon run :test-jit")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LIBRARY = EXAMPLES / "library"
MODELS = EXAMPLES / "models"

COVERED = ("Robertson", "Pendulum", "BouncingBall", "DrainingTank", "RoastingMachine")
"""Models chosen for the constructs they force the emitter to write, not for coverage.

Compiling costs real time, so this is five models rather than the whole corpus, picked so
that between them every shape in the generated source is compiled at least once:
``Robertson`` is explicit blocks and nothing else; ``Pendulum`` is the mode dispatcher, a
Newton iteration and a 3x3 linear solve, which is the only place `np.linalg.solve` with a
2-D right-hand side appears; ``BouncingBall`` is a ``when`` with a ``reinit``;
``DrainingTank`` is a latched relation read through the workspace; and ``RoastingMachine``
is a nine-segment schedule against ``time``, which is the shape that used to be
uncompilable. Compiling every function of each, measured:

    Robertson         5.6 s
    BouncingBall      2.1 s
    DrainingTank      2.4 s
    RoastingMachine   3.9 s
    Pendulum         37.9 s     <- two modes, so every function is written and compiled twice

``RoastingMachine`` is here **because** it was the counterexample, and it is worth recording
what it was a counterexample to. It used to be absent with a finding attached: Numba did not
compile it in any usable sense -- 31 minutes and 3.8 GB of resident memory without finishing,
against seconds for everything else -- and the nine-branch chain also came within about
twenty-five frames of the default recursion limit of 1000 during Numba's *recursive*
control-flow walk, so it raised ``RecursionError`` from a deep call site and proceeded from a
shallow one on the same machine and the same hour.

The diagnosis was wrong, and this is the useful part. The chain in ``rhs`` was never the
problem: it is nine constants and Numba compiles it in a fifth of a second. The problem was
what *differentiating* it produced. A schedule of constants has derivative zero on every
segment, so ``d_dx`` wrote ``0.0 if w[30] > 0.5 else 0.0 if w[31] > 0.5 else ...`` -- a
nine-branch chain whose every answer is the same number -- and ``simplify`` had no fold for
a conditional with two identical arms. Every block Jacobian carried it, so it landed in
``rhs``, ``jacobian``, ``observe``, ``indicators`` and ``transition`` alike. One fold in
``modelica.ir.derive.simplify`` removes it, and the measurements move like this:

                    before          after
    rhs             5.6 s           0.2 s
    jacobian        15.6 s, 4.4 GB, out of memory
                                    2.6 s
    observe         9.2 s           0.4 s
    indicators      5.5 s           0.2 s
    transition      7.5 s           0.5 s

So the claim that "a model whose behaviour is a long schedule against ``time`` is effectively
not compilable" was a statement about a missing fold, not about Numba and not about the model.
The recursion limit was never touched. The model was never flattened into branchless
arithmetic, which is what the earlier note refused to do and was right to refuse.
"""


def _system(stem: str) -> Any:
    from modelica.lang import parse  # noqa: PLC0415 - after the numba skip
    from modelica.sim import causalize, flatten  # noqa: PLC0415 - after the numba skip

    model = MODELS / f"{stem}.mo"
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    files = [LIBRARY / f"{name}.mo" for name in spec["requires"]] + [model]
    return causalize(flatten([parse(f.read_text(encoding="utf-8")) for f in files], stem))


def _pair(stem: str) -> tuple[Any, Any]:
    """The same model built twice: interpreted, and compiled."""
    from modelica.sim import build, emit  # noqa: PLC0415 - after the numba skip

    code = emit(_system(stem))
    return build(code), build(code, jit=True)


@pytest.mark.parametrize("stem", COVERED)
def test_the_compiled_kernel_answers_what_the_interpreted_one_answers(stem: str) -> None:
    """Every generated function, compiled against interpreted, at the model's own start.

    Bit-for-bit, and that is not optimism: the compiled code performs the same operations in
    the same order on the same doubles. A difference here would mean Numba reassociated
    something, which is exactly the kind of surprise worth failing on rather than absorbing
    into a tolerance.

    ``assert_array_equal`` rather than ``pytest.approx(rel=0, abs=0)``, and the difference
    matters here rather than being a style choice: ``approx`` holds that a NaN never equals
    anything, itself included, so it would report a disagreement between two kernels that
    agreed perfectly on producing one.

    **Each kernel gets its own copy of the workspace, every call.** That is not hygiene, it is
    the test being correct at all: ``w`` is not an input, it is state. An implicit block writes
    its answer back into ``w`` as the starting guess for next time, so calling the interpreted
    kernel first *changes what the compiled one is asked*. Shared, this compared one kernel
    against the other's leftovers, and on the pendulum -- where the guess decides which side of
    a singularity Newton walks in from -- that was the difference between an answer and a
    ``LinAlgError``. It looked exactly like a Numba bug for as long as it took to notice.
    """
    import numpy as np  # noqa: PLC0415 - after the numba skip

    slow, fast = _pair(stem)
    p, y, start = slow.start()
    states, observed, events = slow.states, slow.observed, slow.events

    calls: dict[str, Callable[[Any, Any], Any]] = {
        "rhs": lambda k, w: k.rhs(0.0, y, p, w, np.empty(states)),
        "jacobian": lambda k, w: k.jacobian(0.0, y, p, w, np.zeros((states, states))),
        "observe": lambda k, w: k.observe(0.0, y, p, w, np.empty(observed)),
        "observe_jacobian": lambda k, w: k.observe_jacobian(
            0.0, y, p, w, np.zeros((observed, states))
        ),
        "indicators": lambda k, w: k.indicators(0.0, y, p, w, np.empty(events)),
    }
    for name, call in calls.items():
        mine, theirs = start.copy(), start.copy()
        expected = np.asarray(call(slow, mine)).copy()
        actual = np.asarray(call(fast, theirs)).copy()
        np.testing.assert_array_equal(actual, expected, err_msg=f"{stem}.{name}", strict=True)
        np.testing.assert_array_equal(theirs, mine, err_msg=f"{stem}.{name} workspace", strict=True)

    for event in range(events):
        mine, theirs = start.copy(), start.copy()
        expected = np.asarray(slow.transition(event, 0.0, y, p, mine, np.empty(states))).copy()
        actual = np.asarray(fast.transition(event, 0.0, y, p, theirs, np.empty(states))).copy()
        np.testing.assert_array_equal(
            actual, expected, err_msg=f"{stem}.transition({event})", strict=True
        )
        np.testing.assert_array_equal(
            theirs, mine, err_msg=f"{stem}.transition({event}) workspace", strict=True
        )


def test_a_compiled_run_lands_where_an_interpreted_one_does() -> None:
    """Not just the functions -- the whole loop, events and all.

    ``BouncingBall`` because the thing a compiled kernel could plausibly get wrong is not the
    arithmetic but the discontinuity: the workspace it writes through, the indicator SciPy
    roots on, and the transition that reinitialises the state. Agreeing to 1e-12 after four
    bounces means all three survived compilation.
    """
    import numpy as np  # noqa: PLC0415 - after the numba skip

    from modelica.sim import build, emit, simulate  # noqa: PLC0415 - after the numba skip

    code = emit(_system("BouncingBall"))
    spec = tomllib.loads((MODELS / "BouncingBall.toml").read_text(encoding="utf-8"))["simulate"]
    stop = float(spec["stop"])
    grid: list[float] = [stop * step / 39 for step in range(40)]

    slow = simulate(build(code), stop=spec["stop"], tolerance=1e-9, times=grid)
    fast = simulate(build(code, jit=True), stop=spec["stop"], tolerance=1e-9, times=grid)

    assert len(slow.events) == len(fast.events), "a bounce went missing under compilation"
    for one, other in zip(slow.events, fast.events, strict=True):
        assert other.time == pytest.approx(one.time, abs=1e-12), f"bounce moved: {one} vs {other}"
    for name in ("h", "v"):
        mine = np.array([slow.at(float(t))[name] for t in grid])
        theirs = np.array([fast.at(float(t))[name] for t in grid])
        assert theirs == pytest.approx(mine, abs=1e-12), f"{name} diverged after compilation"


def test_a_zero_pivot_is_an_infinity_in_both_tiers() -> None:
    """Division by zero must mean the same thing compiled as interpreted.

    Numba's default ``error_model="python"`` raises ``ZeroDivisionError`` the way CPython
    does; the interpreted kernel divides numpy scalars and gets an infinity. `_accelerate`
    asks for the numpy model to close that fork, and this is the assertion that it did.

    Written against a division this test performs itself rather than against a model, and
    that is a deliberate weakening of an earlier version of it. No model in the corpus
    actually reaches a division by exactly zero -- the pendulum at ``y = -L`` comes closest
    and its Newton lands on an ``x`` near 1e-16, not on 0.0 -- so a test that claimed to catch
    one was testing a situation it had not produced. What is genuinely true is narrower and
    still worth holding: *if* a pivot reaches zero, the two tiers agree about what happens,
    and they only agree because of the flag.
    """
    import numpy as np  # noqa: PLC0415 - after the numba skip

    from modelica.sim.kernel import (  # noqa: PLC0415 - after the numba skip
        _accelerate,  # pyright: ignore[reportPrivateUsage]
    )

    source = "def divide(a, b):\n    return a / b\n"
    namespace: dict[str, Any] = {"np": np}
    exec(compile(source, "<zero>", "exec"), namespace)  # noqa: S102 - the emitter's own road
    interpreted = namespace["divide"]
    _accelerate(namespace)
    compiled = namespace["divide"]

    with np.errstate(divide="ignore", invalid="ignore"):  # the whole point is to divide by zero
        assert interpreted(np.float64(1.0), np.float64(0.0)) == np.inf
    assert compiled(1.0, 0.0) == np.inf, (
        "the compiled tier raises where the interpreted one returns an infinity --"
        " _accelerate has lost error_model='numpy'"
    )
    assert np.isnan(compiled(0.0, 0.0))
