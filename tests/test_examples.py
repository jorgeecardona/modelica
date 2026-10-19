"""The example corpus, run as a test suite.

Two things happen here, and only the first one works today.

The **corpus guard** is unconditional: every ``.mo`` must have a well-formed ``.toml``
beside it, every ``requires`` must name a library that exists, and every reference point
must carry a tolerance and say where its number came from. It fails loudly, because a
malformed spec is a bug in the spec regardless of how much of the toolchain is written.

The **front-end equivalence** check is the other half: each model is also written in
Python with :mod:`modelica.build`, and the two must say the same thing. Until the parser
lands there is no way to compare the two as values, so the comparison runs through
:func:`modelica.lang.unparse` -- render the Python model as Modelica and diff it against
the handwritten file. That is weaker than comparing flat models, and it is meant to be
replaced by the stronger check the moment ``modelica.lang.parse`` exists.

Stages that need code nobody has written yet skip rather than fail, so ``make test`` stays
green and the skip list reads as a progress meter.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from modelica import ir
from modelica.build import Model, Package, to_ir, to_modelica
from modelica.ir.evaluate import evaluate
from modelica.ir.flatten import element

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LIBRARY = EXAMPLES / "library"
MODELS = EXAMPLES / "models"

MODEL_FILES = sorted(MODELS.glob("*.mo"))
LIBRARY_FILES = sorted(LIBRARY.glob("*.mo"))

SOURCES = frozenset({"analytic", "reference", "measured", "published"})
"""Where a reference number is allowed to have come from.

``analytic`` is a closed form and ``reference`` a trusted solver -- both of them things
this project could in principle have got wrong in the same way twice. ``measured`` is a
reading off a real machine, which cannot be wrong in this project's favour, and is the
only one of the three that says the model is about the world rather than about itself.
It buys that with tolerance: a measured point is asserted to within the model's honest
agreement with the hardware, degrees rather than ulps, so a spec that leans on measured
points alone will not catch a regression. Carry both kinds where you have both.

``published`` is the fourth and it closes the gap the first sentence admits: a number
somebody else computed, with somebody else's solver, and printed. It cannot be wrong in
the same way twice because nothing about it came from here. It carries a ``citation``,
required below, because a published number with no publication is a ``reference`` point
wearing a better name -- and the citation has to name *which table*, since a test set
prints run characteristics beside its solutions and only one of those is the answer.
"""
METHODS = frozenset({"RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"})


def _ids(paths: list[Path]) -> list[str]:
    return [p.stem for p in paths]


def _spec(model: Path) -> dict[str, Any]:
    return tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Corpus guard -- always runs
# --------------------------------------------------------------------------------------


def test_corpus_is_not_empty() -> None:
    assert MODEL_FILES, f"no models found under {MODELS}"


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_every_model_has_a_spec(model: Path) -> None:
    spec_file = model.with_suffix(".toml")
    assert spec_file.exists(), f"{model.name} has no {spec_file.name}"
    spec = _spec(model)
    assert spec.get("summary"), "a spec needs a one-line summary"
    assert isinstance(spec.get("teaches"), list), "`teaches` must be a list"
    assert isinstance(spec.get("requires"), list), "`requires` must be a list (use [] if none)"


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_required_libraries_exist(model: Path) -> None:
    for name in _spec(model)["requires"]:
        assert (LIBRARY / f"{name}.mo").exists(), f"{model.name} requires missing library {name}"


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_flatten_section_is_well_formed(model: Path) -> None:
    flatten = _spec(model).get("flatten")
    if flatten is None:
        return
    assert flatten["model"] == model.stem, "`flatten.model` must name the file's own class"
    if "states" in flatten:
        assert flatten["states"], "an empty `states` list asserts nothing -- omit the key"
    if "index" in flatten:
        assert flatten["index"] >= 0
    if "candidates" in flatten:
        assert flatten["candidates"], "an empty `candidates` list asserts nothing -- omit it"
        if "states" in flatten:
            assert sorted(flatten["candidates"][0]) == sorted(flatten["states"]), (
                "`candidates` is ordered with the chosen set first, so its first entry and "
                "`states` are the same claim and must agree"
            )


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_simulate_section_is_well_formed(model: Path) -> None:
    simulate = _spec(model).get("simulate")
    if simulate is None:
        return
    stop = simulate["stop"]
    assert stop > 0, "`stop` must be positive"
    absolute = simulate.get("absolute")
    assert absolute is None or absolute > 0, "`absolute` must be positive when given"
    assert simulate.get("method", "Radau") in METHODS, f"unknown method {simulate['method']!r}"

    for point in simulate.get("point", []):
        assert 0 <= point["t"] <= stop, f"point at t={point['t']} lies outside [0, {stop}]"
        assert point["values"], "a point with no values asserts nothing"
        assert "rtol" in point or "atol" in point, "a reference point needs a tolerance"
        assert point["source"] in SOURCES, f"unknown source {point['source']!r}"
        if point["source"] == "published":
            assert point.get("citation"), (
                "a published point needs a `citation` naming the work and the table it is"
                " printed in -- see SOURCES"
            )

    for invariant in simulate.get("invariant", []):
        assert invariant["expr"], "an invariant needs an expression"
        bounded = "min" in invariant or "max" in invariant
        assert ("value" in invariant) ^ bounded, "an invariant is either a value or bounds"
        after = invariant.get("after")
        assert after is None or 0 <= after < stop, f"`after = {after}` is outside [0, {stop})"

    events = simulate.get("events")
    if events is not None:
        times = events["times"]
        assert times == sorted(times), "event times must be in order"
        assert all(0 <= t <= stop for t in times), "an event lies outside the horizon"


# --------------------------------------------------------------------------------------
# Front-end equivalence: the `.mo` file and its Python twin must say the same thing
# --------------------------------------------------------------------------------------

# Numeric literals, so `1e4` and `10000.0` compare equal. Spelling a number differently is
# not a difference between the front-ends; it is a difference between two ways of writing
# the same value, and only one of them is reachable from Python.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")


def _normalise(text: str) -> list[str]:
    """Reduce Modelica source to what the two front-ends must agree on."""
    numbers_folded = _NUMBER.sub(lambda m: repr(float(m.group())), text)
    return [line.rstrip() for line in numbers_folded.splitlines() if line.strip()]


def _load(path: Path) -> ModuleType:
    """Import an example by path, as if it were on ``sys.path``."""
    if str(LIBRARY) not in sys.path:
        sys.path.append(str(LIBRARY))
    if path.stem in sys.modules:
        return sys.modules[path.stem]
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None, f"cannot import {path}"
    assert spec.loader is not None, f"no loader for {path}"
    module = importlib.util.module_from_spec(spec)
    # Register before executing: `build.package()` looks the module up in `sys.modules`.
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "source", LIBRARY_FILES + MODEL_FILES, ids=_ids(LIBRARY_FILES + MODEL_FILES)
)
def test_python_twin_exists(source: Path) -> None:
    twin = source.with_suffix(".py")
    assert twin.exists(), (
        f"{source.name} has no Python twin. Every example is written twice, once in each "
        f"front-end, and the pair is what proves they agree."
    )


def _twin_class(source: Path) -> Any:
    """The Python twin of ``source`` as the :class:`modelica.ir.Class` it builds."""
    module = _load(source.with_suffix(".py"))
    target: object = getattr(module, "PACKAGE", None) or getattr(module, source.stem)
    if not isinstance(target, Package):
        assert isinstance(target, type), f"{source.stem} is neither a package nor a class"
        assert issubclass(target, Model), f"{source.stem} is not a modelica.build.Model"
    return to_ir(target)


def _disagreement(  # noqa: C901, PLR0911 -- one exit per kind of difference, which is the point
    left: Any, right: Any, path: str = ""
) -> str | None:
    """Where two IR values first disagree, named so the message points at a declaration.

    ``assert a == b`` on two packages prints two screens of dataclass and leaves the reader
    to diff them by eye. What is wanted is "Electrical.Resistor.R", so this walks the two in
    step and reports the first place they part company.
    """
    if left == right:
        return None
    if isinstance(left, ir.Class) and isinstance(right, ir.Class):
        here = f"{path}.{left.name}" if path else left.name
        for field in ("restriction", "partial", "comment", "imports", "extends", "literals"):
            if getattr(left, field) != getattr(right, field):
                return (
                    f"{here}: {field} is {getattr(left, field)!r} against {getattr(right, field)!r}"
                )
        for attribute in ("components", "classes", "redeclarations"):
            mine = {member.name: member for member in getattr(left, attribute)}
            theirs = {member.name: member for member in getattr(right, attribute)}
            if set(mine) != set(theirs):
                only = sorted(set(mine) ^ set(theirs))
                return f"{here}: {attribute} differ by {only}"
            for name, member in mine.items():
                found = _disagreement(member, theirs[name], here)
                if found is not None:
                    return found
        if left.equations != right.equations:
            return f"{here}: the equations differ"
        return f"{here}: differs, and every part compared equal -- look at the annotation"
    if isinstance(left, ir.Component) and isinstance(right, ir.Component):
        for field in (
            "type_name",
            "variability",
            "causality",
            "connector",
            "scoping",
            "comment",
            "dimensions",
            "replaceable",
            "constraining",
            "modification",
            "annotation",
        ):
            mine, theirs = getattr(left, field), getattr(right, field)
            if mine != theirs:
                return f"{path}.{left.name}.{field}:\n        mo: {mine!r}\n        py: {theirs!r}"
    return f"{path}: {left!r} against {right!r}"


@pytest.mark.parametrize(
    "source", LIBRARY_FILES + MODEL_FILES, ids=_ids(LIBRARY_FILES + MODEL_FILES)
)
def test_python_twin_builds_the_same_value(source: Path) -> None:
    """The real equivalence, and the only one a *library* can have.

    A library is a package of classes and has no flat model of its own -- there is nothing
    to flatten until a model instantiates it -- so `test_front_ends_flatten_to_the_same_model`
    cannot reach the four in `examples/library/`. What both front ends do promise for every
    file, library or model, is the same `ir.Class` (`docs/architecture.md:41`), and that is
    a value, so it can simply be compared.

    This replaces a normalised-text diff, which was the wrong instrument twice over: it
    failed on formatting that means nothing, and it pinned `modelica.lang.unparse` -- any
    change to the printer broke tests about the corpus rather than tests about the printer.
    """
    parse = _require("modelica.lang", "parse")
    (expected,) = parse(source.read_text(encoding="utf-8"), source.name).classes
    found = _disagreement(expected, _twin_class(source))
    assert found is None, f"the two front ends disagree at {found}"


@pytest.mark.parametrize(
    "source", LIBRARY_FILES + MODEL_FILES, ids=_ids(LIBRARY_FILES + MODEL_FILES)
)
def test_the_corpus_is_written_the_way_the_printer_prints(source: Path) -> None:
    """Corpus hygiene, and no longer a claim about the Python front end.

    Every example is written in the form `modelica.lang.unparse` produces, so a reader who
    prints a model gets back the file they are looking at. The Python twin does not appear
    here at all: it builds an equal value, and the printer is a function, so equal values
    print equally -- checking the twin's *text* would only re-test the printer.
    """
    parse = _require("modelica.lang", "parse")
    text = source.read_text(encoding="utf-8")
    assert _normalise(to_modelica(parse(text, source.name))) == _normalise(text)


# --------------------------------------------------------------------------------------
# Stages waiting on code that does not exist yet
# --------------------------------------------------------------------------------------


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - depends on which extras are installed
        pytest.skip(f"{name} is not installed")


def _require(module: str, what: str) -> Callable[..., Any]:
    imported = _module(module)
    if not hasattr(imported, what):
        pytest.skip(f"{module}.{what} is not written yet")
    return getattr(imported, what)


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_parses(model: Path) -> None:
    parse = _require("modelica.lang", "parse")
    program = parse(model.read_text(encoding="utf-8"))
    expected = _spec(model)["parse"]["classes"]
    assert [c.name for c in program.classes] == expected


def _sources(model: Path, parse: Callable[..., Any]) -> list[Any]:
    """The model and the libraries its ``requires`` names, parsed from ``.mo``."""
    files = [LIBRARY / f"{lib}.mo" for lib in _spec(model)["requires"]] + [model]
    return [parse(f.read_text(encoding="utf-8")) for f in files]


def _twin_sources(model: Path) -> list[Any]:
    """The same set, built with :mod:`modelica.build` instead."""
    libraries = [_load(LIBRARY / f"{lib}.py").PACKAGE for lib in _spec(model)["requires"]]
    return [*libraries, getattr(_load(model.with_suffix(".py")), model.stem)]


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_front_ends_flatten_to_the_same_model(model: Path) -> None:
    """The real equivalence test: not the same text, the same mathematical object."""
    parse = _require("modelica.lang", "parse")
    flatten = _require("modelica.sim", "flatten")

    from_source = flatten(_sources(model, parse), model.stem)
    from_python = flatten(_twin_sources(model), model.stem)
    assert from_python == from_source


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_flattens_as_specified(model: Path) -> None:
    spec = _spec(model).get("flatten")
    if spec is None:
        pytest.skip("no flatten section")
    parse = _require("modelica.lang", "parse")
    flatten = _require("modelica.sim", "flatten")

    flat = flatten(_sources(model, parse), spec["model"])
    if "equations" in spec:
        assert len(flat.equations) == spec["equations"]
    if "balanced" in spec:
        assert flat.is_balanced() == spec["balanced"]


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_selects_the_specified_states(model: Path) -> None:
    """Which variables end up integrated -- a property of the *causalized* model.

    Flattening only reports what appears differentiated, which is a larger set: aliases
    survive it (``damper.s_rel`` is really ``mass.s``), and an index-3 model like the
    pendulum differentiates four variables to integrate two. Alias elimination and dummy
    derivatives are what cut it down, so this asks the stage that does them.
    """
    spec = _spec(model).get("flatten")
    if spec is None or not spec.keys() & {"states", "state_count", "candidates"}:
        pytest.skip("no state expectation")
    parse = _require("modelica.lang", "parse")
    flatten = _require("modelica.sim", "flatten")
    causalize = _require("modelica.sim", "causalize")

    system = causalize(flatten(_sources(model, parse), spec["model"]))
    if "states" in spec:
        assert sorted(system.states) == sorted(spec["states"])
    if "state_count" in spec:
        assert len(system.states) == spec["state_count"]
    if "candidates" in spec:
        # Ordered, and the order carries the claim: `System.candidates` puts the chosen set
        # first, so this one assertion pins both which sets are admissible and which one the
        # selection rule took. A model whose candidates are told apart by the geometry does
        # not need the key; one whose candidates are the same shape does, because there the
        # answer comes from a tie-break rather than from the model.
        assert [list(c) for c in system.candidates] == [list(c) for c in spec["candidates"]]


def _difference(rhs: Callable[[Any], Any], y: Any, np: Any) -> tuple[Any, Any]:
    """``d(dy)/dy`` by central differences, and how far it admits it might be out.

    Two step sizes, not one, because the crude tool has to be allowed to say how crude it is
    being. A central difference carries a truncation error of order ``h^2 f'''``, so halving
    the step should quarter it: ``|D(h) - D(2h)| / 3`` is that error, measured rather than
    assumed, and ``(4 D(h) - D(2h)) / 3`` is the Richardson extrapolation it implies, good to
    ``h^4``.

    Both are needed, and the reason is `DrainingTank`. Its right-hand side is a square root,
    so as the tank empties the exact derivative goes to infinity and the third derivative
    goes there faster; a difference quotient near the bottom is wrong by parts in ten
    thousand while the emitted derivative is exact. Comparing against a fixed tolerance there
    does not test the Jacobian, it tests the finite difference -- and fails the wrong one of
    the two. Comparing within the quotient's own demonstrated uncertainty is both stricter
    where it is sharp and honest where it is not.
    """
    coarse: list[Any] = []
    fine: list[Any] = []
    for slot in range(len(y)):
        step = 1e-6 * max(1.0, abs(float(y[slot])))
        for width, into in ((step, coarse), (step / 2.0, fine)):
            ahead, behind = y.copy(), y.copy()
            ahead[slot] += width
            behind[slot] -= width
            into.append((rhs(ahead) - rhs(behind)) / (2.0 * width))
    near, far = np.stack(fine, axis=1), np.stack(coarse, axis=1)
    return (4.0 * near - far) / 3.0, np.abs(near - far) / 3.0


@pytest.mark.slow
@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_the_jacobian_agrees_with_a_difference_quotient(model: Path) -> None:
    """The emitted derivative, against the crude one, at points a real run visits.

    A difference quotient is the wrong tool for the job -- it is why the exact one is
    emitted at all -- but it is an *independent* tool, which is what a cross-check needs.
    Where the two are both valid they have to agree, and every way of getting the source
    transformation wrong (a chain rule term dropped, a solved block differentiated as if it
    were explicit, a latch read one way in the value and the other in the derivative) moves
    one and not the other.

    The states come from the simulated trajectory rather than from a perturbation of the
    initial point, because the initial point is where models are at their most degenerate:
    a pendulum hanging still, a ball not yet falling, every rate zero. ``latch`` re-freezes
    the relations at each sample so the discrete configuration matches the state -- and both
    functions then read that same configuration, which is precisely the claim being tested.

    Samples where the right-hand side itself is not finite are skipped. Those are states the
    integrator would never have been asked about (the tank below empty, a state set past the
    point where its own linear algebra is singular), and a difference quotient of an
    infinity says nothing about anything.

    The run is deliberately *not* held to the spec's tolerance. What the trajectory is for is
    supplying states worth asking about, and both functions are then evaluated at the same
    one -- so a state that is off the true trajectory by a millimetre is as good a question
    as a state that is on it. Asking at ``1e-6`` rather than the ``1e-10`` some specs demand
    costs this check nothing and saves the corpus's stiffest models a great deal.
    """
    _agrees_with_a_difference_quotient(model, "rhs", "jacobian")


@pytest.mark.slow
@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_the_output_jacobian_agrees_with_a_difference_quotient(model: Path) -> None:
    """``observe_jacobian`` against a difference quotient of ``observe`` -- the claim for H.

    Everything the test above says applies here unchanged; only the pair of functions moves.
    It is worth having separately rather than folding into that one, because the two reach
    different code. ``jacobian`` stops at the state derivatives, so a variable that no state
    derivative depends on is never differentiated at all. ``observe_jacobian`` reads off
    *every* variable of the model, aliases included -- and the aliases are differentiated
    nowhere else, because causalization eliminated them before the solver ever saw them.

    An alias is also exactly what a model is usually measured through: a probe temperature,
    a branch current, a lag. So the row that an estimator actually needs is the row this
    test is the only one to check.
    """
    _agrees_with_a_difference_quotient(model, "observe", "observe_jacobian")


@pytest.mark.slow
@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_the_indicator_jacobian_agrees_with_a_difference_quotient(model: Path) -> None:
    """``indicator_jacobian`` against a difference quotient of ``indicators`` -- ``g_y``.

    The third of the same shape, and the newest. It is what an event-crossing sensitivity
    needs: when a parameter moves, a crossing happens at a different time, and
    ``dt*/dp = -(g_y S) / (g_t + g_y f)``. Nothing emitted ``g_y`` until
    `13-08-2026-forward-sensitivity-refuses-a-model-with-events` asked for it.

    Nine corpus models have events and the rest skip. Measured when it landed, worst
    disagreement 1.8e-9 on `Pendulum` and at or below 2.5e-10 everywhere else.

    The **zero rows are the interesting output**, not a degenerate case. A crossing over
    ``time`` alone has no state in it, so its row is identically zero and its crossing time
    does not move when a parameter does. Eight of `RoastingMachine`'s nine are that, and
    `Setback`'s only one is -- which is the same census that entry took from the IR, arrived
    at here from the emitted arithmetic instead.
    """
    _agrees_with_a_difference_quotient(model, "indicators", "indicator_jacobian")


def _agrees_with_a_difference_quotient(model: Path, value: str, derivative: str) -> None:
    """One emitted derivative against a crude one, at points a real run visits."""
    spec = _spec(model).get("simulate")
    if spec is None:
        pytest.skip("no simulate section")
    np = _module("numpy")
    parse = _require("modelica.lang", "parse")
    flatten = _require("modelica.sim", "flatten")
    causalize = _require("modelica.sim", "causalize")
    build = _require("modelica.sim", "build")
    simulate = _require("modelica.sim", "simulate")

    system = causalize(flatten(_sources(model, parse), model.stem))
    kernel = build(system)
    result = simulate(
        system,
        stop=spec["stop"],
        tolerance=max(spec.get("tolerance", 1e-6), 1e-6),
        absolute=spec.get("absolute"),
        method=spec.get("method", "Radau"),
    )
    trajectory = np.stack([result[name] for name in system.states])

    p, _, w = kernel.start()
    rows = {"observe": kernel.observed, "indicators": kernel.events}.get(value, kernel.states)
    if not rows:
        pytest.skip(f"{model.stem} has no {value} to differentiate")
    checked, t = 0, 0.0

    def evaluate(state: Any) -> Any:
        # A copy of the workspace per call, because ``observe`` writes its answers back into
        # it: shared, each perturbation would be asked from the previous one's leftovers and
        # the quotient would be of two different functions.
        return getattr(kernel, value)(t, state, p, w.copy(), np.empty(rows)).copy()

    for index in np.unique(np.linspace(0, len(result) - 1, 12).astype(int)):
        t = float(result.time[index])
        y = np.ascontiguousarray(trajectory[:, index])
        kernel.latch(t, y, p, w)
        emitted = getattr(kernel, derivative)
        exact = emitted(t, y, p, w.copy(), np.zeros((rows, len(y)))).copy()
        crude, doubt = _difference(evaluate, y, np)
        if not np.all(np.isfinite(crude)) or not np.all(np.isfinite(exact)):
            continue
        checked += 1
        scale = max(1.0, float(np.max(np.abs(crude))))
        slack = 1e-6 * scale + 4.0 * doubt
        assert np.all(np.abs(exact - crude) <= 1e-4 * np.abs(crude) + slack), (
            f"{model.stem} {derivative} at t = {t:.6g}\n"
            f"emitted:\n{exact}\ndifferenced:\n{crude}\nits own doubt:\n{doubt}"
        )
    assert checked, "every sample was skipped -- the check never ran"


UNSIMULATED: dict[str, str] = {}
"""Models whose *simulation* is a known gap, and precisely which gap."""


def _simulations() -> list[Any]:
    return [
        pytest.param(model, marks=pytest.mark.xfail(reason=reason, strict=True))
        if (reason := UNSIMULATED.get(model.stem))
        else pytest.param(model)
        for model in MODEL_FILES
    ]


def _scalarise(node: ir.Expr) -> ir.Expr:
    """Rewrite ``T[1]`` into the flat variable named ``T[1]``.

    An invariant is written against the *model*, where a cell is subscripted, and read from
    a result, where every variable is a scalar with the index in its name. Flattening does
    exactly this rewrite for the equations, and it exports `element` so that this second
    reader spells the name the same way rather than inventing a parallel convention.
    """

    def go(inner: ir.Expr) -> ir.Expr:
        match inner:
            case ir.Subscript(base=base, indices=indices):
                resolved = [int(evaluate(go(i), {})) for i in indices]
                return ir.Ref((*base.parts[:-1], element(base.parts[-1], resolved)))
            case ir.Binary(op=op, lhs=lhs, rhs=rhs):
                return ir.Binary(op, go(lhs), go(rhs))
            case ir.Unary(op=op, operand=operand):
                return ir.Unary(op, go(operand))
            case ir.Call(name=name, args=args):
                return ir.Call(name, tuple(go(a) for a in args))
            case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
                return ir.IfExpr(go(condition), go(then), go(otherwise))
            case _:
                return inner

    return go(node)


def _reader(expr: str, names: tuple[str, ...]) -> Callable[[Any], float]:
    """Compile a Modelica expression into something callable on a column of the result."""
    parse_expression = _require("modelica.lang", "parse_expression")
    emit_expr = _require("modelica.sim", "emit_expr")

    symbols = {name: f"v[{index}]" for index, name in enumerate(names)}
    source = emit_expr(_scalarise(parse_expression(expr)), symbols)
    code = compile(source, f"<invariant:{expr}>", "eval")
    numpy = _module("numpy")

    def read(column: Any) -> float:
        return float(eval(code, {"np": numpy, "v": column}))  # noqa: S307 - our own source

    return read


def _check_points(spec: dict[str, Any], result: Any, np: Any) -> None:
    for point in spec.get("point", []):
        sampled = result.at(point["t"])
        for name, expected in point["values"].items():
            assert name in result, f"{name} is not a variable of the model"
            got = sampled[name]
            assert np.isclose(
                got, expected, rtol=point.get("rtol", 0.0), atol=point.get("atol", 0.0)
            ), f"{name} at t={point['t']}: got {got!r}, want {expected!r} ({point['source']})"


def _check_events(spec: dict[str, Any], result: Any) -> None:
    events = spec.get("events")
    if events is None:
        return
    expected = events["times"]
    got = [event.time for event in result.events]
    assert len(got) == len(expected), f"fired {len(got)} events, expected {len(expected)}"
    atol = events.get("atol", 1e-6)
    for actual, wanted in zip(got, expected, strict=True):
        assert abs(actual - wanted) <= atol, f"event at {actual!r}, expected {wanted!r}"


def _check_invariants(spec: dict[str, Any], result: Any, np: Any) -> None:
    for invariant in spec.get("invariant", []):
        read = _reader(invariant["expr"], result.names)
        values = np.array([read(result.values[:, j]) for j in range(len(result))])
        # `after` is for something that is only true once the model has settled: a
        # controller reaches its setpoint, a transient dies away. Asserting it from t = 0
        # would assert it of the transient, which is the part that is supposed to move.
        after = invariant.get("after")
        if after is not None:
            values = values[result.time >= after]
            assert len(values), f"nothing is sampled after t = {after}"
        label = f"{invariant['expr']!r} over {len(values)} samples"
        if after is not None:
            label += f" after t = {after}"
        if "value" in invariant:
            wanted = invariant["value"]
            close = np.isclose(
                values, wanted, rtol=invariant.get("rtol", 0.0), atol=invariant.get("atol", 0.0)
            )
            worst = values[np.argmax(np.abs(values - wanted))]
            assert close.all(), f"{label}: drifted to {worst!r}, expected {wanted!r}"
        else:
            # A bound that an event enforces is only ever met to the accuracy the event was
            # located with: the ball's height is zero at the bounce to within the root
            # finder's few ulps, and asking for exactly zero is asking the arithmetic for
            # something it cannot give. `atol` is how far past the bound is still a pass.
            slack = invariant.get("atol", 0.0)
            low = invariant.get("min", -np.inf)
            high = invariant.get("max", np.inf)
            bottom, top = values.min(), values.max()
            assert bottom >= low - slack, f"{label}: fell to {bottom!r}, floor is {low!r}"
            assert top <= high + slack, f"{label}: rose to {top!r}, ceiling is {high!r}"


@pytest.mark.slow
@pytest.mark.parametrize("model", _simulations(), ids=_ids(MODEL_FILES))
def test_simulates_as_specified(model: Path) -> None:
    """The end of the pipeline: numbers, checked against numbers from somewhere else.

    Every reference point in a spec says where it came from -- ``analytic`` for a closed
    form, ``reference`` for a trusted solver, ``published`` for somebody else's printed
    answer -- so a pass here means the toolchain agrees with something that is not itself.
    """
    spec = _spec(model).get("simulate")
    if spec is None:
        pytest.skip("no simulate section")
    np = _module("numpy")
    parse = _require("modelica.lang", "parse")
    flatten = _require("modelica.sim", "flatten")
    causalize = _require("modelica.sim", "causalize")
    simulate = _require("modelica.sim", "simulate")

    system = causalize(flatten(_sources(model, parse), model.stem))
    result = simulate(
        system,
        stop=spec["stop"],
        tolerance=spec.get("tolerance", 1e-6),
        # `absolute` defaults to `tolerance` inside `simulate`, which is right for a model
        # whose states are all order one and wrong for one whose states are not. `E5` has
        # `y1` at 2e-3 and everything else below 4e-10, and its source says so in as many
        # words: "a scalar absolute error control is unsuitable ... for this problem". A spec
        # that needs the two apart now says so instead of being unable to.
        absolute=spec.get("absolute"),
        times=sorted({point["t"] for point in spec.get("point", [])}) or None,
        method=spec.get("method", "Radau"),
    )

    _check_points(spec, result, np)
    _check_events(spec, result)
    _check_invariants(spec, result, np)


def test_the_pendulum_changes_its_mind_about_what_to_integrate() -> None:
    """A full swing needs both state sets, so the run has to move between them.

    ``(y, vy)`` cannot get through the bottom and ``(x, vx)`` cannot get through the sides,
    which the spec already proves indirectly: the invariants in ``Pendulum.toml`` hold over
    two seconds, and no single set stays regular that long. What it does not show is the
    mechanism, and the mechanism is what breaks quietly -- a switch that never fires leaves
    a run that only happens to survive, and one that fires both ways at once chatters.

    So this asks for the crossings by name. Two ordered pairs, each with no latch, because a
    switch is not a question the model keeps asking; alternating in time, because leaving a
    set and immediately returning to it is the failure the hysteresis margin exists to
    prevent; and spread out, rather than bunched at one instant.
    """
    parse = _require("modelica.lang", "parse")
    flatten = _require("modelica.sim", "flatten")
    causalize = _require("modelica.sim", "causalize")
    emit = _require("modelica.sim", "emit")
    simulate = _require("modelica.sim", "simulate")

    model = MODELS / "Pendulum.mo"
    spec = _spec(model)["simulate"]
    system = causalize(flatten(_sources(model, parse), model.stem))

    code = emit(system)
    assert code.switching is not None, "a model with a choice has to record which way it went"
    switches = [trigger for trigger in code.events if trigger.switch is not None]
    assert [str(trigger) for trigger in switches] == [
        "switch to (x, vx)",
        "switch to (y, vy)",
    ]
    assert all(trigger.latch is None for trigger in switches)

    result = simulate(
        system,
        stop=spec["stop"],
        tolerance=spec.get("tolerance", 1e-6),
        absolute=spec.get("absolute"),
    )
    causes = [event.cause for event in result.events]
    assert causes[:2] == ["switch to (x, vx)", "switch to (y, vy)"], (
        f"the pendulum never left its first state set: {causes}"
    )
    assert all(a != b for a, b in pairwise(causes)), (
        f"a state set was left and re-entered without the other in between: {causes}"
    )
    times = [event.time for event in result.events]
    assert all(b - a > 0.1 for a, b in pairwise(times)), (
        f"switches this close are chatter, not a swing: {times}"
    )
