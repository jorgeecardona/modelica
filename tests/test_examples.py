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

from modelica.build import Model, Package, to_modelica

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LIBRARY = EXAMPLES / "library"
MODELS = EXAMPLES / "models"

MODEL_FILES = sorted(MODELS.glob("*.mo"))
LIBRARY_FILES = sorted(LIBRARY.glob("*.mo"))

SOURCES = frozenset({"analytic", "reference", "measured"})
"""Where a reference number is allowed to have come from.

``analytic`` is a closed form and ``reference`` a trusted solver -- both of them things
this project could in principle have got wrong in the same way twice. ``measured`` is a
reading off a real machine, which cannot be wrong in this project's favour, and is the
only one of the three that says the model is about the world rather than about itself.
It buys that with tolerance: a measured point is asserted to within the model's honest
agreement with the hardware, degrees rather than ulps, so a spec that leans on measured
points alone will not catch a regression. Carry both kinds where you have both.
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


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_simulate_section_is_well_formed(model: Path) -> None:
    simulate = _spec(model).get("simulate")
    if simulate is None:
        return
    stop = simulate["stop"]
    assert stop > 0, "`stop` must be positive"
    assert simulate.get("method", "Radau") in METHODS, f"unknown method {simulate['method']!r}"

    for point in simulate.get("point", []):
        assert 0 <= point["t"] <= stop, f"point at t={point['t']} lies outside [0, {stop}]"
        assert point["values"], "a point with no values asserts nothing"
        assert "rtol" in point or "atol" in point, "a reference point needs a tolerance"
        assert point["source"] in SOURCES, f"unknown source {point['source']!r}"

    for invariant in simulate.get("invariant", []):
        assert invariant["expr"], "an invariant needs an expression"
        bounded = "min" in invariant or "max" in invariant
        assert ("value" in invariant) ^ bounded, "an invariant is either a value or bounds"

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


def _twin_text(source: Path) -> str:
    """Render the Python twin of ``source`` as Modelica.

    A library twin exposes its whole package as ``PACKAGE``; a model twin is a single
    class named after the file. Those are the only two shapes an example may take.
    """
    module = _load(source.with_suffix(".py"))
    target: object = getattr(module, "PACKAGE", None) or getattr(module, source.stem)
    if isinstance(target, Package):
        return to_modelica(target)
    assert isinstance(target, type), f"{source.stem} is neither a package nor a class"
    assert issubclass(target, Model), f"{source.stem} is not a modelica.build.Model"
    return to_modelica(target)


@pytest.mark.parametrize(
    "source", LIBRARY_FILES + MODEL_FILES, ids=_ids(LIBRARY_FILES + MODEL_FILES)
)
def test_python_twin_exists(source: Path) -> None:
    twin = source.with_suffix(".py")
    assert twin.exists(), (
        f"{source.name} has no Python twin. Every example is written twice, once in each "
        f"front-end, and the pair is what proves they agree."
    )


@pytest.mark.parametrize(
    "source", LIBRARY_FILES + MODEL_FILES, ids=_ids(LIBRARY_FILES + MODEL_FILES)
)
def test_python_twin_says_the_same_thing(source: Path) -> None:
    expected = _normalise(source.read_text(encoding="utf-8"))
    actual = _normalise(_twin_text(source))
    assert actual == expected


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
    if spec is None or not ("states" in spec or "state_count" in spec):
        pytest.skip("no state expectation")
    parse = _require("modelica.lang", "parse")
    flatten = _require("modelica.sim", "flatten")
    causalize = _require("modelica.sim", "causalize")

    system = causalize(flatten(_sources(model, parse), spec["model"]))
    if "states" in spec:
        assert sorted(system.states) == sorted(spec["states"])
    if "state_count" in spec:
        assert len(system.states) == spec["state_count"]


def _difference(rhs: Callable[[Any], Any], y: Any, np: Any) -> Any:
    """``d(dy)/dy`` by central differences, which is the thing the emitter has to beat."""
    columns: list[Any] = []
    for slot in range(len(y)):
        step = 1e-7 * max(1.0, abs(float(y[slot])))
        ahead, behind = y.copy(), y.copy()
        ahead[slot] += step
        behind[slot] -= step
        columns.append((rhs(ahead) - rhs(behind)) / (2.0 * step))
    return np.stack(columns, axis=1)


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
    """
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
        tolerance=spec.get("tolerance", 1e-6),
        method=spec.get("method", "Radau"),
    )
    trajectory = np.stack([result[name] for name in system.states])

    p, _, w = kernel.start()
    checked, t = 0, 0.0

    def rhs(state: Any) -> Any:
        return kernel.rhs(t, state, p, w, np.empty(len(state))).copy()

    for index in np.unique(np.linspace(0, len(result) - 1, 12).astype(int)):
        t = float(result.time[index])
        y = np.ascontiguousarray(trajectory[:, index])
        kernel.latch(t, y, p, w)
        exact = kernel.jacobian(t, y, p, w, np.zeros((len(y), len(y)))).copy()
        crude = _difference(rhs, y, np)
        if not np.all(np.isfinite(crude)) or not np.all(np.isfinite(exact)):
            continue
        checked += 1
        scale = max(1.0, float(np.max(np.abs(crude))))
        assert np.allclose(exact, crude, rtol=1e-4, atol=1e-6 * scale), (
            f"t = {t:.6g}\nemitted:\n{exact}\ndifferenced:\n{crude}"
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


def _reader(expr: str, names: tuple[str, ...]) -> Callable[[Any], float]:
    """Compile a Modelica expression into something callable on a column of the result."""
    parse_expression = _require("modelica.lang", "parse_expression")
    emit_expr = _require("modelica.sim", "emit_expr")

    symbols = {name: f"v[{index}]" for index, name in enumerate(names)}
    source = emit_expr(parse_expression(expr), symbols)
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
        label = f"{invariant['expr']!r} over {len(result)} samples"
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


@pytest.mark.parametrize("model", _simulations(), ids=_ids(MODEL_FILES))
def test_simulates_as_specified(model: Path) -> None:
    """The end of the pipeline: numbers, checked against numbers from somewhere else.

    Every reference point in a spec says where it came from -- ``analytic`` for a closed
    form, ``reference`` for a trusted solver -- so a pass here means the toolchain agrees
    with something that is not itself.
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

    result = simulate(system, stop=spec["stop"], tolerance=spec.get("tolerance", 1e-6))
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
