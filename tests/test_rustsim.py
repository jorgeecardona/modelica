"""The compiled Rust backend, called from Python, against the interpreter.

`modelica.sim.rustsim` is the packaging the Rust backend was missing: one shared library per
model, built once and cached by content hash, loaded through ctypes, answering with the same
`Result` the interpreter returns. These tests hold the three promises that make it usable --
the trajectory is the interpreter's, the events are the interpreter's (the tally, the instants and
the causes), and the second call for a model does not compile -- on a model of each kind: smooth
(RLCCircuit), with events (BouncingBall), and switching state sets at runtime (Pendulum, compared
on its coordinates).

`rustc` is optional exactly as in `test_rustgen`: no compiler, no test.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import rustsim

# Whole file is slow: it compiles the Rust backend and runs it per model. `make test` skips it;
# CI and `make test-all` run it.
pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
needs_rustc = pytest.mark.skipif(rustsim.compiler() is None, reason="no rustc on this machine")


def _system(stem: str):  # noqa: ANN202 - the return type is System, imported lazily by the package
    model = EXAMPLES / "models" / f"{stem}.mo"
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    flatten = pytest.importorskip("modelica.sim").flatten
    return causalize(flatten(sources, stem)), float(spec["simulate"]["stop"])


@needs_rustc
@pytest.mark.parametrize(
    ("stem", "names"),
    [
        ("RLCCircuit", None),
        ("BouncingBall", None),
        ("Pendulum", ("x", "y", "vx", "vy")),
    ],
)
def test_the_compiled_run_matches_the_interpreter(
    stem: str, names: tuple[str, ...] | None, tmp_path: Path
) -> None:
    """Same grid, same tolerance, same names: the two tiers agree, and so do the events.

    ``names`` picks what is compared: the states by default, and for a model that switches
    state sets the coordinates, which mean the same in every set where the states do not.
    """
    numpy = pytest.importorskip("numpy")
    simulate = pytest.importorskip("modelica.sim").simulate
    system, stop = _system(stem)
    grid = numpy.linspace(0.0, stop, 41)
    reference = simulate(system, stop=stop, times=grid, tolerance=1e-8)
    run = rustsim.simulate(system, stop=stop, times=grid, tolerance=1e-8, cache=tmp_path)

    assert run.events == len(reference.events), (
        f"{stem}: rust fired {run.events} events, interpreter {len(reference.events)}"
    )
    # The whole events, not only the tally: each crossing the driver located -- its instant, the
    # index of the indicator that fired, and its cause -- against the interpreter's. BouncingBall's
    # six bounces are the test that carries.
    assert len(run.event_list) == len(reference.events)
    for got, want in zip(run.event_list, reference.events, strict=True):
        assert got.time == pytest.approx(want.time, abs=1e-6), (
            f"{stem}: event at {got.time}, want {want.time}"
        )
        assert got.index == want.index, f"{stem}: event index {got.index}, want {want.index}"
        assert got.cause == want.cause, f"{stem}: event cause {got.cause!r}, want {want.cause!r}"
    assert run.end == pytest.approx(stop)
    rows = numpy.clip(
        numpy.searchsorted(reference.time, grid, side="left"), 0, len(reference.time) - 1
    )
    for name in names or system.modes[0].states:
        assert name in run.result, f"{stem}: {name} is not in the compiled result"
        numpy.testing.assert_allclose(
            run.result[name], reference[name][rows], rtol=1e-4, atol=1e-3, err_msg=f"{stem} {name}"
        )


@needs_rustc
def test_the_library_is_built_once_and_reused(tmp_path: Path) -> None:
    """The second build of an unchanged model is a hash and a stat, not a compile."""
    system, _ = _system("RLCCircuit")
    first = rustsim.build(system, cache=tmp_path)
    stamp = first.stat().st_mtime_ns
    second = rustsim.build(system, cache=tmp_path)
    assert second == first
    assert second.stat().st_mtime_ns == stamp, "the cached library was rebuilt"
    assert (tmp_path / first.with_suffix(".rs").name).exists(), "the source is kept beside it"


@needs_rustc
def test_a_run_reads_like_a_result(tmp_path: Path) -> None:
    """``run["x"]`` and ``run.time`` give what ``run.result`` gives (#4)."""
    system, _ = _system("RLCCircuit")
    run = rustsim.simulate(system, stop=0.1, cache=tmp_path)
    name = run.result.names[0]
    assert (run[name] == run.result[name]).all()
    assert (run.time == run.result.time).all()


@needs_rustc
def test_a_chattering_model_is_reported_not_run_forever(tmp_path: Path) -> None:
    """The driver's chattering status surfaces as the same error the interpreter raises."""
    from test_chattering import SLIDING  # noqa: PLC0415 - the model lives with its own test

    flatten = pytest.importorskip("modelica.sim").flatten
    error = pytest.importorskip("modelica.sim").SimulationError
    system = causalize(flatten([parse(SLIDING)], "Chattering"))
    with pytest.raises(error, match="chattering"):
        rustsim.simulate(system, stop=5.0, points=11, tolerance=1e-8, cache=tmp_path)
