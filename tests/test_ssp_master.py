"""The co-simulation master's protocol, checked over every interleaving.

The finding this file exists to pin: FMI lets an FMU decline ``canGetAndSetFMUState``, and a
master that rolls back on a discarded step has no move left when a component that already
stepped cannot be put back. It is not a rare race -- it is four moves from the start -- and
it is invisible to any test that happens to use FMUs which all support state saving.
"""

from __future__ import annotations

import pytest

from modelica.ssp import Component, Phase, Progress, moves, start

verify = pytest.importorskip("modelica.verify")

ROLLBACK = (Component("a", rollback=True), Component("b", rollback=True))
MIXED = (Component("a", rollback=True), Component("b", rollback=False))


def test_a_master_over_rollback_capable_components_always_finishes() -> None:
    report = verify.check_master(ROLLBACK)
    assert report.ok, str(report)


def test_a_component_that_cannot_roll_back_strands_the_master() -> None:
    """The whole point. The counterexample is the shortest interleaving that gets there."""
    report = verify.check_master(MIXED)
    assert not report.ok
    assert report.safety.invariant == "a-rollback-can-put-everyone-back"
    trace = [step.action for step in report.safety.trace]
    assert "step b" in trace
    assert "a discards" in trace


def test_nobody_is_ever_more_than_one_point_ahead() -> None:
    """Checked as an invariant over the whole reachable space, asserted here as intent."""
    report = verify.check_master(ROLLBACK)
    assert "agreed-point-is-never-passed-twice" in verify.master(ROLLBACK).invariants
    assert report.safety.ok


def test_a_snapshot_left_over_from_the_last_point_is_not_a_problem() -> None:
    """Because ``restore`` will not use it -- the stale snapshot is simply ignored.

    Worth pinning: the first version of the invariant said every snapshot must be current,
    which made a perfectly healthy master look broken one move after `commit`.
    """
    state = start(ROLLBACK)
    for name in ("save a", "save b", "begin", "step a", "step b", "commit"):
        state = next(after for move, after in _out(state) if move == name)
    assert state.accepted == 1
    assert state.saved == (0, 0), "snapshots are still at the previous point"
    assert state.phase is Phase.SAVE
    assert verify.check_master(ROLLBACK).safety.ok


def test_a_refusal_is_a_legal_answer_not_a_fault() -> None:
    """A discard has to be reachable, or the check is proving something about nothing."""
    state = next(after for move, after in _out(start(ROLLBACK)) if move == "save a")
    state = next(after for move, after in _out(state) if move == "save b")
    state = next(after for move, after in _out(state) if move == "begin")
    assert any(move == "a discards" for move, _ in _out(state))


def _out(state: Progress) -> list[tuple[str, Progress]]:
    return list(moves(state, ROLLBACK, horizon=2, retries=1))
