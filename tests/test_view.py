"""The notebook view: what ends up in the page.

The chart itself is JavaScript and is not tested here -- what is tested is the contract
the Python side owes it. The page must be one self-contained document, the payload must be
valid JSON that survives hostile variable names, non-finite values must arrive as ``null``
rather than as something a JSON parser rejects, and a long run must be thinned without
losing the samples that straddle an event.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from modelica.sim.result import Event, Result
from modelica.sim.view import CELLS, View, view

if TYPE_CHECKING:
    from pathlib import Path

MARKER = '<script id="data" type="application/json">'


def payload(document: str) -> dict[str, Any]:
    """The embedded JSON, read the way the browser reads it: to the next ``</script>``."""
    start = document.index(MARKER) + len(MARKER)
    end = document.index("</script>", start)
    return json.loads(document[start:end])


def series(data: dict[str, Any], name: str) -> list[float | None]:
    return next(s["values"] for s in data["series"] if s["name"] == name)


@pytest.fixture
def result() -> Result:
    """A bounce: two segments meeting at t = 1, where the velocity flips sign."""
    time = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0])
    values = np.array(
        [
            [1.0, 0.75, 0.0, 0.0, 0.75, 1.0],  # h
            [0.0, -1.0, -2.0, 2.0, 1.0, 0.0],  # v
        ]
    )
    return Result(
        name="Bounce",
        time=time,
        names=("h", "v"),
        values=values,
        events=(Event(time=1.0, index=0, cause="when h <= 0"),),
    )


def test_document_carries_the_data(result: Result) -> None:
    data = payload(view(result).document())
    assert data["name"] == "Bounce"
    assert data["time"] == [0.0, 0.5, 1.0, 1.0, 1.5, 2.0]
    assert [s["name"] for s in data["series"]] == ["h", "v"]
    assert series(data, "v") == [0.0, -1.0, -2.0, 2.0, 1.0, 0.0]
    assert data["events"] == [{"t": 1.0, "cause": "when h <= 0"}]
    assert data["shown"] == data["total"] == 6


def test_selects_the_named_variables(result: Result) -> None:
    data = payload(result.plot("v").document())
    assert [s["name"] for s in data["series"]] == ["v"]


def test_rejects_a_name_the_result_does_not_have(result: Result) -> None:
    with pytest.raises(KeyError, match="no variable 'q'"):
        result.plot("q")


def test_non_finite_values_become_null(result: Result) -> None:
    values = result.values.copy()
    values[0, 2] = np.nan
    values[1, 3] = np.inf
    data = payload(view(Result("X", result.time, result.names, values)).document())
    assert series(data, "h")[2] is None
    assert series(data, "v")[3] is None
    # `null`, not the string "NaN": a JSON parser is entitled to reject the latter.
    assert "NaN" not in json.dumps(data)


def test_a_hostile_name_cannot_close_the_script_tag() -> None:
    name = "a</script><script>alert(1)</script>"
    values = np.array([[0.0, 1.0]])
    document = view(Result("X", np.array([0.0, 1.0]), (name,), values)).document()
    data = payload(document)
    assert [s["name"] for s in data["series"]] == [name]
    assert "alert(1)" not in document.split(MARKER)[0]


def test_thinning_keeps_the_ends_and_both_sides_of_an_event() -> None:
    count = 10_000
    time = np.concatenate([np.linspace(0.0, 1.0, count // 2), np.linspace(1.0, 2.0, count // 2)])
    values = np.vstack([time, np.sin(time)])
    result = Result("Long", time, ("t", "s"), values, (Event(1.0, 0, "when t >= 1"),))
    data = payload(View(result, result.names, limit=500).document())

    drawn = data["time"]
    assert data["total"] == count
    assert data["shown"] == len(drawn) <= 500 * 2  # the stride, plus the kept boundaries
    assert drawn[0] == 0.0
    assert drawn[-1] == pytest.approx(2.0)
    # The segment boundary is a repeated time: both copies survive, so the jump stays
    # vertical instead of being smoothed into a ramp.
    assert drawn.count(1.0) == 2


def test_values_are_rounded_but_not_mangled() -> None:
    values = np.array([[math.pi, 1e-17, 6.02214076e23]])
    data = payload(view(Result("X", np.array([0.0, 1.0, 2.0]), ("x",), values)).document())
    assert series(data, "x") == [3.1415927, 1e-17, 6.0221408e23]


def test_the_chart_opens_on_the_variables_that_move() -> None:
    """Parameters are in the legend, but a chart of flat lines is not worth opening on."""
    time = np.array([0.0, 1.0, 2.0])
    values = np.array([[9.81, 9.81, 9.81], [1.0, 0.5, 0.0], [0.8, 0.8, 0.8], [0.0, -1.0, -2.0]])
    result = Result("Ball", time, ("g", "h", "e", "v"), values)

    data = payload(view(result).document())
    assert data["initial"] == ["h", "v", "g", "e"]  # movers first, then the rest to fill
    assert [s["name"] for s in data["series"]] == ["g", "h", "e", "v"]  # legend order is intact
    assert payload(view(result, initial=2).document())["initial"] == ["h", "v"]


def test_the_payload_is_bounded_by_the_cell_budget() -> None:
    """Many variables buy fewer samples, so a notebook file cannot balloon."""
    count, width = 20_000, 500
    time = np.linspace(0.0, 1.0, count)
    names = tuple(f"x{i}" for i in range(width))
    values = np.broadcast_to(time, (width, count))
    data = payload(view(Result("Wide", time, names, np.asarray(values))).document())

    assert data["total"] == count
    # The stride, plus the final sample, which a stride of 40 over 20 000 misses.
    assert data["shown"] <= CELLS // width + 1
    # The header reads off `shown` and `total`, so the thinning is stated, not silent.
    assert data["shown"] < data["total"]


def test_repr_html_is_a_sandboxed_iframe(result: Result) -> None:
    frame = result._repr_html_()  # pyright: ignore[reportPrivateUsage]
    assert frame.startswith("<iframe srcdoc=")
    assert 'sandbox="allow-scripts"' in frame
    # The document is escaped into an attribute, so no raw quote of the document itself can
    # end the attribute early.
    body = frame.split('srcdoc="', 1)[1].split('"', 1)[0]
    assert "&lt;!doctype html&gt;" in body


def test_an_untrusted_notebook_gets_a_sentence_instead_of_a_blank_cell(result: Result) -> None:
    """JupyterLab's sanitizer drops the ``iframe`` and keeps its text. Keep text worth reading."""
    inside = view(result).embed().split(">", 1)[1].removesuffix("</iframe>")
    assert "not trusted" in inside
    assert "Bounce" in inside
    assert ".save(" in inside


def test_save_writes_a_standalone_page(result: Result, tmp_path: Path) -> None:
    path = view(result).save(tmp_path / "bounce.html")
    text = (tmp_path / "bounce.html").read_text(encoding="utf-8")
    assert path.endswith("bounce.html")
    assert text.startswith("<!doctype html>")
    # Self-contained: nothing to fetch, so it still works offline and after the kernel dies.
    assert "<script src" not in text
    assert "<link" not in text
    assert payload(text)["name"] == "Bounce"
