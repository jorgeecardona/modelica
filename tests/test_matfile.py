"""The Dymola result format, checked against a reader this project did not write.

A round trip through our own writer proves only that we are self-consistent, and the entry
that asked for this said so. `DyMat` is the second opinion: it is one of the three maintained
Python packages whose whole job is reading this format, and it is what
`test_dymat_opens_what_we_wrote` runs. It is not a dependency -- the test skips without it,
the way `tests/test_jit.py` skips without Numba.

Two ways of getting this format wrong are what the tests below are shaped around, and both
produce a file that *opens*:

* the character matrices are not all stored the same way round. ``Aclass`` is read row by row
  and ``name`` is read column by column, in the same file, by the same reader.
* `scipy.io.savemat` writes an array's buffer, so a transposed *view* goes to disk in the
  original order under a header claiming the new shape. That file reports the right number of
  variables and names them things like ``Te  v  .``.

Neither is visible from inside. Both were found by asking DyMat.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim.matfile import read

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _system(stem: str) -> System:
    model = EXAMPLES / "models" / f"{stem}.mo"
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    flatten = pytest.importorskip("modelica.sim").flatten
    sources = [
        parse((EXAMPLES / "library" / f"{name}.mo").read_text(encoding="utf-8"))
        for name in spec["requires"]
    ]
    sources.append(parse(model.read_text(encoding="utf-8")))
    return causalize(flatten(sources, stem))


def _run(stem: str = "RLCCircuit"):  # noqa: ANN202 - a `Result`, and the import is lazy
    simulate = pytest.importorskip("modelica.sim.solve").simulate
    return simulate(_system(stem), stop=5.0, points=9)


def test_it_round_trips(tmp_path: Path) -> None:
    """Same names, same order, same numbers, same time. Necessary and not sufficient."""
    numpy = pytest.importorskip("numpy")
    original = _run()
    path = tmp_path / "run.mat"
    original.write(path)

    back = read(path)

    assert back.names == original.names
    numpy.testing.assert_array_equal(back.time, original.time)
    numpy.testing.assert_array_equal(back.values, original.values)


def test_dymat_opens_what_we_wrote(tmp_path: Path) -> None:
    """The load-bearing test. A reader from outside this project, on every variable.

    Skipped where DyMat is absent, which is everywhere it has not been asked for: it is not a
    dependency and this format is not on the core path.
    """
    numpy = pytest.importorskip("numpy")
    dymat = pytest.importorskip("DyMat")
    original = _run()
    path = tmp_path / "run.mat"
    original.write(path)

    opened = dymat.DyMatFile(str(path))

    assert set(opened.names()) == set(original.names)
    for name in original.names:
        numpy.testing.assert_allclose(opened.data(name), original[name], err_msg=name)
    numpy.testing.assert_allclose(opened.abscissa(2, valuesOnly=True), original.time)


def test_a_negative_index_means_the_negation(tmp_path: Path) -> None:
    """The alias packing, and the one convention a round trip cannot check.

    A variable stored as the negative of another is written by pointing at that other one's
    row with a **negative** one-based index -- which is how ``a = -b`` avoids two copies. This
    writer never emits one, so the file here is built by hand, and the check is that our
    reader and DyMat agree about what it means. Two readers agreeing on a sign is a stronger
    statement than either one alone.
    """
    numpy = pytest.importorskip("numpy")
    savemat = pytest.importorskip("scipy.io").savemat

    def characters(strings: list[str], *, down: bool) -> object:
        width = max(len(text) for text in strings)
        grid = numpy.array([list(text.ljust(width)) for text in strings], dtype="U1")
        return numpy.ascontiguousarray(grid.T) if down else grid

    time = numpy.array([0.0, 1.0, 2.0])
    signal = numpy.array([1.0, 2.0, 3.0])
    path = tmp_path / "alias.mat"
    savemat(
        path,
        {
            "Aclass": characters(["Atrajectory", "1.1", "", "binTrans"], down=False),
            "name": characters(["Time", "b", "a"], down=True),
            "description": characters(["Time in [s]", "", ""], down=True),
            # `a` points at `b`'s row (index 2, one-based) with a minus: it is `-b`.
            "dataInfo": numpy.array(
                [[0, 2, 2], [1, 2, -2], [0, 0, 0], [-1, -1, -1]], dtype=numpy.int32
            ),
            "data_1": numpy.array([[0.0, 2.0]]),
            "data_2": numpy.vstack([time, signal]),
        },
        format="4",
    )

    ours = read(path)

    numpy.testing.assert_array_equal(ours["b"], signal)
    numpy.testing.assert_array_equal(ours["a"], -signal)

    dymat = pytest.importorskip("DyMat")
    theirs = dymat.DyMatFile(str(path))
    numpy.testing.assert_array_equal(theirs.data("a"), -signal)


def test_a_constant_comes_back_as_a_trajectory(tmp_path: Path) -> None:
    """``data_1`` holds two points spanning the run; a `Result` has one time axis.

    A caller who asks for a parameter wants an array they can plot beside everything else,
    not a two-element special case.
    """
    numpy = pytest.importorskip("numpy")
    savemat = pytest.importorskip("scipy.io").savemat

    def characters(strings: list[str], *, down: bool) -> object:
        width = max(len(text) for text in strings)
        grid = numpy.array([list(text.ljust(width)) for text in strings], dtype="U1")
        return numpy.ascontiguousarray(grid.T) if down else grid

    time = numpy.array([0.0, 0.5, 1.0, 1.5])
    path = tmp_path / "constant.mat"
    savemat(
        path,
        {
            "Aclass": characters(["Atrajectory", "1.1", "", "binTrans"], down=False),
            "name": characters(["Time", "R"], down=True),
            "description": characters(["Time in [s]", ""], down=True),
            "dataInfo": numpy.array([[0, 1], [1, 2], [0, 0], [-1, -1]], dtype=numpy.int32),
            "data_1": numpy.array([[0.0, 1.5], [7.0, 7.0]]),
            "data_2": numpy.vstack([time]),
        },
        format="4",
    )

    ours = read(path)

    numpy.testing.assert_array_equal(ours["R"], numpy.full(len(time), 7.0))


def test_a_file_that_is_not_one_says_so(tmp_path: Path) -> None:
    """A `.mat` from somewhere else is a common thing to be handed."""
    numpy = pytest.importorskip("numpy")
    savemat = pytest.importorskip("scipy.io").savemat
    path = tmp_path / "other.mat"
    savemat(path, {"x": numpy.arange(4.0)}, format="4")

    with pytest.raises(ValueError, match="not a Dymola result file"):
        read(path)
