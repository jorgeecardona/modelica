"""`build` recompiles when the model changed, and not when anything else did.

The claim under test is `docs/ideas/backends.md:70`: the IR is hashable, so the compiled
kernel is keyed on the *content* of the model. Every test here is a way of asking what the
key actually is -- the same model twice, the same text from two paths, and one character
changed -- because a cache that keys on the wrong thing is worse than no cache at all. It
does not return a stale answer slowly; it returns a wrong answer quickly.

Identity (``is``) rather than equality, deliberately. Two `Kernel` values built from the
same model would compare equal whether or not anything was cached, so equality would pass
against an implementation that cached nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.system import FrozenMap
from modelica.lang import parse
from modelica.sim.kernel import build, cache_info, clear_cache

if TYPE_CHECKING:
    from modelica.ir.system import System

RC = """
model Cached
  parameter Real R = 1.0;
  parameter Real C = 0.5;
  Real v(start = 1.0);
equation
  C * der(v) = -v / R;
end Cached;
"""


def _system(source: str) -> System:
    flatten = pytest.importorskip("modelica.sim").flatten
    return causalize(flatten([parse(source)], "Cached"))


@pytest.fixture(autouse=True)
def cold() -> None:
    """Every test starts with an empty cache, so a hit here was made here."""
    clear_cache()


def test_the_same_model_is_compiled_once() -> None:
    pytest.importorskip("numpy")
    first = build(_system(RC))
    second = build(_system(RC))
    assert first is second
    assert cache_info().hits == 1


def test_the_key_is_the_content_and_not_the_text_it_came_from() -> None:
    """Two spellings of one model, so the source string cannot be what is being keyed on.

    Comments, whitespace and declaration order all differ; the flattened, causalized model
    does not. A cache keyed on the file, the path or the text would miss here, and the point
    of putting the hash on the IR is that it does not.
    """
    pytest.importorskip("numpy")
    spelled = """
    model Cached "the same model, written by somebody else"
      // a comment the first version does not have
      parameter Real R = 1.0;

      parameter Real C = 0.5;
      Real v(start = 1.0);
    equation
      C  *  der( v )  =  -v / R;
    end Cached;
    """
    assert build(_system(RC)) is build(_system(spelled))


def test_one_character_in_one_equation_is_a_different_kernel() -> None:
    """The other half of the claim, and the half a broken cache passes the first half by.

    ``-v / R`` becomes ``-v * R``. Nothing else moves -- same name, same parameters, same
    states -- so a key that is anything coarser than the content of the equations would hand
    back the first kernel and silently integrate the wrong model.
    """
    pytest.importorskip("numpy")
    changed = RC.replace("-v / R", "-v * R")
    assert build(_system(RC)) is not build(_system(changed))
    assert cache_info().misses == 2


def test_a_parameter_value_is_part_of_the_model() -> None:
    """A number in a modification changes the emitted source, so it has to change the key.

    Worth its own test because it is the case a hash over *structure* would get wrong: the
    two models have identical shapes and differ only in a literal.
    """
    pytest.importorskip("numpy")
    assert build(_system(RC)) is not build(_system(RC.replace("R = 1.0", "R = 2.0")))


def test_the_cache_is_bounded() -> None:
    """More models than it holds evicts the oldest rather than growing without end."""
    pytest.importorskip("numpy")
    size = cache_info().maxsize
    assert size is not None
    first = build(_system(RC))
    for k in range(size):
        build(_system(RC.replace("R = 1.0", f"R = {k + 2}.0")))
    assert build(_system(RC)) is not first
    assert cache_info().currsize == size


def test_a_system_is_hashable_like_the_values_on_either_side_of_it() -> None:
    """`FlatModel`, `System` and `Code` all hash, which is what makes the key available.

    `System` was the one that did not, and the reason is worth a test rather than a comment:
    a `dict` field in a frozen dataclass makes the whole dataclass unhashable, so a single
    mapping in the middle of the pipeline broke a property both ends had.
    """
    flatten = pytest.importorskip("modelica.sim").flatten
    emit = pytest.importorskip("modelica.sim.emit").emit
    flat = flatten([parse(RC)], "Cached")
    system = causalize(flat)
    assert hash(flat) == hash(flatten([parse(RC)], "Cached"))
    assert hash(system) == hash(causalize(flatten([parse(RC)], "Cached")))
    assert hash(emit(system)) == hash(emit(system))
    assert isinstance(system.aliases, FrozenMap)
    assert isinstance(system.mode.derivative, FrozenMap)


def test_a_frozen_map_reads_like_the_dict_it_replaced() -> None:
    """Only the hash is new. Equality, lookup, iteration and length are `Mapping`'s.

    Checked because the coercion is silent: `Mode.__post_init__` swaps the type out from
    under every reader, and a reader comparing ``derivative == {...}`` must not notice.
    """
    frozen: FrozenMap[str, int] = FrozenMap({"a": 1, "b": 2})
    assert frozen == {"a": 1, "b": 2}
    assert frozen == FrozenMap({"b": 2, "a": 1})
    assert hash(frozen) == hash(FrozenMap({"b": 2, "a": 1}))
    assert frozen["a"] == 1
    assert list(frozen) == ["a", "b"]
    assert len(frozen) == 2
    assert "b" in frozen
    assert dict(frozen.items()) == {"a": 1, "b": 2}
    assert FrozenMap() == {}
