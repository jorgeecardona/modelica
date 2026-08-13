"""Generated source, turned into functions you can call.

:mod:`modelica.sim.emit` writes the model out as text; this executes that text and hands
back the eight functions bound together, with the array sizes the caller needs to allocate.
Splitting it out is not ceremony -- the same :class:`~modelica.sim.emit.Code` will later
be handed to a C compiler for FMU export instead of to CPython, and everything above this
line stays the same.

The generated source is deliberately restricted to scalars, arithmetic and array indexing
so that ``@njit`` accepts it without leaving nopython mode. Numba is not a dependency,
though: ``jit=True`` is opt-in, and the interpreted functions are what runs by default.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from types import FunctionType
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol

import numpy as np

from modelica.sim.emit import Code, emit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System

__all__ = ["CACHED", "CacheInfo", "Kernel", "build", "cache_info", "clear_cache"]


class CacheInfo(NamedTuple):
    """What `cache_info` returns.

    Declared here rather than imported: `functools` spells it ``_CacheInfo``, and a private
    name from the standard library has no business in this module's signatures.
    """

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


class _Prepare(Protocol):
    def __call__(self, p: NDArray[np.float64], /) -> NDArray[np.float64]: ...


class _Initial(Protocol):
    def __call__(
        self, p: NDArray[np.float64], y: NDArray[np.float64], w: NDArray[np.float64], /
    ) -> NDArray[np.float64]: ...


class _Latch(Protocol):
    def __call__(
        self,
        t: float,
        y: NDArray[np.float64],
        p: NDArray[np.float64],
        w: NDArray[np.float64],
        /,
    ) -> NDArray[np.float64]: ...


class _Evaluate(Protocol):
    def __call__(
        self,
        t: float,
        y: NDArray[np.float64],
        p: NDArray[np.float64],
        w: NDArray[np.float64],
        out: NDArray[np.float64],
        /,
    ) -> NDArray[np.float64]: ...


class _Transition(Protocol):
    def __call__(  # noqa: PLR0913, PLR0917 - the generated calling convention, plus which event
        self,
        event: int,
        t: float,
        y: NDArray[np.float64],
        p: NDArray[np.float64],
        w: NDArray[np.float64],
        out: NDArray[np.float64],
        /,
    ) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class Kernel:
    """The compiled model: nine functions and the shapes of the arrays they expect."""

    code: Code
    parameters: _Prepare
    initial: _Initial
    latch: _Latch
    rhs: _Evaluate
    jacobian: _Evaluate
    jacobian_sparse: _Evaluate
    jacobian_parameters: _Evaluate
    """``d(der)/dp`` over `Code.sensitivity`. Empty columns when nobody asked for any."""

    observe_jacobian_parameters: _Evaluate
    """The same Jacobian as ``jacobian``, written as its non-zeros in `Code.sparsity` order.

    A separate entry point rather than a mode of the first, because the two fill differently
    shaped arrays and a caller has to know which it allocated.
    """
    observe: _Evaluate
    observe_jacobian: _Evaluate | None
    """``d(observed)/dy``, or ``None`` when the caller said not to emit it.

    ``None`` rather than a stub that raises: a kernel should say what it does not have, and a
    name that is present until you touch it is the shape that makes a caller write the check
    in the wrong place.
    """
    indicators: _Evaluate
    transition: _Transition

    @property
    def name(self) -> str:
        return self.code.name

    @property
    def states(self) -> int:
        return len(self.code.states)

    @property
    def observed(self) -> int:
        return len(self.code.observed)

    @property
    def events(self) -> int:
        return len(self.code.events)

    def start(
        self, t: float = 0.0, parameters: NDArray[np.float64] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The parameter vector, the initial state, and a workspace latched at ``t``.

        ``parameters`` replaces the vector the model computes for itself. The initial state
        and the latches are then computed from *that*, which is the point: a start value
        written in terms of a parameter has to follow the override, or a sweep changes the
        equations and not the conditions they start from.
        """
        p = (
            np.array(parameters, dtype=float)
            if parameters is not None
            else self.parameters(np.zeros(len(self.code.parameters)))
        )
        y = np.zeros(self.states)
        w = np.zeros(self.code.workspace)
        self.initial(p, y, w)
        self.latch(t, y, p, w)
        return p, y, w


CACHED: Final = 32
"""How many distinct models `build` keeps compiled at once.

Bounded rather than unbounded, because a cache entry holds the `System` it was keyed on and
every array the generated module allocated, and a long-lived process that builds a hundred
different models should not hold all hundred. Thirty-two is more than a notebook or a test
session has models and small enough that the memory is not worth thinking about; nothing was
measured to choose it, and if a workload ever wants more this is the number to change.
"""


def build(
    model: System | Code,
    *,
    jit: bool = False,
    fold: bool = False,
    sensitivity: Sequence[str] = (),
    observed_jacobian: bool = True,
) -> Kernel:
    """Compile a system (or already-emitted code) into callables.

    Cached on the *content* of ``model``, which is what `docs/ideas/backends.md:70` asks for:
    "The IR nodes are frozen, slotted and hashable, so a `FlatModel` has a content hash and
    needs no extra machinery to get one. That hash is the cache key for the compiled kernel.
    It recompiles when the model changed, and not when the mtime of the file changed."

    So two identical models parsed from two different paths are one entry, and changing one
    character in one equation is a different key. Nothing about a filename or a timestamp
    reaches this.

    `System` had to be made hashable for it -- see `modelica.ir.system.FrozenMap`. It was the
    only value between `FlatModel` and `Code` that was not, so the key the document promised
    existed at both ends of the pipeline and not in the middle.

    Sharing one `Kernel` between callers is safe because the generated module has no mutable
    state: every function writes into an ``out`` array the caller owns, `Kernel.start`
    allocates fresh arrays on each call, and the two module-level constants the emitter writes
    (``_seed`` and ``_null``) are read and never written. `clear_cache` drops everything -- a
    test that wants to time a cold compile has to.

    The keyword is normalised here rather than cached with it, and that is not a style
    choice. `functools.lru_cache` keys on the call as written, so ``build(system)`` and
    ``build(system, jit=False)`` are two entries for one answer -- and both forms are in this
    repo, since `modelica.sim.solve.simulate` always passes the keyword. Decorating `build`
    directly compiled the same model twice and reported a hit rate that looked fine.
    """
    return _compiled(model, jit, fold, tuple(sensitivity), observed_jacobian)


@lru_cache(maxsize=CACHED)
def _compiled(
    model: System | Code,
    jit: bool,  # noqa: FBT001 - positional so `lru_cache` sees one key per call shape
    fold: bool,  # noqa: FBT001 - the same reason
    sensitivity: tuple[str, ...],
    observed_jacobian: bool,  # noqa: FBT001 - the same reason
) -> Kernel:
    code = (
        model
        if isinstance(model, Code)
        else emit(model, fold=fold, sensitivity=sensitivity, observed_jacobian=observed_jacobian)
    )
    namespace: dict[str, Any] = {"np": np}
    exec(compile(code.source, f"<modelica:{code.name}>", "exec"), namespace)  # noqa: S102
    if jit:
        _accelerate(namespace)
    return Kernel(
        code=code,
        parameters=namespace["parameters"],
        initial=namespace["initial"],
        latch=namespace["latch"],
        rhs=namespace["rhs"],
        jacobian=namespace["jacobian"],
        jacobian_sparse=namespace["jacobian_sparse"],
        jacobian_parameters=namespace["jacobian_parameters"],
        observe_jacobian_parameters=namespace["observe_jacobian_parameters"],
        observe=namespace["observe"],
        observe_jacobian=namespace.get("observe_jacobian"),
        indicators=namespace["indicators"],
        transition=namespace["transition"],
    )


def clear_cache() -> None:
    """Forget every compiled kernel. A test timing a cold build calls this first."""
    _compiled.cache_clear()


def cache_info() -> CacheInfo:
    """Hits, misses, and how full the cache is -- `functools.lru_cache`'s own record."""
    return CacheInfo(*_compiled.cache_info())


def _accelerate(namespace: dict[str, Any]) -> None:
    """Compile every function the generated module defines, in place.

    Every one, not the seven the kernel holds. A model with more than one state set writes
    ``rhs_0``, ``rhs_1`` and a ``rhs`` that picks between them, and compiling only the
    picker would leave it calling into the interpreter -- which numba would either refuse
    or fall back on, and either way the point of asking would be gone.

    In place, and not into a copy, because the namespace *is* the functions' ``__globals__``:
    the picker resolves ``rhs_0`` by looking there, so the compiled one has to be what it
    finds. Numba compiles lazily, at the first call, so the order of replacement does not
    matter -- every name is already the compiled one by the time any of them runs.

    Numba is asked for by name rather than ``from numba import njit`` because it is
    genuinely optional: it is in no dependency group, so it is absent from the environment
    the type checker reads, and a real import would have to be suppressed there.

    ``error_model="numpy"`` keeps the two tiers saying the same thing. Numba defaults to
    ``"python"``, where dividing by zero raises ``ZeroDivisionError`` the way CPython does,
    while the interpreted kernel divides *numpy* scalars and gets an infinity. That is a
    genuine semantic fork between the two, and the interpreted tier is the reference one --
    differential debugging (see `docs/ideas/backends.md`) is worth nothing if the fast path
    can answer a question the slow path answers differently.

    ``cache=False`` is not an oversight and cannot be turned on. Numba's on-disk cache needs
    to find the source file a function came from, and these functions come from `exec`. Asking
    for it raises at decoration time rather than falling back -- measured on numba 0.66,
    ``RuntimeError: cannot cache function 'rhs': no locator available for file
    '<modelica:Demo>'``. Writing the generated source to a real file would give numba a
    locator, and that is a disk cache, which is a separate piece of work with an invalidation
    problem of its own. The in-process cache on `build` is what removes the repeated
    compilation this was reached for.

    Stated honestly, the error model is defensive rather than a fix for an observed failure: no model in
    the corpus reaches a division by exactly zero. The pendulum comes closest and does not
    arrive -- at ``y = -L`` its Newton lands on an ``x`` of about 1e-16 rather than on 0.0, so
    both tiers return the same finite nonsense. But an exact zero pivot is *reachable in
    principle* precisely where this engine has something to say: it is a state set going
    singular, which dynamic state selection exists to survive by switching rather than by
    stopping. An infinity there is information; an exception is a crash. The default model
    was chosen for scripts, and this is a numerical kernel.
    """
    try:
        numba = importlib.import_module("numba")
    except ImportError as error:  # pragma: no cover - depends on the environment
        message = "jit=True needs numba: pip install numba"
        raise ImportError(message) from error
    njit: Any = numba.njit
    for name, value in list(namespace.items()):
        if isinstance(value, FunctionType):
            namespace[name] = njit(cache=False, error_model="numpy")(value)
