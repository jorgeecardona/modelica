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
from types import FunctionType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from modelica.sim.emit import Code, emit

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from modelica.ir.system import System

__all__ = ["Kernel", "build"]


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
    observe: _Evaluate
    observe_jacobian: _Evaluate
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
        self, t: float = 0.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The parameter vector, the initial state, and a workspace latched at ``t``."""
        p = self.parameters(np.zeros(len(self.code.parameters)))
        y = np.zeros(self.states)
        w = np.zeros(self.code.workspace)
        self.initial(p, y, w)
        self.latch(t, y, p, w)
        return p, y, w


def build(model: System | Code, *, jit: bool = False) -> Kernel:
    """Compile a system (or already-emitted code) into callables."""
    code = model if isinstance(model, Code) else emit(model)
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
        observe=namespace["observe"],
        observe_jacobian=namespace["observe_jacobian"],
        indicators=namespace["indicators"],
        transition=namespace["transition"],
    )


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

    Stated honestly, it is defensive rather than a fix for an observed failure: no model in
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
