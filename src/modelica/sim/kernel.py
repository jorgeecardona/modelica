"""Generated source, turned into functions you can call.

:mod:`modelica.sim.emit` writes the model out as text; this executes that text and hands
back the seven functions bound together, with the array sizes the caller needs to allocate.
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
    """The compiled model: seven functions and the shapes of the arrays they expect."""

    code: Code
    parameters: _Prepare
    initial: _Initial
    latch: _Latch
    rhs: _Evaluate
    observe: _Evaluate
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
        namespace = _accelerate(namespace)
    return Kernel(
        code=code,
        parameters=namespace["parameters"],
        initial=namespace["initial"],
        latch=namespace["latch"],
        rhs=namespace["rhs"],
        observe=namespace["observe"],
        indicators=namespace["indicators"],
        transition=namespace["transition"],
    )


def _accelerate(namespace: dict[str, Any]) -> dict[str, Any]:
    """The same functions, compiled.

    By name rather than ``from numba import njit`` because numba is genuinely optional:
    it is in no dependency group, so it is absent from the environment the type checker
    reads, and a real import would have to be suppressed there. Asking for it by string
    says the same thing without pretending the module is known to exist.
    """
    try:
        numba = importlib.import_module("numba")
    except ImportError as error:  # pragma: no cover - depends on the environment
        message = "jit=True needs numba: pip install numba"
        raise ImportError(message) from error
    njit: Any = numba.njit
    functions = ("parameters", "initial", "latch", "rhs", "observe", "indicators", "transition")
    return namespace | {name: njit(cache=False)(namespace[name]) for name in functions}
