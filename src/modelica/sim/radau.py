"""Radau IIA in Python, the reference the engine's runtime (`rust_radau`) is ported from.

The three-stage Radau IIA method of order five, Hairer and Wanner's RADAU5, *Solving
Ordinary Differential Equations II*, IV.8, with `rhs` and `jacobian` taken as arguments.

**What it does not do.** No events. `modelica.sim.solve` stops on a zero crossing, applies
the discontinuity and restarts, and that machinery lives on the Python side; a model with
events is refused here rather than integrated wrongly. No mass matrix, no index reduction
beyond what causalization already did, and forward in time only.

**The linear algebra is local.** A step factorises two matrices of size ``n``, one real and
one complex, and the factorisations are written out here rather than called from SciPy --
not to be faster than LAPACK, which at these sizes is not the question, but because
``scipy.linalg.lu_factor`` cannot be called from compiled code, and calling it would put the
Python boundary back exactly where this module exists to remove it.

The two compile modes `docs/ideas/backends.md:66` asks for are the emitter's, not this
module's: ``build(system, fold=True)`` burns the parameters in as literals and folds them,
``fold=False`` keeps them as an argument so one compiled artefact serves a sweep. Both reach
here unchanged -- the folded kernel simply ignores the ``p`` it is handed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from modelica.sim.kernel import Kernel, build
from modelica.sim.result import Result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from modelica.ir.system import System
    from modelica.sim.emit import Code

__all__ = ["FusionError", "Stats", "integrate", "simulate"]


class FusionError(Exception):
    """The fused loop will not run this model, and why."""


_S6 = 6.0**0.5

C = np.array([(4.0 - _S6) / 10.0, (4.0 + _S6) / 10.0, 1.0])
"""The Radau IIA collocation nodes. The last is 1, which is what makes the method stiffly
accurate: the final stage *is* the step's end point, so no extrapolation is done past it."""

E = np.array([-13.0 - 7.0 * _S6, -13.0 + 7.0 * _S6, -1.0]) / 3.0
"""Weights of the embedded third-order estimate the step-size controller reads."""

MU_REAL = 3.0 + 3.0 ** (2.0 / 3.0) - 3.0 ** (1.0 / 3.0)
MU_COMPLEX = (
    3.0
    + 0.5 * (3.0 ** (1.0 / 3.0) - 3.0 ** (2.0 / 3.0))
    - 0.5j * (3.0 ** (5.0 / 6.0) + 3.0 ** (7.0 / 6.0))
)
"""The eigenvalues of the inverse Butcher matrix, one real and one complex conjugate pair.

Diagonalising ``A`` is the whole reason a Radau step is affordable. The collocation system
is ``3n`` unknowns and a direct simplified Newton would factorise a ``3n`` by ``3n`` matrix
every time the step size changes -- twenty-seven times the work of one ``n`` by ``n``
factorisation. In the eigenbasis it splits into one real system of size ``n`` and one
complex system of size ``n``, which is about four times the work of a real one. Nine times
cheaper than the direct form, and the reason RADAU5 is written the way it is.
"""

T = np.array(
    [
        [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
        [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
        [1.0, 1.0, 0.0],
    ]
)
TI = np.array(
    [
        [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
        [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
        [0.50287263494578682, -2.57192694985560522, 0.59603920482822492],
    ]
)
"""``A = T L T^-1``, to the precision Hairer and Wanner publish them at."""

TI_REAL = TI[0].copy()
TI_COMPLEX_R = TI[1].copy()
TI_COMPLEX_I = TI[2].copy()
"""The rows of ``TI`` the iteration actually combines, split so the complex one can be built
from two real rows -- a compiled loop over floats beats one over a complex array here."""

P = np.array(
    [
        [13.0 / 3.0 + 7.0 * _S6 / 3.0, -23.0 / 3.0 - 22.0 * _S6 / 3.0, 10.0 / 3.0 + 5.0 * _S6],
        [13.0 / 3.0 - 7.0 * _S6 / 3.0, -23.0 / 3.0 + 22.0 * _S6 / 3.0, 10.0 / 3.0 - 5.0 * _S6],
        [1.0 / 3.0, -8.0 / 3.0, 10.0 / 3.0],
    ]
)
"""The collocation polynomial, as coefficients in powers of the position within the step.

Dense output is free here and it is not a convenience. Requested output times are filled
from this polynomial rather than by forcing the integrator to land on them, so asking for a
uniform grid does not change a single step the method takes.
"""

NEWTON_MAXITER = 6
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
EPS = float(np.finfo(np.float64).eps)

_ORDER = 3
"""The order of the *error estimate*, which is what the controller's exponent is built from."""

_TRIVIAL = 1e-5
_NEGLIGIBLE = 1e-15
"""Scaled sizes below which the starting-step heuristic stops believing its own division."""

_STALE_ITERATIONS = 2
_STALE_RATE = 1e-3
"""A Newton solve that took more than this many iterations, contracting no faster than this,
is reporting a Jacobian that has drifted. Recomputing it beats grinding the next step."""

_WORTH_CHANGING = 1.2
"""Growth below which the step size is left alone. Changing ``h`` invalidates both
factorisations, so a five-percent improvement in step size costs more than it returns."""

_TOO_SMALL = 1
_NO_CONVERGENCE = 2
"""The two ways `integrate` gives up, as `Stats.status`."""


class _Evaluate(Protocol):
    """`rhs` and `jacobian` as this loop calls them: fill ``out``, hand it back."""

    def __call__(
        self,
        t: float,
        y: NDArray[np.float64],
        p: NDArray[np.float64],
        w: NDArray[np.float64],
        out: Any,  # noqa: ANN401 - a vector for `rhs`, a matrix for `jacobian`
        /,
    ) -> Any: ...  # noqa: ANN401 - the same array back


class Stats:
    """What one integration cost, and whether it finished.

    A plain class and not a `NamedTuple`, because the fused loop returns a tuple of numbers
    from compiled code and this is where they get their names back.
    """

    __slots__ = (
        "evaluations",
        "factorisations",
        "jacobians",
        "rejected",
        "status",
        "steps",
        "time",
    )

    def __init__(  # noqa: PLR0913, PLR0917 - seven counters, one per counted thing
        self,
        status: int,
        steps: int,
        rejected: int,
        evaluations: int,
        jacobians: int,
        factorisations: int,
        time: float,
    ) -> None:
        self.status = status
        """0 finished, 1 the step size went below what the floating-point grid can express,
        2 the Newton iteration would not converge even on a freshly computed Jacobian."""
        self.steps = steps
        self.rejected = rejected
        self.evaluations = evaluations
        self.jacobians = jacobians
        self.factorisations = factorisations
        self.time = time
        """How far it got. Equal to the requested end unless ``status`` says otherwise."""

    def __repr__(self) -> str:
        return (
            f"Stats(status={self.status}, steps={self.steps}, rejected={self.rejected},"
            f" evaluations={self.evaluations}, jacobians={self.jacobians},"
            f" factorisations={self.factorisations}, time={self.time!r})"
        )


# -- linear algebra, small and local ----------------------------------------------------


def _factor(a: Any, piv: Any) -> bool:  # noqa: ANN401 - float64 or complex128, one body for both
    """LU with partial pivoting, in place. False when a column is exactly zero.

    One body serves the real and the complex system: pivoting compares ``abs``, which is a
    magnitude either way, and everything else is arithmetic that both types support. Numba
    compiles a specialisation per element type from the same source, so the two matrices a
    Radau step factorises cost one function to write and none to maintain twice.
    """
    n = a.shape[0]
    for k in range(n):
        big = 0.0
        row = k
        for i in range(k, n):
            size = abs(a[i, k])
            if size > big:
                big = size
                row = i
        piv[k] = row
        if big == 0.0:
            return False
        if row != k:
            for j in range(n):
                held = a[k, j]
                a[k, j] = a[row, j]
                a[row, j] = held
        pivot = a[k, k]
        for i in range(k + 1, n):
            share = a[i, k] / pivot
            a[i, k] = share
            if share != 0.0:
                for j in range(k + 1, n):
                    a[i, j] -= share * a[k, j]
    return True


def _solve(a: Any, piv: Any, b: Any) -> None:  # noqa: ANN401 - the same two types as `_factor`
    """Forward and back substitution against a factorisation, in place on ``b``.

    **Every swap first, then the substitution.** Interleaving them -- swapping ``b[k]`` with
    ``b[piv[k]]`` just before eliminating with column ``k`` -- reads like the same thing and
    is not. `_factor` swaps whole rows of the working matrix, so the multipliers already
    computed in columns below ``k`` move with them; the multiplier now sitting in row ``k``
    belongs to the *permuted* right-hand side, and an earlier elimination that used it
    against an unpermuted ``b`` has already subtracted the wrong thing. LAPACK applies the
    permutation as one pass for exactly this reason.

    The interleaved version was here first and it survived every corpus model, because a
    Radau matrix is ``mu/h * I - J`` and for the step sizes these models take the diagonal
    dominates, so partial pivoting never chooses a row to swap. It failed the first random
    matrix of size four it was shown.
    """
    n = a.shape[0]
    for k in range(n):
        row = piv[k]
        if row != k:
            held = b[k]
            b[k] = b[row]
            b[row] = held
    for k in range(n):
        for i in range(k + 1, n):
            b[i] -= a[i, k] * b[k]
    for k in range(n - 1, -1, -1):
        for j in range(k + 1, n):
            b[k] -= a[k, j] * b[j]
        b[k] /= a[k, k]


def _norm(x: NDArray[np.float64]) -> float:
    """The root-mean-square norm the error test is written in.

    Not the Euclidean one. A norm that grows with the number of states would make the same
    tolerance mean something stricter on a bigger model, and the whole point of scaling by
    ``atol + rtol * |y|`` first is that a component either passes or it does not.
    """
    return float(np.sqrt(np.mean(x * x)))


# -- the method -------------------------------------------------------------------------


def _newton(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917 - one iteration, written out
    rhs: _Evaluate,
    t: float,
    y: Any,  # noqa: ANN401 - annotated loosely so numba types it, not the checker
    h: float,
    p: Any,  # noqa: ANN401
    w: Any,  # noqa: ANN401
    z: Any,  # noqa: ANN401
    scale: Any,  # noqa: ANN401
    tol: float,
    ar: Any,  # noqa: ANN401
    pr: Any,  # noqa: ANN401
    ac: Any,  # noqa: ANN401
    pc: Any,  # noqa: ANN401
) -> tuple[bool, int, float, int]:
    """Simplified Newton on the three collocation stages, in the eigenbasis of ``A``.

    Simplified: the Jacobian is the one held from an earlier point and the factorisations
    are reused across iterations, so an iteration costs three right-hand sides and two
    substitutions rather than anything cubic. That is why a stiff step is affordable at all,
    and it is also why the loop watches the *rate* of contraction rather than only the
    residual -- an iteration that is converging too slowly is telling you the Jacobian is
    stale, and refactorising is cheaper than grinding six iterations to a halt.

    Returns whether it converged, how many iterations it took, the observed rate (negative
    when only one iteration ran and there is no rate yet), and the right-hand side count.
    """
    n = y.shape[0]
    mu_real = MU_REAL / h
    mu_complex_r = MU_COMPLEX.real / h
    mu_complex_i = MU_COMPLEX.imag / h

    stages = np.empty((3, n))
    point = np.empty(n)
    wm = np.zeros((3, n))
    dw = np.empty((3, n))
    br = np.empty(n)
    bc = np.empty(n, dtype=np.complex128)

    for i in range(3):
        for q in range(n):
            total = 0.0
            for j in range(3):
                total += TI[i, j] * z[j, q]
            wm[i, q] = total

    calls = 0
    rate = -1.0
    previous = -1.0
    converged = False
    iterations = 0
    for k in range(NEWTON_MAXITER):
        iterations = k + 1
        for i in range(3):
            for q in range(n):
                point[q] = y[q] + z[i, q]
            rhs(t + h * C[i], point, p, w, stages[i])
        calls += 3
        if not np.all(np.isfinite(stages)):
            break

        for q in range(n):
            real = 0.0
            imag_r = 0.0
            imag_i = 0.0
            for i in range(3):
                real += TI_REAL[i] * stages[i, q]
                imag_r += TI_COMPLEX_R[i] * stages[i, q]
                imag_i += TI_COMPLEX_I[i] * stages[i, q]
            br[q] = real - mu_real * wm[0, q]
            bc[q] = complex(
                imag_r - (mu_complex_r * wm[1, q] - mu_complex_i * wm[2, q]),
                imag_i - (mu_complex_r * wm[2, q] + mu_complex_i * wm[1, q]),
            )
        _solve(ar, pr, br)
        _solve(ac, pc, bc)
        for q in range(n):
            dw[0, q] = br[q]
            dw[1, q] = bc[q].real
            dw[2, q] = bc[q].imag

        size = _norm(dw / scale)
        if previous >= 0.0 and previous > 0.0:
            rate = size / previous
        if rate >= 1.0:
            break
        if rate >= 0.0 and rate ** (NEWTON_MAXITER - k) / (1.0 - rate) * size > tol:
            break

        for i in range(3):
            for q in range(n):
                wm[i, q] += dw[i, q]
        for i in range(3):
            for q in range(n):
                total = 0.0
                for j in range(3):
                    total += T[i, j] * wm[j, q]
                z[i, q] = total

        if size == 0.0 or (rate >= 0.0 and rate / (1.0 - rate) * size < tol):
            converged = True
            break
        previous = size

    return converged, iterations, rate, calls


def _first_step(  # noqa: PLR0913, PLR0917 - the standard starting-step heuristic
    rhs: _Evaluate,
    t0: float,
    y: Any,  # noqa: ANN401
    p: Any,  # noqa: ANN401
    w: Any,  # noqa: ANN401
    f0: Any,  # noqa: ANN401
    rtol: float,
    atol: float,
    span: float,
) -> tuple[float, int]:
    """Hairer and Wanner's starting step: one explicit Euler probe, then a curvature guess.

    Getting this wrong is not fatal -- the controller recovers within a few steps either way
    -- but a first step that is far too large is rejected repeatedly at the point where no
    Jacobian has been reused yet, which is the most expensive place to be wrong.
    """
    n = y.shape[0]
    scale = np.empty(n)
    for q in range(n):
        scale[q] = atol + rtol * abs(y[q])
    d0 = _norm(y / scale)
    d1 = _norm(f0 / scale)
    h0 = 1e-6 if d0 < _TRIVIAL or d1 < _TRIVIAL else 0.01 * d0 / d1

    probe = np.empty(n)
    slope = np.empty(n)
    for q in range(n):
        probe[q] = y[q] + h0 * f0[q]
    rhs(t0 + h0, probe, p, w, slope)
    d2 = _norm((slope - f0) / scale) / h0

    biggest = max(d1, d2)
    h1 = (
        max(1e-6, h0 * 1e-3) if biggest <= _NEGLIGIBLE else (0.01 / biggest) ** (1.0 / (_ORDER + 1))
    )
    return min(100.0 * h0, h1, span), 1


def _factor_pair(  # noqa: PLR0913, PLR0917 - two systems, two factorisations, one step size
    jacobian: Any,  # noqa: ANN401
    h: float,
    ar: Any,  # noqa: ANN401
    pr: Any,  # noqa: ANN401
    ac: Any,  # noqa: ANN401
    pc: Any,  # noqa: ANN401
) -> bool:
    """Build and factorise ``mu/h * I - J`` for the real and the complex eigenvalue."""
    n = jacobian.shape[0]
    mu_real = MU_REAL / h
    mu_complex_r = MU_COMPLEX.real / h
    mu_complex_i = MU_COMPLEX.imag / h
    for i in range(n):
        for j in range(n):
            ar[i, j] = -jacobian[i, j]
            ac[i, j] = complex(-jacobian[i, j], 0.0)
        ar[i, i] += mu_real
        ac[i, i] += complex(mu_complex_r, mu_complex_i)
    return _factor(ar, pr) and _factor(ac, pc)


def _predict(h: float, h_old: float, error: float, error_old: float) -> float:
    """The step-size factor, from this error and the previous one.

    Two-step control -- Gustafsson's -- rather than the textbook single-step formula. The
    single-step rule reacts to one error and overshoots on a problem whose difficulty
    changes smoothly, which is every relaxation oscillator: it grows the step until the
    corner arrives, is rejected, halves, and does it again. Remembering the previous error
    damps that.
    """
    if error == 0.0:
        return MAX_FACTOR
    unknown = error_old < 0.0 or h_old < 0.0
    multiplier = 1.0 if unknown else h / h_old * (error_old / error) ** 0.25
    return min(1.0, multiplier) * error**-0.25


def integrate(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917 - one step function, deliberately not split
    rhs: _Evaluate,
    jacobian: _Evaluate,
    t0: float,
    t1: float,
    y0: Any,  # noqa: ANN401
    p: Any,  # noqa: ANN401
    w: Any,  # noqa: ANN401
    rtol: float,
    atol: float,
    grid: Any,  # noqa: ANN401
    out: Any,  # noqa: ANN401
) -> tuple[int, int, int, int, int, int, float]:
    """Integrate from ``t0`` to ``t1``, writing the state at every time in ``grid``.

    ``grid`` is sorted and within the span; ``out`` is ``(len(grid), n)``. The counters come
    back as a tuple because compiled code cannot build a Python object, and `Stats` gives
    them their names on the other side.

    One long function on purpose. Splitting the controller from the step reads better and
    costs a compiled call on the hot path for every quantity the two would have to pass; the
    parts that *are* split out -- the Newton solve, the factorisation, the substitutions --
    are the ones that are called once per iteration rather than once per anything smaller.
    """
    n = y0.shape[0]
    y = y0.astype(np.float64).copy()
    t = t0
    f = np.zeros(n)
    rhs(t, y, p, w, f)
    evaluations = 1

    jac = np.zeros((n, n))
    jacobian(t, y, p, w, jac)
    jacobians = 1
    fresh = True

    ar = np.empty((n, n))
    ac = np.empty((n, n), dtype=np.complex128)
    pr = np.empty(n, dtype=np.int64)
    pc = np.empty(n, dtype=np.int64)
    factored = False
    factorisations = 0

    z = np.zeros((3, n))
    quad = np.empty((n, 3))
    y_poly = np.empty(n)
    t_poly = t0
    h_poly = 1.0
    extrapolate = False
    y_new = np.empty(n)
    f_new = np.empty(n)
    error = np.empty(n)
    combined = np.empty(n)
    scale = np.empty(n)

    points = grid.shape[0]
    cursor = 0
    while cursor < points and grid[cursor] <= t0:
        for q in range(n):
            out[cursor, q] = y[q]
        cursor += 1

    tol = max(10.0 * EPS / rtol, min(0.03, rtol**0.5))
    h_prop, probes = _first_step(rhs, t0, y, p, w, f, rtol, atol, t1 - t0)
    evaluations += probes
    h_old = -1.0
    error_old = -1.0

    steps = 0
    rejected_total = 0
    status = 0
    while t < t1:
        floor = 10.0 * abs(np.nextafter(t, np.inf) - t)
        h_abs = h_prop
        h_previous = h_old
        error_previous = error_old
        if h_abs < floor:
            h_abs = floor
            h_previous = -1.0
            error_previous = -1.0

        rejected = False
        accepted = False
        iterations = 0
        rate = -1.0
        error_norm = 0.0
        h = h_abs
        t_new = t
        while not accepted:
            if h_abs < floor:
                status = _TOO_SMALL
                break
            h = h_abs
            t_new = t + h
            t_new = min(t_new, t1)
            h = t_new - t
            h_abs = h

            # The previous step's collocation polynomial, evaluated past its own end, is the
            # guess for this one. Starting from zero instead costs one or two extra Newton
            # iterations per step -- and the controller reads the iteration count to decide
            # whether the Jacobian is stale, so a bad guess does not merely cost iterations,
            # it recomputes and refactorises a Jacobian that was perfectly good. Measured on
            # `Robertson` at rtol 1e-8: 240 Jacobians for 253 steps from zero, 4 with this.
            for i in range(3):
                if extrapolate:
                    x = (t + h * C[i] - t_poly) / h_poly
                    for q in range(n):
                        z[i, q] = (
                            y_poly[q] + x * (quad[q, 0] + x * (quad[q, 1] + x * quad[q, 2])) - y[q]
                        )
                else:
                    for q in range(n):
                        z[i, q] = 0.0
            for q in range(n):
                scale[q] = atol + rtol * abs(y[q])

            converged = False
            while not converged:
                if not factored:
                    if not _factor_pair(jac, h, ar, pr, ac, pc):
                        status = _NO_CONVERGENCE
                        break
                    factored = True
                    factorisations += 1
                converged, iterations, rate, calls = _newton(
                    rhs, t, y, h, p, w, z, scale, tol, ar, pr, ac, pc
                )
                evaluations += calls
                if not converged:
                    if fresh:
                        break
                    for i in range(n):
                        for j in range(n):
                            jac[i, j] = 0.0
                    jacobian(t, y, p, w, jac)
                    jacobians += 1
                    fresh = True
                    factored = False
            if status != 0:
                break
            if not converged:
                h_abs *= 0.5
                factored = False
                rejected = True
                continue

            for q in range(n):
                y_new[q] = y[q] + z[2, q]
                combined[q] = f[q] + (E[0] * z[0, q] + E[1] * z[1, q] + E[2] * z[2, q]) / h
                error[q] = combined[q]
            _solve(ar, pr, error)
            for q in range(n):
                scale[q] = atol + rtol * max(abs(y[q]), abs(y_new[q]))
            error_norm = _norm(error / scale)
            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + iterations)

            if rejected and error_norm > 1.0:
                # A second pass through the same factorisation, evaluated at the *errored*
                # point. Hairer's fix for the estimate being unreliable exactly where it
                # matters: on a very stiff step the linear estimate above is itself
                # inaccurate, and on the second rejection in a row that is worth one extra
                # right-hand side to find out.
                for q in range(n):
                    y_new[q] = y[q] + error[q]
                rhs(t, y_new, p, w, f_new)
                evaluations += 1
                for q in range(n):
                    error[q] = f_new[q] + combined[q] - f[q]
                _solve(ar, pr, error)
                error_norm = _norm(error / scale)
                for q in range(n):
                    y_new[q] = y[q] + z[2, q]

            if error_norm > 1.0:
                factor = _predict(h_abs, h_previous, error_norm, error_previous)
                h_abs *= max(MIN_FACTOR, safety * factor)
                factored = False
                rejected = True
                rejected_total += 1
            else:
                accepted = True
        if not accepted:
            break

        steps += 1
        for q in range(n):
            for j in range(3):
                total = 0.0
                for i in range(3):
                    total += z[i, q] * P[i, j]
                quad[q, j] = total
        while cursor < points and grid[cursor] <= t_new:
            x = (grid[cursor] - t) / h
            for q in range(n):
                out[cursor, q] = y[q] + x * (quad[q, 0] + x * (quad[q, 1] + x * quad[q, 2]))
            cursor += 1

        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + iterations)
        again = iterations > _STALE_ITERATIONS and rate > _STALE_RATE
        factor = min(MAX_FACTOR, safety * _predict(h_abs, h_previous, error_norm, error_previous))
        if not again and factor < _WORTH_CHANGING:
            factor = 1.0
        else:
            factored = False

        rhs(t_new, y_new, p, w, f_new)
        evaluations += 1
        if again:
            for i in range(n):
                for j in range(n):
                    jac[i, j] = 0.0
            jacobian(t_new, y_new, p, w, jac)
            jacobians += 1
            fresh = True
        else:
            fresh = False

        h_old = h_prop
        error_old = error_norm
        h_prop = h_abs * factor
        for q in range(n):
            y_poly[q] = y[q]
        t_poly = t
        h_poly = h
        extrapolate = True
        t = t_new
        for q in range(n):
            y[q] = y_new[q]
            f[q] = f_new[q]

    while cursor < points:
        for q in range(n):
            out[cursor, q] = y[q]
        cursor += 1
    return status, steps, rejected_total, evaluations, jacobians, factorisations, t


# -- the two tiers ----------------------------------------------------------------------


def simulate(  # noqa: PLR0913 - the same knobs the interpreted path has
    model: System | Code | Kernel,
    *,
    stop: float,
    start: float = 0.0,
    tolerance: float = 1e-6,
    absolute: float | None = None,
    times: Sequence[float] | NDArray[np.float64] | None = None,
    points: int = 51,
    fold: bool = False,
    parameters: NDArray[np.float64] | None = None,
) -> tuple[Result, Stats]:
    """Run the fused loop and return what `modelica.sim.solve.simulate` returns, plus counts.

    ``times`` or ``points`` decides where the answer is sampled, and unlike the interpreted
    path there is no third option: the steps the method took are not reported. They are not
    hidden out of tidiness -- collecting them means growing an array inside the compiled
    loop, which is the one thing that would put an allocation on the hot path.
    """
    kernel = model if isinstance(model, Kernel) else build(model, fold=fold)
    if kernel.events:
        message = (
            f"{kernel.name} has {kernel.events} event condition(s), and the fused loop has no"
            " event handling -- use `modelica.sim.simulate`, which stops, applies the"
            " discontinuity and restarts"
        )
        raise FusionError(message)
    p, y, w = kernel.start(start, parameters)
    grid = (
        np.linspace(start, stop, points)
        if times is None
        else np.sort(np.asarray(times, dtype=float))
    )
    values = np.empty((len(grid), kernel.states))
    counted = integrate(
        kernel.rhs,
        kernel.jacobian,
        float(start),
        float(stop),
        y,
        p,
        w,
        float(tolerance),
        float(absolute if absolute is not None else tolerance),
        grid,
        values,
    )
    stats = Stats(
        int(counted[0]),
        int(counted[1]),
        int(counted[2]),
        int(counted[3]),
        int(counted[4]),
        int(counted[5]),
        float(counted[6]),
    )
    if stats.status == _TOO_SMALL:
        message = f"{kernel.name}: the step size went below the resolution of t = {stats.time:.12g}"
        raise FusionError(message)
    if stats.status == _NO_CONVERGENCE:
        message = (
            f"{kernel.name}: the collocation system would not converge at t = {stats.time:.12g}"
            " even on a freshly computed Jacobian"
        )
        raise FusionError(message)
    return _result(kernel, grid, values, p, w), stats


def _result(
    kernel: Kernel,
    grid: NDArray[np.float64],
    values: NDArray[np.float64],
    p: NDArray[np.float64],
    w: NDArray[np.float64],
) -> Result:
    """Every observed variable at every sample, from the states the loop wrote.

    One workspace for the whole trajectory, which the interpreted path cannot do: there it
    is rebuilt per segment because a latched relation holds what the last event decided.
    Here there are no events, so the latches never move.
    """
    observed = np.empty((kernel.observed, len(grid)))
    buffer = np.empty(kernel.observed)
    for column, moment in enumerate(grid):
        kernel.observe(float(moment), values[column], p, w, buffer)
        observed[:, column] = buffer
    return Result(
        name=kernel.name,
        time=np.asarray(grid, dtype=float),
        names=kernel.code.observed,
        values=observed,
        events=(),
    )
