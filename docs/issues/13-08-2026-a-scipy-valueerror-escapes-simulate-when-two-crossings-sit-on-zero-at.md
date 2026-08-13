---
title: A SciPy ValueError escapes simulate when two crossings sit on zero at the same instant
status: open
area: sim
owner: null
filed: 2026-08-13
kind: bug
scope: S
verify: test
---

## Measured

```modelica
model Fight
  Real x(start = 0, fixed = true);
  Real y(start = 0, fixed = true);
equation
  der(x) = if y > 0 then -1 else 1;
  der(y) = if x > 0 then 1 else -1;
end Fight;
```

`simulate(system, stop=5.0)` raises, in 0.01 s:

    builtins.ValueError: f(a) and f(b) must have different signs

That is `scipy.optimize.brentq`, reached through `solve_ivp`'s `handle_events` ->
`solve_event_equation`, surfacing out of `modelica.sim.simulate` unchanged. The traceback ends
inside `scipy/optimize/_zeros_py.py`. No model name, no time, no indication of which relation,
and nothing a caller can act on -- and `SimulationError` exists in this package for exactly
this job.

The entry path is `src/modelica/sim/solve.py:331`, the `solve_ivp` call in `_Run._segment`.

Measured on 2026-08-13. `tests/test_chattering.py::test_two_relations_fighting_raise_a_simulation_error`
carries the model, asserts `SimulationError`, and is an expected failure with
`raises=ValueError` recording what comes out today.

## Inferred

Inferred from the exception rather than from instrumenting SciPy: both indicators are exactly
zero at `t = 0`, because both states start at zero and both relations test against zero. SciPy
brackets a root between the previous and current step and hands the bracket to `brentq`, which
requires a sign change across it; with an indicator that is zero at both ends there is none,
and `brentq` refuses the bracket rather than returning the endpoint.

Also inferred: this is not the same defect as
[`the-standstill-counter-cannot-fire`](13-08-2026-the-standstill-counter-cannot-fire-because-a-chattering-run-advances-by-the.md),
though the same file found both. That one is a run that goes forever without complaint; this
one is a run that stops immediately with the wrong complaint. A model can hit this without
chattering at all -- any model whose indicator starts at exactly zero is a candidate, and a
relation like `x >= 0` on a state that starts at `0` is an ordinary thing to write.

Not established: whether the right answer is to nudge the initial indicators the way `_REARM`
nudges a fired one, or to catch `ValueError` around the `solve_ivp` call and re-raise with the
model name and time, or both. The first prevents it; the second makes any other SciPy
internal legible on the way out. They are not alternatives.

## Done

- `simulate` on the model above raises `SimulationError`, and the message names the model and
  the instant. If a crossing genuinely cannot be located, saying so is the answer.
- The expected failure in `tests/test_chattering.py` becomes a passing test, with its marker
  deleted.
- A model whose indicator is exactly zero at `t = 0` and is *not* pathological still
  simulates. `der(x) = 1` with a relation `x >= 0` and `x(start = 0)` is the case to check;
  it must not be broken by whatever nudge or guard goes in.

## Deliberately not done

- No blanket `except Exception` around `solve_ivp`. Turning every SciPy failure into a
  `SimulationError` would hide a genuine bug in this package's own generated code, which is
  what those tracebacks are for. The `ValueError` from the root finder is a specific,
  identifiable condition.
- Nothing about the model itself being ill-posed. It is, and that is not the point: a caller
  who writes an ill-posed model is exactly who needs a legible error.
