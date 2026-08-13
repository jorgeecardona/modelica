---
title: A SciPy ValueError escapes simulate when two crossings sit on zero at the same instant
status: done
area: sim
owner: quiet-tars-187f2
filed: 2026-08-13
fixed_in: b2537f8
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

## Fixed 2026-08-13

`_Run._segment` catches `ValueError` from `solve_ivp` and re-raises a `SimulationError` that
says which relations the root finder could not bracket.

    Fight: no crossing could be located at t = 1e-09 -- 'y > 0', 'x > 0' sit at zero to within
    the solver's tolerance of 1e-06, so there is no sign change to bracket
    (scipy: f(a) and f(b) must have different signs)

`ValueError` and not `Exception`, for the reason this entry's Deliberately-not-done gives:
turning every SciPy failure into a `SimulationError` would hide a genuine bug in the source
this package generates, and those tracebacks are how such a bug gets found.

### The near-zero test is a tolerance, not `== 0.0`

This entry's inference was "both indicators are exactly zero at `t = 0`". Measured, that is
true at `t = 0` and **not true where the failure happens**. The failing segment starts at
`t = 1e-9`, after an event has already fired, with the state at `1e-9` and both indicators
reading `1e-9` against an `atol` of `1e-6`.

So an exact test would have named nothing in the one case the method exists for. The predicate
is `abs(value) <= self.absolute` -- what the solver itself cannot tell from zero -- and the
number is printed, so the reader can see the scale the judgement was made on.

### The must-not-break case, measured

An indicator at exactly zero at `t = 0` is ordinary and is **not** what breaks `Fight`.
`tests/test_chattering.py::ON_THE_THRESHOLD` carries three models whose watched relation reads
`0.0` at the start:

| | at `t = 2` |
| --- | --- |
| `der(x) = 1 + (if x > 0 then 1 else 0)` | `x = 4` |
| `der(x) = -1 - (if x > 0 then 1 else 0)` | `x = -2` |
| `der(q) = if x > 0 then 2 else 0`, `der(x) = 1` | `x = 2`, `q = 4` |

All three simulate and all three are right. What breaks `Fight` is not the zero: it is two
relations at zero where each decides the flow the other crosses on. Those three models are the
guard on any future fix here -- a nudge or a bracket guard that broke them would be a cure
worse than the disease.

1472 pass, typecheck 0.
