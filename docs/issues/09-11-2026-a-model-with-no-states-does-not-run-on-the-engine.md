---
title: A model with no states does not run on the engine
status: done
kind: bug
scope: S
owner: jorgeecardona
filed: 2026-11-09
area: sim
verify: test
blocked_by: null
fixed_in: 0.0.248
---

A model whose only unknowns are discrete stops at t = 0 on `backend="engine"` with "the
required step size fell below the spacing between numbers". `backend="python"` runs it.

```modelica
model Z
  discrete Real a(start = 0, fixed = true);
equation
  when time >= 0.01 then a = 1; end when;
end Z;
```

`md_first_step` gives a step of zero when `n = 0`. Done when `Z` and the `Pulse` model in
`tests/test_sample.py` give the same events on both backends.

Fixed: `md_norm` of an empty vector is 0 (it was 0 / 0), `md_newton` returns converged at once
when `n = 0`, and `rustsim.assemble` builds an empty state block. Test:
`tests/test_sample.py::test_a_model_with_no_states_runs_on_the_engine`.
