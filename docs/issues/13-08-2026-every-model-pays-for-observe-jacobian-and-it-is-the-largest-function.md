---
title: Every model pays for observe_jacobian, and it is the largest function in the module
status: open
area: sim
owner: null
filed: 2026-08-13
kind: story
scope: M
verify: probe
---

## Measured

`Emitter.code` emits nine functions unconditionally. On a method-of-lines bar at n = 1000,
where the lines go:

| function | lines |
| --- | --- |
| `observe_jacobian` | 6,011 |
| `jacobian` | 5,008 |
| `jacobian_sparse` | 5,008 |
| `observe` | 4,010 |
| everything else | 15,071 |
| **total** | **35,108** |

`observe_jacobian` is the largest single function in the module — **17 %** of it — and it is
`d(observed)/dy`, the matrix a state estimator calls `H`. It was added for that
(`no-output-jacobian`, closed 2026-08-04) and it is the right thing to have.

Nothing in `modelica.sim.solve` calls it. A caller who integrates a model and reads the
result never touches it, and every corpus model that is not about estimation pays its emit
time, its `compile()` time and its bytes anyway.

The same is true of `jacobian_parameters` and `observe_jacobian_parameters`, which exist for
`sensitivity=` and are emitted whether or not anybody asked for a sensitivity.

## Inferred

Inferred, not measured: this is the cheapest large saving available on the source-size
problem, because it costs *nothing* at run time. Every other route
(`a-stencil-is-emitted-as-n-unrolled-scalar-blocks` measures two of them) trades source size
against either evaluation speed or a test that proves the sparsity pattern exact. Not
emitting a function nobody called trades against neither.

Also inferred: the shape is already half-built. `emit(system, sensitivity=(...))` exists and
takes the parameters to differentiate against, and emits nothing for the parameter Jacobians
when the tuple is empty — that is the same decision, made once, for one pair of functions.
Whether `observe_jacobian` should follow the same rule (`observe=(...)`, naming what to
differentiate) or a plain flag is a design question this entry does not settle.

## Done

- A caller can build a kernel without `observe_jacobian`, and the default for
  `modelica.sim.simulate` does not emit one, because it does not use one.
- `Kernel` says clearly what it does not have rather than carrying a name that raises when
  touched. An estimator asks for it and gets it.
- The saving is measured on the bar at several sizes and recorded, next to the numbers in
  `a-stencil-is-emitted-as-n-unrolled-scalar-blocks` so the two can be read together.
- `tests/test_examples.py` and `tests/test_sparsity.py` pass unchanged. The corpus models that
  *do* want it — the difference-quotient check over every model — ask for it explicitly.

## Deliberately not done

- No change to what the functions compute. This is about which ones are written, not how.
- Nothing about `jacobian_sparse` versus `jacobian`. Those two are the subject of the other
  entry, they trade against a real property, and this one deliberately takes the case that
  trades against nothing.
