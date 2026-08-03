---
title: "RoastingMachine did not compile: a derivative chain of zeros, not the recursion limit"
status: done
area: sim
owner: quiet-tars-187f2
filed: 2026-08-03
fixed_in: 71dfaaf
kind: bug
scope: XS
verify: probe
---

## What this was filed as, and why that was wrong

Filed as: raise the recursion limit around the Numba compile, in a `try`/`finally`, and add
`RoastingMachine` to `tests/test_jit.py`. Both halves of that were wrong, and the second half
would have hung the suite for half an hour.

Three things were measured before touching anything, in a throwaway environment
(`uv run --isolated --with numba`):

1. **The recursion limit is not what stops it.** `rhs` compiles at the default limit of 1000,
   in 5.6 s. The `RecursionError` reported on 2026-07-30 and 2026-08-01 is real and marginal —
   about 975 frames against 1000, so it depends on the caller's stack — but it is not the
   blocker, and raising the limit fixes nothing.
2. **Numba compiles lazily, at the first call**, which `src/modelica/sim/kernel.py:151`
   already says. So there is no compile to put a `try`/`finally` around in `_accelerate`.
   The bracket the issue asked for could not have existed where it asked for it.
3. **`jacobian` is what blows up**, and only `jacobian`. Per generated function, first call,
   with the address space capped at 6 GB:

       rhs              5.6 s   peak rss 0.27 GB   ok
       jacobian        15.6 s   peak rss 4.40 GB   MemoryError
       observe          9.2 s                      ok
       indicators       5.5 s                      ok
       transition       7.5 s                      ok

## The actual cause

Read the emitted source rather than guessing from the timings. The gas schedule is nine
constant segments, so its derivative is zero on every segment, and `d_dx` wrote that out
faithfully:

    _d_der_T_mach = ... (0.0 if w[30] > 0.5 else 0.0 if w[31] > 0.5 else 0.0 if w[32] > 0.5
                         else ... else 0.0) ...

A nine-branch chain whose every answer is the same number. `modelica.ir.derive.simplify` had
folds for arithmetic and none for a conditional with two identical arms, so the chain
survived. `modelica.sim.emit` builds each implicit block's Jacobian through the same
`simplify(d_dx(...))` (`emit.py:403`, `emit.py:765`), so the chain landed in **every**
generated function and not only in `jacobian`.

The model was never the problem. `rhs`'s own schedule — `20 if w[30] > 0.5 else 27 if ...`,
nine *different* constants — compiles in a fifth of a second and is untouched.

## The fix, and what it moved

One fold in `simplify`: if both arms are structurally equal, return one of them.

                    before                          after
    rhs             5.6 s                           0.2 s
    jacobian        15.6 s, 4.4 GB, out of memory   2.6 s, 0.21 GB
    observe         9.2 s                           0.4 s
    indicators      5.5 s                           0.2 s
    transition      7.5 s                           0.5 s

`RoastingMachine` is in `tests/test_jit.py`'s `COVERED` now and `moon run :test-jit` passes in
35 s for all five models — *less* than the 37.9 s the pendulum alone used to take, because the
same fold shrinks its two modes as well. `code.events` is still 9, so no relation was folded
out of the model: only the derivative's chain of zeros went. 593 tests pass.

## What this leaves

- The `RecursionError` is still reachable in principle, on a schedule long enough. It is no
  longer reachable on anything in the corpus, and it was never the reason `RoastingMachine`
  would not compile. Not worth a limit change on a hypothetical.
- `tests/test_jit.py`'s note claiming "a model whose behaviour is a long schedule against
  `time` is effectively not compilable today" is corrected in the same commit. It was a
  statement about a missing fold.
- The 2026-07-30 decision to keep the branch chain rather than flatten it into arithmetic was
  right, and is now right for a better reason: the chain was never what cost anything.
