---
title: Causalization never returns on a three-equation linear ODE, because alias elimination can put a cycle in the derivative map
status: open
area: ir
owner: null
filed: 2026-08-13
kind: bug
scope: S
verify: test
---

## Measured

This model does not causalize. Not slowly -- at all: it was still running after 90 seconds,
and the loop it is in cannot end.

```modelica
model M
  Real x0(start = 1, fixed = true);
  Real x1(start = 1, fixed = true);
  Real x2(start = 1, fixed = true);
equation
  der(x0) = x2;
  der(x1) = 2 * x0;
  der(x2) = x0;
end M;
```

Three explicit ODEs, index 1, no algebraic loop, nothing a person would look at twice. Its
neighbours are all fine, which is what makes the trigger exact:

| model | |
| --- | --- |
| the one above | **> 90 s, did not return** |
| `der(x1) = x0` (the scale removed) | 2.0 ms |
| `der(x2) = 3 * x0` (the cycle broken) | 1.6 ms |
| two states, `der(x0) = x1; der(x1) = x0` | 1.1 ms |

Found by `tests/test_causalize_properties.py`'s `solvable` generator on the day it was
written, and shrunk by hand to the three lines above.

`faulthandler.dump_traceback_later(20)` says where:

```
File "src/modelica/ir/causalize.py", line 169 in level
File "src/modelica/ir/causalize.py", line 518 in _priority
File "src/modelica/ir/causalize.py", line 560 in <lambda>
File "src/modelica/ir/causalize.py", line 558 in candidates
File "src/modelica/ir/causalize.py", line 577 in augment
File "src/modelica/ir/causalize.py", line 584 in run
File "src/modelica/ir/causalize.py", line 593 in _match
```

`_Work.level` at `src/modelica/ir/causalize.py:164` walks the `integral` map to a root:

```python
def level(self, name: str) -> int:
    """How many times ``name`` has been differentiated: ``der_der_x`` is 2."""
    count = 0
    while name in self.integral:
        name = self.integral[name]
        count += 1
    return count
```

That `while` assumes `integral` is a forest. On this model it is not.

## Inferred

The mechanism, inferred from reading `_permits` and `_pair` rather than from an instrumented
run -- the measurement above is the hang and the stack, not this chain:

- `der(x0) = x2` is an affine relation between two unknowns with unit coefficients, so
  `_pair` takes it. `_permits` lets `der_x0` be the victim (`src/modelica/ir/causalize.py:340`
  -- a derivative may be replaced when the ratio is 1, the shift is zero, and the other name
  is not itself a derivative), `x2` cannot be the victim because it is `fixed`. So
  `der_x0 := x2`, and `derivative_of` installs `integral[x2] = x0`. **`x2` is now the
  derivative of `x0`**, which is the documented and usually desirable move.
- `der(x2) = x0` is then the same shape, and the same reasoning gives `der_x2 := x0` and
  `integral[x0] = x2`.
- `integral` is now `{x2: x0, x0: x2}`, and `level` walks it forever.

The guard that should have caught it is the first line of `_permits`
(`src/modelica/ir/causalize.py:336`):

```python
if work.derivative.get(victim) == other or work.integral.get(victim) == other:
    return False
```

That is a **one-step** check, and this cycle is two steps: at the second elimination the
victim is `der_x2` and the other is `x0`, and `integral[der_x2]` is `x2`, not `x0`. The
existing comment there explains the one-step case at length -- `der(x) = -x` read as an alias
-- so the guard was written for exactly this failure mode and stops one hop short of it.

Also inferred: `der(x1) = 2 * x0` is load-bearing rather than decoration. Remove the scale and
`der_x1 := x0` is eliminable too, so the third equation is consumed and the matching never
runs. The scale is what leaves an equation for `_match` to walk, which is where `level` is
called.

## Done

- The model above causalizes, and a test holds it -- `tests/test_causalize_properties.py`
  carries it today as `xfail(strict=True)` with a wall-clock budget, and closing this means
  deleting the marker rather than writing a new test.
- `integral` and `derivative` are acyclic by construction, whatever `_permits` allows. The
  cheap version is to reject a substitution that would close a cycle by walking to the root
  before installing it; the honest version is that `level` cannot loop even if something else
  ever installs one.
- The `solvable` property run no longer needs its timeout guard, which exists only because a
  generated draw can hang the suite.

## Deliberately not done

- No opinion on which of `x0` and `x2` *should* be the state. Both readings are legitimate --
  each is the other's derivative -- and picking one is state selection, which is
  `_Matching`'s job and not this guard's.
- Nothing about the other refusal the same generator found; that is
  `a-triangular-model-is-refused-when-alias-elimination-renames-a-variable`.
