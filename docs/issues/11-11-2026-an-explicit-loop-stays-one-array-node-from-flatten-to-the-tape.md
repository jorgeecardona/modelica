---
title: An explicit loop stays one array node from flatten to the tape
status: open
kind: story
scope: L
owner: jorgeecardona
filed: 2026-11-11
area: ir
verify: test
blocked_by: null
fixed_in: null
---

# An explicit loop stays one array node from flatten to the tape

Goal: a `for` loop over a state array goes through `flatten`, `causalize` and `vm.lower` as one
node, not as one scalar equation per cell. This is phase 2 of the shaped-arrays note, kept
narrow. It is not general array-aware Pantelides.

## Why

The front end costs more than integration. `scripts/frontend_bench.py`, wall time, on the
engine, `stop=0.01` (`simulate` includes a second `lower`):

| Box | `flatten` | `causalize` | `lower` | `simulate` | per-cell calls in `lower` |
|---|---|---|---|---|---|
| n = 13, 2197 states | 1.27 s | 2.04 s | 2.45 s | 2.83 s | |
| n = 20, 8000 states | 5.05 s | 8.53 s | 11.19 s | 10.41 s | 112,004 |

The cost is per cell. `causalize` builds 8000 scalar blocks, `_array_blocks` finds them again by
position, and `vm.lower` reads every subscript at every cell (`_read_plan` calls `indices_at`
14 times per cell).

Done when: on the n = 20 box, `flatten + causalize + lower` is under 2 s, n = 20 costs less than
twice n = 13, and the per-cell calls in `lower` do not grow with n.

## Plan (Fable, 2026-11-11), one commit each, same bits as today in every step

1. Benchmark script and a bound test on the per-cell calls (`tests/test_frontend_bench.py`).
2. `Mode` holds the loop and expands its scalar blocks only on demand (SciPy path, diagrams).
   Test: the expanded blocks equal today's for the box and TorusHeat.
3. Coarse matching: a loop `der(x[i..]) = f`, with `x` written by no other equation, is one node
   matched to `der(x)`. Anything else expands as today. Test: the same states and block order
   across the corpus.
4. BLT on the coarse graph; a cycle through an array node expands it. Test: a model with an
   algebraic loop through an array gives the same `System`.
5. `vm.lower` lowers from the loop template: no per-cell comparison or reads. Test: the same
   tape and the same bits on the engine.
6. The shaped Jacobian without the scalar path. Test: the same entries on the box.
7. `flatten` keeps the loop instead of writing one equation per cell. Test: the `examples/`
   twins still flatten equal.

## Traps

- `causalize` attaches arrays only when no alias was removed and nothing was differentiated.
  Step 3 keeps that rule or carries a real index map; position does not survive.
- Every reader of `mode.blocks` (SciPy fallback, events, diagrams) must be found before step 2.
- Boundary cells written by separate equations make `x` partly loop-owned: refuse and expand.
- Bit equality is the test oracle in every step.

## Progress (2026-11-11, evening)

Done, each with the same `System` and tape over the corpus: `Loop.matches` builds two cells
(f6a2c5c); `_affine` gives up on more than two unknowns and incidence is cached (82b5734);
`_read_plan` evaluates an index along its loop (2ccd849); no unpack without events (a0ae68e);
`initial` skips start stores an initial equation overwrites (c495ffd). Box n = 20, wall, loaded
machine: `flatten` about 2 s, `causalize` about 2.6 s, `lower` about 5.6 s.

Tried and parked: `_fuse` at the end of `_fold` (pass input is 136k instructions on the box, 75%
single `LOAD_Y`/`STORE_OUT`). It gives the same tape on the box but not on 26 corpus models:
`_compact` numbers a fused range's registers as one block on first touch, so later loads are
numbered apart and the final `_fuse` no longer joins them (Lorenz `jacobian`: 27 against 25
instructions). Also `_hoist` makes loads adjacent that were not at `_fold`. A fix must make
`_compact` number a range in the order the late path would, or accept different tapes.

## Where the incremental track stops (2026-11-12)

Since the evening entry: early fusion done (9f94220, `_compact` numbers a joined run register by
register, tapes identical on the corpus) and `_compact` numbers a group in one update (3be1452).

Box n = 20, quiet machine, wall: `flatten` 1.90 s, `causalize` 2.28 s, `lower` 4.94 s,
integration about 0.6 s. Total 9.1 s, from about 25 s on 2026-11-11. n = 13: 0.52, 0.49,
1.24 s.

`lower` by pass (cProfile, Fable): `_fold` 1.13 s, `_jacobian` 1.06 s, `_compact` 0.88 s,
`_fuse` 0.24 s. Pass input: jacobian 64k instructions (53,600 stores, one per nonzero), initial
32k, observe 24k, rhs 16k.

Parked, with the reason:

- Vectorising `initial`: removing it entirely saves at most 0.47 s. Do it after step 7, when the
  initial loop arrives as an array node.
- `rhs` and `observe` writing `VLOAD_Y`/`VSTORE_OUT` directly: about 0.3 to 0.4 s, because the
  jacobian stores stay one per nonzero, and `_prune` would need a rule to split a partly read
  `VLOAD_Y` to keep tapes identical.

What is left is structural, each estimated at 1 to 2 s: step 6 first (`_Tangents` seeds a
`VLOAD_Y` run and stores runs through `VSCATTER`; test: the same jacobian entries on the box),
then steps 2 and 3.
