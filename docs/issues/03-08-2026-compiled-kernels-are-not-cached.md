---
title: Nothing caches a compiled kernel, though the content hash that would key it exists
status: open
area: sim
owner: null
filed: 2026-08-03
kind: story
scope: M
verify: test
---

## Measured

- `docs/ideas/backends.md:70` lists caching among the five things owning the compiler buys:
  "The IR nodes are frozen, slotted and hashable, so a `FlatModel` has a content hash and
  needs no extra machinery to get one. That hash is the cache key for the compiled kernel. It
  recompiles when the model changed, and not when the mtime of the file changed."
- Nothing does it. `grep -rniE "cache|lru_cache" src/modelica/sim/` finds `cached_property` in
  `result.py` and `njit(cache=False, ...)` in `kernel.py:184`, and nothing else. Every call to
  `build` re-emits the source, re-`exec`s it, and re-compiles it.
- The hash claim was checked on `RLCCircuit` rather than assumed:

      FlatModel   hashable
      System      NOT hashable -- unhashable type: 'dict'
      Code        hashable

  So the key is available at both ends of the pipeline and not in the middle. `System` carries
  a `Mapping` (`Mode.derivative`), and a dict is not hashable.

## Inferred

Where this costs, inferred from what is being paid for rather than from a timing: emitting
and `exec`ing is cheap, and Numba is not. `tests/test_jit.py` records 5.6 s for `Robertson`
and 37.9 s for `Pendulum` — the pendulum has two modes, so every function is written and
compiled twice. Anything that builds the same model more than once in a process pays that
again: a notebook re-running a cell, a parameter sweep that rebuilds, a test session.

Not inferred: `njit(cache=False)` is explicit, so even Numba's own on-disk cache is off. Why
it is off is not written down at that line, and whoever picks this up should find out before
turning it on — the emitted module has a synthetic filename (`<modelica:Name>`), and Numba's
cache keys on the source file, which is the likely reason.

## Done

- `build` returns the same `Kernel` for the same model without recompiling, keyed on the
  content hash of the value it was given.
- The key is the *content*, not the file or its mtime: two identical models from different
  paths hit the same entry, and changing one character in an equation misses it. That is the
  observation the test asserts, and it is the whole claim in `backends.md`.
- `System`'s unhashability is either fixed (the mapping becomes a tuple of pairs, as
  `ir.Call.named` already does for the same reason) or the cache keys on the `FlatModel` and
  the `Code` and says why in a comment. Either is fine; leaving the middle of the pipeline
  quietly unhashable while a document says otherwise is not.
- Whether Numba's own `cache=True` can be turned on is answered in the same change, with the
  reason recorded at that line whichever way it goes.

## Deliberately not done

No cache on disk across processes in the first pass. In-process is where the repeated builds
are, and a disk cache adds an invalidation problem that a content hash makes tractable but not
free.
