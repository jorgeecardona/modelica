---
title: Nothing caches a compiled kernel, though the content hash that would key it exists
status: done
area: sim
owner: cute-replicant-6e76d
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

## What landed

`modelica.sim.kernel.build` is cached on the content of the model. Measured on this machine,
building every model in the corpus once:

| | cold | warm |
| --- | --- | --- |
| whole corpus, interpreted | 272.8 ms | 0.68 ms |
| `AtwoodPendulum` alone | 168.7 ms | 0.17 ms |
| `Robertson`, `jit=True`, first call included | 3.16 s | 0.07 ms |
| `Pendulum`, `jit=True`, first call included | 2.78 s | 0.10 ms |

The two `jit` rows are the ones this was reached for, and they had to be measured in a
throwaway environment because numba is in no dependency group here. They include exercising
every entry point once, since numba compiles at the first call and a build that only returns
the functions would have measured nothing.

Each Done bullet:

- **Same `Kernel` for the same model.** `build` normalises `jit` and defers to an
  `lru_cache`d `_compiled`. `clear_cache` and `cache_info` are exported so a test can start
  cold and read the hit count.
- **Keyed on content, not on the file.** `tests/test_cache.py` asks it three ways: the same
  model built twice is one entry; the same model written with different comments, blank lines
  and spacing is the *same* entry; `-v / R` changed to `-v * R` is a different one. Identity
  (``is``) and not equality, because two kernels from one model compare equal whether or not
  anything was cached.
- **`System` is hashable.** `modelica.ir.system.FrozenMap`, coerced in `__post_init__`.
- **Numba's own `cache=True`** is answered below, and stays off.

### Why `FrozenMap` and not a tuple of pairs

The issue offered the tuple, as `ir.Call.named` does. Ten places read `Mode.derivative` and
`System.aliases` with `[]`, `.items()` and `in`, and turning a mapping into a sequence to win
a hash would have made every one of them say `dict(...)` first. `FrozenMap` is a
`collections.abc.Mapping` that adds `__hash__` and nothing else -- equality stays the one
`Mapping` already defines, so a test comparing `system.aliases` to a dict literal reads
unchanged, and the hash is over a `frozenset` of the pairs so that it agrees with that
equality.

The cost of agreeing is stated in the class docstring rather than hidden: iteration order is
visible downstream, so two systems differing *only* in the order of otherwise identical
aliases would compare equal and share an entry while generating different source.
Causalization is deterministic, so the pipeline cannot produce that pair. It is a hole in the
argument and not one in the behaviour.

### Numba's `cache=True`, answered

It cannot be turned on, and it does not degrade -- it raises at decoration time. Measured on
numba 0.66 against a function built the way `kernel.build` builds them:

    RuntimeError: cannot cache function 'f': no locator available for file '<modelica:Demo>'

The guess in the Inferred section was right: numba's cache keys on the source *file*, and
these functions come from `exec` with a synthetic filename. Writing the generated source to a
real file would give numba a locator, and that is a disk cache -- which this issue put out of
scope for a reason. The reason and the error are recorded at that line in `_accelerate`.

## Deliberately not done, found while doing it

- **`lru_cache` keys on the call as written**, so `build(system)` and `build(system, jit=False)`
  were two entries for one answer -- and both forms are in this repo, since
  `modelica.sim.solve.simulate` always passes the keyword. Decorating `build` directly
  compiled the same model twice while reporting a hit rate that looked fine. `build` is now a
  wrapper that normalises the keyword before the cache sees it, and there is a test named for
  the mistake.
- **The cache is bounded at 32 and nothing measured that number.** An entry holds the `System`
  it was keyed on and every array the generated module allocated. Thirty-two is more than a
  notebook or a test session has models; it is a guess, and it is one constant to change.
- **`FlatModel` is not cached.** `flatten` and `causalize` run again for every call that
  starts from source; only `build` is spared. Those passes are the cheap ones -- 273 ms for
  the whole corpus is the *build*, and the corpus flattens in less -- but a caller that
  re-parses in a loop still pays them.
- **Nothing on disk, as planned.** A second process compiles from scratch.
