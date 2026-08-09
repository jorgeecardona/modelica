---
title: A C emitter — the second backend, and the one FMU export needs
status: done
area: sim
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: a0d8cca 6dc0b21 8e1e4b3 4eeb442 8ad24a1
kind: story
scope: L
verify: test
---

## Measured

- `src/modelica/sim/emit.py:8` states the design: "The same emitter, pointed at a different set
  of templates, is what will produce C mode." No second set of templates exists —
  `grep -rn "\.c\b\|translation unit" src/modelica/sim/` finds nothing.
- The emitter writes five functions into one namespace: `rhs`, `jacobian`, `observe`,
  `indicators`, `transition` (`src/modelica/sim/emit.py:48`), over a flat float workspace `w`
  with parameters in `p`. That interface is already C-shaped: no Python object appears in it.
- `docs/ideas/backends.md:42` lists C as "one translation unit — FMU export, where a shared
  library is the deliverable".
- `src/modelica/fmi/__init__.py` says export is not implemented, and
  `docs/architecture.md:236` describes it as "a `System` plus a generated
  `modelDescription.xml`, with the FMU's C entry points backed by the Python evaluator".

Note the disagreement between those last two, and it is worth naming: the architecture note
allows an FMU whose binary calls back into CPython; the backends note wants a self-contained
translation unit. Both are real options. Only the second produces an FMU somebody else's tool
can load without installing Python.

## Inferred

The emitter is *mechanical* to retarget and the IR was designed for it —
`docs/ideas/backends.md:88` says the surface that has to be correct is the IR and the
interpreter, and the emitters are consequences. The scope of L comes from the parts that are
not the arithmetic: `np.linalg.solve` in the implicit blocks needs a C linear solve (LAPACK, or
a small dense LU written here), and the mode dispatcher needs to become a switch. That is
inference from reading `emit.py`, not from having tried it.

## Done

- `modelica.sim.emit` produces C source for the same five functions, from the same `System`,
  with the same workspace layout.
- The generated unit compiles with `cc -std=c11 -O2 -Wall -Werror` and links with no dependency
  beyond libm and whatever the linear solve needs.
- Differential test, in the style `tests/test_jit.py` already uses: for every model in the
  corpus, the C functions and the interpreted ones agree on the same inputs to the tolerance
  that file already sets. No reference numbers of the emitter's own.
- The compiled unit is reachable from Python (ctypes) so the same corpus simulations can run
  through it end to end.

## Prior art, added 2026-08-03

[Rumoca](https://arxiv.org/abs/2606.14998) writes its backends as **Jinja templates** rather
than as code, and emits CasADi, SymPy, FMI 2.0 and 3.0, JavaScript, JAX, Julia, ONNX and
embedded C from one intermediate representation. That is `docs/ideas/backends.md`'s claim —
"the emitters are mechanical" — taken all the way, and this issue is where the claim gets
tested here: if the C emitter comes out as a near-copy of the Python one with different string
literals, templates were the right shape.

Their fourth IR is worth knowing about and is not proposed here: a register-based opcode
stream with Jacobian-vector products precomputed by forward mode, which is the alternative to
emitting source at all. See `docs/ideas/landscape.md`.

## Deliberately not done

- No build system for it, and no vendored compiler. The test finds `cc` or skips, the way
  `tests/test_jit.py:32` skips without Numba.
- No decision here about which of the two FMU export routes wins. That belongs to the export
  issue; this one only makes the second route possible.

## What landed

Five commits, in the order the functions came out: `a0d8cca` the dialect and `rhs`,
`6dc0b21` the linear solve and the mode dispatcher, `8e1e4b3` `observe` and `indicators`,
`4eeb442` `transition`, and this one `jacobian`. 1037 tests pass.

All four Done bullets, measured:

| bullet | evidence |
| --- | --- |
| five functions, same `System`, same workspace | `modelica.sim.cgen.FUNCTIONS`, and `unit` writes all five for all 21 corpus models |
| compiles under `-std=c11 -O2 -Wall -Werror`, libm only | `tests/test_cgen.py:_library`, every model, every test |
| differential against the interpreter | four point tests over the whole corpus at 1e-12, plus the dialect's own 26 expression cases |
| reachable from Python, corpus simulations run through it | `test_the_corpus_simulates_through_the_compiled_unit` -- `simulate` driving the shared object |

The split the issue guessed at held. `modelica.sim.dialect` is *spelling* -- operators, the
names of built-ins, how a conditional is written -- and it is data, two values in one file,
so every disagreement between the languages is visible in one place. `modelica.sim.cgen` is
*structure*: semicolons, declarations, and the three things that are loops in C and are not
loops in numpy. Nothing about what a block solves for, what order blocks run in, which
relations are latched, or what a tangent differentiates against moved out of
`modelica.sim.emit`. That is the claim `docs/ideas/backends.md:88` makes, and it survived.

Where it did *not* hold is worth as much: three of the five functions needed structure that
no template can carry.

1. **The Newton loop.** A fixed trip count and a convergence break are statements.
2. **The dense solve.** `np.linalg.solve` is one call; C got `md_solve`, thirty lines of LU
   with partial pivoting in the preamble, because a BLAS dependency for a two-by-two would
   be a build system in exchange for nothing.
3. **The Jacobian.** This is the only place the two backends differ in *data layout* rather
   than in wording. A tangent row is a numpy vector in Python -- `_seed[k]`, whole-row
   assignment, `_null` for a structurally zero row -- and a `double[n]` here, so every
   assignment to one becomes a loop, and the right-hand side of an implicit block grew a
   column per state and is solved one column at a time.

Five bugs the differential tests found, each of which had compiled and run:

- **Integer division.** `burner / 100` in C with an `int` literal is integer division.
  `RoastingMachine` came out 23% wrong and silent. `Dialect.integer` now writes `100.0`.
- **`!` binds tighter than a comparison.** `!a > 0.0` is `(!a) > 0.0`. Fixed per dialect
  with `negation_level`, because Python's `not` binds the other way.
- **`&&` inside `||`** is unambiguous and `-Wall` warns anyway, so under `-Werror` the unit
  did not build. `Dialect.clarify` adds the brackets rather than the flag coming out.
- **Dead stores.** `-Werror=unused-but-set-variable` turns a line Python spends a
  microsecond on into a failed build, so `_live` and `_reached` prune -- the second by
  group, because a tangent row is four lines and cannot be pruned one line at a time.
- **Semicolons, wrong in both directions.** Appending one to every line wrote `};`, legal
  and ugly; skipping one for any line ending in a brace broke `double _b[2] = {0}`. The rule
  now asks what the line *is*.

## Deliberately not done, found while doing it

- **`parameters`, `initial`, `latch` and `observe_jacobian` are not in the unit.** Four of
  the nine `Kernel` holds. Three are start-up and the fourth is an output Jacobian nothing
  in this repo calls; the end-to-end test names them where it substitutes the other five.
  An FMU needs `initial` at least, and that is the export issue's to take.
- **No CSE in the C bodies.** `Emitter.sharing` is used by the Python emitter and not here,
  so a repeated subexpression is written out repeatedly. `-O2` is very good at exactly this
  and the generated source is more readable without it, but that is a judgement made on
  reading rather than on measuring.
- **One start point per model.** Every point test takes the model's own initial state. A
  random vector would compare two functions outside the domain either was written for, and
  a *sweep* of valid states would be better than either -- that is a property-test shape and
  a separate piece of work.
