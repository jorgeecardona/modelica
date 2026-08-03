---
title: A C emitter — the second backend, and the one FMU export needs
status: open
area: sim
owner: null
filed: 2026-08-03
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
