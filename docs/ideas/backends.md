# One IR, several backends, and a debugger that still works

!!! note "Tracked in the register"
    The work this page argues for is tracked in [`docs/issues`](https://gitlab.com/jorgeecardona/pymodelica/-/tree/main/docs/issues): the C emitter, common-subexpression elimination, tearing, kernel caching, the two parameter modes, the fused-step JIT, differential debugging, and the sparse Jacobian. This page is the argument; the register is what is owed.

OpenModelica has this pipeline: `.mo` to generated C, then a C compiler, then an
executable, then run it. It works, and it is fast, and it makes debugging genuinely
painful. When the trajectory goes wrong, you have to read thousands of lines of
machine-written C with names like `$DER$Pbody$Pv`, plus a `.mat` file of numbers. The
symptom appears in a language you did not write, three translations away from the equation
you got wrong.

This project takes the compiled path, because
[performance](../architecture.md#performance-where-the-time-goes) leaves no choice. It
refuses to inherit that debugging story. The way out has two parts: own the compiler, and
keep a second, slow, honest evaluator over the same IR.

## Two evaluators over one IR

`modelica.ir.evaluate` is an interpreter that walks the tree. It is slow, it allocates, and
it will never be on the hot path. It is also the **reference tier**, and it is always
there.

* It reports an error in *Modelica* terms, and points at the line: "nothing in the
  environment answers 'Rw' at Motor.mo:5:7". It can, because it still holds the IR node and
  the span the parser put on it. The compiled kernel holds only a NaN.

  What it does not yet do is name the *equation* the reference sits in -- "equation 12 of
  BouncingBall" -- because an equation's span is on the equation and the interpreter is
  handed an expression. That is a caller change, not a missing capability.
* It is what the compiled kernel is tested against. Any expression, any model, both
  evaluators, one answer. That is a property test, and not a hope.
* It gives **differential debugging**. When a compiled run disagrees with an interpreted
  run, step both and find the first equation where they diverge. You cannot even ask that
  question of a generated-C pipeline, because there is nothing to compare against.
* It runs when the compiler is absent. `pip install modelica` with no Numba still
  simulates. It is only slow.

So the fast path is an optimisation of a correct thing, and not the only thing.

## A backend is an emitter, not a rewrite

The IR compiles to *source*, so adding a target means writing an emitter.

| Backend | What it emits | What it is for |
| --- | --- | --- |
| interpreter | nothing; it walks the IR | reference, error messages, install with no dependency |
| Numba | Python source, `@njit` | the default fast path |
| C | one translation unit | FMU export, where a shared library is the deliverable |
| SPIR-V and Vulkan | compute shaders | width: ensembles and MOL systems. See [GPU backends](vulkan.md) |
| CuPy and Jax | array expressions | batched parameter sweeps on NVIDIA |

The shared work happens once, above all of them: flattening, matching, BLT, index
reduction, tearing, symbolic differentiation, and common-subexpression elimination. Only
the last step differs. This is the shape of [vLLM][vllm] and [XLA][xla]: one intermediate
representation carries all the domain knowledge, and the hardware backends stay narrow,
replaceable and separately testable.

## Tailored, not adopted

Borrowing the *shape* of XLA is not the same as using XLA. A general tensor compiler is
built for a workload that is not ours: large dense array operations, batch dimensions, and
fusion across a dataflow graph. A DAE right-hand side is a few hundred scalar operations,
with a sparsity pattern we already know exactly, evaluated millions of times, inside a loop
whose control flow an error norm decides.

Writing our own compiler for that shape buys five things that a general one will not give.

* **The integrator and the model compile together.** RADAU5 is a library, and it calls `F`
  through a function pointer it cannot see through. Ours puts the right-hand side inside
  the step, so the optimiser sees one function.
* **Sparsity is structural, not discovered.** The pattern of the Jacobian falls out of the
  incidence matrix that matching already built. There is no colouring heuristic and no
  probing.
* **Specialisation is a choice, not a policy.** Burn the parameters in as literals and fold
  them, or leave them as an argument so that one compiled artefact serves a whole sweep.
  Two compile modes, and the caller picks. vLLM makes the same decision about what to bake
  into a CUDA graph and what to keep as a runtime input.
* **Caching is free.** The IR nodes are frozen, slotted and hashable, so a `FlatModel` has
  a content hash and needs no extra machinery to get one. That hash is the cache key for
  the compiled kernel. It recompiles when the model changed, and not when the mtime of the
  file changed.
* **Fixed-step co-simulation is a static dispatch sequence.** Nothing branches between
  communication points, so the whole schedule can be recorded once and replayed. That is
  CUDA graphs, or `cml_replay` on the Vulkan side. Only a compiler that knows it is
  compiling a simulation can see that optimisation.

None of these five is a general-purpose optimisation. Each one follows from knowing that
the workload is a DAE. That is exactly the knowledge a general compiler has been careful to
throw away.

## What this costs

A hand-written compiler is a maintenance burden, and adopting one would not be. Three
things reduce that cost. The IR is deliberately small: arithmetic, calls and conditionals,
with no arrays of structs and no user-defined types at the expression level. The
interpreter pins down the semantics on its own, apart from any emitter. And each backend is
tested against the interpreter, and not against the other backends. So the surface that has
to be correct is the IR and the interpreter. The emitters are mechanical.

[Benchmarks](benchmarks.md) says where this is measured, and against what.

[vllm]: https://github.com/vllm-project/vllm
[xla]: https://openxla.org/xla
