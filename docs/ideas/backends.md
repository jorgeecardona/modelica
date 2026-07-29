# One IR, several backends — and a debugger that still works

OpenModelica's pipeline is `.mo` → generated C → a C compiler → an executable → run it.
That works, and it is fast, and it makes debugging genuinely painful: when the trajectory
goes wrong, what you have to look at is thousands of lines of machine-written C with names
like `$DER$Pbody$Pv` and a `.mat` file of numbers. The symptom appears in a language you
did not write, three translations away from the equation you got wrong.

This project takes the compiled path — [performance](../architecture.md#performance-where-the-time-actually-goes)
leaves no choice — but refuses to inherit that debugging story. The way out is to make the
compiler ours, and to keep a second, slow, honest evaluator over the same IR.

## Two evaluators over one IR

`modelica.ir.evaluate` is a tree-walking interpreter. It is slow, it allocates, and it will
never be on the hot path. It is also the **reference tier**, and it is always available:

* It reports errors in *Modelica* terms — `division by zero in equation 12 of BouncingBall:
  ball.v / ball.m` — because it still has the IR node and the source span attached to it,
  where the compiled kernel has only a NaN.
* It is what the compiled kernel is tested against. Any expression, any model, both
  evaluators, same answer. That is a property test, not a hope.
* It gives **differential debugging**: when a compiled run disagrees with the interpreted
  run, step both and find the first equation where they diverge. That question cannot even
  be *asked* of a generated-C pipeline, because there is nothing to compare against.
* It runs when the compiler is not installed. `pip install modelica` with no Numba still
  simulates; it is just slow.

So the fast path is an optimisation over a correct thing, rather than the only thing.

## Backends are emitters, not rewrites

Because the IR compiles to *source*, adding a target is writing an emitter:

| Backend | What it emits | For |
| --- | --- | --- |
| interpreter | nothing — walks the IR | reference, error messages, no-dependency install |
| Numba | Python source, `@njit` | the default fast path |
| C | a translation unit | FMU export, where a shared library is the deliverable |
| SPIR-V / Vulkan | compute shaders | width — ensembles, MOL systems (see [GPU backends](vulkan.md)) |
| CuPy / Jax | array expressions | batched parameter sweeps on NVIDIA |

The shared work — flattening, matching, BLT, index reduction, tearing, symbolic
differentiation, common-subexpression elimination — happens once, above all of them. Only
the last step differs. This is the [vLLM][vllm]/[XLA][xla] shape: one intermediate
representation that carries all the domain knowledge, and hardware backends that are
narrow, replaceable, and individually testable.

## Tailored, not adopted

The point of borrowing the *shape* from XLA is not to use XLA. A general-purpose tensor
compiler is built for a workload that is not ours: large dense array ops, batch dimensions,
fusion across a dataflow graph. A DAE right-hand side is a few hundred scalar operations
with a sparsity pattern we already know exactly, evaluated millions of times, inside a loop
whose control flow is decided by an error norm.

Writing our own compiler for that shape buys things a general one will not give:

* **The integrator and the model compile together.** RADAU5 is a library: it calls `F`
  through a function pointer it cannot see through. Ours inlines the RHS into the step, so
  the optimiser sees one function.
* **Sparsity is structural, not discovered.** The Jacobian pattern falls out of the
  incidence matrix that matching already built. No colouring heuristics, no probing.
* **Specialisation is a choice, not a policy.** Parameters can be *burned in* as literals
  and constant-folded, or left as a kernel *argument* so one compiled artefact serves a
  whole sweep. Two compile modes, picked by the caller — the same decision vLLM makes about
  what to bake into a CUDA graph and what to keep as a runtime input.
* **Caching is free.** The IR nodes are frozen, slotted and hashable, so a `FlatModel` has
  a content hash without any extra machinery. That hash is the cache key for the compiled
  kernel: recompile only when the model actually changed, not when the file's mtime did.
* **Fixed-step co-simulation is a static dispatch sequence.** No branching between
  communication points means the whole schedule can be recorded once and replayed — CUDA
  graphs, or `cml_replay` on the Vulkan side. That optimisation is only visible to a
  compiler that knows it is compiling a simulation.

None of these are general-purpose optimisations. They are all consequences of knowing the
workload is a DAE, which is exactly the knowledge a general compiler has been careful to
throw away.

## What this costs

Honesty about the trade: a hand-written compiler is a maintenance burden that adopting one
would not be. The mitigations are that the IR is deliberately small (arithmetic, calls,
conditionals — no arrays-of-structs, no user-defined types at the expression level), that
the interpreter pins down the semantics independently of any emitter, and that each backend
is tested against the interpreter rather than against the others. The surface that has to
be correct is the IR and the interpreter; the emitters are mechanical.

Where this is measured, and against what, is in [benchmarks](benchmarks.md).

[vllm]: https://github.com/vllm-project/vllm
[xla]: https://openxla.org/xla
