# GPU backends

!!! warning "Dropped direction"
    This page argues for compiling the model (C, Rust, Numba or GPU code). The project dropped
    that: models run on the precompiled engine, a VM that reads the model as a tape
    ([engine](../engine.md)). The page is kept as a record.

!!! note "Tracked in the register"
    The work this page points at is tracked in [`docs/issues`](https://gitlab.com/cardona-oss/modelica/-/tree/main/docs/issues): the ensemble driver first, then the SPIR-V spike, the method-of-lines path, and JAX as the other vehicle for the same workload. This page is the argument; the register is what is owed.

`compute-ml` is a raw-Vulkan compute library in the next directory. The obvious question is
whether this project can borrow it and run simulations on the GPU. This page records the
answer, because the honest version is more interesting than either "yes" or "no".

!!! info "Short version"
    The architecture says yes. The IR already compiles to source, so a SPIR-V emitter is
    one more backend and not a redesign. The numerics say no, at least for the thing you
    would reach for first. One stiff trajectory of a small system is a serial chain of
    dependencies, and a GPU cannot help with that. `compute-ml` is also float32 only, which
    stiff integration cannot survive. The GPU pays when the *width* is real: ensembles,
    method-of-lines PDE systems, and co-simulation across many FMUs.

## Why the coupling is plausible

The architecture already committed to the decision that makes this cheap. `modelica.sim`
does not walk a tree of closures at every evaluation. It **emits source** for the residual,
the Jacobian and the event indicators, and gives that source to a compiler. Numba is the
first backend. C is the second, because FMU export needs it anyway.

A GPU backend has the same shape: the same `FlatModel`, the same emitter interface, and a
different target language. Emitting GLSL is no harder than emitting C. It is arguably
easier, because the generated kernel is straight-line arithmetic over a state vector, with
no allocation and no control flow except `if`. The decision that bought this was made for
CPU reasons. The GPU option is a free consequence of it.

## Where a GPU actually wins

Time integration is serial by construction. Step *n+1* needs step *n*. Nothing parallelises
*along* a trajectory. So the only question is where the width comes from.

| Shape | Width | Verdict |
| --- | --- | --- |
| One trajectory, small system (`Robertson`, `VanDerPol`, n = 2–10) | none | **The GPU loses badly.** The right-hand side is tens of flops. The launch and the synchronisation cost more than the whole step. |
| One trajectory, large system (method of lines, n = 10⁴–10⁶) | the state vector | **The GPU wins**, if the linear algebra is there. |
| An ensemble: parameter sweep, Monte Carlo, sensitivity, calibration | one thread per trajectory | **The GPU wins hardest.** One kernel, different parameters, no communication. Embarrassingly parallel. |
| SSP co-simulation, many FMUs between communication points | one thread per FMU | **The GPU wins**, and `cml_replay` fits it exactly. |

The ensemble case is the one worth designing for. Parameter identification against
measurements, propagation of uncertainty, and a Monte Carlo over component tolerances are
*the* industrial uses of a Modelica model after the first simulation runs. Each of them is
thousands of independent integrations of one identical right-hand side. That is a GPU
workload in a way that a single Robertson run will never be.

The large-system case is the [PDE](pdes.md) direction. `compute-ml` already ships
`examples/heat_equation.c`. It runs a 256×256 explicit finite-difference stencil for 50,000
steps, entirely on the device. That is a method-of-lines integration with the time loop on
the GPU and no round trip to the host per step, and it is exactly the pattern this project
would need. The precedent exists and it works.

## What blocks it today

Four things. They are listed from hardest to fix to easiest.

### 1. float32 only, which is the real blocker

`compute-ml` never requests `shaderFloat64`. `VkDeviceCreateInfo` passes no
`pEnabledFeatures` and no feature chain, the dtype enum has no FP64 entry, and the host API
takes `float*`. Everything is single precision, from end to end.

Single precision gives about seven decimal digits. A stiff solver needs the Newton
iteration to converge to the local error tolerance, and it needs the error estimate itself
to be signal instead of noise. At `rtol = 1e-8` there is nothing left. `Robertson` alone
spans ten orders of magnitude in concentration, so fp32 cannot represent the trajectory,
and integrating it is out of the question. The reference points in
`examples/models/*.toml` are generated at `rtol = 1e-12`, and fp32 cannot be pointed at
that target.

Explicit integration of a non-stiff problem at a loose tolerance is fine in fp32, which is
why the heat-equation example works. That is the honest scope of the library today.

The fix is only half a fix. A consumer GPU runs FP64 at one thirty-second to one
sixty-fourth of its FP32 throughput. Enabling `shaderFloat64` makes the arithmetic
*correct*. It makes the arithmetic fast only on datacenter parts, where FP64 runs at half
rate. On a laptop iGPU most of the advantage disappears.

### 2. There is no way to plug in a generated kernel

The set of kernels is closed. `glslc` compiles the shaders at build time and embeds them as
C arrays. Adding one means editing several files in the library and recompiling it. There
is no runtime SPIR-V loading, no shaderc, and no public entry point that says "here is my
module, and here are its bindings".

Code generation needs exactly that entry point: one shader per model, produced at
simulation time. This is the deepest mismatch with the way this project works. It is also
the most mechanical to add. Runtime `shaderc` plus a call to register a module is enough,
and the pipeline cache and the push-descriptor machinery already carry the rest.

### 3. There is no linear algebra beyond dense GEMM

There is no LU, no triangular solve, and nothing sparse. An implicit integrator is one
Newton solve per step, which is one factorisation per Jacobian. That is the single most
important kernel, and it is absent. For an explicit method it does not matter at all.

### 4. Submission blocks

One queue, one fence, and about 100 µs of round trip per readback that the host can see.
That is fatal for anything with the host in the loop, and irrelevant when the time loop
lives on the device. `cml_replay` records the dispatches of one step and then replays them,
which is the CUDA-graph idea. It is the feature that makes a device-side time loop
practical, and it is already there.

## The position

Not now. Do not close the door.

* Keep the code-generation backend an interface: `FlatModel` to kernel source, with Numba
  as the first backend. That is the only work the GPU question justifies today, and the CPU
  path wants that work anyway.
* The first GPU-worthy target is **not** the solver. It is the ensemble driver: run one
  compiled model over a matrix of parameters. Design it single-threaded first. The shape is
  already the shape a GPU wants.
* The first plausible coupling to `compute-ml` is the method-of-lines path: explicit
  stepping, fp32 acceptable, and `cml_replay` for the time loop. That is the
  `heat_equation.c` pattern, with the stencil generated from a flat model instead of
  written by hand.
* None of this happens before `flatten` and `sim` work on the CPU. A GPU backend for a
  toolchain that cannot yet simulate `NewtonCooling` is a distraction.
