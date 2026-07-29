# GPU backends

`compute-ml` is a raw-Vulkan compute library sitting in the next directory over. The
obvious question is whether this project can borrow it and run simulations on the GPU. This
page records the answer, because the honest version is more interesting than either "yes"
or "no".

!!! info "Short version"
    Architecturally yes — the IR already compiles to source, so a SPIR-V emitter is one
    more backend rather than a redesign. Numerically no, not for the thing you would first
    reach for: one stiff trajectory of a small system is a serial dependency chain that a
    GPU cannot help with, and `compute-ml` is float32-only, which stiff integration cannot
    survive. The GPU pays when the *width* is real — ensembles, method-of-lines PDE
    systems, co-simulation across many FMUs.

## Why the coupling is even plausible

The architecture already committed to the thing that makes this cheap. `modelica.sim` does
not walk a tree of closures at every RHS evaluation; it **emits source** for the residual,
the Jacobian and the event indicators, and hands that source to a compiler. Numba is the
first backend, C is the second (it is how FMU export has to work anyway).

A GPU backend is then the same shape: same `FlatModel`, same emitter interface, different
target language. Emitting GLSL is not harder than emitting C — it is arguably easier, since
the generated kernel is straight-line arithmetic over a state vector with no allocation and
no control flow beyond `if`. The decision that bought this was made for CPU reasons; the
GPU option is a free consequence of it.

## Where a GPU actually wins

Time integration is serial by construction: step $n+1$ needs step $n$. Nothing parallelises
*along* a trajectory. So the only question is where the width comes from.

| Shape | Width | Verdict |
| --- | --- | --- |
| One trajectory, small system (`Robertson`, `VanDerPol`, n = 2–10) | none | **GPU loses badly.** The RHS is tens of flops. Launch and synchronisation cost more than the whole step. |
| One trajectory, large system (method of lines, n = 10⁴–10⁶) | the state vector | **GPU wins**, if the linear algebra is there. |
| Ensemble — parameter sweep, Monte Carlo, sensitivity, calibration | one thread per trajectory | **GPU wins hardest.** Same kernel, different parameters, no communication. Embarrassingly parallel. |
| SSP co-simulation, many FMUs between communication points | one thread per FMU | **GPU wins**, and `cml_replay` fits it exactly. |

The ensemble case is the one worth designing for. Parameter identification against
measurements, uncertainty propagation, a Monte Carlo over component tolerances — these are
*the* industrial uses of a Modelica model after the first simulation runs, and they are
thousands of independent integrations of an identical RHS. That is a GPU workload in the
way a single Robertson run never will be.

The large-system case is the [PDE](pdes.md) direction. `compute-ml` already ships
`examples/heat_equation.c`: a 256×256 explicit finite-difference stencil run for 50,000
steps, entirely on device. That is a method-of-lines integration with the time loop on the
GPU and no per-step host round-trip, which is precisely the pattern this project would
need. The precedent exists and works.

## What blocks it today

Four things, in descending order of how hard they are to fix.

### 1. float32 only — the real blocker

`compute-ml` never requests `shaderFloat64`. `VkDeviceCreateInfo` passes no
`pEnabledFeatures` and no feature chain; the dtype enum has no FP64 entry; the host API
takes `float*`. Everything is single precision, end to end.

Single precision is ~7 decimal digits. A stiff solver needs the Newton iteration to
converge to the local error tolerance and needs the error estimate itself to be signal
rather than noise; at `rtol = 1e-8` there is nothing left. `Robertson` alone spans ten
orders of magnitude in concentration — fp32 cannot represent the trajectory, never mind
integrate it. The reference points in `examples/models/*.toml` are generated at
`rtol = 1e-12`, which is not a target fp32 can be pointed at.

Explicit integration of a non-stiff problem at loose tolerance is fine in fp32, which is
why the heat-equation example works. That is the honest scope of the current library.

And the fix is only half a fix: consumer GPUs run FP64 at 1/32 to 1/64 of FP32 throughput.
Enabling `shaderFloat64` makes the arithmetic *correct*; it makes it fast only on
datacenter parts where FP64 is 1/2 rate. On a laptop iGPU the advantage largely evaporates.

### 2. No way to plug in a generated kernel

The kernel set is closed. Shaders are compiled at build time by `glslc` and embedded as C
arrays; adding one means editing several library files and recompiling. There is no runtime
SPIR-V loading, no shaderc, no public "here is my module, here are its bindings" entry
point.

Code generation needs exactly that entry point — one shader per model, produced at
simulation time. This is the deepest mismatch with how this project works, but it is also
the most mechanical to add: runtime `shaderc` plus a register-module call, and the existing
pipeline cache and push-descriptor machinery carries the rest.

### 3. No linear algebra beyond dense GEMM

There is no LU, no triangular solve, no sparse anything. An implicit integrator is a Newton
solve per step, which is a factorisation per Jacobian — the single most important kernel and
it is absent. For explicit methods this does not matter at all.

### 4. Blocking submit

Single queue, single fence, ~100 µs of CPU-visible round-trip per host readback. Fatal for
anything host-in-the-loop, irrelevant when the time loop lives on the device. `cml_replay`
— record a step's dispatches once, then replay, the CUDA-graph idea — is the feature that
makes device-side time loops practical, and it is already there.

## The position

Not now, but do not close the door.

* Keep the code-generation backend an interface — `FlatModel` → kernel source — with Numba
  as backend one. That is the only work the GPU question justifies today, and it is work
  the CPU path wants regardless.
* The first GPU-worthy target is **not** the solver. It is the ensemble driver: run the same
  compiled model over a parameter matrix. Design it single-threaded first; the shape is
  already the shape a GPU wants.
* The first plausible `compute-ml` coupling is the PDE / method-of-lines path, explicit
  stepping, fp32 acceptable, `cml_replay` for the time loop — the `heat_equation.c` pattern
  with the stencil generated from a flat model instead of hand-written.
* Nothing here happens before `flatten` and `sim` work on the CPU. A GPU backend for a
  toolchain that cannot yet simulate `NewtonCooling` is a distraction.
