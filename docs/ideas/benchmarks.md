# Benchmarks: what "as good as the Fortran codes" has to mean

A claim about solver performance is worthless without a fixed set of problems and a fixed
way of measuring. This page records which ones this project targets, and why the obvious
metric is the wrong one.

## FLOPS is the wrong number

LINPACK has not gone away — HPL still underpins the [TOP500][top500], refreshed twice a
year, and HPCG was added alongside it precisely because HPL had stopped being
representative of real workloads. But a single throughput figure says nothing useful about
an integrator. On Robertson, an explicit method needs on the order of 10⁷–10⁸ steps where a
good stiff method needs a few hundred. A solver that is 10× slower per step and takes 10⁶
fewer steps wins by five orders of magnitude, and no FLOPS count shows that.

The metric the field actually uses is the **work-precision diagram**: the error the solver
*achieved* against the CPU time it took, swept across a range of tolerances, one curve per
method. Lower and to the left wins. It is the honest measurement because it prices in the
two things a scalar timing hides — that a loose tolerance is not free accuracy, and that a
method which fails to converge has not "run fast".

## The problem sets

**[Test Set for IVP Solvers][bari]** — Mazzia & Magherini, University of Bari and INdAM,
release 2.4 (February 2008). The canonical collection: Robertson, HIRES, Ring Modulator
(15-dimensional stiff), E5, Beam (80-dimensional), Medical Akzo Nobel, EMEP. Each problem
ships with a high-precision reference solution, which is what makes an *error* axis
possible at all. Ported to R as `deTestSet` and to Julia as `IVPTestSuite.jl`.

**[Hairer's testset and drivers][hairer]** — the reference harness for RADAU5, DOPRI5,
DOP853, RODAS and SEULEX, from the authors of those codes. If the claim is "faster than
RADAU5", this is the harness that has to say so.

**[SciMLBenchmarks.jl][scimlbench]**, results published at [benchmarks.sciml.ai][sciml] —
the live one. Actively maintained, cross-language (Julia, Python including Jax, MATLAB, R,
and the Fortran codes through wrappers), and it carries the IVP-test-set problems above. It
is also the direct precedent for the position taken here: it is where "a high-level
language can beat the Fortran codes" was argued and defended in public with numbers.

**[casella/ScalableTestSuite][scalable]** — the Modelica-specific one, and the one that
tests the *compiler* rather than only the integrator. Models are parameterised by size, so
both compile time and simulation time can be plotted against *n*. Paper: ["Towards a
benchmark suite for high-performance Modelica compilers"][eoolt], EOOLT. OpenModelica
publishes [per-branch results][omcresults] against it, which gives a public baseline to
measure against rather than a number from a paper.

## Why beating them is a design goal and not bravado

The interesting claim is not "Python can be fast". It is that a compiler which knows it is
compiling a *differential-algebraic system* can do three things a precompiled general
integrator structurally cannot:

* **Inline the right-hand side into the integrator loop.** RADAU5 is a library: it calls
  the user's `F` through a function pointer, across a call boundary it cannot see through,
  with the parameters living in memory it must load. A JIT compiles the model and the
  integrator together, so the RHS is inlined, the parameters are constants, and the
  constant-folding happens once at compile time instead of once per stage per step.
* **Differentiate the model.** The equations are held symbolically, so an analytic — and
  sparse — Jacobian is free. A black-box integrator has to finite-difference it, which
  costs *n* extra RHS evaluations per Jacobian and injects truncation error into the Newton
  iteration exactly where it hurts convergence.
* **Shrink the system before the solver sees it.** Tearing and BLT reduce a simultaneous
  block of *n* equations to a much smaller torn system with an assignment sequence around
  it. That is a symbolic transformation on the model; a bare ODE library is handed the
  system already assembled and has no way to perform it.

None of the three is available to a library that receives a black-box residual. All three
are available to a compiler that owns the model. That is the whole argument for writing our
own rather than binding to an existing one — see [the backends note](backends.md).

The [Rumoca][rumoca] work (Modelica as a universal algebraic frontend, via a Rust-native
compiler) is contemporary evidence that this space is live and that the frontend-plus-IR
framing is not idiosyncratic.

## What gets measured here, and when

Build-order stage 5 is where this becomes real: code generation from the `FlatModel`, with
work-precision diagrams against scipy's Fortran-backed integrators. The staging:

1. **Robertson and van der Pol** first — small, stiff, universally understood, and both are
   in the Bari set with reference solutions. They catch order/step-size control bugs.
2. **HIRES, E5, Ring Modulator** next — enough dimension and enough stiffness that Jacobian
   strategy and linear-solve cost start to matter.
3. **ScalableTestSuite** last — where compile time enters the picture, and where the
   symbolic stages (matching, BLT, tearing, index reduction) are what is actually under
   test rather than the integrator.

Every number published from this project cites the problem, the tolerance sweep, the
reference solution used for the error axis, and the machine. A timing without those is not
a benchmark, it is an anecdote.

[bari]: https://archimede.dm.uniba.it/~testset/
[eoolt]: https://dl.acm.org/doi/10.1145/3365984.3365988
[hairer]: https://www.unige.ch/~hairer/testset/testset.html
[omcresults]: https://libraries.openmodelica.org/branches/master/ScalableTestSuite/ScalableTestSuite.html
[rumoca]: https://arxiv.org/pdf/2606.14998
[scalable]: https://github.com/casella/ScalableTestSuite
[sciml]: https://benchmarks.sciml.ai/
[scimlbench]: https://github.com/SciML/SciMLBenchmarks.jl
[top500]: https://www.top500.org/
