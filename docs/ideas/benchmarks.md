# Benchmarks: what "as good as the Fortran codes" has to mean

A claim about solver performance is worthless without a fixed set of problems and a fixed
way to measure. This page records which problems this project targets, and why the obvious
metric is the wrong one.

## FLOPS is the wrong number

LINPACK has not gone away. HPL still underpins the [TOP500][top500], and the list is
refreshed twice a year. HPCG was added beside it for one reason: HPL had stopped
representing real workloads.

A single throughput figure says nothing useful about an integrator. On Robertson, an
explicit method needs 10⁷ to 10⁸ steps where a good stiff method needs a few hundred. A
solver that is 10 times slower per step, and takes 10⁶ fewer steps, wins by five orders of
magnitude. No FLOPS count shows that.

The field uses the **work-precision diagram** instead. It plots the error the solver
*achieved* against the CPU time it took, swept over a range of tolerances, with one curve
per method. Lower and to the left wins. It is the honest measurement, because it prices in
the two things a scalar timing hides: a loose tolerance is not free accuracy, and a method
that fails to converge has not "run fast".

## The problem sets

**[Test Set for IVP Solvers][bari]**, by Mazzia and Magherini, University of Bari and
INdAM, release 2.4 of February 2008. This is the canonical collection: Robertson, HIRES,
Ring Modulator (stiff, 15 dimensions), E5, Beam (80 dimensions), Medical Akzo Nobel, and
EMEP. Every problem ships with a high-precision reference solution, which is the only thing
that makes an *error* axis possible. It was ported to R as `deTestSet` and to Julia as
`IVPTestSuite.jl`.

**[Hairer's testset and drivers][hairer]** is the reference harness for RADAU5, DOPRI5,
DOP853, RODAS and SEULEX, written by the authors of those codes. If the claim is "faster
than RADAU5", this harness has to say so.

**[SciMLBenchmarks.jl][scimlbench]**, published at [benchmarks.sciml.ai][sciml], is the
live one. It is actively maintained. It crosses languages: Julia, Python including Jax,
MATLAB, R, and the Fortran codes through wrappers. It carries the IVP test set problems
above. It is also the direct precedent for the position taken here, because it is where "a
high-level language can beat the Fortran codes" was argued and defended in public, with
numbers.

**[casella/ScalableTestSuite][scalable]** is the Modelica-specific one, and it tests the
*compiler* and not only the integrator. Its models take a size parameter, so both compile
time and simulation time can be plotted against *n*. The paper is ["Towards a benchmark
suite for high-performance Modelica compilers"][eoolt], at EOOLT. OpenModelica publishes
[per-branch results][omcresults] against it, which gives a public baseline to measure
against instead of a number from a paper.

## Why beating them is a design goal and not bravado

The interesting claim is not that Python can be fast. It is that a compiler which knows it
is compiling a *differential-algebraic system* can do three things that a precompiled
general integrator cannot.

* **It puts the right-hand side inside the integrator loop.** RADAU5 is a library. It calls
  the user's `F` through a function pointer, across a call boundary it cannot see through,
  and the parameters live in memory it must load. A JIT compiles the model and the
  integrator together. The right-hand side is inlined, the parameters are constants, and
  the folding happens once at compile time instead of once per stage per step.
* **It differentiates the model.** The equations are held symbolically, so an analytic and
  sparse Jacobian is free. A black-box integrator must use finite differences. That costs
  *n* extra evaluations per Jacobian, and it injects truncation error into the Newton
  iteration exactly where that hurts convergence.
* **It makes the system smaller before the solver sees it.** Tearing and BLT reduce a
  simultaneous block of *n* equations to a much smaller torn system inside a sequence of
  assignments. That is a symbolic transformation on the model. A bare ODE library receives
  the system already assembled and cannot do it.

A library that receives a black-box residual has none of the three. A compiler that owns
the model has all three. That is the whole argument for writing our own instead of binding
to one that exists. [The backends note](backends.md) carries the rest of it.

The [Rumoca][rumoca] work — Modelica as a universal algebraic front end, through a
Rust-native compiler — is current evidence that this space is live, and that the
front-end-plus-IR framing is not idiosyncratic.

## What gets measured here, and when

Stage 5 of the build order is where this becomes real: code generation from the
`FlatModel`, with work-precision diagrams against the Fortran-backed integrators of scipy.
The staging has three steps.

1. **Robertson and van der Pol** first. They are small, stiff and universally understood,
   and both are in the Bari set with reference solutions. They catch bugs in order control
   and step size control.
2. **HIRES, E5 and the Ring Modulator** next. They have enough dimension and enough
   stiffness that the Jacobian strategy and the cost of the linear solve start to matter.
3. **ScalableTestSuite** last. Compile time enters the picture there, and the symbolic
   stages — matching, BLT, tearing, index reduction — are what is under test, rather than
   the integrator.

Every number published from this project cites four things: the problem, the tolerance
sweep, the reference solution used for the error axis, and the machine. A timing without
them is not a benchmark. It is an anecdote.

[bari]: https://archimede.dm.uniba.it/~testset/
[eoolt]: https://dl.acm.org/doi/10.1145/3365984.3365988
[hairer]: https://www.unige.ch/~hairer/testset/testset.html
[omcresults]: https://libraries.openmodelica.org/branches/master/ScalableTestSuite/ScalableTestSuite.html
[rumoca]: https://arxiv.org/pdf/2606.14998
[scalable]: https://github.com/casella/ScalableTestSuite
[sciml]: https://benchmarks.sciml.ai/
[scimlbench]: https://github.com/SciML/SciMLBenchmarks.jl
[top500]: https://www.top500.org/
