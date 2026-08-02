# PDEs

Every few years somebody proposes partial differential equations for Modelica. It has
happened since 2000. Nothing has landed.

This page records why. The failure is instructive and not accidental. It also settles what
this project does when the question comes up again.

!!! info "Short version"
    The whole value of Modelica is that a tool can flatten *any* declarative DAE and give
    it to DASSL or IDA. PDEs have no universal solver. So a `pder` operator must either lie
    about the numerics, or drag meshing, discretization schemes and stability control into
    the language specification. The Modelica Association standardized exactly one narrow
    PDE, `spatialDistribution` for 1D transport, and refused the general case.

## The graveyard

| Year | Attempt | Outcome |
| --- | --- | --- |
| 2000–01 | Saldamli & Fritzson, *Object-oriented modeling with PDEs*; *Domains and PDEs in Modelica* | papers only |
| 2003 | Casella & Schiavo (FEM heat exchangers); Sheshadri & Fritzson (symbolic PDE generator) | papers only |
| 2004 | PDEModelica sketch in Fritzson's book | no tool ever implemented it |
| 2005–06 | **PDEModelica** — Saldamli PhD (LiU, VISP/Vinnova): `domain`, `boundary`, `composite`, an external mesher (bamg) and an FEM library (rheolef) | "currently not supported by any tool" |
| 2007 | Dshabarow MSc | papers only |
| 2008–09 | Li & Zheng, *Description of / Solving PDE Models in Modelica* (IEEE) | a prototype external translator |
| **Mar 2010** | **`spatialDistribution()` enters Modelica 3.2** | the only survivor |
| 2010 | PDElib | dead |
| 2013–14 | HiFlow3 C++ solver, imported through FMI | PDEs moved *outside* the language |
| Mar 2016 | Specification ticket closed as expired (issue #254) | the official end of the general case |
| 2015–18 | **PDEModelica1** — Šilar (Charles University) in OpenModelica: 1D evolutionary PDEs, method of lines | still experimental. As of OM 1.27 and 1.28 it needs `--grammar=PDEModelica` **and the deprecated old front end** (`-d=-newInst`) |
| Mar 2018 | Association proposal *PDE for Systems* (Heckmann, DLR): reduced-order export and xFMI | 9 commits, and it never became a project |
| 2018–26 | Association effort goes to FMI 3.0, SSP, DCP, eFMI and Base Modelica (MCP-0031) | no PDE change proposal exists |

The epitaph is on the specification tracker. Hans Olsson maintains the specification and
works on Dymola. He wrote this in March 2016, on issue #254, *"Solving PDEs"*:

> There are currently no concrete plans for introducing PDEs in Modelica. After 5 years I
> would close this as expired.

It was closed as expired. There has never been a Modelica Change Proposal for PDEs.

## Why it keeps failing

### 1. There is no `dassl` for PDEs

The promise of Modelica is uniform. You declare equations, the tool flattens them and
reduces the index, and a general integrator solves *whatever came out*. PDEs have no
equivalent. The choice of scheme is a property of the problem: hyperbolic against parabolic
against elliptic, upwinding, flux limiters, explicit against implicit. Šilar said it
plainly in his workshop slides of 2015: "no numerical library able to solve vast majority
of PDEs (as is dassl or IDA in ODE)", and "Modelica tools solve enormous amount of ODE
(DAE) problems (would say almost any) — probably not possible in PDE."

So a PDE construct in the language has two options, and both are bad. It can pick a scheme
silently. Or it can expose meshing, discretization, limiters and stability control as
syntax, which means standardizing a finite-element framework inside a language
specification.

### 2. Silently wrong is worse than unsupported

Under the method of lines, the time integrator controls the error *in time* and knows
nothing about the error *in space*. You can converge in time and still be far from the
solution. Šilar's own convergence tables move between observed orders of 0.49 and 1.6, and
his BTCS scheme rings badly on a discontinuous initial condition. Stability needs a CFL
bound built from the largest eigenvalue of the flux Jacobian. That is a fact at the level
of the solver, and the declarative layer cannot infer it from the text of the model.

Modelica is sold into safety-relevant industrial modelling. A construct that quietly
returns plausible garbage is disqualifying there.

### 3. `connect()` has no dimensions

Modelica connection semantics are Kirchhoff's. Potentials are equal and flows sum to zero,
at a *point*. To couple two distributed subdomains you must match fluxes across a shared
boundary, and the two meshes may not conform. Generalizing connections to interfaces with
more than one dimension was the core of Saldamli's research, and nobody solved it.

### 4. The compilation model assumes a fixed number of equations

Flattening, matching, BLT sorting, tearing, index reduction and the balanced-model rules of
Modelica 3.0 all need known sizes and known scalar counts. A field variable has no size
until somebody discretizes it, and the modeller is not the one who chooses the
discretization.

There is a worse problem. Whether a PDE is well posed — do the boundary conditions match
the type of the equation? — is not a structural property, so none of that machinery can
check it. The one guarantee Modelica does offer, a structurally sound system, does not
extend.

### 5. Geometry has no home in the language

Saldamli's design needed domains, boundary curves, composite regions, an external mesh
generator and an external FEM library. That is a CAD and meshing pipeline, and not a
language extension. Modelica has no concept of geometry, and no obvious place to put one.

### 6. Vendor economics

A PDE extension means touching the parser, the front end, the back end and the code
generation templates, and then integrating a PDE solver into the runtime. Meanwhile the
distributed problems of paying customers are overwhelmingly 1D: pipes, heat exchangers and
batteries. Hand-discretized finite-volume libraries already serve them — `Modelica.Fluid`,
ThermoFluid, TIL, Buildings and ThermofluidStream. Anybody who genuinely needs 3D already
owns COMSOL, Fluent or Abaqus, and is not going to switch. The cost is large and the
marginal value is small.

### 7. Chicken and egg

Standardization needs a working implementation and vendor demand. Non-standard syntax means
that no portable library can depend on it. So a feature implemented in one tool has no path
to adoption, and the demand never appears. PDEModelica1 works, in one tool, behind two
flags, and no library uses it.

### 8. Every attempt was one graduate student

Saldamli, Dshabarow, Li and Zheng, and Šilar. Four groups, four dissertations, four
abandoned branches. Nobody was ever staffed to maintain a compiler fork after graduation.
PDEModelica1 is stranded on the *deprecated* front end of OpenModelica a decade later,
which is the tidiest proof available.

### 9. The escape hatches were good enough

Each workaround released the pressure that would have forced a change to the language.

* A hand-written method of lines, with arrays and `for` equations. It is plain Modelica, it
  works today, and it is what industry actually does.
* `spatialDistribution()`, for transport along pipes and transmission lines.
* FMI import of an external PDE solver: HiFlow3, and later co-simulation with OpenFOAM or
  FEniCS.
* A reduced-order model exported from a finite-element tool and consumed as an FMU.

## What got standardized instead

`spatialDistribution()`, in Modelica 3.2, in March 2010. It approximates the transport of a
quantity along a 1D path at variable speed. It serves pipelines, gas networks,
district-heating networks, sprinklers, conveyor belts and electrical transmission lines.
One class of equation, clear industrial demand, a known-good algorithm, and an
implementation as a transport buffer inside the tool rather than as a general
discretization.

That is the revealed preference of the Association, and it is coherent. Standardize the
narrow case where a correct method exists. Keep the general case outside the language. Then
standardize the *interface* to whatever solves it. FMI, SSP, DCP and eFMI are all interface
standards. Note also that every recent specification issue near this subject — #1066,
#1729, #3111, #3640 — is about the semantics of `spatialDistribution`, and not about
fields.

## The position of this project

Do not design toward native PDE support. Three options, in order of preference.

1. **Do nothing.** A method of lines written by hand, with arrays and `for` equations, is
   already expressible in the language that `modelica.lang` parses. It needs no feature,
   and it keeps the discretization visible in the model, where the modeller can see it and
   change it.
2. **Add `spatialDistribution()` as a builtin operator** in the expression IR, with its
   transport-buffer semantics in `modelica.sim`. This is standard Modelica, so it belongs
   on the roadmap on its own merits. It is a conformance item, and not a PDE feature.
3. **If field variables are ever wanted, write a desugaring layer and never a core
   contract.** `field`, `domain` and `pder` would live *above* the standard AST. They would
   lower to arrays and difference equations before `modelica.ir` ever saw them, with the
   scheme and the grid named explicitly in the source. The core stays compliant with the
   specification, and the choice of discretization stays with the modeller. Every failed
   attempt got that exact point wrong, by hiding the choice in the compiler.

The general lesson is worth keeping outside PDEs as well. The guarantees of this pipeline
come from one property: every stage is a value you can inspect. A construct whose meaning
depends on a numerical method that the tool picks silently cannot be checked at any stage.
So it does not belong in the core.

## Sources

* [Modelica specification issue #254, *Solving PDEs*](https://github.com/modelica/ModelicaSpecification/issues/254) — closed as expired, March 2016
* [Šilar, *Partial Differential Equations in Modelica*, OpenModelica Workshop 2015](https://openmodelica.org/images/docs/openmodelica2015/OpenModelica2015-talk05-PDEInModelica_silar.pdf)
* [Šilar et al., *PDEModelica1*, Int. J. Modelling and Simulation 38(2), 2017](https://www.tandfonline.com/doi/full/10.1080/02286203.2017.1404417)
* [PDEModelica1 in the OpenModelica User's Guide](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/pdemodelica.html) — current status and flags
* [Saldamli, *PDEModelica — A High-Level Language for Modeling with Partial Differential Equations*, PhD thesis, LiU 2006](https://liu.diva-portal.org/smash/record.jsf?pid=diva2:21488)
* [Saldamli & Fritzson, Modelica Workshop 2000](https://modelica.org/events/workshop2000/proceedings/Saldamli.pdf)
* [LiU PELAB, *Partial Differential Equations in Modelica*](https://www.ida.liu.se/labs/pelab/modelica/pde.shtml) — listed under previous projects
* [`modelica/pde-for-systems`](https://github.com/modelica/pde-for-systems) — the 2018 proposal
* [`Foadsf/ModelicaPDE`](https://github.com/Foadsf/ModelicaPDE) — a community survey of every attempt
* [Modelica Language Specification §3, `spatialDistribution`](https://specification.modelica.org/master/operators-and-expressions.html)
