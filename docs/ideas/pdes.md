# PDEs

Every few years someone proposes partial differential equations for Modelica. It has been
happening since 2000. Nothing has landed. This page records why, because the failure is
instructive rather than accidental — and because it settles what this project should do
when the question comes up again.

!!! info "Short version"
    Modelica's whole value is that a tool can flatten *any* declarative DAE and hand it to
    DASSL/IDA. PDEs have no universal solver, so a `pder` operator either lies about the
    numerics or drags meshing, discretization schemes and stability control into the
    language spec. The Modelica Association standardized exactly one narrow PDE
    (`spatialDistribution`, 1D transport) and declined the general case.

## The graveyard

| Year | Attempt | Outcome |
| --- | --- | --- |
| 2000–01 | Saldamli & Fritzson, *Object-oriented modeling with PDEs*; *Domains and PDEs in Modelica* | papers only |
| 2003 | Casella & Schiavo (FEM heat exchangers); Sheshadri & Fritzson (symbolic PDE generator) | papers only |
| 2004 | PDEModelica sketch in Fritzson's book | never implemented in a tool |
| 2005–06 | **PDEModelica** — Saldamli PhD (LiU, VISP/Vinnova): `domain`, `boundary`, `composite`, external mesher (bamg) + FEM library (rheolef) | "currently not supported by any tool" |
| 2007 | Dshabarow MSc | papers only |
| 2008–09 | Li & Zheng, *Description of / Solving PDE Models in Modelica* (IEEE) | external translator prototype |
| **Mar 2010** | **`spatialDistribution()` enters Modelica 3.2** | the only survivor |
| 2010 | PDElib | dead |
| 2013–14 | HiFlow3 C++ solver via FMI import | PDEs moved *outside* the language |
| Mar 2016 | Spec ticket closed as expired (issue #254) | official end of the general case |
| 2015–18 | **PDEModelica1** — Šilar (Charles University) in OpenModelica: 1D evolutionary PDEs, method of lines | still experimental; needs `--grammar=PDEModelica` **and the deprecated old frontend** (`-d=-newInst`) as of OM 1.27/1.28 |
| Mar 2018 | Association proposal *PDE for Systems* (Heckmann, DLR): reduced-order export + xFMI | 9 commits, never became a project |
| 2018–26 | Association effort goes to FMI 3.0, SSP, DCP, eFMI, Base Modelica (MCP-0031) | no PDE change proposal exists |

The epitaph is on the specification tracker. Hans Olsson (Dymola, one of the main spec
maintainers), March 2016, on issue #254 *"Solving PDEs"*:

> There are currently no concrete plans for introducing PDEs in Modelica. After 5 years I
> would close this as expired.

It was closed as expired. There has never been a Modelica Change Proposal for PDEs.

## Why it keeps failing

### 1. There is no `dassl` for PDEs

Modelica's promise is uniform: declare equations, the tool flattens, reduces the index,
and a general-purpose integrator solves *whatever came out*. PDEs have no equivalent.
Scheme choice is a property of the problem — hyperbolic vs. parabolic vs. elliptic,
upwinding, flux limiters, explicit vs. implicit. Šilar's own 2015 workshop slides state it
plainly: *"no numerical library able to solve vast majority of PDEs (as is dassl or IDA in
ODE)"*, and *"Modelica tools solve enormous amount of ODE (DAE) problems (would say almost
any) — probably not possible in PDE."*

So a language-level PDE construct has two options, both bad: silently pick a scheme, or
expose meshing, discretization, limiters and stability control as language syntax — which
means standardizing a finite-element framework inside a language specification.

### 2. Silently wrong is worse than unsupported

Under the method of lines the time integrator controls *temporal* error and knows nothing
about spatial error. You can converge in time and still be far from the solution. Šilar's
own convergence tables wobble between observed orders 0.49 and 1.6, and his BTCS scheme
rings badly on a discontinuous initial condition. Stability needs a CFL bound built from
the maximum eigenvalue of the flux Jacobian — a solver-level fact the declarative layer
cannot infer from the model text.

For a language sold into safety-relevant industrial modeling, a construct that quietly
returns plausible garbage is disqualifying.

### 3. `connect()` is zero-dimensional

Modelica connection semantics are Kirchhoff: potentials equal, flows sum to zero, at a
*point*. Coupling two distributed subdomains means matching fluxes across a shared
boundary, potentially between non-conforming meshes. Generalizing connections to
multi-dimensional interfaces was the core of Saldamli's research and was never solved.

### 4. The compilation model assumes a static, finite equation count

Flattening, matching, BLT sorting, tearing, index reduction and the Modelica 3.0
balanced-model rules all need known sizes and scalar counts. A field variable has no size
until it is discretized, and the modeler is not the one choosing the discretization.
Worse, PDE well-posedness (do the boundary conditions match the equation type?) is not a
structural property that any of that machinery can check — so the one guarantee Modelica
does offer, a structurally sound system, does not extend.

### 5. Geometry has no home in the language

Saldamli's design needed domains, boundary curves, composite regions, an external mesh
generator and an external FEM library. That is a CAD and meshing pipeline, not a language
extension. Modelica has no concept of geometry and no obvious place to put one.

### 6. Vendor economics

Implementing a PDE extension means touching parser, front-end, back-end, code-generation
templates *and* integrating a PDE solver into the runtime. Meanwhile the paying customers'
distributed problems are overwhelmingly 1D — pipes, heat exchangers, batteries — and
already served by hand-discretized finite-volume libraries: `Modelica.Fluid`,
ThermoFluid, TIL, Buildings, ThermofluidStream. Anyone who genuinely needs 3D already owns
COMSOL, Fluent or Abaqus and is not switching. Large cost, small marginal value.

### 7. Chicken and egg

Standardization needs a working implementation plus vendor demand. Non-standard syntax
means no portable library can depend on it, so a feature implemented in one tool has no
adoption path, so demand never materializes. PDEModelica1 works, in one tool, behind two
flags, and no library uses it.

### 8. Every attempt was one graduate student

Saldamli, Dshabarow, Li & Zheng, Šilar — four groups, four dissertations, four abandoned
branches. Nobody was ever staffed to maintain a compiler fork past graduation.
PDEModelica1 being stranded on OpenModelica's *deprecated* frontend a decade later is the
tidiest proof available.

### 9. The escape hatches were good enough

Each workaround bled off the pressure that would have forced a language change:

* hand-written method of lines with arrays and `for`-equations — plain Modelica, works
  today, and is what industry actually does;
* `spatialDistribution()` for transport along pipes and transmission lines;
* FMI import of an external PDE solver (HiFlow3, and later co-simulation with
  OpenFOAM/FEniCS);
* reduced-order models exported from a finite-element tool and consumed as an FMU.

## What got standardized instead

`spatialDistribution()`, in Modelica 3.2 (March 2010). It approximates variable-speed
transport of a quantity along a 1D path — pipelines, gas and district-heating networks,
sprinklers, conveyor belts, electrical transmission lines. One equation class, clear
industrial demand, a known-good algorithm, implemented as a tool-internal transport
buffer rather than as a general discretization.

That is the Association's revealed preference, and it is a coherent one: standardize the
narrow case where a correct method exists, keep the general case outside the language, and
standardize the *interface* to whatever solves it. FMI, SSP, DCP and eFMI are all
interface standards. Note also that every recent PDE-adjacent specification issue (#1066,
#1729, #3111, #3640) is about `spatialDistribution` semantics — not about fields.

## Position for this project

Do not design toward native PDE support. In order of preference:

1. **Nothing.** Method of lines written by hand with arrays and `for`-equations is already
   expressible in the language `modelica.lang` parses. It needs no feature, and it keeps
   the discretization visible in the model where the modeler can see and change it.
2. **`spatialDistribution()` as a builtin operator** in the expression IR, with its
   transport-buffer semantics implemented in `modelica.sim`. This is standard Modelica,
   so it belongs on the roadmap on its own merits — it is a conformance item, not a PDE
   feature.
3. **If field variables are ever wanted: a desugaring layer, never a core contract.**
   `field` / `domain` / `pder` would live *above* the standard AST and lower to arrays and
   difference equations before `modelica.ir` ever sees them, with the scheme and grid
   named explicitly in the source. The core stays specification-compliant and the
   discretization choice stays the modeler's — which is precisely what every failed
   attempt got wrong by hiding it in the compiler.

The general lesson is worth keeping even outside PDEs: this pipeline's guarantees come
from every stage being a value you can inspect. A construct whose meaning depends on a
numerical method the tool picks silently cannot be checked at any stage, and so it does
not belong in the core.

## Sources

* [Modelica specification issue #254, *Solving PDEs*](https://github.com/modelica/ModelicaSpecification/issues/254) — closed as expired, March 2016
* [Šilar, *Partial Differential Equations in Modelica*, OpenModelica Workshop 2015](https://openmodelica.org/images/docs/openmodelica2015/OpenModelica2015-talk05-PDEInModelica_silar.pdf)
* [Šilar et al., *PDEModelica1*, Int. J. Modelling and Simulation 38(2), 2017](https://www.tandfonline.com/doi/full/10.1080/02286203.2017.1404417)
* [PDEModelica1 in the OpenModelica User's Guide](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/pdemodelica.html) — current status and flags
* [Saldamli, *PDEModelica — A High-Level Language for Modeling with Partial Differential Equations*, PhD thesis, LiU 2006](https://liu.diva-portal.org/smash/record.jsf?pid=diva2:21488)
* [Saldamli & Fritzson, Modelica Workshop 2000](https://modelica.org/events/workshop2000/proceedings/Saldamli.pdf)
* [LiU PELAB, *Partial Differential Equations in Modelica*](https://www.ida.liu.se/labs/pelab/modelica/pde.shtml) — listed under previous projects
* [`modelica/pde-for-systems`](https://github.com/modelica/pde-for-systems) — the 2018 proposal
* [`Foadsf/ModelicaPDE`](https://github.com/Foadsf/ModelicaPDE) — community survey of every attempt
* [Modelica Language Specification §3, `spatialDistribution`](https://specification.modelica.org/master/operators-and-expressions.html)
