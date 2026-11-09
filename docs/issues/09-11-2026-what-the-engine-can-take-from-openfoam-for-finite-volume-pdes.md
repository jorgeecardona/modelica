---
title: What the engine can take from OpenFOAM for finite-volume PDEs
status: open
kind: idea
scope: L
owner: jorgeecardona
filed: 2026-11-09
area: sim
verify: test
blocked_by: null
fixed_in: null
---

OpenFOAM's data layout fits a bytecode VM well. Its run-time compiled code (`codeStream`) and
its per-step matrix rebuild do not. This note records what to take, with sources, for PDEs on
the engine after the [simulation IR plan](09-11-2026-a-simulation-ir-with-arrays-events-and-parallel-execution-for-the-engine.md).

## Take

- **Face addressing.** A mesh is face lists: `owner[f]`, `neighbour[f]` (internal faces only,
  `owner < neighbour`), plus `ownerStart` and `losort`/`losortStart` for cell-ordered access.
  Five integer arrays, shipped as program data.
  [lduAddressing](https://api.openfoam.com/2406/lduAddressing_8C_source.html),
  [fvMeshPrimitiveLduAddressing](https://api.openfoam.com/2406/fvMeshPrimitiveLduAddressing_8C_source.html)
- **LDU matrix.** `diag` per cell, `lower` and `upper` per internal face. The product is one
  loop over faces: `y[o] += upper[f] x[n]`, `y[n] += lower[f] x[o]`, plus the diagonal. One
  opcode, no branches. Conservation holds by construction.
  [lduMatrix](https://api.openfoam.com/2406/classFoam_1_1lduMatrix.html),
  [Amul](https://www.openfoam.com/documentation/guides/v2206/api/LduMatrixATmul_8C_source.html)
- **Implicit and explicit operators.** `fvm::laplacian` adds matrix coefficients; `fvc::grad`
  evaluates a field. For us: what goes into the Jacobian against what goes into `rhs`.
  [FVM in OpenFOAM](https://openfoamwiki.net/index.php/Finite_volume_method_(OpenFOAM))
- **Schemes as data.** `fvSchemes` names the discretization per term; a factory table maps the
  name to a class at run time. For us: a scheme name maps to an opcode or a coefficient table.
  [fvSchemes](https://doc.cfd.direct/openfoam/user-guide-v11/fvschemes),
  [run-time selection](https://mobile.openfoamwiki.net/index.php/Snip_runtime_selection_mechanism)
- **Boundary patches.** Named ranges of boundary faces with a rule. A processor patch (the cut
  between two parallel domains) is the same kind of object with a send and a receive.
  [parallel](https://www.openfoam.com/documentation/user-guide/running-applications-parallel.php)
- **Iterative solvers for large meshes.** PCG with DIC (symmetric), PBiCGStab with DILU, GAMG.
  Direct LU fill-in is too large on 3-D meshes.
  [fvSolution](https://doc.cfd.direct/openfoam/user-guide-v13/fvsolution)

## Do differently

- **No run-time compilation.** `codedFixedValue` writes C++, calls the compiler and loads the
  library ([source](https://cpp.openfoam.org/v11/codedFixedValueFvPatchField_8C_source.html)).
  An unknown scheme or boundary rule is an error, never a compile.
- **Build the sparse structure once.** OpenFOAM rebuilds `fvMatrix` every step; GPU ports found
  assembly up to half the time and per-step allocation a large cost (one 22M-cell case: 823 s to
  141 s with a memory pool). Keep the pattern fixed per mesh and only refill values.
  [SPUMA](https://arxiv.org/html/2512.22215)
- **Keep one stiff integrator.** OpenFOAM time-steps with Euler, backward or Crank-Nicolson and
  couples pressure and velocity in segregated loops (PISO, SIMPLE, PIMPLE); stiff chemistry gets
  a separate Rosenbrock solver per cell ([chemistryModel](https://cpp.openfoam.org/v12/chemistryModel_8C_source.html)).
  Radau with a mass matrix covers both, and keeps events.
- **Threads as well as MPI.** OpenFOAM is mostly MPI; our engine already runs one program per
  thread. Domain cuts as processor patches would work the same way across threads.

## Possible steps

1. A face-list `Gather`/`Scatter` opcode pair and an `LDU_AMUL` opcode over shipped arrays.
2. The Jacobian in LDU form for a mesh model; PCG/DIC and BiCGStab/DILU in the engine for
   systems above the size where banded LU stops paying.
3. A Python front end that reads a mesh (owner, neighbour, patches) and writes the arrays.
4. Mass-matrix Radau (`M y' = f`) for finite element models.

Not confirmed in this research: ExaFOAM and petsc4Foam details.
