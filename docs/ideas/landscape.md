# What else exists in Python

!!! note "Tracked in the register"
    What came out of this survey is tracked in [`docs/issues`](https://gitlab.com/jorgeecardona/pymodelica/-/tree/main/docs/issues): the README sentence, a reachability-and-agreement measurement, a result writer other tools can read, and JAX as an ensemble backend. This page is the evidence; the register is what is owed.

This project says in its README that Python has two halves of a Modelica toolchain and not
the middle. That is a claim about other people's software, so it should be checked, and
checked again as they keep working. This page is the check, made in August 2026, and it is
also a reading list: several of these tools have solved problems that are on this project's
own list, and their answers are public.

!!! info "Short version"
    The README's claim is too strong. **`pymoca` with RTC-Tools takes a `.mo` file to a
    trajectory today**, and Deltares runs it for water-system control. It gets there with a
    fixed step and smooth residuals, because its consumer is an optimizer. The narrower
    statement is the true one and is a better description of this project anyway: what is
    missing in Python is the *hybrid* path — adaptive stiff integration, index reduction and
    events. Two projects outside Python, `Rumoca` in Rust and `jaxdae` in JAX, have arrived
    at architectures close to this one, and both are worth reading.

## The map

Five kinds of thing, and only the first is what this project is.

| | What it is | Examples |
| --- | --- | --- |
| **compiler** | reads Modelica, produces equations | pymoca, Rumoca |
| **FMU runner** | runs what another tool compiled | FMPy, PyFMI |
| **session handler** | drives an installed compiler | OMPython, OMSimulator, BuildingsPy |
| **result reader** | reads a `.mat` another tool wrote | DyMat, ModelicaRes |
| **modelling library** | acausal modelling, no Modelica syntax | jaxdae |

## pymoca — the nearest neighbour

[pymoca][pymoca] is a Modelica-to-computer-algebra translator in Python: ANTLR for the
grammar, an AST module, a tree module that flattens, and backends for CasADi, SymPy and
ModelicaXML. Deltares uses it under [RTC-Tools][rtctools] for optimal control of water
systems. BSD licensed.

**It reads much more of the language than this does.** Its test corpus names the cases:
`Array3D`, `ArrayExpand`, `ArrayExpressions`, `MatrixExpressions`, `ForLoop`,
`NestedForLoop`, `DelayForLoop`, `IndexedConnect`, `ArrayConnectParam`,
`RedeclareComponents`, `RedeclareClass`, `RedeclareInExtends`, `RedeclareTypeCompatibility`,
`ExpandableConnectorBus`, `FunctionCall`. Every one of those is a construct
`modelica.lang.parse` refuses by name today, and every one of them has a worked
implementation and a test there to read.

**How RTC-Tools integrates, and why.** From `simulation_problem.py`: the derivative is a
backward difference at a fixed step, `derivative_state - (X[index] - X_prev[index]) / dt`;
the residual and the derivative approximations go into one root-finding problem per step,
solved by CasADi's `rootfinder`; initialization is a separate least-squares solve. There is
no event detection — conditionals become smooth CasADi expressions.

That reads like a limitation only if you forget what it is for. A gradient-based optimal
control problem *needs* a smooth residual; an integrator that stops at events and
reinitialises a state is a discontinuity an NLP solver cannot differentiate through. Smooth
and fixed-step is the right answer to their question. It is a different question from
"what does this model actually do", which is the one here.

So the two sets of gaps are close to disjoint:

| | pymoca | this |
| --- | --- | --- |
| arrays, `for`, `redeclare`, functions | yes | refused by name |
| events, `when`, `reinit` | not evident | yes |
| index reduction, dummy derivatives | not evident | yes |
| BLT | no | yes |
| tearing | no | not yet |
| integrator | fixed step, smooth residual | adaptive, stiff, event-aware |
| what it is for | optimal control | a trajectory |

The two "not evident" rows are inference from a file listing plus what RTC-Tools has to do
downstream, not a reading of the source. Anybody who needs that answer should confirm it
before repeating it.

## Rumoca — the same rule, reached independently

[Rumoca][rumoca] compiles Modelica in Rust and emits CasADi, SymPy, FMI 2.0 and 3.0,
JavaScript, JAX, Julia/ModelingToolkit, ONNX and embedded C. Three things in it are worth
importing.

**It reached "every stage is a value you can hold" on its own.** Four intermediate
representations — parsed, flattened, DAE, solver — and the stated reason is this project's
reason: each phase produces an accessible artifact, so a failure localises to a phase. Two
projects reaching the same rule from different languages is the best evidence available that
the rule is right.

**Backends are templates.** A new target is a Jinja template rather than a pass, which is
how eight backends stay maintainable. This project runs models on its own VM instead of generating code.

**It publishes its coverage, and that is a good habit to copy.** Rumoca reports that **28 %
of MSL examples reach end-to-end simulation**, and that **about 33 % of those that simulate
disagree with OpenModelica**. Numbers like that are unflattering and useful, and they are the
kind of thing this project cannot currently say about itself: the corpus here is fifteen
models written by the same people who wrote the compiler, which measures the corpus more than
the toolchain.

Their fourth IR is a design worth weighing later: a register-based opcode stream with
Jacobian-vector products precomputed by forward mode, which is the alternative to emitting
source text.

## jaxdae — a very similar engine, without a front end

[jaxdae][jaxdae] (July 2026) is not a Modelica tool. It is acausal modelling with typed
effort-flow connectors and a `connect()`, assembled through a Python API — which is what
`modelica.build` is. Underneath: **Pantelides with dummy derivatives**, adaptive BDF, Radau
and Rosenbrock, state events with a jump map, differentiable adjoints through the
discontinuities, and Newton-Krylov on exact forward-mode Jacobian-vector products.

Two of its results bear on decisions recorded elsewhere here.

* **It gives the ensemble question an answer.** Batched-gradient wall time "stays nearly flat
  from batch 1 to 1000" through XLA fusion. That is the ensemble driver of [GPU
  backends](vulkan.md), working, in double precision, with no Vulkan and no shader — which
  makes JAX a cheaper vehicle to try first than a SPIR-V emitter.
* **It does not form the Jacobian.** GMRES on Jacobian-vector products, with dense LU below
  about a hundred states. This project forms the matrix, which is right at the corpus's size
  and is a decision with a horizon.

## The rest, and what each is good for

**[FMPy][fmpy]** runs FMUs: pure Python and `ctypes`, FMI 1.0, 2.0 and 3.0, both interface
types, and it validates `modelDescription.xml` against the schema rather than trusting it.
BSD 3-clause. It also ships a `cross_check` module that drives the
[fmi-cross-check][crosscheck] repository — FMUs from many vendors, with reference results.
That repository is the acceptance test for any FMI importer, and FMPy is the independent
second opinion an *exporter* needs. Neither has to be built here.

**[PyFMI][pyfmi]** is the same job with Assimulo's solvers behind it, and it is the execution
engine of Modelon Impact. LGPL, which is worth knowing before reading it for ideas.

**OMPython, OMSimulator and BuildingsPy** drive an installed compiler: a session handler for
`omc`, the FMI and SSP co-simulation environment, and library regression testing against
Dymola, OpenModelica or OPTIMICA. When somebody asks how to use Modelica from Python, this is
usually the answer, and it is a good one — a complete compiler and the whole MSL. What it is
not is a model you can hold as a Python value, which is the thing this project is for.
OMPython is under the OSMC Public Runtime License.

**DyMat and [ModelicaRes][modelicares]** read the Dymola and OpenModelica `.mat` result
format and plot it. Every analysis script in this field speaks that format, and a `Result`
that can be written in it becomes readable by tools nobody here has to write.

## What this changes here

Nothing about the architecture. Four things about the work.

1. **The README's claim needs one more word.** "Nothing takes Modelica source all the way to
   a trajectory" is not true; "nothing does it with adaptive stiff integration, index
   reduction and events" is, and it describes the work better.
2. **The refused constructs have worked implementations to read.** Arrays, `for`, `redeclare`
   and functions are done in pymoca, with tests, under a permissive licence. That makes them
   cheaper than their `L` scope suggests, and they are what stands between this toolchain and
   models people already have.
3. **There is a published way to say where a young compiler is.** Reachability and agreement
   against a reference tool, on a library nobody here wrote.
4. **The ensemble path has a cheaper vehicle to try first than Vulkan**, and somebody has
   already measured it.

[crosscheck]: https://github.com/modelica/fmi-cross-check
[fmpy]: https://github.com/CATIA-Systems/FMPy
[jaxdae]: https://arxiv.org/abs/2607.23202
[modelicares]: https://github.com/kdavies4/ModelicaRes
[pyfmi]: https://github.com/modelon-community/PyFMI
[pymoca]: https://github.com/pymoca/pymoca
[rtctools]: https://github.com/rtc-tools/rtc-tools
[rumoca]: https://github.com/cognipilot/rumoca
