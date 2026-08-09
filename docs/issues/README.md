# docs/issues — follow-ups with an owner-shaped edge

One file per follow-up, self-contained, with YAML front matter. This folder is for work that is
**known, scoped, and not being done right now** — the residue a session leaves behind when it lands
its main change. Without a home, that residue lives in a chat log nobody re-reads, and "we should
fix X" decays into nobody having ever owned X.

**The files are the store.** No database, no second copy — an issue is markdown in git, readable and
editable by hand. Everything derived (the index below above all) is *generated from* the files and
is never authoritative. A duplicate store is a fault dimension; the way to avoid one is not to
create it.

## The rules that make one worth writing

1. **Answerable cold.** Assume the reader has none of the author's context and no access to the
   session that found it. Concrete instances, file:line, the actual commands.
2. **Say what is MEASURED and what is INFERRED.** Separately, and in those words. An issue that
   states an inference as a finding sends the next person to fix the wrong thing, and they will
   believe you because you wrote it down.
3. **Define done.** Not "improve X" — the specific observation that will be true afterwards, and
   how to make it. `verify:` says which kind of evidence closes it, because they are different
   claims: `test` (a red-then-green test is enough), `probe` (only sensing the real thing counts),
   `inspect` (the artifact itself is the evidence).
4. **Name what you deliberately did NOT do,** with the reason. Silence reads as an oversight and
   gets re-litigated.
5. **Close by marking it done,** with `fixed_in:` naming the sha that landed the fix. A register
   that cannot say what fixed something has stopped being evidence.

## The format

```yaml
---
title: One line — the finding, not the fix
status: open          # open | in-progress (name an owner) | blocked | done (needs fixed_in)
kind: story           # bug | story | idea | spike — see below
scope: M              # XS | S | M | L | ?
owner: null           # a handle, or null when unowned
filed: 2026-08-03     # when it was filed; it never changes
area: lang
verify: probe         # test | probe | inspect — see rule 3
blocked_by: null      # slug(s) of issues that must close first
fixed_in: null        # the sha on the mainline; required once status is done
---
```

**`kind` is what claim the issue makes,** and the four close differently.

| | The claim | What closes it |
| --- | --- | --- |
| `bug` | the code is wrong today | the behaviour changed, and a test holds it |
| `story` | a named capability is missing | the capability is there and is exercised |
| `idea` | this direction is worth taking | it became a story, or it was refused *in writing* |
| `spike` | a question is unanswered | the answer, yes **or** no. A spike that returns "no" has succeeded |

An `idea` that cannot say what would turn it into a story is not an idea; it is a wish, and it
belongs in `docs/ideas/` where the argument has room. A `spike` states its question as a
question, and names the measurement that answers it.

**`blocked_by` is what makes the board honest.** `status: blocked` names a *person* who can unblock
something; `blocked_by` names the *work* it waits on. Only the second can be computed, which is why
"what can I pick up cold right now" is answerable at all.

**Naming:** `DD-MM-YYYY-short-kebab-summary.md` — the date it was filed, then a slug.

## Tooling — the folder checks itself

Managed by `cabildo issues`. The CLI, the `issues` MCP server, and any pre-commit gate are one
module, so the tool an agent calls and the gate that grades it cannot disagree.
`docs/issues/config.toml` holds this folder's specifics.

```bash
cabildo issues                # what's ready, what's yours, what everything else waits on
cabildo issues check          # the invariants; non-zero on problems
cabildo issues reindex        # regenerate the index below
cabildo issues show <slug>
```

## Open now

<!-- BEGIN GENERATED INDEX — `cabildo issues reindex` -->
| Issue | Area | Kind | Scope | Verify | Status | Waiting on |
|---|---|---|---|---|---|---|
| [There is no honest number for how much Modelica this reads, because the corpus is ours](03-08-2026-reachability-and-agreement-against-a-reference-tool.md) | ci | story | L | probe | open | — |
| [Diagrams draw boxes, and the icons that would make them read like circuits now survive parsing](04-08-2026-diagrams-draw-boxes-not-icons.md) | diagram | story | L | inspect | open | — |
| [The EKF results live in a log and a deleted scratch file, not in the documentation](03-08-2026-an-estimation-notebook.md) | docs | story | M | inspect | open | — |
| [The Association has not been told this exists, and the trademark note assumes they will be](03-08-2026-tell-the-modelica-association.md) | docs | idea | XS | inspect | open | — |
| [The README says nothing in Python reaches a trajectory, and something does](03-08-2026-the-readme-claim-is-too-strong.md) | docs | bug | S | inspect | open | — |
| [FMI 3.0 export — a System as an FMU, and the two routes disagree in the docs](03-08-2026-fmi-export.md) | fmi | story | L | test | open | — |
| [FMI 3.0 import — the mode table exists, the FMU reader does not](03-08-2026-fmi-import.md) | fmi | story | L | test | open | — |
| [IR nodes carry no source span, so no error below the parser can point at the equation](03-08-2026-ir-nodes-carry-no-source-span.md) | ir | bug | M | test | open | — |
| [Library twins are compared as normalised text, because a library has no flat model](03-08-2026-library-twins-are-only-compared-as-text.md) | ir | story | M | test | open | — |
| [Tearing is priced in by three documents and implemented by none](03-08-2026-tearing-is-priced-in-by-three-documents-and-implemented-by-none.md) | ir | story | L | test | open | — |
| [Redeclaring a class is parsed and refused, and it is what a medium package is](09-08-2026-redeclaring-a-class-is-parsed-and-refused-and-it-is-what-a.md) | ir | story | L | test | open | — |
| [A C emitter — the second backend, and the one FMU export needs](03-08-2026-a-c-emitter.md) | sim | story | L | test | open | — |
| [Nothing caches a compiled kernel, though the content hash that would key it exists](03-08-2026-compiled-kernels-are-not-cached.md) | sim | story | M | test | open | — |
| [Detecting charge and drop from the bean probe — the models are fine, the decision layer is wrong](03-08-2026-detecting-charge-and-drop-from-the-bean-probe.md) | sim | idea | L | probe | open | — |
| [Differential debugging is the argument for two tiers, and there is no tool that does it](03-08-2026-differential-debugging-has-no-tool.md) | sim | story | M | test | open | ir-nodes-carry-no-source-span |
| [RoastingMachine's moisture is assumed, and Cropster has it measured per lot](03-08-2026-green-moisture-as-a-measurement.md) | sim | idea | S | probe | open | — |
| [JAX gives the ensemble what Vulkan was being asked for, in double precision, today](03-08-2026-jax-is-the-ensemble-vehicle.md) | sim | idea | M | probe | open | the-ensemble-driver |
| [The method-of-lines path is the one GPU coupling with a working precedent](03-08-2026-method-of-lines-on-the-device.md) | sim | idea | L | probe | open | — |
| [Parameters are always an argument, and burning them in as literals is the other half](03-08-2026-parameters-as-constants-or-as-arguments.md) | sim | story | S | probe | open | — |
| [A phase-change kinetics example — the reachable half of the welding animation](03-08-2026-phase-change-kinetics-example.md) | sim | idea | M | test | open | — |
| [`spatialDistribution()` — the one PDE the Association standardized, and we do not have it](03-08-2026-spatialdistribution.md) | sim | story | M | test | open | — |
| [Spike: can an emitted SPIR-V kernel run an ensemble on compute-ml at all?](03-08-2026-spike-a-spirv-emitter-against-compute-ml.md) | sim | spike | L | probe | open | the-ensemble-driver |
| [The ensemble driver — one compiled model over a matrix of parameters](03-08-2026-the-ensemble-driver.md) | sim | story | M | test | open | — |
| [The Jacobian is emitted dense, though matching already knows where the zeros are](03-08-2026-the-jacobian-is-emitted-dense.md) | sim | story | L | probe | open | — |
| [The JIT compiles the kernel, not the step — the integrator loop is still Python](03-08-2026-the-jit-compiles-the-kernel-not-the-step.md) | sim | story | L | probe | open | work-precision-harness |
| [No work-precision harness, so every speed claim in the docs is unmeasured](03-08-2026-work-precision-harness.md) | sim | story | L | probe | open | — |
| [A Result is readable only by this package, and the whole field reads one `.mat` format](03-08-2026-write-results-other-tools-can-read.md) | sim | story | M | test | open | — |
| [An `initial equation` is parsed, flattened, carried on System — and read by nothing](04-08-2026-initial-equations-are-carried-and-never-read.md) | sim | bug | M | test | open | — |
| [A model can be integrated and cannot be asked where it settles](09-08-2026-a-model-can-be-integrated-and-cannot-be-asked-where-it-settles.md) | sim | story | M | test | open | — |
| [No component reads a measured signal, so nothing computed elsewhere can drive a model](09-08-2026-no-component-reads-a-measured-signal-so-nothing-computed-elsewhere-can-drive.md) | sim | story | S | test | open | — |
| [No corpus model is worth solving two ways — nothing has several zones under one schedule](09-08-2026-no-corpus-model-is-worth-solving-two-ways-nothing-has-several-zones.md) | sim | idea | M | test | open | a-model-can-be-integrated-and-cannot-be-asked-where-it-settles |
| [Nothing fits a parameter to data, and the reason that is hard is that parameters trade off](09-08-2026-nothing-fits-a-parameter-to-data-and-the-reason-that-is-hard.md) | sim | idea | L | probe | open | the-parameter-derivative-has-a-name-reserved-for-it-in-the-emitter |
| [The parameter derivative has a name reserved for it in the emitter and no code behind it](09-08-2026-the-parameter-derivative-has-a-name-reserved-for-it-in-the-emitter.md) | sim | story | S | test | open | — |
| [A fixed-step co-simulation schedule never branches, so it could be recorded once](03-08-2026-replay-a-fixed-step-cosimulation-schedule.md) | ssp | idea | M | probe | open | ssp-read-the-archive-and-run-the-master |
| [SSP 2.0 — the master's protocol is specified and checked, and nothing reads an .ssp](03-08-2026-ssp-read-the-archive-and-run-the-master.md) | ssp | story | L | test | open | fmi-import |
| [Chattering across instants is watched by a counter, and the checker cannot see it](03-08-2026-chattering-across-instants.md) | verify | idea | M | probe | open | — |
| [Composed FMUs with rollback is the workload that would justify partial-order reduction](03-08-2026-partial-order-reduction-for-composed-fmus.md) | verify | idea | L | probe | open | ssp-read-the-archive-and-run-the-master |
| [Causalization is tested by examples only, and its properties are stateable](03-08-2026-property-based-testing-for-causalization.md) | verify | story | M | test | open | — |

_38 open — 38 open; 9 done. Generated; edit the files, then `cabildo issues reindex`._
<!-- END GENERATED INDEX -->
