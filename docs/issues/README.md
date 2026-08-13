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
| [The corpus has no way to hold a model that costs a minute to simulate, and Ring Modulator costs 74 seconds](13-08-2026-the-corpus-has-no-way-to-hold-a-model-that-costs-a.md) | ci | story | M | probe | in-progress · @quiet-tars-187f2 | — |
| [Diagrams draw boxes, and the icons that would make them read like circuits now survive parsing](04-08-2026-diagrams-draw-boxes-not-icons.md) | diagram | story | L | inspect | open | — |
| [Straightness is a placement property, and the layouts here decide it in the router](10-08-2026-straightness-is-a-placement-property-and-the-layouts-here-decide-it-in.md) | diagram | story | L | probe | open | — |
| [The EKF results live in a log and a deleted scratch file, not in the documentation](03-08-2026-an-estimation-notebook.md) | docs | story | M | inspect | open | — |
| [The Association has not been told this exists, and the trademark note assumes they will be](03-08-2026-tell-the-modelica-association.md) | docs | idea | XS | inspect | open | — |
| [FMI 3.0 export — a System as an FMU, and the two routes disagree in the docs](03-08-2026-fmi-export.md) | fmi | story | L | test | open | — |
| [FMI 3.0 import — the mode table exists, the FMU reader does not](03-08-2026-fmi-import.md) | fmi | story | L | test | open | — |
| [RoastingMachine's moisture is assumed, and Cropster has it measured per lot](03-08-2026-green-moisture-as-a-measurement.md) | sim | idea | S | probe | blocked · @jorge | — |
| [`spatialDistribution()` — the one PDE the Association standardized, and we do not have it](03-08-2026-spatialdistribution.md) | sim | story | L | test | blocked | — |
| [A stencil is emitted as n unrolled scalar blocks, so a method-of-lines model is 230,000 lines of Python at n = 10,000](09-08-2026-a-stencil-is-emitted-as-n-unrolled-scalar-blocks-so-a-method.md) | sim | story | L | probe | in-progress · @quiet-tars-187f2 | — |
| [No component reads a measured signal, so nothing computed elsewhere can drive a model](09-08-2026-no-component-reads-a-measured-signal-so-nothing-computed-elsewhere-can-drive.md) | sim | story | S | test | blocked | — |
| [No corpus model is worth solving two ways — nothing has several zones under one schedule](09-08-2026-no-corpus-model-is-worth-solving-two-ways-nothing-has-several-zones.md) | sim | idea | M | test | open | — |
| [Nothing fits a parameter to data, and the reason that is hard is that parameters trade off](09-08-2026-nothing-fits-a-parameter-to-data-and-the-reason-that-is-hard.md) | sim | idea | L | probe | open | — |
| [Stages 2 and 3 of the benchmark: HIRES, E5, Ring Modulator, and ScalableTestSuite](10-08-2026-stages-2-and-3-of-the-benchmark-hires-e5-ring-modulator-and.md) | sim | story | L | probe | in-progress · @quiet-tars-187f2 | — |
| [A TypeScript backend is a third dialect plus a port of radau.py, and it would put models in the browser](11-08-2026-a-typescript-backend-is-a-third-dialect-plus-a-port-of-radau.md) | sim | idea | L | probe | open | — |
| [A fixed-step co-simulation schedule never branches, so it could be recorded once](03-08-2026-replay-a-fixed-step-cosimulation-schedule.md) | ssp | idea | M | probe | open | ssp-read-the-archive-and-run-the-master |
| [SSP 2.0 — the master's protocol is specified and checked, and nothing reads an .ssp](03-08-2026-ssp-read-the-archive-and-run-the-master.md) | ssp | story | L | test | open | fmi-import |
| [Composed FMUs with rollback is the workload that would justify partial-order reduction](03-08-2026-partial-order-reduction-for-composed-fmus.md) | verify | idea | L | probe | open | ssp-read-the-archive-and-run-the-master |

_19 open — 13 open, 3 in-progress, 3 blocked; 49 done. Generated; edit the files, then `cabildo issues reindex`._
<!-- END GENERATED INDEX -->
