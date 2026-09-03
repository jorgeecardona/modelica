---
title: A VSCode extension is a diagram preview and an LSP today, and in-editor simulation once the Rust backend reaches WASM
status: open
kind: idea
scope: L
owner: null
filed: 2026-09-03
area: sim
verify: probe
blocked_by: null
fixed_in: null
---

## Measured

Three pieces an extension would lean on already exist, and the cheapest ones are done.

- **Diagrams are already theme-neutral SVG.** `modelica.diagram` draws the component and
  connection graph and writes an SVG that names no colour, so it reads on a light editor and a
  dark one (`README.md`). `diagram.of(path, *libs).write(...)` is the whole call. A webview that
  shows that SVG and re-renders on save is a wrapper, not a feature.
- **Diagnostics are already computed with source spans.** The hand-written lexer and parser
  carry positions (`README.md`, `modelica.lang`), and the interpreter reports an error in
  Modelica terms with the line it sits on ("nothing in the environment answers 'Rw' at
  Motor.mo:5:7", `docs/ideas/backends.md`). That is an LSP `Diagnostic` with a range, already
  produced -- an LSP server is a transport around it, not a new analysis.
- **Simulation runs today, in Python.** `modelica.sim.simulate` takes a `.mo` to a trajectory.
  A "run and plot" command backed by a local Python process is reachable now.

## Inferred

Inferred, not measured: the extension has two routes, and they differ in what the user must
have installed.

- **Python-LSP route, available now.** A `pygls` server wraps the toolchain; the extension
  talks to a local Python process. Diagram preview, diagnostics, hover, go-to-definition, and a
  "run and plot" command all work against the current package. Cost to the user: a Python
  environment on their machine.
- **Rust-to-WASM route, the endgame.** The Rust backend
  (`docs/issues/03-09-2026-a-rust-backend-is-a-third-dialect-plus-a-port-of-radau.md`) compiles a
  model to WebAssembly, so the model runs in the webview with no Python, no server, and no round
  trip. This is what makes a parameter slider that moves the trajectory live feel instant, and
  it is the same "interactive documentation" argument `docs/ideas/backends.md` makes, moved into
  the editor. It is blocked on that backend reaching an integrator and a WASM target, not on the
  extension.

Also inferred: this is where the monorepo the owner wants pays for itself. One moon-managed
repo with the Python package and a TypeScript extension as sibling projects, and `moon run :ci`
over both, keeps the emitter and its consumer in one place and one pipeline.

## What would turn this into a story

A spike, in this order, each shippable on its own:

1. **The diagram preview.** An extension that, on opening a `.mo`, renders
   `modelica.diagram.of(...)` into a webview and refreshes on save. No LSP, no simulation. If a
   thin wrapper around the existing SVG is the only new code, that is the finding.
2. **The language server.** A `pygls` server surfacing parse/flatten errors as diagnostics with
   the spans the toolchain already carries, plus hover and go-to-definition.
3. **Run and plot.** A command that calls `simulate` and shows the trajectory, Python-backed.
4. **In-editor simulation, WASM.** The same run, but the model is the Rust backend's WASM module
   running in the webview, with a parameter slider. Blocked on the Rust backend.

## Deliberately not done

- **No language-server rewrite in another language.** The analysis is the Python toolchain; the
  server is a transport. Reimplementing the parser in TypeScript to get a browser-only LSP would
  duplicate the one thing this project is careful to own once.
- **No packaging or marketplace publish yet.** A `.vsix` a person can install locally is the
  proof; the store is a later decision, the same way the Rust backend defers FMU/WASM packaging.
- **No dependence on the WASM route for the first release.** Steps 1 to 3 ship against the Python
  toolchain so the extension is useful before the Rust backend lands; step 4 is the upgrade, not
  the gate.
