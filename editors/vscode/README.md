# pymodelica Simulation Scope (VS Code)

Simulate the Modelica `.mo` file you are editing and see its trajectory, in an interactive
chart inside VS Code. Same chart engine as the notebook widget (`modelica.sim.view`), so the
two match: drag to zoom, double-click to reset, hover for a cursor readout, click a variable to
toggle it in or out, and switch the vertical axis between **linear**, **log |y|**,
**normalised**, **by unit** (variables that share a physical unit share a strip and a scale --
the temperatures together, mass on its own), and **split** (each variable its own strip). A
single open `.mo` opens on its states grouped by unit, so nothing gets crushed.

Plain JavaScript, no build step, no dependencies. Simulation runs through the Python simulator
(`modelica.sim`, the reference tier) -- nothing is compiled per model.

## Use it

Open a `.mo` file, then run **pymodelica: Simulate This Model** -- from the `⧗` button in the
editor title bar, the right-click menu, or `Cmd/Ctrl+Shift+P`. The Scope opens beside the editor
showing the model's trajectory. Turn on **pymodelica.simulateOnSave** to re-run on every save.

**pymodelica: Open Simulation Scope (corpus gallery)** opens the bundled gallery of example
models instead.

### Settings

| Setting | Default | What it does |
| --- | --- | --- |
| `pymodelica.runtime` | `auto` | Which runtime simulates. `auto` picks the fastest available (Python today; a Rust/C interpreter later). |
| `pymodelica.python` | `["uv","run","--extra","sim","python"]` | Command that runs a Python with `modelica` importable. |
| `pymodelica.stop` | `10` | Stop time, in the model's time units. |
| `pymodelica.library` | `[]` | Extra `.mo` files/folders for dependencies (a sibling `library/` is auto-loaded). Set to `examples/library` for the corpus models. |
| `pymodelica.simulateOnSave` | `false` | Re-simulate when you save a `.mo` (while the Scope is open). |

## Install it locally

**Package a `.vsix` and install it from the UI (most reliable):**
```bash
cd editors/vscode
pnpm run package                                 # pnpm dlx @vscode/vsce package -> the .vsix
```
Then in VS Code (the window you actually use -- stable and Insiders are separate installs):
Extensions (`Cmd/Ctrl+Shift+X`) -> the `⋯` menu -> **Install from VSIX…** -> the `.vsix`.

`pnpm run install-local` does the packaging and then tries the `code`/`code-insiders` CLI, and
prints the UI steps if neither is on PATH.

## Refreshing the corpus gallery data

`media/scope.data.json` is a snapshot the Python simulator produces via `scripts/build-data.py`.
Regenerate it and the gallery updates. The `.mo` viewer never reads it; it simulates the open
file live.
