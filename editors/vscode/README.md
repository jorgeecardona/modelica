# pymodelica Simulation Scope (VS Code)

Simulate the Modelica `.mo` file you are editing and read its trajectory in an interactive
chart beside the editor. The same chart engine as the notebook widget (`modelica.sim.view`),
so the two match. Plain JavaScript, no build step, no dependencies: simulation runs through the
Python simulator (`modelica.sim`, the reference tier), nothing is compiled per model.

## Use it

Open a `.mo` file and run **pymodelica: Simulate This Model**: the `⧗` button in the editor
title bar, the right-click menu, or the command palette (`Cmd/Ctrl+Shift+P`). The Scope opens
beside the editor with the model integrated to its own stop time. Turn on
**pymodelica.simulateOnSave** to re-run on every save while the panel is open.

**pymodelica: Open Simulation Scope (corpus gallery)** opens the bundled gallery of the
example models instead, each integrated to its reference points.

**pymodelica: Overlay a Measured Log (CSV)…** puts measured data on the chart (see below).

A model that references the real **Modelica Standard Library** names resolves against a
bundled slice, loaded automatically: `Modelica.Electrical.Analog`, `Modelica.Mechanics.
Translational` and `Rotational`, and `Modelica.Thermal.HeatTransfer` -- the physical domains,
whose connectors carry a potential and a flow. It is a curated subset, not the whole MSL, so a
model reaching for a part that is not there fails by name (`Modelica has no nested class X`);
turn it off with **pymodelica.standardLibrary**. Blocks is not in it yet: its causal signal
connector needs a language feature the front end does not have.

The model's sibling `<name>.toml`, when there is one, sets the run: `[simulate].stop`,
`method` and `absolute` are honoured, and its `tolerance` is the **exact** speed below. A
`library/` folder holding the components a model uses is loaded automatically, whether it sits
beside the `.mo` or one directory up (the corpus keeps it at `examples/library`, a sibling of
`examples/models`, so a component model like `BumpStop.mo` resolves with nothing configured);
**pymodelica.library** adds more files or folders.

## The panel

Two badges in the header tell you what drew the picture: the extension version (`v0.1.20`) and
the integrator in use (`Radau · 1e-6`, or the model's own, `LSODA · 1e-9` for VanDerPol).
After upgrading the extension the old extension host keeps running until the window reloads,
so an old version number here means reload the window (**Developer: Reload Window**).

### Speed

The **speed** control sets how hard the integrator works: **fast** (relative tolerance 1e-6,
the default), **fine** (1e-8), or **exact** (the model's own `.toml` tolerance, the run its
reference points were taken at). A model's reference tolerance can be far tighter than looking
at a trajectory needs -- the chaotic double pendulum asks for 1e-10, about 2.5 s, where fast
draws in a quarter second. The badge by the version shows the tolerance in use, with a `*` when
it is a display tolerance rather than the model's own. On a chaotic model each speed draws a
*different* path: the trajectory only means one thing at the tolerance its references were taken
at, so switch to **exact** to see that one. The model's method (LSODA for VanDerPol) and its
absolute tolerance (E5's 1.7e-24) are kept at every speed.

### Time horizon

- **`t →` stop field.** Type a stop time and press Enter, or use the arrow keys: edits are
  coalesced, so a held key becomes one run. Extending the stop with nothing else changed
  continues the run from where it ended rather than starting over at t = 0; any other change
  (a parameter, a shorter stop) is a fresh run. The stop is also a **draggable marker** on the
  plot: drag it outward to extend.
- **`→ event`** runs on to the next event past the current stop, with a small tail so the
  event line is visible. **`⇥ settle`** runs until every state stays within 1 % of its final
  value. Both probe a doubling horizon bounded by **pymodelica.autoStopMax** (0 means ten
  times the current stop) and say in the footer what they found, or that nothing was found
  before the cap. The probes run at a coarse tolerance, since the question is only whether
  something happened; the run that is then shown uses the model's own.

### Axes and units

- **Vertical scale:** `linear`, `log |y|`, `normalised`, `by unit` (variables that share a
  physical unit share a strip and a scale, so the temperatures sit together and mass on its
  own), or `split` (one strip per variable). A single open `.mo` opens on its states grouped by
  unit, so nothing is crushed.
- **Plot mode, `t · xy · xyz`.** `t` draws the variables against time; **log t** (shown in
  that mode) puts time on a log axis for a stiff run over a long horizon (E5 runs to 1e13 s
  and finishes its kinetics by 1e3, which a linear axis puts in one pixel column; the ticks
  become the decades). `xy` is a **phase plot**: the `x` slot picks the variable on the x axis
  and the selected variables draw against it, such as the rate of rise against bean temperature
  or Lorenz's z against x; `⇄` swaps x with the first drawn variable. `xyz` (offered with three
  or more variables) is the **3D phase view**: three slots pick the axes, defaulting to the
  first three states -- Lorenz's x, y, z, the butterfly -- with each option showing its unit
  and `⇄` swapping x and y. The trajectory is coloured by time (blue to orange, with a small
  colourbar) so direction reads off the line; pinned runs draw dashed. Drag to rotate, wheel to
  zoom, double-click to reset the camera; the `view…` menu has a reset, top / front / side
  presets, and an auto-rotate that any drag stops. This is the view a chaotic model is meant to
  be seen in; against time those states look like noise. Events and reference points hide in a
  phase or 3D plot; in `xy` the cursor picks the nearest sample by the x variable's value.
- **Time in seconds or minutes.** The data stays in seconds; only the axis, the readouts and
  the rates convert: a `K/s` rate reads `K/min` in minutes.
- **Display units:** `SI units` or `K as °C`. A bare kelvin value is shown in Celsius; a rate
  such as `K/s` is never offset. With minutes and °C together a roaster's rate of rise reads
  in °C/min.

### Reading the chart

- **Drag** to zoom in time (or in x, in a phase plot). **Double-click** the chart to reset the
  zoom and clear the marks. **Hover** for a cursor readout of every drawn variable, in the
  display units, at the nearest sample: the values appear beside each variable in the legend,
  and, while the legend is hidden in a narrow pane, in the footer instead.
- **Measurement marks.** Click the chart to place a mark at the nearest sample; click again
  for a second and the footer shows Δt and the change in every drawn variable between them.
  Marks are sample indices, so they work in phase plots too.
- **Events** are vertical lines, listed in the legend with their instant and the condition in
  Modelica that fired, as the model states it (`when probe >= T_drop`).
- **Legend.** Click a variable to toggle it; `all`, `none` and `reset` act on the set. The
  legend pane resizes by dragging its splitter and collapses on a double-click. Below 780 px
  of panel width it is hidden by default and a `legend` button appears in the toolbar: it
  opens the legend as a strip under the chart, with every section below reachable.

### Parameters and comparison

- **Parameters** lists every parameter with its value, unit and description. Edit a value and
  press Enter to re-run; the undo arrow restores the model's own. A parameter whose binding is
  an expression of others is *derived* and shown read-only, since it follows what it depends on
  (`m_dry` follows `m_green`). Overrides go through the simulator's sweep vector, so everything
  downstream is recomputed.
- **`+ compare`** pins the current run: it keeps drawing, dashed, while you change parameters,
  and the legend shows each run's value at the cursor. Pinned runs appear as chips above the
  chart, each naming what it overrides, with a `×` that drops the run. **Click a chip** to edit
  that run's parameters: the panel switches to that instance (its title shows the letter), and a
  change re-runs only it, so A and B can hold different parameters. Pinned runs extend with the
  stop the same way the live one does.

### Studies

- **Sweep.** Pick a parameter, a range and a number of runs; **run sweep** simulates the model
  once per value, each as a pinned run, and plots a **response** against the parameter: a
  variable's final value or the time an event first fires. The response defaults to a model
  event when one fires (the roaster's drop) until you choose otherwise.
- **Sensitivity.** Pick a response and a relative step; **compute** runs every editable
  parameter a step either way (two simulations each) and draws a **tornado**: parameters
  sorted by how much they move the response, with an undefined end (an event that never
  fires) marked as such.

### Measured log overlay

`log…` (or the command) opens a CSV, TSV or TXT file and draws its columns as hollow points on
the chart, in each strip's own scale, so a measured roast lies over the simulated one. The
format:

- One header row. The delimiter is detected: comma, semicolon or tab. With a semicolon or tab
  delimiter, comma decimals are accepted.
- The time column is the first header named `t`, `time`, `s`, `sec`, `seconds`, `min` or
  `minutes` (case-insensitive), else the first column. A header starting with `min` is read
  as minutes and converted to seconds.
- Every other numeric column is a series named by its header, matched to the model's
  variables case-insensitively (`probe` overlays `probe`). A matched variable is switched on
  in the legend when the log arrives, so its points show at once. Non-numeric cells are gaps.

Loaded logs are listed in the legend under **measured**, with their clock in the current time
unit.

## Settings

| Setting | Default | What it does |
| --- | --- | --- |
| `pymodelica.runtime` | `auto` | Which runtime simulates. `auto` picks the fastest available (Python today; a Rust/C interpreter later). |
| `pymodelica.python` | `["uv","run","--extra","sim","python"]` | Command that runs a Python with `modelica` importable. The simulator script and arguments are appended. |
| `pymodelica.stop` | `0` | Stop time. `0` means the model's own: its sibling `.toml`'s `[simulate].stop`, else 10. Changeable live in the panel. |
| `pymodelica.autoStopMax` | `0` | How far `→ event` and `⇥ settle` may probe past the current stop. `0` means ten times the current stop. |
| `pymodelica.library` | `[]` | Extra `.mo` files or folders for dependencies (a sibling `library/` is loaded automatically). `examples/library` for the corpus models. |
| `pymodelica.cwd` | `""` | Working directory for the simulator. Empty means the workspace folder of the `.mo` file. |
| `pymodelica.simulateOnSave` | `false` | Re-simulate when you save a `.mo`, while the Scope is open. |

The simulator needs a Python where `modelica` imports with its simulation extra. The default
command assumes the pymodelica repository is the workspace (`uv run --extra sim python`); point
**pymodelica.python** at another interpreter otherwise.

## Install it locally

Package a `.vsix` and install it:

```bash
cd editors/vscode
pnpm run install-local      # packages, then installs into every VS Code found (stable, Insiders)
```

If no `code` CLI is found the script prints the UI steps: Extensions (`Cmd/Ctrl+Shift+X`), the
`⋯` menu, **Install from VSIX…**, the `.vsix`. Do this in the window you actually use; stable
and Insiders are separate installs. Reload the window afterwards and check the version badge.

## Publish it

```bash
cd editors/vscode
VSCE_PAT=<marketplace token> pnpm run publish
```

The token is a Visual Studio Marketplace Personal Access Token (Azure DevOps, scope
Marketplace → Manage) for the publisher named in `package.json`. It is read from the
environment and never stored in the repository.

## Previewing the panel outside VS Code

`scripts/preview.py` fills the webview template the way the extension does and writes a
standalone page, for a browser or for Playwright:

```bash
uv run --extra sim python python/simulate_file.py ../../examples/models/RoastingMachine.mo > run.json
uv run python scripts/preview.py --data run.json --out preview.html --host
python3 -m http.server 8731          # the page's content-security policy blocks file://
```

`--gallery` builds the corpus gallery from `media/scope.data.json`. `--host` adds a stand-in
extension host that answers the panel's messages from the embedded record, so pinning,
parameter overrides, the sweep, the sensitivity study, the automatic stops and a continued run
(a larger stop, answered with a flat tail) run end to end without VS Code; the messages the
panel posted are in `window.__posted`.

## Refreshing the corpus gallery data

`media/scope.data.json` is a snapshot the Python simulator produces via `scripts/build-data.py`.
Regenerate it and the gallery updates. The `.mo` viewer never reads it; it simulates the open
file live.
