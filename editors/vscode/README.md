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
whose connectors carry a potential and a flow -- and `Modelica.Blocks`, the causal signal
domain (`Sources.Constant`, `Math.Gain`/`Feedback`, `Continuous.Integrator`/`FirstOrder`), wired
by the real-valued `RealInput`/`RealOutput` connectors. It is a curated subset, not the whole
MSL, so a model reaching for a part that is not there fails by name (`Modelica has no nested
class X`); turn it off with **pymodelica.standardLibrary**.

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
  or Lorenz's z against x; `⇄` swaps x with the first drawn variable. A few arrowheads along the
  live run's curve point the way time runs, and a hollow ring marks where the trajectory begins and
  a filled dot where it ends, so a decaying spiral shows its start at the rim and its rest at the
  centre. A `coords` menu reads the two slots as **polar** -- a radius and an angle -- and draws the
  plane the trajectory lives in, on a real polar grid: concentric circles at round radii and spokes
  every 30 degrees (labelled in degrees or as fractions of pi, following the angle toggle), the
  origin at the centre. SchwarzschildOrbit's r and phi become its precessing rosette, the perihelion
  and aphelion read straight off the circles. `xyz` (offered with three
  or more variables) is the **3D phase view**: three slots pick the axes, defaulting to the
  first three states -- Lorenz's x, y, z, the butterfly -- with each option showing its unit
  and `⇄` swapping x and y. Any of the three axes can be **time** instead of a variable, which
  extrudes a phase curve into a tube over time: put x and y on two axes and t on the third and
  Lorenz's butterfly climbs as the run goes, the two lobes stacked in time. A faint reference frame -- the bounding cube with a grid on its floor,
  and the trajectory's shadow cast straight down onto it -- gives depth, so the curve reads as
  sitting in a space, its footprint says where it is in the plane at each height, and the camera's
  rotation shows in the near and far edges. The trajectory is coloured by time (blue to orange, with a small
  colourbar) so direction reads off the line; pinned runs draw dashed. Drag to rotate, wheel to
  zoom, double-click to reset the camera; the `view…` menu has a reset, top / front / side
  presets, and an auto-rotate that any drag stops. This is the view a chaotic model is meant to
  be seen in; against time those states look like noise. Events and reference points hide in a
  phase or 3D plot; in `xy` the cursor picks the nearest sample by the x variable's value.
- **Time in seconds or minutes.** The data stays in seconds; only the axis, the readouts and
  the rates convert: a `K/s` rate reads `K/min` in minutes.
- **Display units:** `SI units`, `°C`, `°F`, or `US units`. Temperature converts (kelvin or
  degC to °C or °F); `US units` also shows metres as feet, kilograms as pounds, newtons as
  pounds-force, watts as horsepower. Only the printed value converts, never the data. A rate
  keeps the offset out -- `K/s` becomes `°F/s`, scaled but not shifted -- and follows the time
  axis, so with minutes a roaster's rate of rise reads in °C/min or °F/min. Adding a system is
  one entry in the display table.
- **Angle units.** A second toggle, shown only when a variable is an angle, prints angles in `rad`
  or `deg` -- a heading, a pendulum's swing, an orbit's phase read in degrees when that is how you
  think, `rad/s` becoming `°/s`. Orthogonal to the unit system above (you can have °C and degrees at
  once), and the data stays in radians: only what is printed converts, and a polar phase plot is
  still drawn from the radians.
- **Remembered.** The display units, the angle units, the time unit, the solver speed and the legend
  width are kept across reopens, so a panel you work in °F and minutes stays that way. Which
  variables are drawn, and how a model is looked at (the plot mode and its coordinate system), are
  kept **per model**: an orbit reopens as its rosette and a decay against time, each the way you
  last left it. The vertical scale is not remembered: a freshly opened model picks its own (a
  single `.mo` opens split by unit).

### Reading the chart

The chart carries about ten mouse gestures. The **`?`** button, or the **`?`** key, opens a card
over the chart listing them all -- so the ones below need not be memorised.

- **Drag** to zoom in time (or in x, in a phase plot). Zooming into a window the run only
  samples sparsely re-reads it at full density over just that span, so the sub-second spike of
  a two-minute run is sharp without touching the resolution knob (the automatic form of `Δt ≤`):
  the integration is unchanged, only the output is read more finely where you are looking. A
  live `.mo` only, and only when the window is actually sparse. Once zoomed, the **left and right
  arrow keys** pan the window a quarter of its width at a time, stopping at the ends of the run.
  **Double-click** the chart to
  reset the zoom and clear the marks. **Hover** for a cursor readout of every drawn variable, in
  the display units, at the nearest sample: a line marks the instant and a dot sits on each
  variable's curve where it crosses -- in every scale, the by-unit and split strips included,
  each dot on its own strip -- and the values appear beside each variable in the legend, and,
  while the legend is hidden in a narrow pane, in the footer instead.
- **Measurement marks.** Click the chart to place a mark at the nearest sample; click again
  for a second, and a readout shows Δt, and for every drawn variable its two values, their
  change, the **average rate** across the interval -- the secant slope, `Δvar/Δt`, in the
  variable's unit over the time unit, which is a roast's rate of rise in °C/min -- and the
  **integral** `∫var dt`, the complement of the rate: the charge under a current, the volume
  under a flow, the energy under a power. Marks are sample indices, so they work in phase plots too.
- **Level lines.** Shift-click to drop a horizontal line at the value under the pointer, on the
  strip it lands in: the level a variable is asked to reach. A dot marks where each drawn
  variable of that strip first crosses the level (both the fall and the rise, so a signal that
  dips through and climbs back is caught twice), with the crossing time on the axis and every
  crossing in the footer, so "how long to 400 K" reads straight off the plot. Shift-click a line
  again to lift it; double-click clears the marks and lines together. Level lines are a
  time-view feature on a linear or by-unit / split scale (a horizontal line has no single time
  in a phase plot, nor one value per pixel on a log axis).
- **Events** are vertical lines, listed in the legend with their instant and the condition in
  Modelica that fired, as the model states it (`when probe >= T_drop`). Hovering a line shows the
  same, so an event says what it is without a trip to the legend. Clicking an event in the list
  zooms the time window onto it -- tight when its neighbours are close (a bouncing ball's last few
  contacts), wider for a lone event -- without touching the run; double-click restores the view.
- **Guards.** A model can carry `assert(condition, message)` checks of its own -- a buck converter
  asserting its duty cycle stays in `[0, 1]`, a machine asserting it keeps synchronism. The
  **guards** section of the legend lists each condition and its message, so a model states the
  range it holds to; clicking a guard whose condition names a variable plots that variable, so you
  can watch the duty cycle it keeps in `[0, 1]` rather than hunt for it in the legend. If a run
  drives a guard false the simulation stops and the message appears in
  a banner over the chart; any other run failure (a singular model, a solver that gives up) shows
  there too, wrapped so the whole message is readable.
- **Legend.** Click a variable to toggle it; **alt-click** to *solo* it -- show only that one,
  which on a component model of dozens of variables beats clicking the rest off; alt-click it
  again to bring back the set that was on before. Just **hovering** a row picks its curve out of
  the plot: that variable's line keeps its weight and every other fades while you point, so one
  signal is found in a crowd without changing what is drawn (time and Cartesian phase views).
  `all`, `none` and `reset` act on the set. The
  **filter** box by the `variables` heading narrows the list to the names that contain what you
  type (case-insensitive), and `all` / `none` then act on just what it shows -- a component
  model flattens to dozens of connector-internal variables (`supply.p.i`, `converter.flange.tau`,
  the `der_` rates), and filtering to `der_` and clicking `none` drops all the rates at once. The
  heading shows the shown-of-total count while a filter is active; a filtered-out variable that is
  selected still draws. The legend pane resizes by dragging its splitter and collapses on a
  double-click. Below 780 px of panel width it is hidden by default and a `legend` button appears
  in the toolbar: it opens the legend as a strip under the chart, with every section reachable.

### Parameters and comparison

- **Parameters** lists every parameter with its value, unit and description. Edit a value and
  press Enter to re-run; the undo arrow restores the model's own. A parameter whose binding is
  an expression of others is *derived* and shown read-only, since it follows what it depends on
  (`m_dry` follows `m_green`). Overrides go through the simulator's sweep vector, so everything
  downstream is recomputed.
- **`+ compare`** pins the current run: it keeps drawing, dashed, while you change parameters,
  and the legend shows each run's value at the cursor -- and, with exactly two runs side by side,
  their difference (`Δ`), which is the effect of the change read straight off. Pinned runs appear as chips above the
  chart, each naming what it overrides, with a `×` that drops the run. **Click a chip** to edit
  that run's parameters: the panel switches to that instance (its title shows the letter), and a
  change re-runs only it, so A and B can hold different parameters. Pinned runs extend with the
  stop the same way the live one does.

### Studies

- **Step response.** The **`step`** toggle reads the classic step metrics off each drawn variable:
  its **final** value (the mean of the last 2 % of the run), its **overshoot** past that, its
  **`tr`** 10-90 % rise time and its **`ts`** 2 % settling time -- the numbers a controller is
  judged by, and the reason the servo drive and the roaster earn their keep. They are listed under
  **step response** in the legend, and the latest settling instant is marked on the plot. Only a
  variable that actually changed level gets the overshoot-and-time line; a transient that returns
  near where it started (an armature current that spikes then decays) shows its final value alone,
  since its overshoot against a near-zero move would be a meaningless thousands of percent. Time
  view only.
- **Stats.** The **`stats`** toggle reads, for each drawn variable over the window on screen, its
  **min**, its **peak** (with the time the peak occurs) and its **mean** -- the peak current, the
  mean power, the lowest temperature in the region you are looking at. Zooming the time axis
  narrows the window and the numbers follow it; the heading names the span they cover. Time view only.
- **Play.** The **`▶`** button (or the **space** bar) sweeps the cursor through time on its own, so
  a model whose state is an array field animates: the 1-D spatial-profile line and the 2-D heatmap
  that a hover draws now breathe without a hand on the mouse -- a standing wave swinging, a heat
  field decaying, a tracer circulating a ring. It loops, sweeps the visible window when zoomed, and
  pauses on a click or the space bar. Time view only.
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

### Saving the run

**`csv`** saves the drawn run to a CSV file (a save dialog opens, defaulting beside the `.mo`):
a `time` column and one column per variable, in SI units -- the honest data behind the chart,
not the display units. The header names the variables as the model does and the clock column is
`time`, so the file re-imports through `log…` as a measured log; a gap in a series is an empty
cell.

**`svg`** saves the chart itself as an SVG image, for a report or a slide. It is self-contained:
every colour is inlined and an opaque background is laid behind it, so it renders the same
anywhere, with no dependence on the panel's stylesheet or theme. The legend is not part of the
image; what is saved is the plot as drawn, at its current zoom, scale and units.

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
