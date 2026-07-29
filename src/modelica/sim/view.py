"""A simulation result you can poke at, in a notebook.

A trajectory is a picture long before it is a table, and the questions people actually ask
of one -- what is `h` when the ball first bounces, does `c.v` really settle, why is that
step so small there -- are questions about a *place* in the plot. So this renders the
result as a small interactive chart: pick variables, drag to zoom, and read every selected
variable at the cursor.

Two deliberate constraints:

* **No new dependencies.** Not matplotlib, not ipywidgets, not plotly. The chart is SVG
  drawn by a few hundred lines of vanilla JavaScript embedded in the output itself. A
  result stays readable after the kernel dies, after `nbconvert`, and in any front-end
  that renders HTML.
* **Rendered inside an ``<iframe srcdoc>``.** This is the shape the documented behaviour
  points at, not a guess. IPython's display protocol says an object shows itself by
  returning a string from ``_repr_html_``, and JupyterLab only runs ``<script>`` tags in
  *trusted* output -- ``renderHTML`` calls ``evalInnerHTMLScriptTags`` under
  ``if (trusted)``, because "this is not done automatically by the browser when script
  tags are created by setting ``innerHTML``". An iframe sidesteps that entirely: its
  ``srcdoc`` is a document in its own right, so the browser parses and runs it wherever
  the tag survives -- JupyterLab, Notebook 7, VS Code, Colab, ``nbconvert --to html``. It
  is also what folium and branca have done for years. And it isolates: no CSS, no ids and
  no globals of ours can reach the notebook's own page.

The one place this loses is an **untrusted** notebook, where JupyterLab sanitizes HTML
output against an allow-list that does not include ``iframe``. Sanitization drops the tag
but keeps its text, so :meth:`View.embed` puts a sentence inside the tag saying how to get
the chart back -- a blank cell with no explanation would be the worse failure.

The chart is a viewer, not the data: values are written out to 8 significant digits and a
long run is thinned to :attr:`View.limit` samples (keeping both sides of every event, so
a discontinuity stays vertical). When that happens it says so in the header. Anything that
needs the real numbers should ask :class:`~modelica.sim.result.Result` for them.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from numpy.typing import NDArray

    from modelica.sim.result import Result

__all__ = ["View"]

LIMIT = 4000
"""Samples drawn before thinning. Roughly a pixel each on a wide screen."""

INITIAL = 6
"""Variables ticked on to begin with -- the moving ones, see :func:`_opening`.

The rest are one click away in the legend, which lists every variable the result has.
"""

HEIGHT = 420
"""Pixels. Tall enough to read, short enough that a notebook of them still scrolls."""

CELLS = 250_000
"""Numbers one page may carry.

Samples times variables, and the reason :attr:`View.limit` is an upper bound rather than a
promise: the payload is written into the notebook file, so a model with four hundred
variables gets fewer samples rather than a ten-megabyte cell. The header says how many.
"""


@dataclass(frozen=True)
class View:
    """One result, one chart. Displays itself in a notebook; writes itself to a file."""

    result: Result
    names: tuple[str, ...]
    height: int = HEIGHT
    limit: int = LIMIT
    initial: int = INITIAL

    def document(self) -> str:
        """The chart as a standalone HTML page, dependency-free and offline."""
        return _PAGE.replace("__PAYLOAD__", self._payload())

    def save(self, path: str | Path) -> str:
        """Write :meth:`document` to ``path`` and return the path written."""
        target = str(path)
        with open(target, "w", encoding="utf-8") as file:  # noqa: PTH123 - str or Path, both
            file.write(self.document())
        return target

    def embed(self) -> str:
        """The chart as one ``<iframe>`` tag, for dropping into a page of other things.

        The sentence inside the tag is not decoration. A browser ignores the children of
        an ``<iframe>``; JupyterLab's sanitizer, which an untrusted notebook applies and
        which has no ``iframe`` in its allow-list, drops the tag and keeps them. So the
        one reader who cannot see the chart is the one who gets told why.
        """
        doc = html.escape(self.document(), quote=True)
        fallback = html.escape(
            f"[{self.result.name}: chart hidden. This notebook is not trusted, and an "
            f"untrusted notebook strips the frame the chart lives in. Trust it, or write "
            f"the chart to a file with result.plot().save('{self.result.name}.html').]"
        )
        return (
            f'<iframe srcdoc="{doc}" sandbox="allow-scripts" loading="lazy" '
            f'style="width:100%;height:{self.height}px;border:0;display:block">'
            f"{fallback}</iframe>"
        )

    def _repr_html_(self) -> str:
        """What Jupyter asks for."""
        return self.embed()

    def _payload(self) -> str:
        r"""Everything the page needs, as JSON embedded in a script tag.

        ``</`` is escaped because the JSON sits inside ``<script>``: a variable named in a
        way that spells ``</script>`` would otherwise end the tag early. ``<\/`` is a legal
        escape inside a JSON string and means exactly the same thing to the parser.
        """
        budget = max(2, min(self.limit, CELLS // max(len(self.names), 1)))
        keep = _thin(self.result.time, budget)
        time = self.result.time[keep]
        window = (float(time[0]), float(time[-1])) if len(time) else (0.0, 1.0)
        rows = [(name, _numbers(self.result[name][keep])) for name in self.names]
        data = {
            "name": self.result.name,
            "time": _numbers(time),
            "series": [{"name": name, "values": values} for name, values in rows],
            "events": [
                {"t": float(event.time), "cause": event.cause}
                for event in self.result.events
                if window[0] <= event.time <= window[1]
            ],
            "shown": len(keep),
            "total": len(self.result.time),
            "initial": _opening(rows, self.initial),
        }
        return json.dumps(data, allow_nan=False).replace("</", "<\\/")


def view(result: Result, names: Iterable[str] = (), **options: int) -> View:
    """Build a :class:`View` of ``result``, defaulting to every variable it has."""
    chosen = tuple(names) or result.names
    missing = [name for name in chosen if name not in result]
    if missing:
        raise KeyError(f"{result.name} has no variable {missing[0]!r}")
    return View(result, chosen, **options)


def _opening(rows: list[tuple[str, list[float | None]]], count: int) -> list[str]:
    """Which variables are ticked on when the chart opens: the ones that move.

    A flat model declares its parameters wherever the modeller wrote them, which is
    usually first, and a chart that opens on six horizontal lines invites the reader to
    close it. Constants keep their place in the legend and fill the opening selection
    only if the model has fewer moving variables than there is room for.
    """
    moving = [name for name, values in rows if len({v for v in values if v is not None}) > 1]
    known = set(moving)
    return [*moving, *(name for name, _ in rows if name not in known)][:count]


def _thin(time: NDArray[np.float64], limit: int) -> NDArray[np.intp]:
    """Which samples to draw: a stride, plus the ones that carry the discontinuities.

    A plain stride is the wrong answer on its own. A bouncing ball spends most of its
    samples in the smooth arcs and all of its meaning in the instant the sign of the
    velocity flips; drop the two samples that straddle a bounce and the plot shows a
    ball that eases through the floor. Every segment boundary -- where the time array
    repeats a value, because the integrator stopped and restarted there -- is kept.
    """
    count = len(time)
    if count <= limit:
        return np.arange(count)
    stride = int(np.ceil(count / limit))
    breaks = np.flatnonzero(np.diff(time) <= 0.0)
    ends = np.array([0, max(count - 1, 0)], dtype=np.intp)
    keep = np.union1d(np.arange(0, count, stride), np.union1d(np.union1d(breaks, breaks + 1), ends))
    return keep.astype(np.intp)


def _numbers(values: NDArray[np.float64]) -> list[float | None]:
    """A row, rounded for the page, with the non-finite ones as ``null``.

    ``null`` rather than ``NaN`` because JSON has no NaN and the chart wants the break
    anyway: a gap in the line is the honest way to draw a value that does not exist.
    """
    finite: list[bool] = np.isfinite(values).tolist()
    numbers: list[float] = np.where(np.isfinite(values), values, 0.0).tolist()
    return [float(f"{v:.8g}") if ok else None for v, ok in zip(numbers, finite, strict=True)]


_PAGE = """<!doctype html>
<meta charset="utf-8">
<style>
  :root { color-scheme: light dark; --ink: #1a1a1a; --dim: #6b6b6b; --line: #d4d4d4;
          --bg: #ffffff; --panel: #f6f6f6; }
  @media (prefers-color-scheme: dark) {
    :root { --ink: #e6e6e6; --dim: #9a9a9a; --line: #3a3a3a; --bg: #1b1b1b; --panel: #242424; }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { background: var(--bg); color: var(--ink); display: flex; flex-direction: column;
         font: 12px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .head { display: flex; align-items: center; gap: 8px; padding: 6px 8px;
          border-bottom: 1px solid var(--line); }
  .title { font-weight: 600; }
  .note, .foot { color: var(--dim); }
  .tools { margin-left: auto; display: flex; gap: 4px; }
  button, select { font: inherit; color: inherit; background: var(--panel);
                   border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px;
                   cursor: pointer; }
  button:hover { border-color: var(--dim); }
  .body { flex: 1; display: flex; min-height: 0; }
  svg { flex: 1; min-width: 0; display: block; touch-action: none; }
  .legend { width: 200px; overflow-y: auto; border-left: 1px solid var(--line);
            padding: 4px 0; }
  .row { display: flex; align-items: baseline; gap: 6px; padding: 1px 8px; cursor: pointer;
         white-space: nowrap; }
  .row:hover { background: var(--panel); }
  .row.off { opacity: .4; }
  .swatch { width: 9px; height: 9px; border-radius: 2px; flex: none; }
  .row .name { overflow: hidden; text-overflow: ellipsis; }
  .row .value { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--dim); }
  .foot { padding: 4px 8px; border-top: 1px solid var(--line); height: 24px;
          overflow: hidden; white-space: nowrap; }
  .axis { stroke: var(--line); }
  .tick { fill: var(--dim); font-size: 10px; }
  .grid { stroke: var(--line); stroke-dasharray: 2 3; opacity: .7; }
  .event { stroke: var(--dim); stroke-dasharray: 4 3; }
  .cursor { stroke: var(--dim); }
  .band { fill: var(--dim); opacity: .18; }
</style>
<div class="head">
  <span class="title" id="title"></span>
  <span class="note" id="note"></span>
  <span class="tools">
    <select id="scale" title="how the vertical axis is built">
      <option value="linear">linear</option>
      <option value="log">log |y|</option>
      <option value="norm">normalised</option>
    </select>
    <button id="all">all</button>
    <button id="none">none</button>
    <button id="reset" title="or double-click the chart">reset</button>
  </span>
</div>
<div class="body">
  <svg id="chart"><g id="plot"></g><g id="hud"></g></svg>
  <div class="legend" id="legend"></div>
</div>
<div class="foot" id="foot">drag to zoom, double-click to reset</div>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const PALETTE = ["#4c78a8","#f58518","#54a24b","#e45756","#72b7b2","#eeca3b",
                 "#b279a2","#ff9da6","#9d755d","#bab0ac"];
const M = {l: 62, r: 10, t: 8, b: 22};
const svg = document.getElementById("chart");
const plot = document.getElementById("plot");
const hud = document.getElementById("hud");
const legend = document.getElementById("legend");
const foot = document.getElementById("foot");
const T = D.time;
const N = T.length;

let on = new Set(D.initial);
let scale = "linear";
let win = N ? [T[0], T[N - 1]] : [0, 1];
let drag = null;
let box = {w: 0, h: 0};

document.getElementById("title").textContent = D.name;
document.getElementById("note").textContent =
  (D.shown < D.total ? D.shown + " of " + D.total + " samples" : D.total + " samples") +
  ", " + D.series.length + " variables" +
  (D.events.length ? ", " + D.events.length + " event" + (D.events.length > 1 ? "s" : "") : "");

// -- helpers ---------------------------------------------------------------------------

function fmt(v) {
  if (v === null || v === undefined || !isFinite(v)) return "--";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-4 || a >= 1e6)) return v.toExponential(4);
  return String(Number(v.toPrecision(8)));
}

function ticks(lo, hi, want) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / want;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  const step = (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}

function span(lo, hi) {           // never a zero-height axis
  if (!(isFinite(lo) && isFinite(hi))) return [0, 1];
  if (hi > lo) { const pad = (hi - lo) * 0.05; return [lo - pad, hi + pad]; }
  const pad = Math.abs(lo) * 0.05 || 1;
  return [lo - pad, lo + pad];
}

function index(t) {               // first sample at or after t (time is non-decreasing)
  let lo = 0, hi = N - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (T[mid] < t) lo = mid + 1; else hi = mid; }
  return lo;
}

function visible() {
  const a = Math.max(0, index(win[0]) - 1);
  const b = Math.min(N - 1, index(win[1]) + 1);
  return [a, b];
}

function selected() {
  return D.series.filter((s) => on.has(s.name));
}

function el(tag, attrs, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (text !== undefined) node.textContent = text;
  return node;
}

// -- scales ----------------------------------------------------------------------------

function bounds() {
  const [a, b] = visible();
  const rows = selected();
  if (scale === "norm") return [0, 1];
  let lo = Infinity, hi = -Infinity;
  for (const s of rows) {
    for (let i = a; i <= b; i++) {
      let v = s.values[i];
      if (v === null) continue;
      if (scale === "log") { v = Math.abs(v); if (!(v > 0)) continue; v = Math.log10(v); }
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!isFinite(lo)) return [0, 1];
  return span(lo, hi);
}

function extent(s) {              // per-series range, for "normalised"
  const [a, b] = visible();
  let lo = Infinity, hi = -Infinity;
  for (let i = a; i <= b; i++) {
    const v = s.values[i];
    if (v === null) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}

function project(s, dom) {
  if (scale === "norm") { const [lo, hi] = extent(s); return (v) => (v - lo) / (hi - lo); }
  if (scale === "log") return (v) => (Math.log10(Math.abs(v)) - dom[0]) / (dom[1] - dom[0]);
  return (v) => (v - dom[0]) / (dom[1] - dom[0]);
}

const X = (t) => M.l + ((t - win[0]) / (win[1] - win[0])) * (box.w - M.l - M.r);
const Y = (u) => box.h - M.b - u * (box.h - M.t - M.b);

// -- drawing ---------------------------------------------------------------------------

function draw() {
  box = {w: svg.clientWidth, h: svg.clientHeight};
  svg.setAttribute("viewBox", "0 0 " + box.w + " " + box.h);
  const dom = bounds();
  const rows = selected();
  const frag = document.createDocumentFragment();

  for (const t of ticks(win[0], win[1], Math.max(2, Math.round(box.w / 90)))) {
    const x = X(t);
    if (x < M.l - 0.5 || x > box.w - M.r + 0.5) continue;
    frag.appendChild(el("line", {class: "grid", x1: x, x2: x, y1: M.t, y2: box.h - M.b}));
    frag.appendChild(el("text", {class: "tick", x: x, y: box.h - 8, "text-anchor": "middle"},
                        fmt(t)));
  }
  if (scale !== "norm") {
    for (const v of ticks(dom[0], dom[1], Math.max(2, Math.round(box.h / 44)))) {
      const y = Y((v - dom[0]) / (dom[1] - dom[0]));
      if (y < M.t - 0.5 || y > box.h - M.b + 0.5) continue;
      frag.appendChild(el("line", {class: "grid", x1: M.l, x2: box.w - M.r, y1: y, y2: y}));
      frag.appendChild(el("text", {class: "tick", x: M.l - 6, y: y + 3, "text-anchor": "end"},
                          scale === "log" ? "1e" + fmt(v) : fmt(v)));
    }
  }
  frag.appendChild(el("line", {class: "axis", x1: M.l, x2: M.l, y1: M.t, y2: box.h - M.b}));
  frag.appendChild(el("line", {class: "axis", x1: M.l, x2: box.w - M.r,
                               y1: box.h - M.b, y2: box.h - M.b}));

  for (const e of D.events) {
    if (e.t < win[0] || e.t > win[1]) continue;
    const x = X(e.t);
    frag.appendChild(el("line", {class: "event", x1: x, x2: x, y1: M.t, y2: box.h - M.b}));
  }

  const [a, b] = visible();
  for (const s of rows) {
    const to = project(s, dom);
    let d = "", pen = false;
    for (let i = a; i <= b; i++) {
      const v = s.values[i];
      let u = v === null ? null : to(v);
      if (u === null || !isFinite(u)) { pen = false; continue; }
      const p = X(T[i]).toFixed(2) + " " + Y(u).toFixed(2);
      d += (pen ? "L" : "M") + p;
      pen = true;
    }
    frag.appendChild(el("path", {d: d, fill: "none", "stroke-width": 1.4,
                                 stroke: colour(s.name), "stroke-linejoin": "round"}));
  }

  plot.replaceChildren(frag);
  hud.replaceChildren();
  paint();
}

function colour(name) {
  return PALETTE[D.series.findIndex((s) => s.name === name) % PALETTE.length];
}

// -- legend, cursor --------------------------------------------------------------------

function paint(at) {
  const frag = document.createDocumentFragment();
  for (const s of D.series) {
    const row = document.createElement("div");
    row.className = "row" + (on.has(s.name) ? "" : " off");
    row.title = s.name;
    const dot = document.createElement("span");
    dot.className = "swatch";
    dot.style.background = colour(s.name);
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = s.name;
    const value = document.createElement("span");
    value.className = "value";
    if (at !== undefined && on.has(s.name)) value.textContent = fmt(s.values[at]);
    row.append(dot, name, value);
    row.onclick = () => { on.has(s.name) ? on.delete(s.name) : on.add(s.name); draw(); };
    frag.appendChild(row);
  }
  legend.replaceChildren(frag);
}

function cursor(t) {
  const i = index(t);
  const j = i > 0 && Math.abs(T[i - 1] - t) < Math.abs(T[i] - t) ? i - 1 : i;
  const x = X(T[j]);
  const frag = document.createDocumentFragment();
  frag.appendChild(el("line", {class: "cursor", x1: x, x2: x, y1: M.t, y2: box.h - M.b}));
  const dom = bounds();
  for (const s of selected()) {
    const v = s.values[j];
    if (v === null) continue;
    const u = project(s, dom)(v);
    if (!isFinite(u)) continue;
    frag.appendChild(el("circle", {cx: x, cy: Y(u), r: 2.5, fill: colour(s.name)}));
  }
  hud.replaceChildren(frag);
  const near = D.events.filter((e) => Math.abs(X(e.t) - x) < 4);
  foot.textContent = "t = " + fmt(T[j]) + (near.length ? "   [" + near[0].cause + "]" : "");
  paint(j);
}

// -- interaction -----------------------------------------------------------------------

function time(event) {
  const r = svg.getBoundingClientRect();
  const x = Math.min(Math.max(event.clientX - r.left, M.l), r.width - M.r);
  return win[0] + ((x - M.l) / (r.width - M.l - M.r)) * (win[1] - win[0]);
}

svg.addEventListener("pointermove", (e) => {
  if (drag !== null) {
    const t = time(e);
    const x0 = X(Math.min(drag, t)), x1 = X(Math.max(drag, t));
    const band = el("rect", {class: "band", x: x0, y: M.t, width: Math.max(x1 - x0, 0),
                             height: box.h - M.t - M.b});
    hud.replaceChildren(band);
    foot.textContent = "zoom to [" + fmt(Math.min(drag, t)) + ", " + fmt(Math.max(drag, t)) + "]";
    return;
  }
  if (N) cursor(time(e));
});
svg.addEventListener("pointerdown", (e) => { drag = time(e); svg.setPointerCapture(e.pointerId); });
svg.addEventListener("pointerup", (e) => {
  const t = time(e);
  const lo = Math.min(drag, t), hi = Math.max(drag, t);
  drag = null;
  if (hi - lo > (win[1] - win[0]) * 1e-4) { win = [lo, hi]; }
  draw();
});
svg.addEventListener("pointerleave", () => {
  if (drag === null) { hud.replaceChildren(); foot.textContent =
    "drag to zoom, double-click to reset"; paint(); }
});
svg.addEventListener("dblclick", () => { win = N ? [T[0], T[N - 1]] : [0, 1]; draw(); });
document.getElementById("reset").onclick = () => { win = N ? [T[0], T[N - 1]] : [0, 1]; draw(); };
document.getElementById("all").onclick = () => { on = new Set(D.series.map((s) => s.name)); draw(); };
document.getElementById("none").onclick = () => { on = new Set(); draw(); };
document.getElementById("scale").onchange = (e) => { scale = e.target.value; draw(); };
window.addEventListener("resize", draw);
draw();
</script>
"""
