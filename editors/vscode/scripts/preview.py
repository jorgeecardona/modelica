#!/usr/bin/env python
"""Build a standalone preview of the Scope webview, for a browser and for Playwright.

    uv run python editors/vscode/scripts/preview.py --data run.json --out preview.html [--gallery] [--host]

``run.json`` is what ``python/simulate_file.py`` prints (or ``media/scope.data.json`` with
``--gallery``). The template's placeholders are filled the way `extension.js` fills them, so
the page is the panel VS Code would show, minus VS Code. Serve the output over HTTP -- the
webview's content-security policy blocks ``file://``:

    python3 -m http.server 8731

``--host`` adds a stand-in for the extension host: `acquireVsCodeApi` exists, every
``simulate`` message the panel posts is recorded in ``window.__posted`` and answered from the
embedded record -- an override scales every series, ``--until`` returns a note and a longer
stop, ``--sensitivity`` returns one row per editable parameter, and a ``resume`` checkpoint
returns only a flat tail past it -- so the compare, sweep, sensitivity, automatic-stop and
continuation flows run end to end in a plain browser.
"""

# A build script: printing what it wrote is the point.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import json
from pathlib import Path

MEDIA = Path(__file__).resolve().parents[1] / "media"

HOST = """\
window.__posted = [];
window.acquireVsCodeApi = () => ({ postMessage: (m) => { window.__posted.push(m); setTimeout(() => window.__answer(m), 25); } });
window.__answer = (m) => {
  if (m.type !== "simulate") return;
  const base = JSON.parse(document.getElementById("data").textContent).models[0];
  const sets = m.sets || {};
  const p0 = Object.fromEntries((base.parameters || []).map((p) => [p.name, p.value]));
  const key = Object.keys(sets)[0];
  const f = key && p0[key] ? sets[key] / p0[key] : 1;
  const rec = JSON.parse(JSON.stringify(base));
  rec.overrides = sets;
  rec.parameters = (base.parameters || []).map((p) => ({ ...p, value: sets[p.name] !== undefined ? sets[p.name] : p.value }));
  for (const k of Object.keys(rec.series)) rec.series[k] = rec.series[k].map((v) => (v === null ? null : v * f));
  rec.stop = m.stop || base.stop;
  // Echo the tolerance the panel asked for, so the speed control's badge and re-run can be seen
  // (the stand-in does not actually re-integrate at it).
  rec.tolerance = typeof m.tolerance === "number" ? m.tolerance : null;
  if (rec.solver && rec.tolerance !== null) rec.solver = { ...rec.solver, tolerance: rec.tolerance };
  rec.autoNote = m.until ? (m.until === "event" ? "next event: (stand-in)" : "signals settled") : "";
  if (m.until) rec.stop = (m.from || base.stop) * 1.25;
  if (m.resume && typeof m.resume.t === "number" && rec.stop > m.resume.t) {
    // A continued run: only the tail past the checkpoint, flat at the last values, so the
    // panel's stitching and the stop marker's move can be watched without a real solver.
    const t0 = m.resume.t, k = 20;
    rec.t = Array.from({ length: k + 1 }, (_, i) => t0 + ((rec.stop - t0) * i) / k);
    for (const name of Object.keys(rec.series)) {
      const col = base.series[name]; const last = col.length ? col[col.length - 1] : null;
      rec.series[name] = rec.t.map(() => (last === null ? null : last * f));
    }
    rec.events = [];
    rec.resumedFrom = t0;
    rec.checkpoint = { t: rec.stop, y: m.resume.y, w: m.resume.w };
  }
  if (m.sensitivity) {
    const editable = rec.parameters.filter((p) => !p.derived && p.value);
    rec.sensitivity = { response: m.sensitivity, step: m.sstep || 0.05, base: 100,
      rows: editable.map((p, i) => ({ name: p.name, unit: p.unit, value: p.value, minus: 100 - (i + 1) * 3, plus: 100 + (i + 1) * 2 })) };
  }
  window.dispatchEvent(new MessageEvent("message", { data: { type: "data", models: [rec], runId: m.runId, sets } }));
};
"""


def build(data: str, *, live: bool, host: bool, version: str) -> str:
    """The filled template: what `extension.js`'s `render` produces, plus the optional stand-in."""
    template = (MEDIA / "scope.html").read_text(encoding="utf-8")
    page = (
        template.replace("__NONCE__", "preview")
        .replace("__LIVE__", "true" if live else "false")
        .replace("__VERSION__", version)
        .replace("__DATA__", data)
    )
    if host:
        # Before the panel's own script, which asks for `acquireVsCodeApi` as it starts.
        page = page.replace(
            '<script id="data"',
            '<script nonce="preview">' + HOST + '</script>\n<script id="data"',
            1,
        )
    return page


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a standalone preview of the Scope webview.")
    ap.add_argument("--data", required=True, help="a simulate_file.py JSON, or scope.data.json")
    ap.add_argument("--out", required=True, help="where to write the HTML")
    ap.add_argument("--gallery", action="store_true", help="corpus gallery mode (no live controls)")
    ap.add_argument("--host", action="store_true", help="add a stand-in extension host")
    args = ap.parse_args()
    version = json.loads((MEDIA.parent / "package.json").read_text(encoding="utf-8"))["version"]
    data = Path(args.data).read_text(encoding="utf-8")
    Path(args.out).write_text(
        build(data, live=not args.gallery, host=args.host, version=version), encoding="utf-8"
    )
    print(
        f"wrote {args.out} (v{version}, {'gallery' if args.gallery else 'live'}{', with host' if args.host else ''})"
    )


if __name__ == "__main__":
    main()
