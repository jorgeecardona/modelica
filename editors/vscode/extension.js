// pymodelica Simulation Scope -- a VS Code webview over Modelica models.
//
// Plain JavaScript on purpose: the extension host reads a self-contained HTML page and hands
// it to a webview. No bundler, no framework, no build step -- the same choice
// `modelica.sim.view` makes for the notebook widget, and the same chart engine renders in both.
//
// Two commands: browse the bundled corpus gallery, and simulate the .mo file you have open.
// Simulation goes through a *runtime* -- Python today (`modelica.sim`, the reference tier, no
// per-model compile), with `auto` reserved to pick the fastest available later (a Rust or C
// interpreter over the same flat IR). The panel can re-run at a new stop time: it posts a
// message to the extension, which re-simulates and posts the new trajectory back.

const vscode = require("vscode");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");
const os = require("os");

let panel = null;
let live = null; // { context, file } -- the .mo the panel is currently showing, for re-runs
let resumeCounter = 0; // names the temp file a resume checkpoint travels by, one per call
// The relative tolerance the panel opens at -- "fast", matching scope.html's speed default, so
// the first draw is quick even when the model's .toml asks for a tight reference tolerance.
const DISPLAY_TOLERANCE = 1e-6;

function nonce() {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let s = "";
  for (let i = 0; i < 24; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}

function ensurePanel(context) {
  if (panel) {
    panel.reveal(vscode.ViewColumn.Beside, true);
    return panel;
  }
  panel = vscode.window.createWebviewPanel(
    "pymodelicaScope",
    "Simulation Scope",
    { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
    { enableScripts: true, retainContextWhenHidden: true }
  );
  panel.onDidDispose(() => {
    panel = null;
    live = null;
  });
  panel.webview.onDidReceiveMessage((msg) => {
    if (msg && msg.type === "simulate" && live) {
      // `until` asks the simulator to pick the stop itself (next event past `from`, or where
      // the signals settle) instead of taking the number the panel holds.
      // `sets` are parameter overrides for this run; `runId` says which of the panel's runs
      // (the live one, or a pinned comparison) the answer belongs to, and is echoed back.
      const opts = {
        stop: Number(msg.stop), until: msg.until, from: Number(msg.from), sets: msg.sets || {},
        // A sensitivity study: the response to read off, and the relative step for each parameter.
        sensitivity: msg.sensitivity || null, sstep: Number(msg.sstep) || 0.05,
        // A checkpoint {t, y, w} from the run this one extends: integrate only the new tail.
        resume: msg.resume || null,
        // The relative tolerance the panel asked for; null means the model's reference tolerance.
        tolerance: typeof msg.tolerance === "number" ? msg.tolerance : null,
      };
      runHelper(live.context, live.file, opts, (data) => {
        panel.webview.postMessage(
          data.error
            ? { type: "error", error: data.error, runId: msg.runId }
            : { type: "data", models: data.models, runId: msg.runId, sets: opts.sets }
        );
      });
    } else if (msg && msg.type === "importLog") {
      importLog();
    }
  });
  return panel;
}

// A measured log, overlaid. The panel asks; the extension owns the file dialog and the file
// system, parses the CSV, and posts the columns back. The first column named like time (t,
// time, seconds) is the clock -- the first column otherwise -- and every other numeric column
// is a series named by its header, which the panel matches to the model's variables.
function importLog() {
  if (!panel) return;
  vscode.window
    .showOpenDialog({
      canSelectMany: false,
      openLabel: "Overlay log",
      filters: { "Data files": ["csv", "tsv", "txt"] },
    })
    .then((picked) => {
      if (!picked || !picked.length) return;
      const file = picked[0].fsPath;
      let text;
      try {
        text = fs.readFileSync(file, "utf8");
      } catch (error) {
        panel.webview.postMessage({ type: "error", error: `Could not read ${file}: ${error.message}` });
        return;
      }
      const parsed = parseCsv(text);
      if (!parsed) {
        panel.webview.postMessage({ type: "error", error: `${path.basename(file)}: no numeric columns found` });
        return;
      }
      panel.webview.postMessage({ type: "log", name: path.basename(file), ...parsed });
    });
}

// Comma, semicolon or tab separated, one header row. A semicolon file usually carries comma
// decimals, so those are accepted when the delimiter is not a comma.
function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length);
  if (lines.length < 2) return null;
  const delim = [",", ";", "\t"].sort((a, b) => lines[0].split(b).length - lines[0].split(a).length)[0];
  const header = lines[0].split(delim).map((h) => h.trim().replace(/^"|"$/g, ""));
  const num = (s) => {
    const cleaned = String(s).trim().replace(/^"|"$/g, "");
    return Number(delim === "," ? cleaned : cleaned.replace(",", "."));
  };
  const rows = lines.slice(1).map((line) => line.split(delim).map(num)).filter((r) => r.length === header.length);
  if (!rows.length) return null;
  let tcol = header.findIndex((h) => /^(t|time|seconds?|sec|s|min|minutes?)$/i.test(h));
  if (tcol < 0) tcol = 0;
  const minutes = /^min/i.test(header[tcol]);
  const t = rows.map((r) => (minutes ? r[tcol] * 60 : r[tcol]));
  const series = {};
  header.forEach((h, i) => {
    if (i === tcol) return;
    const col = rows.map((r) => (Number.isFinite(r[i]) ? r[i] : null));
    if (col.some((v) => v !== null)) series[h] = col;
  });
  if (!Object.keys(series).length || t.some((v) => !Number.isFinite(v))) return null;
  return { t, series };
}

// Read the webview template, inject a nonce, the data, and whether this is a live .mo (which
// turns on the panel's stop-time control).
function render(context, dataString, isLive) {
  const media = path.join(context.extensionPath, "media");
  const template = fs.readFileSync(path.join(media, "scope.html"), "utf8");
  const n = nonce();
  // The running extension's own version, so the panel can show which build drew it. After a
  // `.vsix` upgrade the old extension host keeps running until the window reloads; seeing an
  // old version here is how you tell the panel is on stale code.
  const version = require("./package.json").version;
  ensurePanel(context).webview.html = template
    .split("__NONCE__")
    .join(n)
    .split("__LIVE__")
    .join(isLive ? "true" : "false")
    .split("__VERSION__")
    .join(version)
    .replace("__DATA__", () => dataString);
}

function openScope(context) {
  live = null;
  const data = fs.readFileSync(
    path.join(context.extensionPath, "media", "scope.data.json"),
    "utf8"
  );
  render(context, data, false);
}

// The command and arguments for the chosen runtime. Python is the only one today; `auto`
// resolves to it. A Rust/C interpreter over the flat IR slots in here later without touching
// the rest of the extension. `opts.stop <= 0` means "let the model decide" (no --stop);
// `opts.until` ("event" | "settle") has the simulator choose the stop from the signals, with
// `opts.from` the current stop it should look past.
function runtimeCommand(context, cfg, file, opts) {
  const runtime = cfg.get("runtime", "auto");
  if (runtime !== "auto" && runtime !== "python") {
    throw new Error(`runtime '${runtime}' is not available yet; use 'python'`);
  }
  const python = cfg.get("python", ["uv", "run", "--extra", "sim", "python"]);
  const helper = path.join(context.extensionPath, "python", "simulate_file.py");
  const libs = cfg.get("library", []).flatMap((dir) => ["--library", dir]);
  const stopArg = opts.stop > 0 ? ["--stop", String(opts.stop)] : [];
  // The farthest an automatic stop may probe; 0 lets the simulator pick ten times the current
  // stop. A cap, so a model that never settles comes back in seconds rather than running away.
  const max = Number(cfg.get("autoStopMax", 0)) || 0;
  const untilArg = opts.until
    ? ["--until", String(opts.until), "--from", String(opts.from > 0 ? opts.from : 0),
       "--max", String(max > 0 ? max : 0)]
    : [];
  // Parameter overrides, one --set name=value each; the simulator recomputes what depends on them.
  const setArgs = Object.entries(opts.sets || {}).flatMap(([name, value]) => [
    "--set", `${name}=${value}`,
  ]);
  // A sensitivity study runs every editable parameter a step either way and reads a response.
  const sensArgs = opts.sensitivity
    ? ["--sensitivity", String(opts.sensitivity), "--sensitivity-step", String(opts.sstep || 0.05)]
    : [];
  // A resume checkpoint travels by file: the state and workspace vectors are too long for an
  // argument and the simulator reads JSON anyway. One file per call, in the OS temp dir.
  let resumeArgs = [];
  if (opts.resume && typeof opts.resume.t === "number") {
    const ckpt = path.join(os.tmpdir(), `pymodelica-resume-${process.pid}-${++resumeCounter}.json`);
    fs.writeFileSync(ckpt, JSON.stringify(opts.resume));
    resumeArgs = ["--resume", ckpt];
  }
  // The relative tolerance the panel draws at; omitted lets the model's reference tolerance stand.
  const tolArg = typeof opts.tolerance === "number" && opts.tolerance > 0
    ? ["--tolerance", String(opts.tolerance)]
    : [];
  return {
    cmd: python[0],
    args: [...python.slice(1), helper, file, ...stopArg, ...untilArg, ...setArgs, ...sensArgs, ...resumeArgs, ...tolArg, ...libs],
  };
}

function runHelper(context, file, opts, done) {
  const cfg = vscode.workspace.getConfiguration("pymodelica");
  const folder = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(file));
  const cwd = cfg.get("cwd") || (folder && folder.uri.fsPath) || path.dirname(file);
  let plan;
  try {
    plan = runtimeCommand(context, cfg, file, opts);
  } catch (error) {
    done({ error: String(error.message || error) });
    return;
  }
  vscode.window.withProgress(
    { location: vscode.ProgressLocation.Window, title: `Simulating ${path.basename(file)}…` },
    () =>
      new Promise((resolve) => {
        execFile(plan.cmd, plan.args, { cwd, maxBuffer: 64 * 1024 * 1024 }, (err, stdout, stderr) => {
          if (err && !stdout) {
            done({ error: `Could not run the simulator (${plan.cmd}). ${stderr || err.message}` });
          } else {
            try {
              done(JSON.parse(stdout));
            } catch (_e) {
              done({ error: `Simulator returned no JSON. ${stderr || ""}`.trim() });
            }
          }
          resolve();
        });
      })
  );
}

function simulateActive(context, document) {
  const doc = document || (vscode.window.activeTextEditor && vscode.window.activeTextEditor.document);
  if (!doc || !doc.fileName.endsWith(".mo")) {
    vscode.window.showWarningMessage("Open a .mo file to simulate it.");
    return;
  }
  const file = doc.fileName;
  const cfg = vscode.workspace.getConfiguration("pymodelica");
  const stop = cfg.get("stop", 0);
  // The panel opens at the "fast" display tolerance so the first draw is quick even for a
  // chaotic model whose .toml asks for a tight reference tolerance; the speed control in the
  // panel changes it, and DISPLAY_TOLERANCE matches the panel's default so the two agree.
  runHelper(context, file, { stop: Number(stop), tolerance: DISPLAY_TOLERANCE }, (data) => {
    if (data.error) {
      vscode.window.showErrorMessage(`${path.basename(file)}: ${data.error}`);
      return;
    }
    live = { context, file };
    render(context, JSON.stringify(data), true);
  });
}

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("pymodelica.openScope", () => openScope(context)),
    vscode.commands.registerCommand("pymodelica.simulateModel", () => simulateActive(context)),
    vscode.commands.registerCommand("pymodelica.importLog", () => {
      if (!panel) { vscode.window.showWarningMessage("Open the Simulation Scope first, then overlay a log."); return; }
      importLog();
    }),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      const cfg = vscode.workspace.getConfiguration("pymodelica");
      if (cfg.get("simulateOnSave", false) && doc.fileName.endsWith(".mo") && panel) {
        simulateActive(context, doc);
      }
    })
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
