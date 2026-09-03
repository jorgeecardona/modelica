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

let panel = null;
let live = null; // { context, file } -- the .mo the panel is currently showing, for re-runs

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
      runHelper(live.context, live.file, Number(msg.stop), (data) => {
        panel.webview.postMessage(
          data.error ? { type: "error", error: data.error } : { type: "data", models: data.models }
        );
      });
    }
  });
  return panel;
}

// Read the webview template, inject a nonce, the data, and whether this is a live .mo (which
// turns on the panel's stop-time control).
function render(context, dataString, isLive) {
  const media = path.join(context.extensionPath, "media");
  const template = fs.readFileSync(path.join(media, "scope.html"), "utf8");
  const n = nonce();
  ensurePanel(context).webview.html = template
    .split("__NONCE__")
    .join(n)
    .split("__LIVE__")
    .join(isLive ? "true" : "false")
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
// the rest of the extension. `stop <= 0` means "let the model decide" (no --stop).
function runtimeCommand(context, cfg, file, stop) {
  const runtime = cfg.get("runtime", "auto");
  if (runtime !== "auto" && runtime !== "python") {
    throw new Error(`runtime '${runtime}' is not available yet; use 'python'`);
  }
  const python = cfg.get("python", ["uv", "run", "--extra", "sim", "python"]);
  const helper = path.join(context.extensionPath, "python", "simulate_file.py");
  const libs = cfg.get("library", []).flatMap((dir) => ["--library", dir]);
  const stopArg = stop > 0 ? ["--stop", String(stop)] : [];
  return { cmd: python[0], args: [...python.slice(1), helper, file, ...stopArg, ...libs] };
}

function runHelper(context, file, stop, done) {
  const cfg = vscode.workspace.getConfiguration("pymodelica");
  const folder = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(file));
  const cwd = cfg.get("cwd") || (folder && folder.uri.fsPath) || path.dirname(file);
  let plan;
  try {
    plan = runtimeCommand(context, cfg, file, stop);
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
  const stop = vscode.workspace.getConfiguration("pymodelica").get("stop", 0);
  runHelper(context, file, Number(stop), (data) => {
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
