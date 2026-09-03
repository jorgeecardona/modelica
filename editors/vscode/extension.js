// pymodelica Simulation Scope -- a VS Code webview over Modelica models.
//
// Plain JavaScript on purpose: the extension host reads a self-contained HTML page and hands
// it to a webview. No bundler, no framework, no build step -- the same choice
// `modelica.sim.view` makes for the notebook widget, and the same chart engine renders in both.
//
// Two commands: browse the bundled corpus gallery, and simulate the .mo file you have open.
// Simulation goes through a *runtime* -- Python today (`modelica.sim`, the reference tier, no
// per-model compile), with `auto` reserved to pick the fastest available later (a Rust or C
// interpreter over the same flat IR). The webview is sandboxed, so its inline script runs only
// under a per-open nonce named in the CSP.

const vscode = require("vscode");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");

let panel = null;

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
  });
  return panel;
}

function render(context, dataString) {
  const media = path.join(context.extensionPath, "media");
  const template = fs.readFileSync(path.join(media, "scope.html"), "utf8");
  const n = nonce();
  // Function replacers so a `$` in the data or a name is a literal, not a group reference.
  ensurePanel(context).webview.html = template
    .split("__NONCE__")
    .join(n)
    .replace("__DATA__", () => dataString);
}

function openScope(context) {
  const data = fs.readFileSync(
    path.join(context.extensionPath, "media", "scope.data.json"),
    "utf8"
  );
  render(context, data);
}

// The command and arguments for the chosen runtime. Python is the only one today; `auto`
// resolves to it. A Rust/C interpreter over the flat IR slots in here later without touching
// the rest of the extension.
function runtimeCommand(context, cfg, file, stop) {
  const runtime = cfg.get("runtime", "auto");
  if (runtime !== "auto" && runtime !== "python") {
    throw new Error(`runtime '${runtime}' is not available yet; use 'python'`);
  }
  const python = cfg.get("python", ["uv", "run", "--extra", "sim", "python"]);
  const helper = path.join(context.extensionPath, "python", "simulate_file.py");
  const libs = cfg.get("library", []).flatMap((dir) => ["--library", dir]);
  return { cmd: python[0], args: [...python.slice(1), helper, file, "--stop", String(stop), ...libs] };
}

function simulateActive(context, document) {
  const doc = document || (vscode.window.activeTextEditor && vscode.window.activeTextEditor.document);
  if (!doc || !doc.fileName.endsWith(".mo")) {
    vscode.window.showWarningMessage("Open a .mo file to simulate it.");
    return;
  }
  const file = doc.fileName;
  const cfg = vscode.workspace.getConfiguration("pymodelica");
  const stop = cfg.get("stop", 10);
  const folder = vscode.workspace.getWorkspaceFolder(doc.uri);
  const cwd = cfg.get("cwd") || (folder && folder.uri.fsPath) || path.dirname(file);

  let plan;
  try {
    plan = runtimeCommand(context, cfg, file, stop);
  } catch (error) {
    vscode.window.showErrorMessage(String(error.message || error));
    return;
  }

  vscode.window.withProgress(
    { location: vscode.ProgressLocation.Window, title: `Simulating ${path.basename(file)}…` },
    () =>
      new Promise((resolve) => {
        execFile(
          plan.cmd,
          plan.args,
          { cwd, maxBuffer: 64 * 1024 * 1024 },
          (err, stdout, stderr) => {
            if (err && !stdout) {
              vscode.window.showErrorMessage(
                `Could not run the simulator (${plan.cmd}). ${stderr || err.message}`
              );
              resolve();
              return;
            }
            let data;
            try {
              data = JSON.parse(stdout);
            } catch (_e) {
              vscode.window.showErrorMessage(`Simulator returned no JSON. ${stderr || ""}`.trim());
              resolve();
              return;
            }
            if (data.error) {
              vscode.window.showErrorMessage(`${path.basename(file)}: ${data.error}`);
              resolve();
              return;
            }
            render(context, JSON.stringify(data));
            resolve();
          }
        );
      })
  );
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
