// pymodelica Simulation Scope -- a VS Code webview over the model corpus.
//
// Plain JavaScript on purpose: the extension host reads a self-contained HTML page and hands
// it to a webview. No bundler, no framework, no build step -- the same choice `modelica.sim.view`
// makes for the notebook widget, and the same chart engine renders in both. The webview is
// sandboxed, so the page's inline script runs only under a per-open nonce named in the CSP.

const vscode = require("vscode");
const fs = require("fs");
const path = require("path");

function nonce() {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let s = "";
  for (let i = 0; i < 24; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}

function activate(context) {
  const command = vscode.commands.registerCommand("pymodelica.openScope", () => {
    const panel = vscode.window.createWebviewPanel(
      "pymodelicaScope",
      "Simulation Scope",
      vscode.ViewColumn.Active,
      { enableScripts: true, retainContextWhenHidden: true }
    );
    const media = path.join(context.extensionPath, "media");
    const template = fs.readFileSync(path.join(media, "scope.html"), "utf8");
    const data = fs.readFileSync(path.join(media, "scope.data.json"), "utf8");
    const n = nonce();
    // Function replacers so a `$` inside a model name or the data is a literal, not a group ref.
    panel.webview.html = template
      .split("__NONCE__")
      .join(n)
      .replace("__DATA__", () => data);
  });
  context.subscriptions.push(command);
}

function deactivate() {}

module.exports = { activate, deactivate };
