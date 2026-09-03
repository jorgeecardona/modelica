#!/bin/sh
# Install the packaged .vsix into VS Code. Finds the `code` CLI even when it is not on PATH
# (macOS binds nothing to install it by default), and if it truly cannot be found, prints how
# to install from the UI instead of failing the script.
set -e
cd "$(dirname "$0")/.."

VSIX="$(ls -t ./*.vsix 2>/dev/null | head -1)"
if [ -z "$VSIX" ]; then
  echo "no .vsix here -- run 'pnpm run package' first" >&2
  exit 1
fi

CODE="$(command -v code 2>/dev/null || true)"
if [ -z "$CODE" ]; then
  for candidate in \
    "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
    "/Applications/Visual Studio Code - Insiders.app/Contents/Resources/app/bin/code" \
    "$HOME/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
    "/usr/share/code/bin/code" \
    "/usr/bin/code" \
    "/snap/bin/code"; do
    if [ -x "$candidate" ]; then CODE="$candidate"; break; fi
  done
fi

if [ -z "$CODE" ]; then
  echo ""
  echo "Built: $(pwd)/${VSIX#./}"
  echo "The 'code' CLI was not found, so install it from the UI:"
  echo "  Extensions (Cmd+Shift+X) -> the ... menu -> Install from VSIX... -> ${VSIX#./}"
  echo "Or add the CLI once: VS Code -> Cmd+Shift+P -> \"Shell Command: Install 'code' command in PATH\", then rerun."
  exit 0
fi

"$CODE" --install-extension "$VSIX"
echo "Installed. In VS Code: Cmd+Shift+P -> 'pymodelica: Open Simulation Scope'"
