#!/bin/sh
# Install the packaged .vsix into every VS Code variant found (stable and Insiders). macOS
# ships no `code`/`code-insiders` on PATH by default, so also look inside the app bundles; if
# none is found, print the Install-from-VSIX UI steps instead of failing.
set -e
cd "$(dirname "$0")/.."

VSIX="$(ls -t ./*.vsix 2>/dev/null | head -1)"
if [ -z "$VSIX" ]; then
  echo "no .vsix here -- run 'pnpm run package' first" >&2
  exit 1
fi

found=""
for cli in \
  "code" \
  "code-insiders" \
  "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
  "/Applications/Visual Studio Code - Insiders.app/Contents/Resources/app/bin/code-insiders" \
  "$HOME/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
  "$HOME/Applications/Visual Studio Code - Insiders.app/Contents/Resources/app/bin/code-insiders" \
  "/usr/share/code/bin/code" \
  "/snap/bin/code"; do
  # `code`/`code-insiders` (bare names) resolve via PATH; absolute paths must be executable.
  case "$cli" in
    /*) [ -x "$cli" ] || continue ;;
    *) command -v "$cli" >/dev/null 2>&1 || continue ;;
  esac
  echo "installing into: $cli"
  "$cli" --install-extension "$VSIX" || true
  found="yes"
done

if [ -z "$found" ]; then
  echo ""
  echo "Built: $(pwd)/${VSIX#./}"
  echo "No 'code' or 'code-insiders' CLI found. Install from the UI instead:"
  echo "  Extensions (Cmd+Shift+X) -> the ... menu -> Install from VSIX... -> ${VSIX#./}"
  echo "  (do this in the SAME window you use -- stable and Insiders are separate installs)"
  exit 0
fi
echo "done. In VS Code: Cmd+Shift+P -> 'pymodelica: Open Simulation Scope'"
