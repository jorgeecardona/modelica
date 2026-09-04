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

# Every CLI worth trying, one per line (paths can hold spaces). PATH names first; then, on
# macOS, whatever app bundles Spotlight knows about -- wherever they were dragged to, including
# ~/Downloads, which is where an unmoved download runs from -- checking both binary names, since
# the Insiders bundle ships its CLI as `bin/code`; then the fixed Linux locations.
candidates() {
  echo "code"
  echo "code-insiders"
  if command -v mdfind >/dev/null 2>&1; then
    for id in com.microsoft.VSCode com.microsoft.VSCodeInsiders; do
      mdfind "kMDItemCFBundleIdentifier == '$id'" 2>/dev/null | while IFS= read -r app; do
        for bin in code code-insiders; do
          [ -x "$app/Contents/Resources/app/bin/$bin" ] && echo "$app/Contents/Resources/app/bin/$bin"
        done
      done
    done
  fi
  # Fixed places too, since Spotlight can be paused or its index in an "unknown state" for a
  # folder (seen on ~/Downloads): the two Applications folders, and Downloads itself, which is
  # where an unmoved download runs from.
  for app in "/Applications" "$HOME/Applications" "$HOME/Downloads"; do
    for name in "Visual Studio Code" "Visual Studio Code - Insiders"; do
      for bin in code code-insiders; do
        echo "$app/$name.app/Contents/Resources/app/bin/$bin"
      done
    done
  done
  echo "/usr/share/code/bin/code"
  echo "/snap/bin/code"
}

# Read the list from a file rather than a pipe, so `found` set in the loop survives it.
list="$(mktemp)"
candidates | awk '!seen[$0]++' > "$list"
found=""
while IFS= read -r cli; do
  # A bare name resolves via PATH; an absolute path must be executable.
  case "$cli" in
    /*) [ -x "$cli" ] || continue ;;
    *) command -v "$cli" >/dev/null 2>&1 || continue ;;
  esac
  echo "installing into: $cli"
  "$cli" --install-extension "$VSIX" --force || true
  found="yes"
done < "$list"
rm -f "$list"

if [ -z "$found" ]; then
  echo ""
  echo "Built: $(pwd)/${VSIX#./}"
  echo "No 'code' or 'code-insiders' CLI found. Install from the UI instead:"
  echo "  Extensions (Cmd+Shift+X) -> the ... menu -> Install from VSIX... -> ${VSIX#./}"
  echo "  (do this in the SAME window you use -- stable and Insiders are separate installs)"
  exit 0
fi
echo "done. In VS Code: Cmd+Shift+P -> 'pymodelica: Open Simulation Scope'"
