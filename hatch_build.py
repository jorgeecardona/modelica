"""Build the engine (``engine/``) into the wheel when ``cargo`` is available.

The engine is a C-ABI shared library that `modelica.sim.engine` opens with ``ctypes``. It does not
link against Python, so a wheel that carries it is tagged ``py3-none-<platform>``: one wheel per
platform serves every Python version. Without ``cargo`` the wheel is the pure ``py3-none-any`` one,
and `modelica.sim.engine` builds the engine from source on first use if ``rustc`` is there.

Environment:

- ``MODELICA_ENGINE=0`` builds the pure wheel even when ``cargo`` exists.
- ``MODELICA_ENGINE=1`` fails the build when ``cargo`` is missing.
- ``MODELICA_WHEEL_PLATFORM`` sets the platform tag, for example ``manylinux_2_28_x86_64`` in a
  manylinux image. The default is the build machine's (``linux_x86_64``).
- ``MODELICA_ENGINE_LIBRARY`` is a library built elsewhere, for example cross-compiled for
  Windows. It goes in the wheel as it is, and ``cargo`` does not run. Set the platform tag with
  it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

LIBRARY = {"linux": "libmodelica_engine.so", "darwin": "libmodelica_engine.dylib"}
WINDOWS = "modelica_engine.dll"


class EngineHook(BuildHookInterface):  # pyright: ignore[reportMissingTypeArgument]
    """Compile ``engine/`` with ``cargo`` and put the library at ``modelica/_engine.<suffix>``."""

    PLUGIN_NAME = "engine"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Add the compiled engine to a wheel build; leave sdists and editables alone."""
        if self.target_name != "wheel" or version == "editable":
            return
        wanted = os.environ.get("MODELICA_ENGINE", "")
        if wanted == "0":
            return
        prebuilt = os.environ.get("MODELICA_ENGINE_LIBRARY")
        if prebuilt:
            library = Path(prebuilt).resolve()
            tag = os.environ.get("MODELICA_WHEEL_PLATFORM")
            if not tag:
                message = "MODELICA_ENGINE_LIBRARY needs MODELICA_WHEEL_PLATFORM"
                raise RuntimeError(message)
            build_data["force_include"][str(library)] = f"modelica/_engine{library.suffix}"
            build_data["pure_python"] = False
            build_data["tag"] = f"py3-none-{tag}"
            return
        cargo = shutil.which("cargo")
        if cargo is None:
            if wanted == "1":
                message = "MODELICA_ENGINE=1 but there is no cargo on PATH"
                raise RuntimeError(message)
            return
        root = Path(self.root)
        manifest = root / "engine" / "Cargo.toml"
        subprocess.run(  # noqa: S603 - cargo from PATH, on this repository's own crate
            [cargo, "build", "--release", "--locked", "--manifest-path", str(manifest)],
            check=True,
        )
        platform = sysconfig.get_platform()
        if platform.startswith("win"):
            name, suffix = WINDOWS, ".dll"
        elif platform.startswith("macosx"):
            name, suffix = LIBRARY["darwin"], ".dylib"
        else:
            name, suffix = LIBRARY["linux"], ".so"
        built = root / "engine" / "target" / "release" / name
        build_data["force_include"][str(built)] = f"modelica/_engine{suffix}"
        build_data["pure_python"] = False
        tag = os.environ.get("MODELICA_WHEEL_PLATFORM") or platform.replace("-", "_").replace(
            ".", "_"
        )
        build_data["tag"] = f"py3-none-{tag}"
