"""
gdal_tools.py

Finds GDAL's command-line tools (ogr2ogr, ogrinfo) of the QGIS install the
plugin runs in, and the environment to run them with.

QGIS on Windows (OSGeo4W) and Linux puts its GDAL tools on the PATH; the
macOS app does not: a program started from the Finder gets the system's
short PATH, and the tools are inside the app bundle (QGIS.app/Contents/...).
They are looked for next to QGIS first (inside its app bundle, whatever the
app is called), so the tools of QGIS's own GDAL are used, then next to
GDAL_DATA, then on the PATH and in common install folders. Each one found is
run once (--version) and used only if it runs: a tool of another GDAL install
that fails to start (e.g. Homebrew's, missing its libraries) is passed over.
"""

import os
import shutil
import subprocess
import sys
import threading
from os.path import abspath, dirname, isdir, isfile, join, normpath
from typing import Dict, List, Optional

from qgis.core import QgsApplication

_FOUND: Dict[str, str] = {}


def _data_env() -> Dict[str, str]:
    """Where this process's GDAL and PROJ read their data. Read once, at
    import (on the importing thread), never alongside other threads' GDAL
    work."""
    found: Dict[str, str] = {}
    try:
        from osgeo import gdal, osr  # pylint: disable=import-outside-toplevel
        gdal_data = gdal.GetConfigOption("GDAL_DATA")
        if gdal_data and isdir(gdal_data):
            found["GDAL_DATA"] = gdal_data
        proj_paths = [p for p in (osr.GetPROJSearchPaths() or []) if isdir(p)]
        if proj_paths:
            found["PROJ_DATA"] = found["PROJ_LIB"] = os.pathsep.join(proj_paths)
    except (ImportError, AttributeError, RuntimeError):
        pass
    return found


_DATA_ENV = _data_env()
_LOCK = threading.Lock()
# How deep below an app bundle's Contents folder a tool is looked for.
_BUNDLE_DEPTH = 4
# Folders other GDAL installs keep their tools in (checked last).
_COMMON_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin")
# Seconds a tool's --version run may take.
_VERSION_TIMEOUT = 30
_CREATE_NO_WINDOW = 0x08000000


def _executable(path: str) -> bool:
    return isfile(path) and os.access(path, os.X_OK)


def _candidate_dirs() -> List[str]:
    """Folders a QGIS install keeps its GDAL tools in, most likely first."""
    roots = [dirname(sys.executable), QgsApplication.prefixPath(), sys.prefix]
    dirs = []
    for root in roots:
        if not root:
            continue
        for sub in ("", "bin", join("..", "bin"), join("..", "..", "bin"), join("..", "MacOS"), join("..", "MacOS", "bin"),
                    join("..", "Frameworks", "bin"), join("..", "Resources", "bin")):
            folder = normpath(join(root, sub))
            if isdir(folder) and folder not in dirs:
                dirs.append(folder)
    return dirs


def _bundle_contents() -> Optional[str]:
    """The Contents folder of the macOS app bundle QGIS runs from, if any."""
    for root in (sys.executable, QgsApplication.prefixPath(), sys.prefix):
        path = abspath(root or "")
        while path and dirname(path) != path:
            if path.endswith(".app"):
                contents = join(path, "Contents")
                return contents if isdir(contents) else None
            path = dirname(path)
    return None


def _search_bundle(contents: str, name: str) -> Optional[str]:
    """``name``'s executable anywhere in the bundle, a few levels deep."""
    base_depth = contents.rstrip(os.sep).count(os.sep)
    for folder, subdirs, files in os.walk(contents):
        if folder.count(os.sep) - base_depth >= _BUNDLE_DEPTH:
            subdirs[:] = []
        if name in files and _executable(join(folder, name)):
            return join(folder, name)
    return None


def _gdal_data_dirs() -> List[str]:
    """Tool folders of the GDAL whose data QGIS uses (.../share/gdal)."""
    gdal_data = _DATA_ENV.get("GDAL_DATA") or os.environ.get("GDAL_DATA")
    if not gdal_data:
        return []
    dirs = []
    for sub in (join("..", "..", "bin"), join("..", "..", "..", "bin"), join("..", "..", "..", "MacOS")):
        folder = normpath(join(gdal_data, sub))
        if isdir(folder) and folder not in dirs:
            dirs.append(folder)
    return dirs


def _candidates(exe: str) -> List[str]:
    """Every ``exe`` executable found, in the order they are tried."""
    found: List[str] = []

    def add(path: Optional[str]) -> None:
        if path and _executable(path) and abspath(path) not in found:
            found.append(abspath(path))

    for folder in _candidate_dirs():
        add(join(folder, exe))
    contents = _bundle_contents()
    if contents:
        add(_search_bundle(contents, exe))
    for folder in _gdal_data_dirs():
        add(join(folder, exe))
    add(shutil.which(exe))
    if os.name != "nt":
        for folder in _COMMON_DIRS:
            add(join(folder, exe))
    return found


def _runs(path: str) -> bool:
    """Whether the tool at ``path`` starts and answers --version."""
    kwargs = {"creationflags": _CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        done = subprocess.run([path, "--version"], env=gdal_tool_env(), stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=_VERSION_TIMEOUT, check=False, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def gdal_tool(name: str) -> str:
    """The path of GDAL's ``name`` tool (e.g. "ogr2ogr"): the first found
    that runs. Raises RuntimeError, saying where it was looked for, if none
    does."""
    with _LOCK:
        if name in _FOUND:
            return _FOUND[name]
        exe = f"{name}.exe" if os.name == "nt" else name
        candidates = _candidates(exe)
        path = next((candidate for candidate in candidates if _runs(candidate)), None)
        if path is None:
            if candidates:
                raise RuntimeError(
                    f"GDAL's {name} tool was found but does not run: {', '.join(candidates)}. "
                    f"QGIS2VectorTiles needs the {name} that comes with QGIS (or GDAL's, on the PATH).")
            raise RuntimeError(
                f"GDAL's {name} tool was not found. Looked in {', '.join(_candidate_dirs())}, "
                f"inside the QGIS app, next to GDAL_DATA and on the PATH. "
                f"QGIS2VectorTiles needs the {name} that comes with QGIS (or GDAL's, on the PATH).")
        _FOUND[name] = path
        return path


def gdal_tool_env() -> Dict[str, str]:
    """This process's environment, with where GDAL and PROJ keep their data
    (as QGIS set them in this process) for a GDAL tool started from it."""
    env = os.environ.copy()
    if "GDAL_DATA" not in env and "GDAL_DATA" in _DATA_ENV:
        env["GDAL_DATA"] = _DATA_ENV["GDAL_DATA"]
    if "PROJ_DATA" not in env and "PROJ_LIB" not in env and "PROJ_DATA" in _DATA_ENV:
        env["PROJ_DATA"] = env["PROJ_LIB"] = _DATA_ENV["PROJ_DATA"]
    return env
