"""
gdal_tools.py

Finds GDAL's command-line tools (ogr2ogr, ogrinfo) of the QGIS install the
plugin runs in, and the environment to run them with.

QGIS on Windows (OSGeo4W) and Linux puts its GDAL tools on the PATH; the
macOS app does not: a program started from the Finder gets the system's
short PATH, and the tools are inside the app bundle (QGIS.app/Contents/...).
They are looked for next to QGIS first, so the tools of QGIS's own GDAL are
used, then on the PATH.
"""

import os
import shutil
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


def gdal_tool(name: str) -> str:
    """The path of GDAL's ``name`` tool (e.g. "ogr2ogr"); raises
    RuntimeError, saying where it was looked for, if it isn't found."""
    with _LOCK:
        if name in _FOUND:
            return _FOUND[name]
        exe = f"{name}.exe" if os.name == "nt" else name
        dirs = _candidate_dirs()
        path = next((join(d, exe) for d in dirs if _executable(join(d, exe))), None)
        if path is None:
            contents = _bundle_contents()
            if contents:
                path = _search_bundle(contents, exe)
        if path is None:
            path = shutil.which(exe)
        if path is None:
            raise RuntimeError(
                f"GDAL's {name} tool was not found. Looked in {', '.join(dirs)} and on the PATH. "
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
