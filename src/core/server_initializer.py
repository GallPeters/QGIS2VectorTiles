"""Lay out the export as a web map folder and launch the local server.

The export folder is a complete static web map, ready for any web server or
static host (e.g. GitHub Pages):

    index.html                 the viewer page (MapLibre or OpenLayers)
    ml/, ol/                   the viewers' scripts
    style/                     style.json, sprite/, glyphs/, search.json
    tiles.pmtiles              the vector tiles
    server/                    the local server, for viewing it on this computer
    activate_server.vbs|.sh    starts the local server and opens the map
"""

import platform
import subprocess

from os import makedirs
from os.path import join, basename, exists, normpath
from shutil import copy2, copytree
from sys import prefix
from typing import Optional

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCsException,
    QgsProject,
)

from ..utils.config import _RESOURCES, _PORT, _EPSG_CRS, _SERVER


class ServerInitializer:
    """Copy the viewer and server into the export and launch the server."""

    LINUX_ACTIVATOR = r'''
#!/bin/bash
lsof -ti :18111991 | xargs kill -9 2>/dev/null
cd "$(dirname "$0")"
echo "Starting the web map server..."
_Q2VT_PYTHON _Q2VT_UTILS --port 18111991 &
sleep 2
URL="http://localhost:18111991/?v=10031993"
command -v xdg-open &>/dev/null && xdg-open "$URL" || command -v open &>/dev/null && open "$URL"
'''
    WINDOWS_ACTIVATOR = r'''
@echo off
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :18111991 ^| findstr LISTENING') do taskkill /PID %%a /F
cd /d %~dp0
echo Starting the web map server...
start "Web map server" "_Q2VT_PYTHON" "_Q2VT_UTILS" --port 18111991
timeout /t 2 /nobreak > nul
start "" "http://localhost:18111991/?v=10031993"
pause
'''

    WINDOWS_WRAPPER = r'''
Set WshShell = CreateObject("WScript.Shell")
base = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
WshShell.Run "cmd /c """ & base & "server\activate_server.bat""", 0
Set WshShell = Nothing
'''

    def __init__(self, extent, min_zoom: int, output_dir: str):
        self.extent = extent
        self.min_zoom = min_zoom
        # One viewer page for both map libraries (see resources/viewer).
        self.viewer_dir = join(_RESOURCES, 'viewer')
        self.output_dir = output_dir
        self.port = _PORT

    def serve_tiles(self):
        """Lay out the web map folder and launch the local server."""
        server_dir = join(self.output_dir, "server")
        makedirs(server_dir, exist_ok=True)
        self._write_viewer(self.output_dir)
        dest_activator, dest_wrapper = self._write_server_files(self.output_dir, server_dir)
        center = self._get_center_wgs84()
        python_exe = self._get_python_exe()
        self._configure_server_placeholders(
            self.output_dir, server_dir, dest_activator, dest_wrapper, center, python_exe
        )
        self._launch_server(self.output_dir, dest_activator, dest_wrapper)

    def _write_viewer(self, output_folder: str):
        """The viewer page as the folder's index.html, with its scripts."""
        copy2(join(self.viewer_dir, "viewer.html"), join(output_folder, "index.html"))
        for library in ("ml", "ol"):
            copytree(join(self.viewer_dir, library), join(output_folder, library), dirs_exist_ok=True)

    def _write_server_files(self, output_folder: str, server_dir: str):
        """Copy/write the server and launcher files to the output directory."""
        if platform.system() == "Windows":
            dest_activator = join(server_dir, 'activate_server.bat')
            with open(dest_activator, "w", encoding="utf-8", newline="") as activator:
                activator.write(self.WINDOWS_ACTIVATOR)
            dest_wrapper = join(output_folder, 'activate_server.vbs')
            with open(dest_wrapper, "w", encoding="utf-8", newline="") as wrapper_file:
                wrapper_file.write(self.WINDOWS_WRAPPER)
        else:
            dest_wrapper = None
            dest_activator = join(output_folder, 'activate_server.sh')
            with open(dest_activator, "w", encoding="utf-8", newline="") as activator:
                activator.write(self.LINUX_ACTIVATOR)
        copy2(_SERVER, server_dir)
        return dest_activator, dest_wrapper

    def _get_center_wgs84(self) -> str:
        """Return the map extent center as a '[lon, lat]' string in EPSG:4326."""
        src_crs = QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}")
        dest_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        transform = QgsCoordinateTransform(
            src_crs, dest_crs, QgsProject.instance().transformContext()
        )
        center = transform.transform(self.extent.center())
        return f"[{center.x()}, {center.y()}]"

    def _get_bounds_wgs84(self) -> str:
        """The map extent as a '[west, south, east, north]' string in EPSG:4326."""
        transform = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}"),
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsProject.instance().transformContext(),
        )
        try:
            box = transform.transformBoundingBox(self.extent)
        except QgsCsException:
            return "null"
        return f"[{box.xMinimum()}, {box.yMinimum()}, {box.xMaximum()}, {box.yMaximum()}]"

    @staticmethod
    def _get_python_exe() -> str:
        """Return the Python executable path for the current platform."""
        system = platform.system()
        if system == "Windows":
            return join(prefix, "pythonw3.exe")
        if system == "Linux":
            return join(prefix, "bin", "python3")
        return join(prefix, "python3")

    def _configure_server_placeholders(
        self,
        output_folder: str,
        server_dir: str,
        activator: str,
        wrapper: Optional[str],
        center: str,
        python_exe: str,
    ):
        """Replace template placeholders in the viewer, style and launchers."""
        # The URLs' version: unique per export (its folder), so browsers may
        # keep the tiles and never show an earlier export's.
        version = basename(normpath(output_folder))
        # Both viewers read their settings from the page; the output
        # folder's name keys the drawings the browser keeps for this map.
        self.replace_in_file(
            join(output_folder, "index.html"),
            {
                "_Q2VT_MINZOOM": str(self.min_zoom),
                "_Q2VT_CENTER": center,
                "_Q2VT_BOUNDS": self._get_bounds_wgs84(),
                "_Q2VT_STORAGE_KEY": version,
            },
        )
        activator_dir = server_dir if wrapper else output_folder
        self.replace_in_file(
            join(activator_dir, basename(activator)),
            {
                "18111991": str(self.port),
                "_Q2VT_PYTHON": python_exe,
                "_Q2VT_UTILS": join(server_dir, "tiles_server.py"),
                "10031993": version,
            },
        )
        self.replace_in_file(
            join(output_folder, 'style', 'style.json'),
            {
                "10031993": version,
            },
        )

    @staticmethod
    def _launch_server(output_folder: str, activator: str, wrapper: Optional[str]):
        """Launch the tile server subprocess."""
        command = "wscript.exe" if wrapper else "bash"
        launch_file = wrapper if wrapper else activator
        subprocess.Popen(
            [command, join(output_folder, basename(launch_file))], cwd=output_folder
        )

    @staticmethod
    def replace_in_file(file_path: str, replacements: dict) -> None:
        """Replace all keys with their values in the given file."""
        if not exists(file_path):
            raise FileNotFoundError(file_path)
        if not replacements:
            raise ValueError("empty replacements")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            for old, new in replacements.items():
                content = content.replace(old, new)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            raise OSError(f"failed: {file_path}") from exc
