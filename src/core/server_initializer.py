"""Copy server utilities and launch the local tile server."""

import platform
import subprocess
import json

from os import makedirs
from os.path import join, basename, exists
from shutil import copy2, copytree
from sys import prefix
from typing import Optional


from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsLayerDefinition,
    QgsMapBoxGlStyleConverter,
    QgsVectorTileLayer,
    QgsMapBoxGlStyleConversionContext
)

from qgis.PyQt.QtCore import qVersion

if int(qVersion()[0]) == 5:
    from PyQt5.QtGui import QImage
else:
    from PyQt6.QtGui import QImage

from ..utils.config import _RESOURCES, _PORT, _EPSG_CRS, _SERVER, _MAPLIBRE_LABELS_FACTOR

class ServerInitializer:
    """Copy server utilities and launch the local tile server."""

    LINUX_ACTIVATOR = r'''
#!/bin/bash
lsof -ti :18111991 | xargs kill -9 2>/dev/null
cd "$(dirname "$0")"
echo "Starting MBTiles server..."
_Q2VT_PYTHON _Q2VT_UTILS --port 18111991 &
sleep 2
URL="http://localhost:18111991/viewer/viewer.html?v=10031993"
command -v xdg-open &>/dev/null && xdg-open "$URL" || command -v open &>/dev/null && open "$URL"
'''
    WINDOWS_ACTIVATOR = r'''
@echo off
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :18111991 ^| findstr LISTENING') do taskkill /PID %%a /F
cd /d %~dp0
echo Starting MBTiles server...
start "MBTiles Server" "_Q2VT_PYTHON" "_Q2VT_UTILS" --port 18111991
timeout /t 2 /nobreak > nul
start "" "http://localhost:18111991/viewer/viewer.html?v=10031993"
pause
'''

    WINDOWS_WRAPPER = r'''
Set WshShell = CreateObject("WScript.Shell")
base = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
WshShell.Run "cmd /c """ & base & "utils\activate_server.bat""", 0
Set WshShell = Nothing
'''

    def __init__(self, extent, min_zoom: int, viewer: int, output_dir: str):
        self.extent = extent
        self.min_zoom = min_zoom
        self.viewer = viewer
        self.viewer_dir = join(_RESOURCES, 'ml_viewer' if viewer == 0 else 'ol_viewer')
        self.output_dir = output_dir
        self.port = _PORT  # use different port for each viewer to allow running both simultaneously

    def serve_tiles(self):
        """Copy server utilities and launch the local tile server."""
        utils_dir = join(self.output_dir, "utils")
        makedirs(utils_dir, exist_ok=True)
        dest_activator, dest_wrapper = self._write_server_files(
            self.output_dir, utils_dir
        )
        center = self._get_center_wgs84()
        python_exe = self._get_python_exe()
        self._configure_server_placeholders(
            self.output_dir, utils_dir, dest_activator, dest_wrapper, center, python_exe
        )
        self._save_as_local_qlr()
        self._launch_server(self.output_dir, dest_activator, dest_wrapper)

    def _save_as_local_qlr(self):
        """ Save vector tiles as ocal styled QgsVectorLayer"""
        # Create vector tiles layer
        output = join(self.output_dir, "tiles.mbtiles")
        uri = f"type=mbtiles&url={output}"
        layer = QgsVectorTileLayer(uri, "Vector Tiles")
        layer = QgsProject.instance().addMapLayer(layer, False)
        QgsProject.instance().layerTreeRoot().insertLayer(0, layer)

        # Convert local style path to unc path so QGIS could read the sprite correctly and apply it on tiles layer
        style_dir = join(self.output_dir, 'style')
        local_style = join(style_dir, 'style.json')
        context = QgsMapBoxGlStyleConversionContext()
        if exists(join(style_dir, 'sprite')):
            sprite_img = QImage(join(style_dir, 'sprite', 'sprite@2x.png'))
            sprite_json = join(style_dir, 'sprite', 'sprite@2x.json')
            sprite_dict = json.load(open(sprite_json, 'r', encoding='utf-8'))
            context.setSprites(sprite_img,sprite_dict)

        with open(local_style, 'r', encoding='utf-8') as f:
            style_data = json.load(f)
            if self.viewer == 0:
                # Fix style due to mismatch between maplibre and qgis glyphs rendering method
                for style_layer in style_data['layers']:
                    paint = style_layer.get('paint')
                    layout = style_layer.get('layout')                
                    if paint:
                        halo = paint.get('text-halo-width')
                        if halo:
                            paint['text-halo-width'] = paint['text-halo-width']*_MAPLIBRE_LABELS_FACTOR
                    if layout:
                        size = layout.get('text-size')
                        if size:
                            layout['text-size'] = size*_MAPLIBRE_LABELS_FACTOR
                        
            json_string = json.dumps(style_data)
            converter = QgsMapBoxGlStyleConverter()
            converter.convert(json_string, context)
            if converter.renderer():
                layer.setRenderer(converter.renderer())                
            if converter.labeling():
                layer.setLabeling(converter.labeling())
    
        # Save layer as QLR
        style_dir = join(self.output_dir, "style")
        makedirs(style_dir, exist_ok=True)
        qlr_path = join(style_dir, "tiles.qlr")
        node = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
        QgsLayerDefinition().exportLayerDefinition(qlr_path, [node])

    def _write_server_files(self, output_folder: str, utils_dir: str):
        """Copy/write the server, viewer, and launcher files to the output directory."""
        if platform.system() == "Windows":
            dest_activator = join(utils_dir, 'activate_server.bat')
            with open(dest_activator, "w", encoding="utf-8", newline="") as activator:
                activator.write(self.WINDOWS_ACTIVATOR)
            del activator
            dest_wrapper = join(output_folder, 'activate_server.vbs')
            with open(dest_wrapper, "w", encoding="utf-8", newline="") as wrapper_file:
                wrapper_file.write(self.WINDOWS_WRAPPER)
            del wrapper_file
        else:
            dest_wrapper = None
            dest_activator = join(output_folder, 'activate_server.sh')
            with open(dest_activator, "w", encoding="utf-8", newline="") as activator:
                activator.write(self.LINUX_ACTIVATOR)
            del activator
        copy2(_SERVER, utils_dir)
        copytree(self.viewer_dir, join(utils_dir, "viewer"), dirs_exist_ok=True)
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
        utils_dir: str,
        activator: str,
        wrapper: Optional[str],
        center: str,
        python_exe: str,
    ):
        """Replace template placeholders in server, viewer, and launcher files."""
        viewer = join(utils_dir, "viewer", 'viewer.html')
        self.replace_in_file(
            viewer,
            {
                "_Q2VT_MINZOOM": str(self.min_zoom + self.viewer),
                "_Q2VT_CENTER": center,
                "18111991": str(self.port),

            },
        )
        self.replace_in_file(
            join(utils_dir, basename(_SERVER)),
            {"18111991": str(self.port)},
        )
        activator_dir = utils_dir if wrapper else output_folder
        self.replace_in_file(
            join(activator_dir, basename(activator)),
            {
                "18111991": str(self.port),
                "_Q2VT_PYTHON": python_exe,
                "_Q2VT_UTILS": join(utils_dir, "tiles_server.py"),
                "10031993": str(self.viewer + 1),
            },
        )
        self.replace_in_file(
            join(output_folder, 'style', 'style.json'),
            {
                "10031993": str(self.viewer + 1),
            },
        )
        if self.viewer_dir.endswith("ol_viewer"):
            self.replace_in_file(
                join(utils_dir, "viewer", "bundle.js"),
                {
                    "18111991": str(self.port),
                    "_Q2VT_MINZOOM": str(self.min_zoom + self.viewer),
                    "_Q2VT_CENTER": center,
                }
            )
            self.replace_in_file(
                join(utils_dir, "viewer", "viewer.js"),
                {
                    "18111991": str(self.port),
                    "_Q2VT_MINZOOM": str(self.min_zoom + self.viewer),
                    "_Q2VT_CENTER": center,
                }
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