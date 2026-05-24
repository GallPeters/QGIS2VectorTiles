"""Copy server utilities and launch the local tile server."""

import platform
import subprocess
from os import makedirs
from os.path import join, basename, exists
from shutil import copy2, copytree
from sys import prefix
from typing import Optional

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
)
from ..utils.config import _RESOURCES, _PORT, _EPSG_CRS, _SERVER, _BAT, _SH, _VB

class ServerInitializer:
    """Copy server utilities and launch the local tile server."""

    def __init__(self, extent, min_zoom: int, viewer: int):
        self.extent = extent
        self.min_zoom = min_zoom
        self.viewer = viewer
        self.viewer_dir = join(_RESOURCES, 'ml_viewer' if viewer == 0 else 'ol_viewer')
        self.port = _PORT  # use different port for each viewer to allow running both simultaneously

    def serve_tiles(self, output_folder: str):
        """Copy server utilities and launch the local tile server."""
        utils_dir = join(output_folder, "utils")
        makedirs(utils_dir, exist_ok=True)
        activator = _BAT if platform.system() == "Windows" else _SH
        wrapper = _VB if platform.system() == "Windows" else None
        dest_activator, dest_wrapper = self._copy_server_files(
            output_folder, utils_dir, activator, wrapper
        )
        center = self._get_center_wgs84()
        python_exe = self._get_python_exe()
        self._configure_server_placeholders(
            output_folder, utils_dir, dest_activator, dest_wrapper, center, python_exe
        )
        self._launch_server(output_folder, dest_activator, dest_wrapper)

    def _copy_server_files(
        self, output_folder: str, utils_dir: str, activator: str, wrapper: Optional[str]
    ):
        """Copy the server, viewer, and launcher files to the output directory."""
        if wrapper:
            dest_activator = join(utils_dir, basename(activator).replace("_win.txt", ".bat"))
            copy2(activator, dest_activator)
            dest_wrapper = join(output_folder, basename(wrapper).replace("_vbs.txt", ".vbs"))
            copy2(wrapper, dest_wrapper)
        else:
            dest_wrapper = None
            dest_activator = join(output_folder, basename(activator).replace("_lin.txt", ".sh"))
            copy2(activator, dest_activator)
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