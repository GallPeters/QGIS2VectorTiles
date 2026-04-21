"""
qgis2vectortiles.py

QGIS2VectorTiles — thin orchestrator class that drives the full conversion
pipeline from QGIS project styling to vector tiles:

  1. Flatten rules  (RulesFlattener)
  2. Export rules   (RulesExporter)
  3. Generate tiles (GDALTilesGenerator)
  4. Style tiles    (TilesStyler)
  5. Export MapLibre style (QgisMapLibreStyleExporter)
  6. Serve tiles via local HTTP server
"""

import platform
import subprocess
from datetime import datetime
from os import makedirs, listdir
from os.path import join, exists, basename
from shutil import rmtree, copy2
from sys import prefix
from time import perf_counter
from typing import List, Optional
from uuid import uuid4

from qgis.utils import iface
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsVectorTileLayer,
    QgsProcessingFeedback,
    QgsProcessingUtils,
    QgsProcessingException,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
)

from .utils.config import _VIEWER, _MAPLIBRE, _PORT, _EPSG_CRS, _SERVER, _BAT, _SH, _VB
from .utils.flattened_rule import FlattenedRule
from .core.rules_flattener import RulesFlattener
from .core.rules_exporter import RulesExporter
from .core.tiles_generator import GDALTilesGenerator
from .core.tiles_styler import TilesStyler

try:
    from .core.maplibre_converter import QgisMapLibreStyleExporter
except ImportError:
    pass


class QGIS2VectorTiles:
    """Orchestrate the conversion from QGIS project styling to vector tiles."""

    def __init__(
        self,
        min_zoom: int = 0,
        max_zoom: int = 5,
        extent=None,
        output_dir: str = None,
        include_required_fields_only=0,
        cpu_percent: int = 100,
        cent_source: int = 0,
        background_type: int = 0,
        feedback: QgsProcessingFeedback = None,
    ):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.extent = extent or iface.mapCanvas().extent()
        self.utils_dir = self._get_utils_dir()
        self.output_dir = output_dir or self.utils_dir
        self.include_required_fields_only = include_required_fields_only
        self.cpu_percent = min(cpu_percent, 90)
        self.cent_source = cent_source
        self.background_type = background_type
        self.feedback = feedback or QgsProcessingFeedback()

    def convert_project_to_vector_tiles(self) -> Optional[QgsVectorTileLayer]:
        """Run the full conversion pipeline; return the styled tiles layer or None."""
        try:
            self._clear_project()
            temp_dir = self._create_temp_directory()
            self._log(". Starting conversion process...")
            start_time = perf_counter()

            self._log(". Flattening rules...")
            rules = self._flatten_rules()
            if not rules:
                self._log(". No visible vector layers found in project.")
                return None
            self._log(f". Successfully extracted {len(rules)} rules "
                      f"({self._elapsed_minutes(start_time)} minutes).")

            flatten_time = perf_counter()
            self._log(". Exporting rules to datasets...")
            layers, rules = self._export_rules(rules)
            self._log(f". Successfully exported {len(layers)} layers "
                      f"({self._elapsed_minutes(flatten_time)} minutes).")

            tiles_uri = None
            export_time = perf_counter()
            if self._has_features(layers):
                self._log(". Generating tiles...")
                tiles_uri = self._generate_tiles(layers, temp_dir)
                self._log(f". Successfully generated tiles "
                          f"({self._elapsed_minutes(export_time)} minutes).")

            self._log(". Loading and styling tiles...")
            styled_layer = self._style_tiles(rules, temp_dir, tiles_uri)
            self._log(". Successfully loaded and styled tiles.")

            self._log(". Exporting tiles style to MapLibre style package...")
            self._export_maplibre_style(temp_dir, styled_layer)
            self._log(". Successfully exported MapLibre style package.")

            self._log(f". Process completed successfully "
                      f"({self._elapsed_minutes(start_time)} minutes).")
            self._clear_project()
            self.serve_tiles(temp_dir)

        except QgsProcessingException as e:
            self._log(f". Processing failed: {str(e)}")
            self._clear_project()
            return None

    def _clear_project(self):
        """Remove all map layers not visible in the project legend."""
        legend_ids = {
            node.layer().id()
            for node in QgsProject.instance().layerTreeRoot().findLayers()
        }
        for layer_id in list(QgsProject.instance().mapLayers()):
            if layer_id not in legend_ids:
                QgsProject.instance().removeMapLayer(layer_id)

    def _get_utils_dir(self) -> str:
        """Clear the QGIS temp folder and create a fresh working directory."""
        for entry in listdir(QgsProcessingUtils.tempFolder()):
            try:
                rmtree(join(QgsProcessingUtils.tempFolder(), entry))
            except (PermissionError, NotADirectoryError):
                continue
        utils_dir = join(QgsProcessingUtils.tempFolder(), f"q2styledtiles_{uuid4().hex}")
        makedirs(utils_dir, exist_ok=True)
        return utils_dir

    def _create_temp_directory(self) -> str:
        temp_dir = join(self.output_dir, datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f"))
        makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _flatten_rules(self) -> List[FlattenedRule]:
        return RulesFlattener(
            self.min_zoom, self.max_zoom, self.utils_dir, self.feedback
        ).flatten_all_rules()

    def _export_rules(self, rules: List[FlattenedRule]):
        return RulesExporter(
            rules, self.extent, self.include_required_fields_only,
            self.max_zoom, self.utils_dir, self.cent_source, self.feedback,
        ).export()

    def _has_features(self, layers: List[QgsVectorLayer]) -> bool:
        return any(layer.featureCount() > 0 for layer in layers)

    def _generate_tiles(self, layers: List[QgsVectorLayer], temp_dir: str) -> str:
        tiles_uri, min_zoom = GDALTilesGenerator(
            layers, temp_dir, self.extent, self.cpu_percent, self.feedback
        ).generate()
        self.min_zoom = min_zoom
        return tiles_uri

    def _style_tiles(self, rules, temp_dir, tiles_uri) -> Optional[QgsVectorTileLayer]:
        return TilesStyler(rules, temp_dir, tiles_uri).apply_styling()

    def _export_maplibre_style(self, temp_dir, styled_layer):
        QgisMapLibreStyleExporter(temp_dir, styled_layer, self.background_type).export()

    def _log(self, message: str):
        if __name__ != "__console__":
            self.feedback.pushInfo(message)
        else:
            print(message)

    def _elapsed_minutes(self, start: float) -> str:
        """Return elapsed time in minutes since start, rounded to 2 decimal places."""
        return f"{round((perf_counter() - start) / 60, 2)}"

    # --- Tile server ---

    def serve_tiles(self, output_folder: str):
        """Copy server utilities and launch the local tile server."""
        utils_dir = join(output_folder, "utils")
        makedirs(utils_dir, exist_ok=True)

        activator = _BAT if platform.system() == "Windows" else _SH
        wrapper = _VB if platform.system() == "Windows" else None

        self._copy_server_files(output_folder, utils_dir, activator, wrapper)

        center = self._get_center_wgs84()
        python_exe = self._get_python_exe()
        self._configure_server_placeholders(
            output_folder, utils_dir, activator, wrapper, center, python_exe
        )
        self._launch_server(output_folder, activator, wrapper)

    def _copy_server_files(
        self, output_folder: str, utils_dir: str, activator: str, wrapper: Optional[str]
    ):
        """Copy the server, viewer, and launcher files to the output directory."""
        if wrapper:
            copy2(activator, utils_dir)
            copy2(wrapper, output_folder)
        else:
            copy2(activator, output_folder)
        copy2(_SERVER, utils_dir)
        copy2(_VIEWER, utils_dir)
        copy2(f"{_MAPLIBRE}.js", utils_dir)
        copy2(f"{_MAPLIBRE}.css", utils_dir)

    def _get_center_wgs84(self) -> str:
        """Return the map extent center as a '[lon, lat]' string in EPSG:4326."""
        src_crs = QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}")
        dest_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        transform = QgsCoordinateTransform(
            src_crs, dest_crs, QgsProject.instance().transformContext()
        )
        center = transform.transform(self.extent.center())
        return f"[{center.x()}, {center.y()}]"

    def _get_python_exe(self) -> str:
        """Return the Python executable path for the current platform."""
        system = platform.system()
        if system == "Windows":
            return join(prefix, "pythonw.exe")
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
        self.replace_in_file(
            join(utils_dir, basename(_VIEWER)),
            {
                "_Q2VT_MINZOOM": str(self.min_zoom),
                "_Q2VT_CENTER": center,
                "18111991": str(_PORT),
            },
        )
        self.replace_in_file(
            join(utils_dir, basename(_SERVER)),
            {"18111991": str(_PORT)},
        )
        activator_dir = utils_dir if wrapper else output_folder
        self.replace_in_file(
            join(activator_dir, basename(activator)),
            {
                "18111991": str(_PORT),
                "_Q2VT_PYTHON": python_exe,
                "_Q2VT_UTILS": join(utils_dir, "mbtiles_server.py"),
            },
        )

    def _launch_server(
        self, output_folder: str, activator: str, wrapper: Optional[str]
    ):
        """Launch the tile server subprocess."""
        command = "wscript.exe" if wrapper else "bash"
        launch_file = wrapper if wrapper else activator
        subprocess.Popen(
            [command, join(output_folder, basename(launch_file))], cwd=output_folder
        )

    def replace_in_file(self, file_path: str, replacements: dict) -> None:
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


if __name__ == "__console__":
    adapter = QGIS2VectorTiles(output_dir=QgsProcessingUtils.tempFolder())
    adapter.convert_project_to_vector_tiles()
