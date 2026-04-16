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
    """
    Main adapter class that orchestrates the conversion process from QGIS
    vector layer styling to vector tiles format.
    """

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
        self.max_zoom = max_zoom + 1
        self.extent = extent or iface.mapCanvas().extent()
        self.utils_dir = self._get_utils_dir()
        self.output_dir = output_dir or self.utils_dir
        self.include_required_fields_only = include_required_fields_only
        self.cpu_percent = min(cpu_percent, 90)
        self.cent_source = cent_source
        self.background_type = background_type
        self.feedback = feedback or QgsProcessingFeedback()

    def convert_project_to_vector_tiles(self) -> Optional[QgsVectorTileLayer]:
        """
        Convert current QGIS project to vector tiles format.

        Returns:
            QgsVectorTileLayer: The created vector tiles layer, or None if failed
        """
        try:
            self._clear_project()
            temp_dir = self._create_temp_directory()
            self._log(". Starting conversion process...")
            start_time = perf_counter()

            # Step 1: Flatten all rules
            self._log(". Flattening rules...")
            rules = self._flatten_rules()
            if not rules:
                self._log(". No visible vector layers found in project.")
                return

            flatten_finish_time = perf_counter()
            flatten_time = self._elapsed_minutes(start_time, flatten_finish_time)
            self._log(f". Successfully extracted {len(rules)} rules ({flatten_time} minutes).")
            tiles_uri = layers = None

            # Step 2: Export rules to datasets
            self._log(". Exporting rules to datasets...")
            layers, rules = self._export_rules(rules)
            export_finish_time = perf_counter()
            export_time = self._elapsed_minutes(flatten_finish_time, export_finish_time)
            self._log(f". Successfully exported {len(layers)} layers ({export_time} minutes).")

            # Step 3: Generate and style tiles
            if self._has_features(layers):
                self._log(". Generating tiles...")
                tiles_uri = self._generate_tiles(layers, temp_dir)
                tiles_finish_time = perf_counter()
                tiles_time = self._elapsed_minutes(export_finish_time, tiles_finish_time)
                self._log(f". Successfully generated tiles ({tiles_time} minutes).")

            self._log(". Loading and styling tiles...")
            styled_layer = self._sytle_tiles(rules, temp_dir, tiles_uri)
            self._log(". Successfully loaded and styled tiles.")

            self._log(". Exporting tiles style to MapLibre style package...")
            self._export_maplibre_style(temp_dir, styled_layer)
            self._log(". Successfully exported MapLibre style package.")

            total_time = self._elapsed_minutes(start_time, perf_counter())
            self._log(f". Process completed successfully ({total_time} minutes).")
            self._clear_project()
            self.serve_tiles(temp_dir)
        except QgsProcessingException as e:
            self._log(f". Processing failed: {str(e)}")
            self._clear_project()
            return None

    def _clear_project(self):
        """Clear temp layers which are not visible in the project legend."""
        legend_layers = QgsProject.instance().layerTreeRoot().findLayers()
        legend_layers_ids = [layer.layer().id() for layer in legend_layers]
        for layer in list(QgsProject.instance().mapLayers().values()):
            if layer.id() not in legend_layers_ids:
                QgsProject.instance().removeMapLayer(layer)

    def _get_utils_dir(self) -> str:
        """Clear utils dir."""
        for subfile in listdir(QgsProcessingUtils.tempFolder()):
            try:
                path = join(QgsProcessingUtils.tempFolder(), subfile)
                rmtree(path)
            except (PermissionError, NotADirectoryError):
                continue
        utils_dir = join(QgsProcessingUtils.tempFolder(), f"q2styledtiles_{uuid4().hex}")
        makedirs(utils_dir, exist_ok=True)
        return utils_dir

    def _create_temp_directory(self) -> str:
        """Create temporary directory for processing."""
        temp_dir = join(self.output_dir, datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f"))
        makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _flatten_rules(self) -> List[FlattenedRule]:
        """Flatten all rules from project layers."""
        flattener = RulesFlattener(self.min_zoom, self.max_zoom, self.utils_dir, self.feedback)
        return flattener.flatten_all_rules()

    def _export_rules(self, rules: List[FlattenedRule]) -> List[QgsVectorLayer]:
        """Export rules to vector layers."""
        exporter = RulesExporter(
            rules,
            self.extent,
            self.include_required_fields_only,
            self.max_zoom,
            self.utils_dir,
            self.cent_source,
            self.feedback,
        )
        return exporter.export()

    def _has_features(self, layers: List[QgsVectorLayer]) -> bool:
        """Check if any layer has features."""
        return any(layer.featureCount() > 0 for layer in layers)

    def _generate_tiles(self, layers: List[QgsVectorLayer], temp_dir: str) -> str:
        """Generate vector tiles."""
        generator = GDALTilesGenerator(
            layers, temp_dir, self.extent, self.cpu_percent, self.feedback
        )
        tiles_uri, min_zoom = generator.generate()
        self.min_zoom = min_zoom
        return tiles_uri

    def _sytle_tiles(self, rules, temp_dir, tiles_uri) -> Optional[QgsVectorTileLayer]:
        """Style tiles."""
        styler = TilesStyler(rules, temp_dir, tiles_uri)
        styled_layer = styler.apply_styling()
        return styled_layer

    def _export_maplibre_style(self, temp_dir, styled_layer):
        """Export MapLibre style."""
        exporter = QgisMapLibreStyleExporter(temp_dir, styled_layer, self.background_type)
        exporter.export()

    def _log(self, message: str):
        """Log message to feedback or console."""
        if __name__ != "__console__":
            self.feedback.pushInfo(message)
        else:
            print(message)

    @staticmethod
    def _elapsed_minutes(start: float, end: float) -> str:
        """Calculate elapsed time in minutes."""
        return f"{round((end - start) / 60, 2)}"

    def serve_tiles(self, output_folder):
        """Serve generated tiles using a simple HTTP server (cross-platform)."""
        # Create utilities directory
        utils_dir_name = "utils"
        utils_dir = join(output_folder, utils_dir_name)
        makedirs(utils_dir, exist_ok=True)

        # Copy utils files to utils directory
        activator = _BAT if platform.system() == "Windows" else _SH
        wrapper = _VB if platform.system() == "Windows" else None
        if wrapper:
            copy2(activator, utils_dir)
            copy2(wrapper, output_folder)
        else:
            copy2(activator, output_folder)
        copy2(_SERVER, utils_dir)
        copy2(_VIEWER, utils_dir)
        copy2(f"{_MAPLIBRE}.js", utils_dir)
        copy2(f"{_MAPLIBRE}.css", utils_dir)

        # Get the extent of the tiles in EPSG:4326 for the MapLibre viewer
        src_crs = QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}")
        dest_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        coord_transform = QgsCoordinateTransform(
            src_crs, dest_crs, QgsProject.instance().transformContext()
        )
        center_src_crs = self.extent.center()
        center_4326 = coord_transform.transform(center_src_crs)
        center_corrd = f"[{center_4326.x()}, {center_4326.y()}]"

        # Get python exe path
        if platform.system() == "Windows":
            python_exe = join(prefix, "pythonw.exe")
        elif platform.system() == "Linux":
            python_exe = join(prefix, "bin", "python3")
        else:
            python_exe = join(prefix, "python3")

        # Replace placeholders in the utils
        self.replace_in_file(
            join(utils_dir, basename(_VIEWER)),
            {
                "_Q2VT_MINZOOM": str(self.min_zoom),
                "_Q2VT_CENTER": center_corrd,
                "18111991": str(_PORT),
            },
        )

        self.replace_in_file(join(utils_dir, basename(_SERVER)), {"18111991": str(_PORT)})
        self.replace_in_file(
            join(utils_dir if wrapper else output_folder, basename(activator)),
            {
                "18111991": str(_PORT),
                "_Q2VT_PYTHON": python_exe,
                "_Q2VT_UTILS": join(utils_dir, "mbtiles_server.py"),
            },
        )

        # Launch server and viewer

        command = "wscript.exe" if wrapper else "bash"
        popen_file = wrapper if wrapper else activator
        subprocess.Popen([command, join(output_folder, basename(popen_file))], cwd=output_folder)

    def replace_in_file(self, file_path: str, replacements: dict[str, str]) -> None:
        """Replace all occurrences of keys in a file with their corresponding values."""
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
