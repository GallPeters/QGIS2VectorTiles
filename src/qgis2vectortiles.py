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

from datetime import datetime
from os import makedirs, listdir
from os.path import join
from shutil import rmtree
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
    QgsProcessingException
)


from .utils.flattened_rule import FlattenedRule
from .core.rules_flattener import RulesFlattener
from .core.rules_exporter import RulesExporter
from .core.tiles_generator import GDALTilesGenerator
from .core.tiles_styler import TilesStyler
from .core.maplibre_converter import QgisMapLibreStyleExporter
from .core.server_initializer import ServerInitializer

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
        viewer: int = 0,
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
        self.viewer = viewer
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
    
    def serve_tiles(self, temp_dir: str):
        """Serve the generated tiles via a local HTTP server."""
        ServerInitializer(self.extent, self.min_zoom, self.viewer).serve_tiles(temp_dir)

if __name__ == "__console__":
    adapter = QGIS2VectorTiles(output_dir=QgsProcessingUtils.tempFolder())
    adapter.convert_project_to_vector_tiles()
