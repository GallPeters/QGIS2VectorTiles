"""
qgis2vectortiles.py

QGIS2VectorTiles — thin orchestrator class that drives the full conversion
pipeline from QGIS project styling to vector tiles:

  1. Flatten rules  (RulesFlattener)
  2. Export rules   (RulesExporter)
  3. Generate tiles (GDALTilesGenerator)
  4. Style tiles    (TilesStyler)
  5. Export MapLibre style (QgisMapLibreStyleExporter)
     and the viewers' search index (SearchIndexBuilder)
  6. Lay out the web map folder and serve it locally
"""

from copy import deepcopy
from datetime import datetime
from os import makedirs, listdir
from os.path import join
from shutil import rmtree
from threading import Thread
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
from .core.search_index import SearchIndexBuilder


class _Background:
    """A call run on its own thread; result() waits for it and returns its
    result or raises its exception."""

    def __init__(self, function, *args):
        self._result = None
        self._error: Optional[BaseException] = None
        self._thread = Thread(target=self._run, args=(function, args), daemon=True)
        self._thread.start()

    def _run(self, function, args):
        try:
            self._result = function(*args)
        except BaseException as e:  # noqa: BLE001  (re-raised by result())
            self._error = e

    def wait(self):
        self._thread.join()

    def result(self):
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._result


class QGIS2VectorTiles:
    """Orchestrate the conversion from QGIS project styling to vector tiles."""

    def __init__(
        self,
        min_zoom: int = 0,
        max_zoom: int = 5,
        extent=None,
        output_dir: str = None,
        included_fields: int = 0,
        cpu_percent: int = 100,
        feedback: QgsProcessingFeedback = None,
    ):
        # Zoom levels are MapLibre's (512 px tiles), in both web viewers:
        # the OpenLayers viewer converts them for its own numbering.
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.extent = extent or iface.mapCanvas().extent()
        self.utils_dir = self._get_utils_dir()
        self.output_dir = output_dir or self.utils_dir
        self.included_fields = included_fields
        self.cpu_percent = min(cpu_percent, 90)
        self.feedback = feedback or QgsProcessingFeedback()
        # The export folder, set when a run completes.
        self.temp_dir: Optional[str] = None

    def convert_project_to_vector_tiles(self) -> Optional[QgsVectorTileLayer]:
        """Run the full conversion pipeline; return the styled tiles layer or None.

        Safe to run in a background thread: it never changes the project.
        Project changes are done by clear_project() before the run and
        finish_in_main_thread() after it, both on QGIS's main thread.
        """
        try:
            temp_dir = self._create_temp_directory()
            # The processing algorithm logs the start as soon as it runs.
            start_time = perf_counter()

            self._log(". Extracting rules...")
            rules = self._flatten_rules()
            if not rules:
                self._log(". No visible vector layers found in project.")
                return None
            self._log(f". Successfully extracted {len(rules)} rules "
                      f"({self._elapsed_minutes(start_time)} minutes).")

            step_time = perf_counter()
            self._log(". Exporting rules to datasets...")
            layers, rules = self._export_rules(rules)
            self._log(f". Successfully exported {len(rules)} rules "
                      f"({self._elapsed_minutes(step_time)} minutes).")

            step_time = perf_counter()
            self._log(". Styling tiles...")
            styled_layer = self._style_tiles(rules, temp_dir)
            self._log(f". Successfully styled tiles "
                      f"({self._elapsed_minutes(step_time)} minutes).")

            # The tiles need only the converted style (the fields it uses);
            # ogr2ogr makes them in other processes while this thread writes
            # the style package (sprites, glyphs) and the search index.
            style_exporter = self._maplibre_style_exporter(temp_dir, styled_layer)
            style = style_exporter.convert()
            tiles, generator = None, None
            tiles_time = perf_counter()
            if self._has_features(layers):
                self._log(". Generating tiles...")
                generator = GDALTilesGenerator(
                    layers, deepcopy(style), temp_dir, self.extent, self.cpu_percent, self.feedback
                )
                tiles = _Background(generator.generate)

            try:
                step_time = perf_counter()
                self._log(". Exporting style to client-side style package...")
                style_exporter.save_to_file()
                self._log(f". Successfully exported client-side style package "
                          f"({self._elapsed_minutes(step_time)} minutes).")

                step_time = perf_counter()
                self._log(". Building the search index...")
                if self._build_search_index(layers, rules, temp_dir):
                    self._log(f". Successfully built the search index "
                              f"({self._elapsed_minutes(step_time)} minutes).")
            except BaseException:
                if generator is not None:
                    generator.cancel()
                    tiles.wait()
                raise

            if tiles is not None:
                _, self.min_zoom = tiles.result()
                self._log(f". Successfully generated tiles "
                          f"({self._elapsed_minutes(tiles_time)} minutes).")
            self._log(f". Vector tiles package generation completed successfully "
                      f"({self._elapsed_minutes(start_time)} minutes).")
            self._log("")
            self.temp_dir = temp_dir
            self.serve_tiles(temp_dir)

        except QgsProcessingException as e:
            self._log(f". Processing failed: {str(e)}")
            return None

    def finish_in_main_thread(self):
        """Project changes after a run. Call on QGIS's main thread only."""
        self.finish(self.extent, self.min_zoom, self.temp_dir)

    @classmethod
    def finish(cls, extent, min_zoom: int, temp_dir: Optional[str]):
        """Project changes after a run. Call on QGIS's main thread only.

        The project and its layer tree belong to the main thread; changing
        them from the processing thread can crash QGIS later (e.g. on exit).
        The result is the web map folder; nothing is added to the project.
        """
        cls.clear_project()

    @staticmethod
    def clear_project():
        """Remove all map layers not visible in the project legend.

        Call on QGIS's main thread only.
        """
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
            rules, self.extent, self.included_fields,
            self.max_zoom, self.utils_dir, self.feedback,
        ).export()

    def _has_features(self, layers: List[QgsVectorLayer]) -> bool:
        return any(layer.featureCount() > 0 for layer in layers)

    def _style_tiles(self, rules, temp_dir) -> Optional[QgsVectorTileLayer]:
        return TilesStyler(rules, temp_dir).apply_styling()

    def _maplibre_style_exporter(self, temp_dir, styled_layer) -> QgisMapLibreStyleExporter:
        return QgisMapLibreStyleExporter(
            temp_dir, self.utils_dir, styled_layer, self.min_zoom, self.max_zoom,
            extent=self.extent,
        )

    def _build_search_index(self, layers, rules, temp_dir: str) -> Optional[str]:
        return SearchIndexBuilder(
            layers, rules, join(temp_dir, "style", "search.json"), self.feedback
        ).build()

    def _log(self, message: str):
        if __name__ != "__console__":
            self.feedback.pushInfo(message)
        else:
            print(message)

    def _elapsed_minutes(self, start: float) -> str:
        """Return elapsed time in minutes since start, rounded to 2 decimal places."""
        return f"{round((perf_counter() - start) / 60, 2)}"
    
    def serve_tiles(self, temp_dir: str):
        """Serve the generated tiles via a local HTTP server (no project changes)."""
        ServerInitializer(self.extent, self.min_zoom, temp_dir).serve_tiles()

if __name__ == "__console__":
    print(". Starting vector tiles package generation process...")
    QGIS2VectorTiles.clear_project()
    adapter = QGIS2VectorTiles(output_dir=QgsProcessingUtils.tempFolder())
    adapter.convert_project_to_vector_tiles()
    adapter.finish_in_main_thread()
