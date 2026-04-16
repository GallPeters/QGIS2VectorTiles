"""
rules_exporter.py

_ParallelExportTask — lightweight QgsTask wrapper that runs a callable in a
                      background QGIS task (internal to this module).

RulesExporter — exports every FlattenedRule to a FlatGeobuf dataset on disk,
applying geometry transformations, field mappings, and data-defined-property
fields.  Uses _ParallelExportTask for parallelism.

Depends on: config, zoom_levels, flattened_rule, data_defined_properties
"""

import os
import threading
from os.path import join, exists
from time import sleep
from typing import List, Optional, Tuple, Union
from uuid import uuid4

from processing import run as run_processing
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsTask,
    QgsApplication,
    QgsVectorLayer,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsCoordinateReferenceSystem,
    QgsProject,
)

from ..utils.config import _DATA_SIMPLIFICATION_TOLERANCE, _EPSG_CRS, sip
from ..utils.flattened_rule import FlattenedRule
from ..utils.zoom_levels import ZoomLevels
from .ddp_fetcher import DataDefinedPropertiesFetcher


class _ParallelExportTask(QgsTask):
    """Internal task for executing export processing in the background safely."""

    def __init__(self, description, func, *args, **kwargs):
        super().__init__(description, QgsTask.CanCancel)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None

    def run(self):
        """Run the task, executing the provided function with arguments,
        and capture results or exceptions."""
        # Minor jitter to prevent concurrent initialization race conditions in GDAL
        sleep(0.5)
        self.result = self.func(*self.args, **self.kwargs)


class RulesExporter:
    """Export all rules to datasets with geometry transformations using parallel processing."""

    FIELD_PREFIX = "q2vt"

    def __init__(
        self,
        flattened_rules: List[FlattenedRule],
        extent,
        include_required_fields_only,
        max_zoom,
        utils_dir,
        cent_source,
        feedback: QgsProcessingFeedback,
        cpu_percent: int = 100,
    ):
        self.flattened_rules = flattened_rules
        self.extent = extent
        self.include_required_fields_only = include_required_fields_only
        self.max_zoom = max_zoom
        self.cent_source = cent_source
        self.utils_dir = utils_dir
        self.feedback = feedback
        self.cpu_percent = cpu_percent
        self.processed_layers = []
        self._source_locks: dict = {}
        self._source_locks_mutex = threading.Lock()

    def export(self) -> Tuple[List[QgsVectorLayer], List[FlattenedRule]]:
        """Export all rules to datasets using QgsTask parallel workers."""
        num_workers = max(1, int(os.cpu_count() * self.cpu_percent / 100))

        # 1. Export base layers parallelly
        output_datasets = self._export_base_layers(num_workers)

        # 2. Export rules parallelly
        total_datasets = len(output_datasets)
        rule_tasks = []

        for index, flat_rules in enumerate(output_datasets.values()):
            task_desc = f"Exporting rule {index + 1}/{total_datasets}"
            task = _ParallelExportTask(task_desc, self._export_rule_thread_safe, flat_rules)
            rule_tasks.append((task, flat_rules))

        self._run_tasks(rule_tasks, num_workers)

        successful_flat_rules = []
        for task, flat_rules in rule_tasks:
            rules_dataset = join(self.utils_dir, f"{flat_rules[0].output_dataset}.fgb")
            if exists(rules_dataset):
                layer_path = task.result
                layer = QgsVectorLayer(layer_path, flat_rules[0].output_dataset, "ogr")
                if layer.isValid() and layer.featureCount() > 0:
                    self.processed_layers.append(layer)
                    successful_flat_rules.extend(flat_rules)
            else:
                for rule in flat_rules:
                    if rule in self.flattened_rules:
                        self.flattened_rules.remove(rule)

        return self.processed_layers, successful_flat_rules

    def _run_tasks(self, tasks_with_meta: list, num_workers: int):
        """Throttle and process QgsTasks safely waiting for their execution."""
        manager = QgsApplication.taskManager()
        active_tasks = []
        task_queue = list(tasks_with_meta)

        while task_queue or active_tasks:
            if self.feedback.isCanceled():
                for t, _ in active_tasks:
                    t.cancel()
                break

            while len(active_tasks) < num_workers and task_queue:
                t_tuple = task_queue.pop(0)
                t = t_tuple[0]
                manager.addTask(t)
                active_tasks.append(t_tuple)

            for t_tuple in active_tasks[:]:
                t = t_tuple[0]
                if not t or sip.isdeleted(t):  # pylint: disable=I1101
                    active_tasks.remove(t_tuple)
                    continue

                if t.status() in (QgsTask.Complete, QgsTask.Terminated):
                    active_tasks.remove(t_tuple)

            QCoreApplication.processEvents()
            sleep(0.5)

    def _export_base_layers(self, num_workers: int):
        """Export base vector layers to FlatGeobuf format utilizing Tasks."""
        output_datasets = {flat_rule.output_dataset: [] for flat_rule in self.flattened_rules}

        tasks_meta = []
        unique_layer_ids = set()

        for flat_rule in self.flattened_rules:
            output_datasets[flat_rule.output_dataset].append(flat_rule)
            layer_id = flat_rule.layer.id()

            if layer_id not in unique_layer_ids:
                output_path = join(self.utils_dir, f"map_layer_{layer_id}.fgb")
                if not exists(output_path):
                    input_source = flat_rule.layer.source()
                    extent = self.extent
                    output_crs = f"EPSG:{_EPSG_CRS}"

                    task = _ParallelExportTask(
                        f"Exporting layer {flat_rule.layer.name()}",
                        self._process_base_layer_thread,
                        input_source,
                        output_path,
                        extent,
                        output_crs,
                    )
                    tasks_meta.append((task, None))
                unique_layer_ids.add(layer_id)

        if tasks_meta:
            self._run_tasks(tasks_meta, num_workers)

        return output_datasets

    def _get_source_lock(self, input_source: str) -> threading.Lock:
        """Return a per-file lock for the given source path.

        GeoPackage files are backed by SQLite, which uses file-level locking.
        When two tasks open the same .gpkg concurrently (e.g. two layers from
        the same file), GDAL may request a write lock for metadata updates,
        causing the second task to block indefinitely.  Serialising all access
        to the same base file path avoids the deadlock.
        """
        base_path = input_source.split("|")[0]
        with self._source_locks_mutex:
            if base_path not in self._source_locks:
                self._source_locks[base_path] = threading.Lock()
            return self._source_locks[base_path]

    def _process_base_layer_thread(self, input_source, output_path, extent, output_crs) -> str:
        """Run inside QgsTask: Thread-safe base layer creation."""
        # Serialise the initial read from the source file.  GeoPackage is
        # backed by SQLite with file-level locking; two concurrent tasks
        # opening the same .gpkg (e.g. two different layers from one file)
        # can cause GDAL to deadlock waiting for the write lock that it
        # requests when updating metadata.  Once the first algorithm has
        # produced an intermediate file all subsequent steps are safe to
        # run in parallel.
        lock = self._get_source_lock(input_source)
        with lock:
            fix_geom_linework_params = {"INPUT": input_source, "METHOD": 0}
            fix_geom_linework = self._run_alg("fixgeometries", "native", **fix_geom_linework_params)

        fix_geom_structure_params = {"INPUT": fix_geom_linework, "METHOD": 1}
        fix_geom_structure = self._run_alg("fixgeometries", "native", **fix_geom_structure_params)

        reproject_params = {
            "INPUT": fix_geom_structure,
            "TARGET_CRS": QgsCoordinateReferenceSystem(output_crs),
        }
        reprojected = self._run_alg("reprojectlayer", "native", **reproject_params)

        clip_params = {"INPUT": reprojected, "EXTENT": extent, "CLIP": False}
        clipped = self._run_alg("extractbyextent", "native", **clip_params)

        singleparts_params = {"INPUT": clipped}
        singleparted = self._run_alg("multiparttosingleparts", "native", **singleparts_params)

        tolerance = _DATA_SIMPLIFICATION_TOLERANCE
        simplify_params = {
            "INPUT": singleparted,
            "METHOD": 0,
            "TOLERANCE": tolerance,
            "OUTPUT": output_path,
        }
        simplified = self._run_alg("simplifygeometries", "native", **simplify_params)

        return simplified

    def _export_rule_thread_safe(self, flat_rules) -> Optional[str]:
        """Export group of rules sharing the same dataset inside a thread-safe task."""
        flat_rule = flat_rules[0]
        source_path = join(self.utils_dir, f"map_layer_{flat_rule.layer.id()}.fgb")

        if not exists(source_path):
            return None

        temp_layer = QgsVectorLayer(source_path, "temp", "ogr")
        if not temp_layer.isValid() or temp_layer.featureCount() <= 0:
            return None

        self._updated_map_scale_variable(flat_rules)
        fields = self._create_expression_fields(flat_rules)

        if flat_rule.get_attr("t") == 1:
            fields = self._add_label_expression_field(flat_rule, fields)

        transformation = self._get_geometry_transformation(flat_rule)
        if transformation:
            return self._apply_field_mapping(source_path, fields, transformation, flat_rule)

        return None

    def _apply_field_mapping(
        self, source_path: str, fields: Optional[list], transformation, flat_rule
    ) -> Optional[str]:
        """Apply field mapping and geometry transformation without touching UI memory layers."""
        output_dataset = join(self.utils_dir, f"{flat_rule.output_dataset}.fgb")
        if exists(output_dataset):
            return output_dataset

        field_mapping = [(4, '"fid"', f"{self.FIELD_PREFIX}_fid")]
        if fields:
            field_mapping.extend(fields)
        rule_description = f"'{flat_rule.get_description()}'"
        field_mapping.append((10, rule_description, f"{self.FIELD_PREFIX}_description"))

        temp_layer = QgsVectorLayer(source_path, "temp", "ogr")
        if self.include_required_fields_only != 0:
            all_fields = [(f.type(), f'"{f.name()}"', f"{f.name()}") for f in temp_layer.fields()] # pylint: disable=E1101
            field_mapping.extend(all_fields)

        field_mapping = [{"type": f[0], "expression": f[1], "name": f[2]} for f in field_mapping]
        current_input = source_path

        if flat_rule.rule.filterExpression():
            filtered_output = join(self.utils_dir, f"filt_{uuid4().hex[:8]}.fgb")
            params = {
                "INPUT": current_input,
                "EXPRESSION": flat_rule.rule.filterExpression(),
                "OUTPUT": filtered_output,
            }
            extracted_path = self._run_alg("extractbyexpression", "native", **params)

            check_layer = QgsVectorLayer(extracted_path, "check", "ogr")
            if not check_layer.isValid() or check_layer.featureCount() <= 0:
                return None
            current_input = extracted_path

        refactored = self._run_alg(
            "refactorfields", INPUT=current_input, FIELDS_MAPPING=field_mapping
        )
        return self._apply_transformation_thread_safe(refactored, transformation, output_dataset)

    def _apply_transformation_thread_safe(
        self, current_input: str, transformation, output_dataset: str
    ) -> Optional[str]:
        """Apply geometry transformation purely resolving file paths inside tasks."""
        geom_type, expression = abs(transformation[0] - 2), transformation[1]
        params = {"INPUT": current_input, "OUTPUT_GEOMETRY": geom_type, "EXPRESSION": expression}

        geom_transformed = self._run_alg("geometrybyexpression", **params)

        check_layer = QgsVectorLayer(geom_transformed, "check", "ogr")
        if not check_layer.isValid() or check_layer.featureCount() <= 0:
            return None

        removenull_parameters = {"INPUT": check_layer, "REMOVE_EMPTY": True}
        removed_nulls = self._run_alg("removenullgeometries", "native", **removenull_parameters)

        singleparts_params = {"INPUT": removed_nulls, "OUTPUT": output_dataset}
        singleparted = self._run_alg("multiparttosingleparts", "native", **singleparts_params)

        return singleparted

    def _updated_map_scale_variable(self, flat_rules):
        """Update expressions including @map_scale and replace with the current scale."""
        for flat_rule in flat_rules:
            rule_type = flat_rule.get_attr("t")
            zoom_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            if rule_type == 1 and flat_rule.rule.settings():
                settings = flat_rule.rule.settings()
                label_exp = settings.getLabelExpression().expression()
                if label_exp:
                    updated_exp = label_exp.replace("@map_scale", zoom_scale)
                    settings.fieldName = updated_exp
                if settings.geometryGeneratorEnabled:
                    updated_exp = settings.geometryGenerator.replace("@map_scale", zoom_scale)
                    settings.geometryGenerator = updated_exp
            else:
                symbol = flat_rule.rule.symbol()
                if not symbol:
                    continue
                symbol_layers = flat_rule.rule.symbol().symbolLayers()
                for layer in filter(lambda x: x.layerType() == "GeometryGenerator", symbol_layers):
                    generator_exp = layer.geometryExpression()
                    updated_exp = generator_exp.replace("@map_scale", zoom_scale)
                    layer.setGeometryExpression(updated_exp)

    def _get_polygon_centroids_expression(self):
        """Get polygon centroids expression based on
        user preference - visible polygon/whole polygon"""
        if self.cent_source == 1:
            extent_wkt = self.extent.asWktPolygon()
            polygons = f"intersection(@geometry, geom_from_wkt('{extent_wkt}'))"
        else:
            polygons = "@geometry"
        centroids = f"with_variable('source', {polygons}, if(intersects(centroid(@source), @source), centroid(@source),  point_on_surface(@source)))"  # pylint: disable=C0301
        return centroids

    def _add_label_expression_field(self, flat_rule: FlattenedRule, fields: list) -> list:
        """Add label expression as a calculated field."""
        field_name = f"{self.FIELD_PREFIX}_label"
        if not flat_rule.rule.settings():
            return fields
        label_exp = flat_rule.rule.settings().getLabelExpression().expression()
        if not label_exp:
            return fields
        filter_exp = f'"{label_exp}"' if not flat_rule.rule.settings().isExpression else label_exp
        fields.append([10, filter_exp, field_name])
        flat_rule.rule.settings().isExpression = False
        flat_rule.rule.settings().fieldName = field_name
        return fields

    def _get_geometry_transformation(self, flat_rule: FlattenedRule) -> Union[str, Tuple, None]:
        """Determine geometry transformation needed for rule."""
        transformation = None
        rule_type = flat_rule.get_attr("t")
        if rule_type == 0 and flat_rule.rule.symbol():
            transformation = self._get_renderer_transformation(flat_rule)
        elif rule_type == 1:
            transformation = self._get_labeling_transformation(flat_rule)
        if not transformation:
            return None
        extent_wkt = self.extent.asWktPolygon()
        clipped_geom = f"with_variable('clip',intersection({transformation[1]}, geom_from_wkt('{extent_wkt}')), if(not is_empty_or_null(@clip), @clip, NULL))"  # pylint: disable=C0301
        transformation[1] = clipped_geom
        return tuple(transformation)

    def _get_labeling_transformation(self, flat_rule: FlattenedRule) -> Union[Tuple, str, None]:
        """Get geometry transformation for labeling rules."""
        settings = flat_rule.rule.settings()
        target_geom = flat_rule.get_attr("g")
        transform_expr = "@geometry"

        if settings and settings.geometryGeneratorEnabled:
            target_geom = settings.geometryGeneratorType
            transform_expr = settings.geometryGenerator
            settings.geometryGeneratorEnabled = False
            flat_rule.set_attr("c", target_geom)
        elif target_geom == 2:  # Polygon to centroid
            flat_rule.set_attr("c", 0)
            target_geom = 0
            transform_expr = self._get_polygon_centroids_expression()
        return [target_geom, transform_expr]

    def _get_renderer_transformation(self, flat_rule: FlattenedRule) -> Union[Tuple, str, None]:
        """Get geometry transformation for renderer rules."""
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return None
        symbol_layer = flat_rule.rule.symbol().symbolLayers()[0]
        target_geom = flat_rule.get_attr("g")
        transform_expr = "@geometry"

        if symbol_layer.layerType() == "GeometryGenerator":
            target_geom = symbol_layer.subSymbol().type()
            transform_expr = symbol_layer.geometryExpression()
        else:
            target_geom = flat_rule.get_attr("c")
            source_geom = flat_rule.get_attr("g")
            if source_geom != target_geom:
                if target_geom == 0:
                    transform_expr = self._get_polygon_centroids_expression()
                elif target_geom == 1:
                    transform_expr = "boundary(@geometry)"
        return [target_geom, transform_expr]

    def _create_expression_fields(self, flat_rules) -> list:
        """Create calculated fields from data-driven properties."""
        fields = []
        for flat_rule in flat_rules:
            min_zoom = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            fields_list = DataDefinedPropertiesFetcher(flat_rule.rule, min_zoom).fetch()
            if not fields_list:
                continue
            fields.extend(fields_list)
        return fields

    def _run_alg(self, algorithm: str, algorithm_type: str = "native", **params):
        """Runs the processing algorithms securely inside isolated thread contexts."""
        # CRITICAL FIX: Instantiate an isolated Context and Feedback for the current thread.
        # This prevents `processing.run()` from attempting to default to the main thread's
        # QgsProject map layer registry (which triggers the fatal access violation).
        context = QgsProcessingContext()
        context.setExpressionContext(QgsProject.instance().createExpressionContext())

        task_feedback = QgsProcessingFeedback()

        if not params.get("OUTPUT") or params.get("OUTPUT") == "TEMPORARY_OUTPUT":
            ext = ".fgb" if algorithm == "convertformat" else ".gpkg"
            params["OUTPUT"] = join(self.utils_dir, f"temp_{uuid4().hex[:8]}{ext}")

        # pylint: disable=E1136
        output = run_processing(
            f"{algorithm_type}:{algorithm}", params, context=context, feedback=task_feedback
        )["OUTPUT"]

        if isinstance(output, QgsVectorLayer):
            return output.source()

        return output