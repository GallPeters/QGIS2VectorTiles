"""
rules_exporter.py

_ParallelExportTask — lightweight QgsTask wrapper that runs a callable in a
                      background QGIS task (internal to this module).

RulesExporter — exports every FlattenedRule to a GeoParquet dataset on disk,
applying geometry transformations, field mappings, and data-defined-property
fields.  Uses _ParallelExportTask for parallelism.

Depends on: config, zoom_levels, flattened_rule, ddp_fetcher
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
    """Internal task for executing export processing in a background thread safely."""

    def __init__(self, description, func, *args, **kwargs):
        super().__init__(description, QgsTask.CanCancel)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None

    def run(self):
        """Execute the provided function and capture its result."""
        sleep(0.5)  # Minor jitter to avoid concurrent GDAL initialization races.
        self.result = self.func(*self.args, **self.kwargs)


class RulesExporter:
    """Export all rules to GeoParquet datasets with geometry transformations."""

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
        self.processed_layers: List[QgsVectorLayer] = []
        self._source_locks: dict = {}
        self._source_locks_mutex = threading.Lock()

    def export(self) -> Tuple[List[QgsVectorLayer], List[FlattenedRule]]:
        """Export all rules using parallel QgsTask workers; return (layers, rules) on success."""
        num_workers = max(1, int(os.cpu_count() * self.cpu_percent / 100))

        output_datasets = self._export_base_layers(num_workers)

        rule_tasks = []
        total_datasets = len(output_datasets)
        for index, flat_rules in enumerate(output_datasets.values()):
            desc = f"Exporting rule {index + 1}/{total_datasets}"
            task = _ParallelExportTask(desc, self._export_rule_thread_safe, flat_rules)
            rule_tasks.append((task, flat_rules))

        self._run_tasks(rule_tasks, num_workers)

        successful_rules: List[FlattenedRule] = []
        for task, flat_rules in rule_tasks:
            dataset_path = join(self.utils_dir, f"{flat_rules[0].output_dataset}.parquet")
            if exists(dataset_path):
                layer = QgsVectorLayer(task.result, flat_rules[0].output_dataset, "ogr")
                if layer.isValid() and layer.featureCount() > 0:
                    self.processed_layers.append(layer)
                    successful_rules.extend(flat_rules)
            else:
                for rule in flat_rules:
                    if rule in self.flattened_rules:
                        self.flattened_rules.remove(rule)

        return self.processed_layers, successful_rules

    def _run_tasks(self, tasks_with_meta: list, num_workers: int):
        """Throttle and process QgsTasks, waiting for their completion."""
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
                manager.addTask(t_tuple[0])
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

    def _export_base_layers(self, num_workers: int) -> dict:
        """Export unique source vector layers to GeoParquet using parallel tasks."""
        output_datasets = {rule.output_dataset: [] for rule in self.flattened_rules}
        tasks_meta = []
        unique_layer_ids: set = set()

        for flat_rule in self.flattened_rules:
            output_datasets[flat_rule.output_dataset].append(flat_rule)
            layer_id = flat_rule.layer.id()
            if layer_id not in unique_layer_ids:
                output_path = join(self.utils_dir, f"map_layer_{layer_id}.parquet")
                if not exists(output_path):
                    task = _ParallelExportTask(
                        f"Exporting layer {flat_rule.layer.name()}",
                        self._process_base_layer,
                        flat_rule.layer,
                        output_path,
                        self.extent,
                        f"EPSG:{_EPSG_CRS}",
                    )
                    tasks_meta.append((task, None))
                unique_layer_ids.add(layer_id)

        if tasks_meta:
            self._run_tasks(tasks_meta, num_workers)

        return output_datasets

    def _get_source_lock(self, input_source: str) -> threading.Lock:
        """Return a per-file lock for the given source path.

        GeoPackage files use SQLite file-level locking; serialising concurrent
        access to the same .gpkg avoids GDAL metadata-write deadlocks.
        """
        base_path = input_source.split("|")[0]
        with self._source_locks_mutex:
            if base_path not in self._source_locks:
                self._source_locks[base_path] = threading.Lock()
            return self._source_locks[base_path]

    def _process_base_layer(
        self, input_layer, output_path: str, extent, output_crs: str
    ) -> str:
        """Thread-safe base layer creation: fix, reproject, clip, simplify."""
        lock = self._get_source_lock(input_layer.source())
        with lock:
            fixed_linework = self._run_alg(
                "fixgeometries", "native", INPUT=input_layer, METHOD=0
            )

        fixed_structure = self._run_alg(
            "fixgeometries", "native", INPUT=fixed_linework, METHOD=1
        )
        reprojected = self._run_alg(
            "reprojectlayer", "native",
            INPUT=fixed_structure,
            TARGET_CRS=QgsCoordinateReferenceSystem(output_crs),
        )
        clipped = self._run_alg(
            "extractbyextent", "native", INPUT=reprojected, EXTENT=extent, CLIP=False
        )
        singleparted = self._run_alg(
            "multiparttosingleparts", "native", INPUT=clipped
        )
        return self._run_alg(
            "simplifygeometries", "native",
            INPUT=singleparted,
            METHOD=0,
            TOLERANCE=_DATA_SIMPLIFICATION_TOLERANCE,
            OUTPUT=output_path,
        )

    def _export_rule_thread_safe(self, flat_rules: list) -> Optional[str]:
        """Export a group of rules sharing the same output dataset."""
        flat_rule = flat_rules[0]
        source_path = join(self.utils_dir, f"map_layer_{flat_rule.layer.id()}.parquet")

        if not exists(source_path):
            return None

        source_layer = QgsVectorLayer(source_path, "temp", "ogr")
        if not source_layer.isValid() or source_layer.featureCount() <= 0:
            return None

        self._resolve_map_scale_in_rules(flat_rules)
        fields = self._create_expression_fields(flat_rules)

        if flat_rule.get_attr("t") == 1:
            fields = self._add_label_expression_field(flat_rule, fields)

        transformation = self._get_geometry_transformation(flat_rule)
        if transformation:
            return self._apply_field_mapping(source_path, fields, transformation, flat_rule)

        return None

    def _apply_field_mapping(
        self, source_path: str, fields: Optional[list], transformation, flat_rule: FlattenedRule
    ) -> Optional[str]:
        """Apply field mapping and geometry transformation to produce the output dataset."""
        output_dataset = join(self.utils_dir, f"{flat_rule.output_dataset}.parquet")
        if exists(output_dataset):
            return output_dataset

        field_mapping = [(4, '"fid"', f"{self.FIELD_PREFIX}_fid")]
        if fields:
            field_mapping.extend(fields)
        field_mapping.append(
            (10, f"'{flat_rule.get_description()}'", f"{self.FIELD_PREFIX}_description")
        )

        if self.include_required_fields_only != 0:
            temp_layer = QgsVectorLayer(source_path, "temp", "ogr")
            all_fields = [
                (f.type(), f'"{f.name()}"', f"{f.name()}") for f in temp_layer.fields()  # pylint: disable=E1101
            ]
            field_mapping.extend(all_fields)

        field_mapping = [{"type": f[0], "expression": f[1], "name": f[2]} for f in field_mapping]
        current_input = source_path

        if flat_rule.rule.filterExpression():
            filtered_output = join(self.utils_dir, f"filt_{uuid4().hex[:8]}.parquet")
            extracted = self._run_alg(
                "extractbyexpression", "native",
                INPUT=current_input,
                EXPRESSION=flat_rule.rule.filterExpression(),
                OUTPUT=filtered_output,
            )
            check = QgsVectorLayer(extracted, "check", "ogr")
            if not check.isValid() or check.featureCount() <= 0:
                return None
            current_input = extracted

        refactored = self._run_alg(
            "refactorfields", INPUT=current_input, FIELDS_MAPPING=field_mapping
        )
        return self._apply_geometry_transformation(refactored, transformation, output_dataset)

    def _apply_geometry_transformation(
        self, current_input: str, transformation, output_dataset: str
    ) -> Optional[str]:
        """Apply geometry transformation expression to produce the final output file."""
        geom_type = abs(transformation[0] - 2)
        expression = transformation[1]
        geom_transformed = self._run_alg(
            "geometrybyexpression",
            INPUT=current_input,
            OUTPUT_GEOMETRY=geom_type,
            EXPRESSION=expression,
        )

        check = QgsVectorLayer(geom_transformed, "check", "ogr")
        if not check.isValid() or check.featureCount() <= 0:
            return None

        removed_nulls = self._run_alg(
            "removenullgeometries", "native", INPUT=check, REMOVE_EMPTY=True
        )
        return self._run_alg(
            "multiparttosingleparts", "native", INPUT=removed_nulls, OUTPUT=output_dataset
        )

    def _resolve_map_scale_in_rules(self, flat_rules: list):
        """Replace @map_scale references in all expressions with the rule's zoom scale."""
        for flat_rule in flat_rules:
            rule_type = flat_rule.get_attr("t")
            zoom_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            if rule_type == 1 and flat_rule.rule.settings():
                settings = flat_rule.rule.settings()
                label_exp = settings.getLabelExpression().expression()
                if label_exp:
                    settings.fieldName = label_exp.replace("@map_scale", zoom_scale)
                if settings.geometryGeneratorEnabled:
                    settings.geometryGenerator = settings.geometryGenerator.replace(
                        "@map_scale", zoom_scale
                    )
            else:
                symbol = flat_rule.rule.symbol()
                if not symbol:
                    continue
                for layer in symbol.symbolLayers():
                    if layer.layerType() == "GeometryGenerator":
                        layer.setGeometryExpression(
                            layer.geometryExpression().replace("@map_scale", zoom_scale)
                        )

    def _get_polygon_centroids_expression(self) -> str:
        """Return a centroid expression based on the user's centroid source preference."""
        if self.cent_source == 1:
            polygons = f"intersection(@geometry, geom_from_wkt('{self.extent.asWktPolygon()}'))"
        else:
            polygons = "@geometry"
        return (
            f"with_variable('source', {polygons}, "
            f"if(intersects(centroid(@source), @source), "
            f"centroid(@source), point_on_surface(@source)))"
        )

    def _add_label_expression_field(
        self, flat_rule: FlattenedRule, fields: list
    ) -> list:
        """Add the label expression as a calculated field and update settings to reference it."""
        if not flat_rule.rule.settings():
            return fields
        label_exp = flat_rule.rule.settings().getLabelExpression().expression()
        if not label_exp:
            return fields

        field_name = f"{self.FIELD_PREFIX}_label"
        filter_exp = (
            f'"{label_exp}"' if not flat_rule.rule.settings().isExpression else label_exp
        )
        fields.append([10, filter_exp, field_name])
        flat_rule.rule.settings().isExpression = False
        flat_rule.rule.settings().fieldName = field_name
        return fields

    def _get_geometry_transformation(
        self, flat_rule: FlattenedRule
    ) -> Union[tuple, None]:
        """Determine the geometry transformation tuple for the rule."""
        rule_type = flat_rule.get_attr("t")
        if rule_type == 0 and flat_rule.rule.symbol():
            transformation = self._get_renderer_transformation(flat_rule)
        elif rule_type == 1:
            transformation = self._get_labeling_transformation(flat_rule)
        else:
            return None

        if not transformation:
            return None

        extent_wkt = self.extent.asWktPolygon()
        clipped = (
            f"with_variable('clip', intersection({transformation[1]}, "
            f"geom_from_wkt('{extent_wkt}')), "
            f"if(not is_empty_or_null(@clip), @clip, NULL))"
        )
        transformation[1] = clipped
        return tuple(transformation)

    def _get_labeling_transformation(
        self, flat_rule: FlattenedRule
    ) -> Union[list, None]:
        """Get geometry transformation for labeling rules."""
        settings = flat_rule.rule.settings()
        target_geom = flat_rule.get_attr("g")
        transform_expr = "@geometry"

        if settings and settings.geometryGeneratorEnabled:
            target_geom = settings.geometryGeneratorType
            transform_expr = settings.geometryGenerator
            settings.geometryGeneratorEnabled = False
            flat_rule.set_attr("c", target_geom)
        elif target_geom == 2:  # Polygon → centroid
            flat_rule.set_attr("c", 0)
            target_geom = 0
            transform_expr = self._get_polygon_centroids_expression()

        return [target_geom, transform_expr]

    def _get_renderer_transformation(
        self, flat_rule: FlattenedRule
    ) -> Union[list, None]:
        """Get geometry transformation for renderer rules."""
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return None

        symbol_layer = symbol.symbolLayers()[0]
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

    def _create_expression_fields(self, flat_rules: list) -> list:
        """Build calculated-field entries from data-driven properties across all rules."""
        fields = []
        for flat_rule in flat_rules:
            min_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            rule_fields = DataDefinedPropertiesFetcher(flat_rule.rule, min_scale).fetch()
            if rule_fields:
                fields.extend(rule_fields)
        return fields

    def _run_alg(self, algorithm: str, algorithm_type: str = "native", **params) -> str:
        """Run a processing algorithm in an isolated thread-safe context."""
        context = QgsProcessingContext()
        context.setExpressionContext(QgsProject.instance().createExpressionContext())
        task_feedback = QgsProcessingFeedback()

        if params.get("OUTPUT") in (None, "TEMPORARY_OUTPUT"):
            params["OUTPUT"] = join(self.utils_dir, f"temp_{uuid4().hex[:8]}.parquet")

        # pylint: disable=E1136
        output = run_processing(
            f"{algorithm_type}:{algorithm}", params, context=context, feedback=task_feedback
        )["OUTPUT"]

        if isinstance(output, QgsVectorLayer):
            return output.source()

        return output
