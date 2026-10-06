"""
rules_exporter.py — Production-grade exporter for FlattenedRules.

Architecture
============
This module exports a large number of FlattenedRules to GeoParquet datasets
in a way that respects QGIS's strict thread-affinity rules and parallelises
*only* the work that is genuinely thread-safe to parallelise.

The legacy implementation crashed with native access violations on Windows
and suffered from "stuck task" deadlocks. The root causes were:

  1.  Live QgsVectorLayer objects (especially Postgres-backed) were passed
      from the main thread into worker QgsTask threads, then handed to
      processing.run(...). QObject thread-affinity rules forbid this; the
      Postgres provider in particular is not safe under cross-thread access
      and will eventually corrupt its connection state, producing access
      violations somewhere deep inside libpq.

  2.  Each worker called QgsProject.instance().createExpressionContext(),
      reaching into a main-thread QObject from a background thread.

  3.  QgsTask.run() was not returning a bool, and Python exceptions raised
      inside run() left tasks in inconsistent states. That is the
      "clock-icon, never starts" stuck-task symptom.

  4.  The polling loop with QCoreApplication.processEvents() introduced
      re-entrancy hazards when export() was itself running inside a
      QgsProcessingAlgorithm.

The redesign separates the export pipeline into clearly typed phases:

   ┌──────────────────────────────────────────────────────────────────────┐
   │ Phase 0 — Caller-thread snapshot                                     │
   │   * Mutate rule symbols / labeling settings (resolve @map_scale).    │
   │   * Capture every QObject we need into plain-Python data.            │
   │   * Per source: the export extent in the source CRS and the fields   │
   │     the rule expressions actually reference.                         │
   │   * After this phase NO QObject crosses a thread boundary.           │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 1 — Source materialisation (SERIAL, caller thread)             │
   │   * Each source is read from a fresh QgsVectorLayer constructed FROM │
   │     URI, filtered to the extent bbox and the referenced fields only, │
   │     and dumped to a local file. The bbox filter runs inside the      │
   │     provider (server-side for Postgres), so out-of-extent rows are   │
   │     never transferred.                                               │
   │   * Postgres / remote providers are not parallel-safe; we never read │
   │     more than one source concurrently. Each materialised source is   │
   │     handed to Phase 2 immediately, so reads overlap with processing. │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 2 — Base-layer pipeline (PARALLEL, file → file)                │
   │   * For each materialised source: fixgeometries → reproject →        │
   │     orig_id → singleparts → simplify.                                │
   │   * All inputs and outputs are file paths; no live layers cross      │
   │     threads.                                                         │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 3 — Rule export (PARALLEL, file → file)                        │
   │   * For each rule group, ONE streaming pass over the base layer:     │
   │     filter → field expressions → geometry transform → drop null /    │
   │     empty → explode multiparts, written straight to the output file. │
   │   * "Keep biggest part" groups first pick the largest part per       │
   │     orig_id (no dissolve / geometry union needed).                   │
   │   * Inputs are base-layer file paths and pure-data RuleGroupSnapshot │
   │     objects. No QObject access.                                      │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 4 — Result collection (caller thread)                          │
   │   * Wrap output Parquet files in QgsVectorLayer for the caller.      │
   │   * Cleanup of temp files.                                           │
   └──────────────────────────────────────────────────────────────────────┘

Phases 2 and 3 use a concurrent.futures.ThreadPoolExecutor rather than
QgsTask. This is intentional:

   * Predictable lifecycle: futures complete or raise — no "clock-icon"
     limbo state.
   * Per-future timeouts prevent any single hung algorithm from stalling
     the whole export.
   * Cancellation is a single shared-flag check between processing calls.
   * No QGIS task-manager re-entrancy with the parent processing
     algorithm.
   * No QCoreApplication.processEvents() polling loop.
"""

import os
import threading
import traceback
import platform
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from os.path import exists, join, splitext
from typing import Any, Dict, Iterator, List, Optional, Tuple
from uuid import uuid4
from qgis.PyQt.QtCore import QMetaType, QVariant
from processing import run as run_processing
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCsException,
    QgsDistanceArea,
    QgsExpressionContext,
    QgsExpression,
    QgsExpressionContextUtils,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsFeatureRequest,
    QgsRectangle,
    QgsCoordinateTransform,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsProject,
)

from ..utils.config import _DATA_SIMPLIFICATION_TOLERANCE, _EPSG_CRS, _FIELD_PREFIX
from ..utils.flattened_rule import FlattenedRule
from ..utils.zoom_levels import ZoomLevels
from .ddp_fetcher import DataDefinedPropertiesFetcher


# ============================================================================
# Module-level configuration
# ============================================================================

# Per processing.run() hard timeout. If a single algorithm exceeds this, the
# rule that triggered it is dropped from the export rather than allowed to
# stall the pipeline. Guarantees export() returns in bounded time regardless
# of bad data, network blips, or upstream bugs.
_PER_ALG_TIMEOUT_S = 600  # 10 minutes

# Hard cap on parallel workers, irrespective of cpu_percent. Beyond this
# point Windows runs out of OS handles, GDAL contention dominates, and the
# export actually gets slower. Empirically 4-6 is the sweet spot.
_MAX_WORKERS_HARD_CAP = 6

# Providers we treat as "must read serially" — anything backed by a remote
# database or HTTP endpoint where the underlying client library is not
# robust to concurrent use from multiple threads.
_SERIAL_READ_PROVIDERS = frozenset(
    {"postgres", "mssql", "oracle", "wfs", "spatialite", "hana", "db2"}
)

# Temp files prefered to be parquet but in linux which not support parquet they are became gpkg.
_TEMP_LAYER_FORMAT = 'sqlite'
_TEMP_RULE_FORMAT = 'gpkg'

# Features buffered per writer.addFeatures() call, and how often streaming
# loops poll for cancellation.
_WRITE_BATCH_SIZE = 1000

# Output WKB type per geometrybyexpression OUTPUT_GEOMETRY code
# (0 = polygon, 1 = line, 2 = point), already exploded to single parts.
_RULE_OUTPUT_WKB = {
    0: Qgis.WkbType.Polygon,
    1: Qgis.WkbType.LineString,
    2: Qgis.WkbType.Point,
}

# How an output field's value is produced in the rule-export pass.
_FIELD_CONSTANT, _FIELD_COPY, _FIELD_EXPRESSION = range(3)

# Expression fragments that read attributes dynamically, so their field
# usage can't be determined statically by referencedColumns().
_DYNAMIC_ATTRIBUTE_TOKENS = ("@feature", "$currentfeature")

# ============================================================================
# Snapshots — pure-Python data, no QObject references
# ============================================================================

@dataclass(frozen=True)
class _SourceSnapshot:
    """Everything a worker needs to read a base layer; carries no Qt objects."""
    layer_id: str
    name: str
    source_uri: str
    provider: str
    # Export extent in the source CRS as (xmin, ymin, xmax, ymax); None reads
    # the whole source (e.g. when the extent can't be transformed).
    extent: Optional[Tuple[float, float, float, float]] = None
    # Source field names referenced by any rule expression; None keeps all.
    required_fields: Optional[Tuple[str, ...]] = None

    @property
    def needs_serial_read(self) -> bool:
        return self.provider in _SERIAL_READ_PROVIDERS


@dataclass
class _RuleGroupSnapshot:
    """Everything a worker needs to export one output dataset.

    Every field is a primitive type, string, list or dataclass — no
    QObjects, no QgsVectorLayer references. Safe to consume from any thread.
    """
    output_dataset: str
    layer_id: str
    rule_type: int
    filter_expression: Optional[str]
    geometry_target: int
    geometry_expression: str
    expression_fields: List[Tuple[int, str, str]]
    description: str
    include_required_fields_only: int
    # Kept ONLY to drive the success/failure return value of export(); workers
    # MUST NOT read any live state from these.
    flat_rules: List[FlattenedRule]
    # Export only the largest part of each original feature (labels without
    # "label every part", centroid fills without "point on all parts").
    keep_biggest_part: bool = False


class _Cancelled(Exception):
    """Raised inside workers when the caller has signalled cancellation."""


# ============================================================================
# RulesExporter
# ============================================================================

class RulesExporter:
    """Export FlattenedRules to GeoParquet datasets, safely and in parallel.

    Public API is unchanged from the legacy implementation:

        exporter = RulesExporter(...)
        layers, rules = exporter.export()

    Internally the pipeline is split into phases that respect QGIS's strict
    thread-affinity rules. See module docstring for the architecture.
    """

    FIELD_PREFIX = "q2vt"

    def __init__(
        self,
        flattened_rules: List[FlattenedRule],
        extent: QgsRectangle,
        include_required_fields_only: int,
        max_zoom,
        utils_dir: str,
        cent_source: int,
        feedback: QgsProcessingFeedback,
        cpu_percent: int = 100,
    ):
        self.flattened_rules = flattened_rules
        # QgsRectangle is a value type — safe to share across threads.
        self.extent = extent
        self.include_required_fields_only = include_required_fields_only
        self.max_zoom = max_zoom
        self.cent_source = cent_source
        self.utils_dir = utils_dir
        self.feedback = feedback
        self.cpu_percent = cpu_percent

        self.processed_layers: List[QgsVectorLayer] = []

        # Captured once on the caller thread; QgsCoordinateTransformContext is
        # an implicitly shared value type, safe to read from workers.
        self._transform_context = QgsProject.instance().transformContext()

        # Temp-file tracking for cleanup.
        self._temp_files: set = set()
        self._temp_files_lock = threading.Lock()

        # Single lock used to serialise reads from "needs_serial_read"
        # providers, regardless of how many workers exist. Conservative but
        # absolutely safe — Postgres/WFS/etc. are read one at a time, full stop.
        self._serial_read_lock = threading.Lock()

        # Cancellation flag. Workers check this between processing.run calls.
        self._cancelled = threading.Event()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def export(self) -> Tuple[List[QgsVectorLayer], List[FlattenedRule]]:
        """Run the full export pipeline. Synchronous. Always returns."""
        try:
            # Phase 0 — caller-thread snapshot.
            if self._is_cancelled():
                return [], []
            sources, rule_groups = self._snapshot_caller_thread()

            # Phases 1 + 2 — source materialisation (serial providers on the
            # caller thread, the rest inside workers) and the parallel
            # base-layer pipeline (file → file).
            if self._is_cancelled():
                return [], []

            base_layers = self._build_base_layers(sources)

            # Phase 3 — parallel rule export (file → file).
            if self._is_cancelled():
                return [], []
            
            rule_outputs = self._export_rules_parallel(rule_groups, base_layers)
            # Phase 4 — collect results on caller thread.
            return self._collect_results(rule_groups, rule_outputs)
        finally:
            self._cleanup_temp_files()

    # -------------------------------------------------------------------
    # Cancellation
    # -------------------------------------------------------------------
    def _is_cancelled(self) -> bool:
        if self._cancelled.is_set():
            return True
        if self.feedback is not None and self.feedback.isCanceled():
            self._cancelled.set()
            return True
        return False

    def _check_cancel(self) -> None:
        if self._is_cancelled():
            raise _Cancelled()

    # -------------------------------------------------------------------
    # Phase 0 — snapshot (caller thread only)
    # -------------------------------------------------------------------
    def _snapshot_caller_thread(
        self,
    ) -> Tuple[Dict[str, _SourceSnapshot], List[_RuleGroupSnapshot]]:
        """Convert all live-QObject state into pure-Python snapshots.

        After this call, no worker thread will ever read from a live
        QgsVectorLayer, QgsRuleBasedRenderer.Rule, or QgsProject instance.
        """
        # Mutate rule symbols / labeling settings to bake in zoom scale.
        # MUST run on caller thread because it touches QObjects.
        self._resolve_map_scale_in_rules(self.flattened_rules)

        # Group rules by output dataset.
        rules_by_dataset: Dict[str, List[FlattenedRule]] = {}
        for r in self.flattened_rules:
            rules_by_dataset.setdefault(r.output_dataset, []).append(r)

        # Snapshot rule groups.
        rule_groups: List[_RuleGroupSnapshot] = []
        for output_dataset, flat_rules in rules_by_dataset.items():
            primary = flat_rules[0]

            # Compute expression fields (data-defined properties, optional
            # label field) — these read from the rule symbol/settings.
            expr_fields = self._create_expression_fields(flat_rules)
            if primary.get_attr("t") == 1:
                expr_fields = self._add_label_expression_field(
                    primary, expr_fields
                )

            # Compute geometry transformation tuple.
            transformation = self._get_geometry_transformation(primary)
            if transformation is None:
                # No geometry transformation → cannot be exported.
                continue
            geom_target, geom_expr = transformation

            rule_groups.append(_RuleGroupSnapshot(
                output_dataset=primary.output_dataset,
                layer_id=primary.layer.id(),
                rule_type=primary.get_attr("t"),
                filter_expression=primary.rule.filterExpression() or None,
                geometry_target=geom_target,
                geometry_expression=geom_expr,
                expression_fields=expr_fields,
                description=primary.get_description(),
                include_required_fields_only=self.include_required_fields_only,
                flat_rules=flat_rules,
                keep_biggest_part=self._keeps_biggest_part(primary),
            ))

        # Snapshot unique sources — only those feeding an exportable group.
        groups_by_layer: Dict[str, List[_RuleGroupSnapshot]] = {}
        for grp in rule_groups:
            groups_by_layer.setdefault(grp.layer_id, []).append(grp)

        sources: Dict[str, _SourceSnapshot] = {}
        for lid, groups in groups_by_layer.items():
            layer = groups[0].flat_rules[0].layer
            sources[lid] = _SourceSnapshot(
                layer_id=lid,
                name=layer.name(),
                source_uri=layer.source(),
                provider=layer.providerType(),
                extent=self._extent_in_crs(layer.crs()),
                required_fields=self._required_source_fields(groups),
            )

        return sources, rule_groups

    @staticmethod
    def _keeps_biggest_part(flat_rule: FlattenedRule) -> bool:
        rule_type = flat_rule.get_attr("t")
        if rule_type == 1:
            settings = flat_rule.rule.settings()
            return bool(settings and not settings.labelPerPart)
        if rule_type == 0 and flat_rule.rule.symbol():
            symbol_layer = flat_rule.rule.symbol().symbolLayers()[0]
            return (
                symbol_layer.layerType() == 'CentroidFill'
                and not symbol_layer.pointOnAllParts()
            )
        return False

    def _extent_in_crs(
        self, crs: QgsCoordinateReferenceSystem
    ) -> Optional[Tuple[float, float, float, float]]:
        """Export extent (EPSG:3857) as a bbox in ``crs``; None = no filter."""
        if not crs.isValid():
            return None
        extent_crs = QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}")
        rect = self.extent
        if crs != extent_crs:
            try:
                rect = QgsCoordinateTransform(
                    extent_crs, crs, self._transform_context
                ).transformBoundingBox(self.extent)
            except QgsCsException:
                return None
        if rect.isNull() or rect.isEmpty():
            return None
        return (rect.xMinimum(), rect.yMinimum(), rect.xMaximum(), rect.yMaximum())

    def _required_source_fields(
        self, groups: List[_RuleGroupSnapshot]
    ) -> Optional[Tuple[str, ...]]:
        """Names of source fields any of ``groups`` evaluates; None = all."""
        if self.include_required_fields_only != 0:
            return None  # All source fields are exported.
        names = set()
        for grp in groups:
            expressions = [grp.filter_expression, grp.geometry_expression]
            expressions.extend(expr for _, expr, _ in grp.expression_fields)
            for expr_str in expressions:
                if not isinstance(expr_str, str) or not expr_str.strip():
                    continue
                if any(token in expr_str for token in _DYNAMIC_ATTRIBUTE_TOKENS):
                    return None
                expr = QgsExpression(expr_str)
                if expr.hasParserError():
                    continue  # The group fails on this expression anyway.
                columns = expr.referencedColumns()
                if QgsFeatureRequest.ALL_ATTRIBUTES in columns:
                    return None
                names.update(columns)
        return tuple(sorted(names))

    # -------------------------------------------------------------------
    # Phase 1 — source materialisation
    # -------------------------------------------------------------------
    def _materialized_path(self, src: _SourceSnapshot) -> str:
        return join(self.utils_dir, f"materialized_{src.layer_id}.{_TEMP_LAYER_FORMAT}")

    def _materialize_source(self, src: _SourceSnapshot) -> Optional[str]:
        """Dump the in-extent, referenced-fields-only part of a source to a
        local file. Returns its path, or None if the source can't be read.

        Caller thread only: this is the only place in the pipeline where we
        touch a database/network provider.
        """
        self._check_cancel()
        out_path = self._materialized_path(src)
        if exists(out_path):
            # Idempotent restart support.
            return out_path

        # Open a FRESH layer in this thread. The original FlattenedRule.layer
        # reference may have main-thread affinity; here we deliberately don't
        # reuse it. The newly constructed layer is owned by this thread.
        layer = QgsVectorLayer(src.source_uri, src.name, src.provider)
        if not layer.isValid():
            self.feedback.pushWarning(
                f"Cannot open source '{src.name}' "
                f"(provider={src.provider}); skipping."
            )
            return None

        fields = layer.fields()
        if src.required_fields is None:
            indices = list(range(fields.count()))
        else:
            indices = sorted(
                {fields.lookupField(name) for name in src.required_fields} - {-1}
            )
        subset = len(indices) != fields.count()
        out_fields = QgsFields()
        for idx in indices:
            out_fields.append(fields.at(idx))

        request = QgsFeatureRequest()
        request.setInvalidGeometryCheck(
            QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck
        )
        if src.extent is not None:
            # Bbox-only test: the provider evaluates it with its spatial index
            # (server-side for databases) and it never trips on invalid
            # geometries. Rule geometry expressions clip to the exact extent.
            request.setFilterRect(QgsRectangle(*src.extent))
        if subset:
            request.setSubsetOfAttributes(indices)

        writer = self._create_writer(out_path, out_fields, layer.wkbType(), layer.crs())
        completed = False
        try:
            batch: List[QgsFeature] = []
            for feature in layer.getFeatures(request):
                if subset:
                    out = QgsFeature(out_fields, feature.id())
                    out.setAttributes([feature.attribute(i) for i in indices])
                    out.setGeometry(feature.geometry())
                    feature = out
                batch.append(feature)
                if len(batch) >= _WRITE_BATCH_SIZE:
                    self._check_cancel()
                    self._write_batch(writer, batch)
            self._write_batch(writer, batch)
            completed = True
        finally:
            del writer  # Closes the file.
            if not completed:
                self._remove_file(out_path)
        return out_path

    # -------------------------------------------------------------------
    # Phase 2 — parallel base-layer pipeline (file → file)
    # -------------------------------------------------------------------
    def _build_base_layers(
        self, sources: Dict[str, _SourceSnapshot]
    ) -> Dict[str, str]:
        """Materialise every source and run fix → reproject → orig_id →
        singleparts → simplify on it.

        Sources are materialised one at a time on this (caller) thread —
        opening project sources from worker threads can deadlock — and each
        is submitted to the pool as soon as it is on disk, so the caller
        reads the next source while workers process the previous ones.
        """
        target_paths: Dict[str, str] = {
            lid: join(self.utils_dir, f"map_layer_{lid}.{_TEMP_LAYER_FORMAT}")
            for lid in sources
        }
        # Idempotent skip.
        todo = {
            lid: src for lid, src in sources.items()
            if not exists(target_paths[lid])
        }

        if not todo:
            return target_paths

        max_workers = self._compute_pool_size(len(todo))

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rules-base"
        ) as pool:
            futures: Dict[Future, str] = {}
            for lid, src in todo.items():
                try:
                    if src.needs_serial_read:
                        with self._serial_read_lock:
                            src_path = self._materialize_source(src)
                    else:
                        src_path = self._materialize_source(src)
                except _Cancelled:
                    self.feedback.pushInfo("Base-layer build cancelled.")
                    return target_paths
                except Exception:  # noqa: BLE001  (we want to swallow per-source)
                    self.feedback.reportError(
                        f"Failed to export source '{src.name}':\n"
                        f"{traceback.format_exc()}"
                    )
                    continue
                if src_path is not None:
                    futures[pool.submit(
                        self._build_one_base_layer, src_path, target_paths[lid]
                    )] = lid

            for fut in self._iter_completed(futures):
                lid = futures[fut]
                try:
                    fut.result(timeout=_PER_ALG_TIMEOUT_S)
                except _Cancelled:
                    self.feedback.pushInfo("Base-layer build cancelled.")
                    return target_paths
                except Exception:  # noqa: BLE001
                    self.feedback.reportError(
                        f"Base-layer build failed for layer_id={lid}:\n"
                        f"{traceback.format_exc()}"
                    )
        return target_paths

    def _build_one_base_layer(self, src_path: str, dst_path: str) -> None:
        """Worker: run the cleanup chain on a materialised local file."""
        self._check_cancel()
        # Single geometry-fix pass, on in-extent features only.
        fixed = self._run_alg_safe(
            "fixgeometries", "native", INPUT=src_path, METHOD=0
        )
        self._check_cancel()
        reprojected = self._run_alg_safe(
            "reprojectlayer", "native",
            INPUT=fixed,
            TARGET_CRS=QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}"),
        )
        orig_id = self._run_alg_safe(
            "fieldcalculator", "native",
            INPUT=reprojected,
            FIELD_NAME=f'{_FIELD_PREFIX}_orig_id',
            FIELD_TYPE=0,
            FORMULA='to_int(@id)'
            )
        self._check_cancel()
        singleparted = self._run_alg_safe(
            "multiparttosingleparts", "native", INPUT=orig_id
        )
        self._check_cancel()
        self._run_alg_safe(
            "simplifygeometries", "native",
            INPUT=singleparted,
            METHOD=0,
            TOLERANCE=_DATA_SIMPLIFICATION_TOLERANCE,
            OUTPUT=dst_path,
        )

    # -------------------------------------------------------------------
    # Phase 3 — parallel rule export (file → file)
    # -------------------------------------------------------------------
    def _export_rules_parallel(
        self,
        rule_groups: List[_RuleGroupSnapshot],
        base_layers: Dict[str, str],
    ) -> Dict[str, Optional[str]]:
        """For each rule group, run filter → refactor → geometry chain in parallel."""
        outputs: Dict[str, Optional[str]] = {}
        if not rule_groups:
            return outputs

        max_workers = self._compute_pool_size(len(rule_groups))

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rules-export"
        ) as pool:
            futures: Dict[Future, _RuleGroupSnapshot] = {}
            for grp in rule_groups:
                src_path = base_layers.get(grp.layer_id)
                if not src_path or not exists(src_path):
                    outputs[grp.output_dataset] = None
                    continue
                fut = pool.submit(
                    self._export_one_rule_group, grp, src_path
                )
                futures[fut] = grp

            for fut in self._iter_completed(futures):
                grp = futures[fut]
                try:
                    outputs[grp.output_dataset] = fut.result(
                        timeout=_PER_ALG_TIMEOUT_S
                    )
                except _Cancelled:
                    self.feedback.pushInfo("Rule export cancelled.")
                    for pending_fut, pending_grp in futures.items():
                        outputs.setdefault(pending_grp.output_dataset, None)
                    return outputs
                except Exception:  # noqa: BLE001
                    self.feedback.reportError(
                        f"Rule export failed for '{grp.output_dataset}':\n"
                        f"{traceback.format_exc()}"
                    )
                    outputs[grp.output_dataset] = None
        return outputs

    def validate_expression(self, grp, expr_str: str):
        layer_name = grp.flat_rules[0].layer.name() or grp.layer_id
        rule_type = 'labeling' if grp.rule_type == 1 else 'symbology'
        warning_msg = f'The expression "{expr_str}" within the {rule_type} of the "{layer_name}" layer'

        if not isinstance(expr_str, str):
            self.feedback.pushWarning(f"{warning_msg} must be a string.")

        expr_str = expr_str.strip()

        if not expr_str:
            return None

        expr = QgsExpression(expr_str)

        if expr.hasParserError():
            self.feedback.pushWarning(f"{warning_msg} is not valid.")
            return None

        return expr_str

    def _export_one_rule_group(
        self, grp: _RuleGroupSnapshot, source_path: str
    ) -> Optional[str]:
        """Worker: export one rule group in a single streaming pass.

        Equivalent to the chain extractbyexpression → refactorfields →
        [dissolve → keepnbiggestparts] → geometrybyexpression →
        removenullgeometries → multiparttosingleparts, but every feature is
        read once and only the final output is written to disk.
        """
        self._check_cancel()
        output_path = join(self.utils_dir, f"{grp.output_dataset}.{_TEMP_RULE_FORMAT}")
        if exists(output_path):
            return output_path

        if grp.filter_expression and not self.validate_expression(grp, grp.filter_expression):
            return None
        if not self.validate_expression(grp, grp.geometry_expression):
            return None
        # layer.geometryType() returns 0 for point and 2 for polygon but
        # geometrybyexpression codes 0 as polygon and 2 as point, so flip it.
        out_wkb = _RULE_OUTPUT_WKB.get(abs(self._enum_int(grp.geometry_target) - 2))
        if out_wkb is None:
            return None

        layer = QgsVectorLayer(source_path, "base", "ogr")
        if not layer.isValid():
            return None
        src_fields = layer.fields()

        context = QgsProject.instance().createExpressionContext()
        context.appendScope(QgsExpressionContextUtils.layerScope(layer))
        context.setFields(src_fields)
        # Planar measurements, as with a default QgsProcessingContext.
        distance_area = QgsDistanceArea()
        distance_area.setSourceCrs(layer.crs(), self._transform_context)

        # Per output field: how to compute its value, cheapest kind first —
        # a constant, a straight copy of a source attribute, or an expression.
        out_fields = QgsFields()
        field_plan: List[Tuple[int, QgsField, Any, bool]] = []
        for m in self._build_field_mapping(grp, src_fields):
            field = QgsField(m["name"], QMetaType.Type(m["type"]))
            if not out_fields.append(field):
                continue  # Duplicate name: first definition wins.
            field_plan.append(self._plan_field(field, m["expression"], src_fields, context, distance_area))
        geom_expr = self._prepare_expression(grp.geometry_expression, context, distance_area)

        request = QgsFeatureRequest()
        request.setInvalidGeometryCheck(
            QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck
        )
        if grp.filter_expression:
            request.setFilterExpression(grp.filter_expression)
            request.setExpressionContext(context)
        if grp.keep_biggest_part:
            biggest = self._biggest_part_ids(layer, request)
            if biggest is not None:
                request = QgsFeatureRequest()
                request.setInvalidGeometryCheck(
                    QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck
                )
                request.setFilterFids(biggest)

        written = 0
        completed = False
        writer = self._create_writer(output_path, out_fields, out_wkb, layer.crs())
        try:
            batch: List[QgsFeature] = []
            for n, feature in enumerate(layer.getFeatures(request)):
                if n % _WRITE_BATCH_SIZE == 0:
                    self._check_cancel()
                context.setFeature(feature)

                geometry = geom_expr.evaluate(context)
                if geom_expr.hasEvalError():
                    raise RuntimeError(
                        f"Evaluation error in geometry expression: {geom_expr.evalErrorString()}"
                    )
                if geometry is None:
                    continue
                if not isinstance(geometry, QgsGeometry):
                    raise RuntimeError(f"{geometry!r} is not a geometry")
                if geometry.isNull() or geometry.isEmpty():
                    continue

                src_values = feature.attributes()
                attributes = []
                for kind, field, source, convert in field_plan:
                    if kind == _FIELD_CONSTANT:
                        attributes.append(source)
                        continue
                    if kind == _FIELD_COPY:
                        value = src_values[source]
                    else:
                        value = source.evaluate(context)
                        if source.hasEvalError():
                            raise RuntimeError(
                                f"Evaluation error in expression \"{source.expression()}\": "
                                f"{source.evalErrorString()}"
                            )
                    attributes.append(self._convert_value(field, value) if convert else value)

                parts = geometry.asGeometryCollection() if geometry.isMultipart() else [geometry]
                for part in parts:
                    out = QgsFeature(out_fields)
                    out.setAttributes(attributes)
                    out.setGeometry(part)
                    batch.append(out)
                if len(batch) >= _WRITE_BATCH_SIZE:
                    written += self._write_batch(writer, batch)
            written += self._write_batch(writer, batch)
            completed = True
        finally:
            del writer  # Closes the file.
            if not completed or written == 0:
                self._remove_file(output_path)
        return output_path if written else None

    def _biggest_part_ids(
        self, layer: QgsVectorLayer, request: QgsFeatureRequest
    ) -> Optional[List[int]]:
        """Feature ids of the largest part of each original feature.

        Base layers are exploded to single parts tagged with q2vt_orig_id, so
        the biggest part is simply the max-area (polygons) / max-length
        (lines) / first (points) feature per orig_id. Returns None if the
        layer has no orig_id field.
        """
        orig_idx = layer.fields().lookupField(f"{_FIELD_PREFIX}_orig_id")
        if orig_idx < 0:
            return None
        geometry_type = layer.geometryType()
        if geometry_type == Qgis.GeometryType.Polygon:
            measure = QgsGeometry.area
        elif geometry_type == Qgis.GeometryType.Line:
            measure = QgsGeometry.length
        else:
            measure = None

        ids_request = QgsFeatureRequest(request)
        # Filter-expression columns are added back automatically.
        ids_request.setSubsetOfAttributes([orig_idx])
        best: Dict[Any, Tuple[float, int]] = {}
        for n, feature in enumerate(layer.getFeatures(ids_request)):
            if n % _WRITE_BATCH_SIZE == 0:
                self._check_cancel()
            geometry = feature.geometry()
            if geometry.isNull():
                size = -1.0
            elif measure is None:
                size = 0.0
            else:
                size = measure(geometry)
            key = feature.attribute(orig_idx)
            current = best.get(key)
            if current is None or size > current[0]:
                best[key] = (size, feature.id())
        return [fid for _, fid in best.values()]

    def _plan_field(
        self,
        field: QgsField,
        expr_str: str,
        src_fields: QgsFields,
        context: QgsExpressionContext,
        distance_area: QgsDistanceArea,
    ) -> Tuple[int, QgsField, Any, bool]:
        """(kind, field, source, convert) for one output field.

        Constants are evaluated and converted once; plain references to a
        source field are copied by index (and only converted when the types
        differ); anything else is evaluated per feature.
        """
        expr = self._prepare_expression(expr_str, context, distance_area)
        root = expr.rootNode()
        if root is not None and root.hasCachedStaticValue():
            return (_FIELD_CONSTANT, field, self._convert_value(field, root.cachedStaticValue()), False)
        if expr.isField():
            idx = src_fields.lookupField(next(iter(expr.referencedColumns())))
            if idx >= 0:
                same_type = self._enum_int(src_fields.at(idx).type()) == self._enum_int(field.type())
                return (_FIELD_COPY, field, idx, not same_type)
        return (_FIELD_EXPRESSION, field, expr, True)

    @staticmethod
    def _convert_value(field: QgsField, value):
        try:
            return field.convertCompatible(value)
        except ValueError as e:
            raise RuntimeError(
                f"Could not convert value for field {field.name()}: {e}"
            ) from e

    @staticmethod
    def _prepare_expression(
        expr_str: str, context: QgsExpressionContext, distance_area: QgsDistanceArea
    ) -> QgsExpression:
        expr = QgsExpression(expr_str)
        if expr.hasParserError():
            raise RuntimeError(
                f"Parser error in expression \"{expr_str}\": {expr.parserErrorString()}"
            )
        expr.setGeomCalculator(distance_area)
        expr.prepare(context)
        return expr

    def _build_field_mapping(
        self, grp: _RuleGroupSnapshot, src_fields: QgsFields
    ) -> List[Dict[str, Any]]:
        """Worker: assemble the output field definitions (type, expression, name)."""
        mapping: List[Tuple[int, str, str]] = []
        mapping.append(
            (10, grp.description, f"{_FIELD_PREFIX}_description")
        )
        mapping.extend(grp.expression_fields)
        if grp.include_required_fields_only != 0:
            for f in src_fields:
                if 'ogc_fid' not in f.name().lower():
                    mapping.append((f.type(), f'"{f.name()}"', f.name()))

        mapping.append(
            (6, f'"{_FIELD_PREFIX}_orig_id"', f"{_FIELD_PREFIX}_orig_id")
        )
        return [
            {
                # Untyped data-defined properties are exported as strings.
                "type": 10 if m[0] is None else self._enum_int(m[0]),
                "expression": m[1],
                "name": m[2],
            }
            for m in mapping
        ]

    # -------------------------------------------------------------------
    # Phase 4 — result collection (caller thread)
    # -------------------------------------------------------------------
    def _collect_results(
        self,
        rule_groups: List[_RuleGroupSnapshot],
        rule_outputs: Dict[str, Optional[str]],
    ) -> Tuple[List[QgsVectorLayer], List[FlattenedRule]]:
        """Wrap successful outputs in QgsVectorLayer; report failures."""
        successful_rules: List[FlattenedRule] = []
        for grp in rule_groups:
            out_path = rule_outputs.get(grp.output_dataset)
            on_disk = join(self.utils_dir, f"{grp.output_dataset}.{_TEMP_RULE_FORMAT}")
            if not out_path or not exists(on_disk):
                # Drop these rules from the caller's flat list.
                for rule in grp.flat_rules:
                    if rule in self.flattened_rules:
                        self.flattened_rules.remove(rule)
                continue
            layer = QgsVectorLayer(on_disk, grp.output_dataset, "ogr")
            if layer.isValid() and layer.featureCount() > 0:
                self.processed_layers.append(layer)
                successful_rules.extend(grp.flat_rules)
            else:
                for rule in grp.flat_rules:
                    if rule in self.flattened_rules:
                        self.flattened_rules.remove(rule)
        return self.processed_layers, successful_rules

    # -------------------------------------------------------------------
    # Worker-safe processing runner
    # -------------------------------------------------------------------
    def _run_alg_safe(
        self,
        algorithm: str,
        algorithm_type: str = "native",
        **params,
    ) -> str:
        """Run a processing algorithm with NO main-thread state access.

        * Fresh QgsProcessingContext per call.
        * Minimal expression context (global scope only) — never
          QgsProject.instance().
        * Per-call QgsProcessingFeedback.
        * Returns an output path (string), never a live layer reference.
        """
        self._check_cancel()
        context = QgsProcessingContext()
        context.setExpressionContext(QgsProject.instance().createExpressionContext())
        context.setInvalidGeometryCheck(QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck)
        feedback = QgsProcessingFeedback()

        if params.get("OUTPUT") in (None, "TEMPORARY_OUTPUT"):
            params["OUTPUT"] = self._temp_path("temp")

        full_name = f"{algorithm_type}:{algorithm}"
        # pylint: disable=E1111
        result = run_processing(
            full_name, params, context=context, feedback=feedback
        )
        output = result.get("OUTPUT")
        # If processing returned a layer, surface its source path. We never
        # let a live QgsVectorLayer escape into our pipeline data flow.
        if isinstance(output, QgsVectorLayer):
            return output.source()
        return output

    @staticmethod
    def _make_worker_expression_context() -> QgsExpressionContext:
        """Minimal expression context safe for worker-thread use.

        Crucially does NOT call QgsProject.instance().createExpressionContext()
        — that walks scopes which include layer references and is the original
        implementation's biggest thread-affinity violation.
        """
        ctx = QgsExpressionContext()
        ctx.appendScope(QgsExpressionContextUtils.globalScope())
        return ctx

    # -------------------------------------------------------------------
    # Snapshot helpers — caller-thread only
    # -------------------------------------------------------------------
    def _resolve_map_scale_in_rules(self, flat_rules: list) -> None:
        """Replace @map_scale references with each rule's zoom scale.

        Runs on caller thread because it mutates QObject state (rule symbols
        and labeling settings).
        """
        for flat_rule in flat_rules:
            rule_type = flat_rule.get_attr("t")
            zoom_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            if rule_type == 1 and flat_rule.rule.settings():
                settings = flat_rule.rule.settings()
                label_exp = settings.getLabelExpression().expression()
                if label_exp:
                    settings.fieldName = label_exp.replace(
                        "@map_scale", zoom_scale
                    )
                if settings.geometryGeneratorEnabled:
                    settings.geometryGenerator = (
                        settings.geometryGenerator.replace(
                            "@map_scale", zoom_scale
                        )
                    )
            else:
                symbol = flat_rule.rule.symbol()
                if not symbol:
                    continue
                for layer in symbol.symbolLayers():
                    if layer.layerType() == "GeometryGenerator":
                        layer.setGeometryExpression(
                            layer.geometryExpression().replace(
                                "@map_scale", zoom_scale
                            )
                        )

    def _create_expression_fields(
        self, flat_rules: list
    ) -> List[Tuple[int, str, str]]:
        """Build calculated-field entries from data-driven properties."""
        fields: List[Tuple[int, str, str]] = []
        for flat_rule in flat_rules:
            rule_type = flat_rule.get_attr("t")
            suffix = flat_rule.get_attr("s") if rule_type == 0 else flat_rule.get_attr("f")
            min_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            rule_fields = DataDefinedPropertiesFetcher(
                flat_rule.rule, min_scale, suffix
            ).fetch()
            if rule_fields:
                # Normalise to tuples of primitives so the snapshot is
                # guaranteed-immutable.
                fields.extend(tuple(f) for f in rule_fields)
        return fields

    def _add_label_expression_field(
        self,
        flat_rule: FlattenedRule,
        fields: List[Tuple[int, str, str]],
    ) -> List[Tuple[int, str, str]]:
        if not flat_rule.rule.settings():
            return fields
        label_exp = flat_rule.rule.settings().getLabelExpression().expression()
        if not label_exp:
            return fields
        field_name = f"{_FIELD_PREFIX}_label"
        filter_exp = (
            f'"{label_exp}"'
            if not flat_rule.rule.settings().isExpression
            else label_exp
        )
        fields.append((10, filter_exp, field_name))
        flat_rule.rule.settings().isExpression = False
        flat_rule.rule.settings().fieldName = field_name
        return fields

    def _get_geometry_transformation(
        self, flat_rule: FlattenedRule
    ) -> Optional[Tuple[int, str]]:
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

    def _get_labeling_transformation(self, flat_rule: FlattenedRule):
        settings = flat_rule.rule.settings()
        target_geom = flat_rule.get_attr("g")
        transform_expr = "@geometry"
        if settings and settings.geometryGeneratorEnabled:
            target_geom = settings.geometryGeneratorType
           
            generator_exp = settings.geometryGenerator
            layer_crs = flat_rule.layer.crs().authid()
            generator_exp = generator_exp.replace('@geometry', f"transform(@geometry, 'EPSG:3857', '{layer_crs}')")
            transform_expr =  f"transform({generator_exp}, '{layer_crs}',  'EPSG:3857')"
            settings.geometryGeneratorEnabled = False
            flat_rule.set_attr("c", target_geom)
        elif target_geom == 2:
            flat_rule.set_attr("c", 0)
            target_geom = 0
            transform_expr = self._get_polygon_centroids_expression()
        return [target_geom, transform_expr]

    def _get_renderer_transformation(self, flat_rule: FlattenedRule):
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return None
        symbol_layer = symbol.symbolLayers()[0]
        target_geom = flat_rule.get_attr("g")
        transform_expr = "@geometry"
        if symbol_layer.layerType() == "GeometryGenerator":
            target_geom = symbol_layer.subSymbol().type()
            generator_exp = symbol_layer.geometryExpression()
            layer_crs = flat_rule.layer.crs().authid()
            generator_exp = generator_exp.replace('@geometry', f"transform(@geometry, 'EPSG:3857', '{layer_crs}')")
            transform_expr =  f"transform({generator_exp}, '{layer_crs}',  'EPSG:3857')"
        else:
            target_geom = flat_rule.get_attr("c")
            source_geom = flat_rule.get_attr("g")
            if source_geom != target_geom:
                if target_geom == 0:
                    transform_expr = self._get_polygon_centroids_expression()
                elif target_geom == 1:
                    transform_expr = "boundary(@geometry)"
        return [target_geom, transform_expr]

    def _get_polygon_centroids_expression(self) -> str:
        if self.cent_source == 1:
            polygons = (
                f"intersection(@geometry, "
                f"geom_from_wkt('{self.extent.asWktPolygon()}'))"
            )
        else:
            polygons = "@geometry"
        return (
            f"with_variable('source', {polygons}, "
            f"if(intersects(centroid(@source), @source), "
            f"centroid(@source), point_on_surface(@source)))"
        )

    # -------------------------------------------------------------------
    # Pool sizing, future iteration, temp tracking
    # -------------------------------------------------------------------
    def _compute_pool_size(self, num_jobs: int) -> int:
        if num_jobs <= 0:
            return 1
        cpu_n = os.cpu_count() or 1
        from_user = max(1, int(cpu_n * self.cpu_percent / 100))
        return min(from_user, _MAX_WORKERS_HARD_CAP, num_jobs)

    def _iter_completed(
        self, futures: Dict[Future, Any]
    ) -> Iterator[Future]:
        """Yield futures as they complete, polling cancellation each second.

        Unlike concurrent.futures.as_completed, this checks our cancel flag
        between waits so an external cancellation request is responsive even
        when current futures are still running.
        """
        pending = set(futures.keys())
        while pending:
            done, pending = wait(
                pending, timeout=1.0, return_when=FIRST_COMPLETED
            )
            for fut in done:
                yield fut
            if self._is_cancelled():
                # Best-effort: cancel anything not yet started. Already-running
                # futures will exit at their next _check_cancel().
                for fut in pending:
                    fut.cancel()
                return

    def _temp_path(self, prefix: str = "temp") -> str:
        """Allocate a tracked temp path inside utils_dir."""
        p = join(self.utils_dir, f"{prefix}_{uuid4().hex}.{_TEMP_RULE_FORMAT}")
        with self._temp_files_lock:
            self._temp_files.add(p)
        return p

    def _cleanup_temp_files(self) -> None:
        with self._temp_files_lock:
            paths = list(self._temp_files)
            self._temp_files.clear()
        for p in paths:
            self._remove_file(p)

    @staticmethod
    def _remove_file(path: str) -> None:
        try:
            if exists(path):
                os.remove(path)
        except OSError:
            pass

    # -------------------------------------------------------------------
    # Direct file writing (worker-safe: every object is thread-local)
    # -------------------------------------------------------------------
    def _create_writer(
        self,
        path: str,
        fields: QgsFields,
        wkb_type,
        crs: QgsCoordinateReferenceSystem,
    ) -> QgsVectorFileWriter:
        """Open a vector file writer configured like a processing output."""
        driver = QgsVectorFileWriter.driverForExtension(splitext(path)[1])
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = driver
        options.fileEncoding = "UTF-8"
        options.datasourceOptions = QgsVectorFileWriter.defaultDatasetOptions(driver)
        options.layerOptions = QgsVectorFileWriter.defaultLayerOptions(driver)
        writer = QgsVectorFileWriter.create(
            path, fields, wkb_type, crs, self._transform_context, options,
            # Exploded parts share source attributes, including any "fid".
            QgsFeatureSink.SinkFlag.RegeneratePrimaryKey,
        )
        if writer is None or writer.hasError() != QgsVectorFileWriter.WriterError.NoError:
            message = writer.errorMessage() if writer is not None else ""
            del writer
            raise RuntimeError(f"Cannot create '{path}': {message}")
        return writer

    @staticmethod
    def _write_batch(writer: QgsVectorFileWriter, batch: List[QgsFeature]) -> int:
        """Flush ``batch`` to ``writer`` and clear it; returns features written."""
        if not batch:
            return 0
        if not writer.addFeatures(batch, QgsFeatureSink.Flag.FastInsert):
            raise RuntimeError(f"Cannot write features: {writer.lastError()}")
        count = len(batch)
        batch.clear()
        return count

    @staticmethod
    def _enum_int(value) -> int:
        """int() of a plain int or a (PyQt5 / PyQt6) enum member."""
        return int(getattr(value, "value", value))