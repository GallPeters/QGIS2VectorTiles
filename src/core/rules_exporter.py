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
   │ Phase 1 — Source reading (SERIAL, caller thread)                     │
   │   * Each source is read from a fresh QgsVectorLayer constructed FROM │
   │     URI, filtered to the extent bbox and the referenced fields only. │
   │     The bbox filter runs inside the provider (server-side for        │
   │     Postgres), so out-of-extent rows are never transferred.          │
   │   * Postgres / remote providers are not parallel-safe; we never read │
   │     more than one source concurrently. Features go to Phase 2 in     │
   │     batches as they are read, so reads overlap with processing.      │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 2 — Base-layer pipeline (PARALLEL batches, caller writes)      │
   │   * Workers take each batch through fix geometry → reproject →       │
   │     explode multiparts → simplify; batches of every source, and      │
   │     several batches of one large source, are processed at once.      │
   │   * The caller thread writes finished batches to the base layer in   │
   │     reading order, numbering orig_id as it goes. No intermediate     │
   │     copy of the source is written.                                   │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 3 — Rule export (PARALLEL, file → file)                        │
   │   * Rule groups of one base layer that share a filter are exported   │
   │     in ONE streaming pass: filter → geometry transform → clip to the │
   │     extent → drop null / empty → explode multiparts → field          │
   │     expressions, written straight to each group's output file.       │
   │     Expressions shared by several groups (typically the geometry     │
   │     transform) are evaluated once per feature.                       │
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
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from os.path import exists, join, splitext
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple
from uuid import uuid4
from qgis.PyQt import sip
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
    QgsWkbTypes,
)

from ..utils.config import _DATA_SIMPLIFICATION_TOLERANCE, _EPSG_CRS, _FIELD_PREFIX
from ..utils.flattened_rule import FlattenedRule
from ..utils.zoom_levels import ZoomLevels
from .base_layer_cache import BaseLayerCache
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

# Base layers and rule outputs. FlatGeobuf, not SQLite/SpatiaLite/GeoPackage:
# many threads create and read these files at once, and SpatiaLite's
# per-connection setup isn't thread-safe (it calls setlocale), which corrupted
# the heap. Written without a spatial index (nothing reads them by area, and
# the index would reorder features), so writing is a plain append.
_TEMP_LAYER_FORMAT = 'fgb'
_TEMP_RULE_FORMAT = 'fgb'

# Features buffered per writer.addFeatures() call, features per base-layer
# batch handed to a worker, and how often streaming loops poll for
# cancellation.
_WRITE_BATCH_SIZE = 1000

# Most rule outputs one pass writes at once, as each holds an open file and a
# write buffer; groups beyond this go to further passes.
_MAX_GROUPS_PER_PASS = 64

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


def _lower_thread_priority() -> None:
    """Pool-thread initializer: below-normal priority, so that when the CPU is
    saturated QGIS's interface threads are scheduled first (Windows only)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentThread.restype = wintypes.HANDLE
        kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -1)  # BELOW_NORMAL
    except (OSError, AttributeError):
        pass

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
    # Source layer name, for user-facing messages.
    layer_name: str = ""

    @property
    def label(self) -> str:
        rule_type = 'labeling' if self.rule_type == 1 else 'symbology'
        return f'{rule_type} of the "{self.layer_name or self.layer_id}" layer'


class _BaseBuild:
    """One base layer being written by the caller thread (Phases 1 + 2)."""

    def __init__(self, src: _SourceSnapshot, path: str):
        self.src = src
        self.path = path
        self.writer: Optional[QgsVectorFileWriter] = None
        self.orig_idx = -1
        self.pending = 0            # Batches submitted but not yet written.
        self.reading_done = False
        self.orig_id = 0
        self.dropped = 0
        self.failed = False
        self.finished = False


class _RuleOutput:
    """One rule group's output while a Phase 3 pass writes it."""

    def __init__(self, grp: _RuleGroupSnapshot, path: str, wkb_type):
        self.grp = grp
        self.path = path
        self.wkb_type = wkb_type
        self.geometry_type = QgsWkbTypes.geometryType(wkb_type)
        self.fields = QgsFields()
        self.plan: List[Tuple[int, QgsField, Any, bool]] = []
        self.geometry: Optional[QgsExpression] = None
        self.writer: Optional[QgsVectorFileWriter] = None
        self.batch: List[QgsFeature] = []
        self.written = 0
        self.failed = False
        # Problems that don't stop the export, reported once per group as
        # {what: [occurrences, first error]}.
        self.problems: Dict[str, List[Any]] = {}

    def note(self, what: str, error: str) -> None:
        entry = self.problems.setdefault(what, [0, error])
        entry[0] += 1


class _Cancelled(Exception):
    """Raised inside workers when the caller has signalled cancellation."""


# Marks a per-feature cache entry not computed yet.
_MISSING = object()


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
                layer_name=primary.layer.name(),
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
    # Phases 1 + 2 — read sources, build base layers
    # -------------------------------------------------------------------
    def _build_base_layers(
        self, sources: Dict[str, _SourceSnapshot]
    ) -> Dict[str, str]:
        """Read every source and run fix → reproject → orig_id → singleparts
        → simplify on it.

        Sources are read one at a time on this (caller) thread — opening
        project sources from worker threads can deadlock. Their features go
        to the pool in batches, and this thread writes the finished batches
        to the base layers in reading order. Workers process batches of
        every source at once, including several of one large source.

        Base layers of unchanged local sources come from BaseLayerCache;
        new ones are built under a temporary name and then committed to it.
        """
        cache = BaseLayerCache(_TEMP_LAYER_FORMAT)
        cache.prune()
        target_paths: Dict[str, str] = {}
        cached: Dict[str, bool] = {}
        for lid, src in sources.items():
            cache_path = cache.path_for(src)
            cached[lid] = cache_path is not None
            target_paths[lid] = cache_path or join(
                self.utils_dir, f"map_layer_{lid}.{_TEMP_LAYER_FORMAT}"
            )
        # Idempotent skip; cache hits skip reading and preparing the source.
        todo = {
            lid: src for lid, src in sources.items()
            if not (cache.hit(target_paths[lid]) if cached[lid] else exists(target_paths[lid]))
        }
        reused = sum(1 for lid in sources if cached[lid] and lid not in todo)
        if reused:
            self.feedback.pushInfo(
                f". Reused {reused} of {len(sources)} prepared layers from the cache."
            )
        if not todo:
            return target_paths

        max_workers = self._compute_pool_size(os.cpu_count() or 1)
        builds: Dict[str, _BaseBuild] = {}
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rules-base",
            initializer=_lower_thread_priority,
        ) as pool:
            # (build, future) in submission order — the order they are written.
            pending: Deque[Tuple[_BaseBuild, Future]] = deque()
            try:
                for lid, src in todo.items():
                    # Cached entries are built under a temporary name.
                    path = cache.temp_path_for(target_paths[lid]) if cached[lid] else target_paths[lid]
                    build = builds[lid] = _BaseBuild(src, path)
                    try:
                        if src.needs_serial_read:
                            with self._serial_read_lock:
                                self._read_source(build, pool, pending, max_workers)
                        else:
                            self._read_source(build, pool, pending, max_workers)
                    except _Cancelled:
                        raise
                    except Exception as e:  # noqa: BLE001  (we want to swallow per-source)
                        self._fail_base_build(build, e)
                    build.reading_done = True
                    if build.pending == 0:
                        self._finish_base_build(build)
                self._write_ready_batches(pending, keep=0)
            except _Cancelled:
                self.feedback.pushInfo("Base-layer build cancelled.")
                for _, fut in pending:
                    fut.cancel()
                for build in builds.values():
                    if not build.finished:
                        self._close_base_build(build, remove=True)

        # Base layers finished before a cancellation are still worth keeping.
        for lid, build in builds.items():
            if build.finished and cached[lid]:
                target_paths[lid] = cache.commit(build.path, target_paths[lid])
        return target_paths

    def _read_source(
        self,
        build: _BaseBuild,
        pool: ThreadPoolExecutor,
        pending: "Deque[Tuple[_BaseBuild, Future]]",
        max_workers: int,
    ) -> None:
        """Read the in-extent, referenced-fields-only part of a source and
        submit it to the pool in batches.

        Caller thread only: this is the only place in the pipeline where we
        touch a database/network provider.
        """
        self._check_cancel()
        src = build.src
        # Open a FRESH layer in this thread. The original FlattenedRule.layer
        # reference may have main-thread affinity; here we deliberately don't
        # reuse it. The newly constructed layer is owned by this thread.
        layer = QgsVectorLayer(src.source_uri, src.name, src.provider)
        try:
            if not layer.isValid():
                build.failed = True
                self.feedback.pushWarning(
                    f"Cannot open source '{src.name}' "
                    f"(provider={src.provider}); skipping."
                )
                return

            fields = layer.fields()
            if src.required_fields is None:
                indices = list(range(fields.count()))
            else:
                indices = sorted(
                    {fields.lookupField(name) for name in src.required_fields} - {-1}
                )
            subset = len(indices) != fields.count()

            # Output schema: the read fields + q2vt_orig_id, typed as
            # fieldcalculator's FIELD_TYPE=0 (decimal, length 10, precision 3).
            out_fields = QgsFields()
            for idx in indices:
                out_fields.append(fields.at(idx))
            orig_name = f"{_FIELD_PREFIX}_orig_id"
            build.orig_idx = out_fields.lookupField(orig_name)
            if build.orig_idx < 0:
                out_fields.append(QgsField(orig_name, QMetaType.Type.Double, "", 10, 3))
                build.orig_idx = out_fields.count() - 1

            request = QgsFeatureRequest()
            request.setInvalidGeometryCheck(
                QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck
            )
            if src.extent is not None:
                # Bbox-only test: the provider evaluates it with its spatial
                # index (server-side for databases) and it never trips on
                # invalid geometries. Rule outputs are clipped to the exact
                # extent.
                request.setFilterRect(QgsRectangle(*src.extent))
            if subset:
                request.setSubsetOfAttributes(indices)

            build.writer = self._create_writer(
                build.path, out_fields, QgsWkbTypes.singleType(layer.wkbType()),
                QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}"),
            )
            source_crs = layer.crs()
            batch: List[Tuple[List[Any], QgsGeometry]] = []
            for feature in layer.getFeatures(request):
                attributes = (
                    [feature.attribute(i) for i in indices] if subset else feature.attributes()
                )
                batch.append((attributes, feature.geometry()))
                if len(batch) >= _WRITE_BATCH_SIZE:
                    self._check_cancel()
                    self._submit_base_batch(build, batch, source_crs, out_fields, pool, pending, max_workers)
                    batch = []
                    if build.failed:
                        return  # Writing an earlier batch failed.
            if batch:
                self._submit_base_batch(build, batch, source_crs, out_fields, pool, pending, max_workers)
        finally:
            self._dispose_layer(layer)

    def _submit_base_batch(
        self, build, batch, source_crs, out_fields, pool, pending, max_workers
    ) -> None:
        """Hand a batch to the pool, then write whatever batches are ready."""
        pending.append((build, pool.submit(
            self._prepare_base_batch, batch, source_crs, out_fields, build.orig_idx
        )))
        build.pending += 1
        # Bounds the features held in memory while workers catch up.
        self._write_ready_batches(pending, keep=2 * max_workers)

    def _prepare_base_batch(
        self,
        batch: List[Tuple[List[Any], QgsGeometry]],
        source_crs: QgsCoordinateReferenceSystem,
        out_fields: QgsFields,
        orig_idx: int,
    ) -> List[Optional[List[QgsFeature]]]:
        """Worker: fix → reproject → singleparts → simplify one batch.

        Returns, per feature that survives, its output parts; q2vt_orig_id
        is filled in by the writer, which knows the feature's position.
        """
        self._check_cancel()
        transform = QgsCoordinateTransform(
            source_crs, QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}"),
            self._transform_context,
        )
        size = out_fields.count()
        results: List[Optional[List[QgsFeature]]] = []
        for attributes, geometry in batch:
            if not geometry.isNull():
                geometry = self._fix_geometry(geometry)
                if not geometry.isNull():
                    try:
                        geometry.transform(transform)
                    except QgsCsException:
                        results.append(None)  # reprojectlayer drops these too
                        continue
            attributes += [None] * (size - len(attributes))
            parts = (
                geometry.asGeometryCollection()
                if not geometry.isNull() and geometry.isMultipart()
                else [geometry]
            )
            features = []
            for part in parts:
                out = QgsFeature(out_fields)
                out.setAttributes(attributes)
                out.setGeometry(
                    part if part.isNull() else part.simplify(_DATA_SIMPLIFICATION_TOLERANCE)
                )
                features.append(out)
            results.append(features)
        return results

    def _write_ready_batches(
        self, pending: "Deque[Tuple[_BaseBuild, Future]]", keep: int
    ) -> None:
        """Write finished batches in submission order, waiting for the oldest
        while more than ``keep`` are pending."""
        while pending and (len(pending) > keep or pending[0][1].done()):
            build, fut = pending.popleft()
            build.pending -= 1
            try:
                results = fut.result(timeout=_PER_ALG_TIMEOUT_S)
                if not build.failed:
                    self._write_base_results(build, results)
            except _Cancelled:
                raise
            except Exception as e:  # noqa: BLE001
                self._fail_base_build(build, e)
            if build.reading_done and build.pending == 0:
                self._finish_base_build(build)

    def _write_base_results(
        self, build: _BaseBuild, results: List[Optional[List[QgsFeature]]]
    ) -> None:
        batch: List[QgsFeature] = []
        for features in results:
            if features is None:
                build.dropped += 1
                continue
            # @id of the reprojected layer: 1-based position of the feature.
            build.orig_id += 1
            for feature in features:
                feature.setAttribute(build.orig_idx, float(build.orig_id))
            batch.extend(features)
        self._write_batch(build.writer, batch)

    def _fail_base_build(self, build: _BaseBuild, error: Exception) -> None:
        if build.failed:
            return
        build.failed = True
        self._close_base_build(build, remove=True)
        self.feedback.pushWarning(f'The "{build.src.name}" layer was skipped: {error}')
        self.feedback.pushDebugInfo(
            f"Base-layer build failed for '{build.src.name}':\n{traceback.format_exc()}"
        )

    def _finish_base_build(self, build: _BaseBuild) -> None:
        if build.failed or build.finished:
            return
        build.finished = True
        self._close_base_build(build, remove=False)
        if build.dropped:
            self.feedback.pushWarning(
                f"{build.dropped} features of '{build.src.name}' could not be "
                f"reprojected to EPSG:{_EPSG_CRS} and were skipped."
            )

    def _close_base_build(self, build: _BaseBuild, remove: bool) -> None:
        build.writer = None  # Closes the file.
        if remove:
            self._remove_file(build.path)

    @staticmethod
    def _fix_geometry(geometry: QgsGeometry) -> QgsGeometry:
        """fixgeometries(METHOD=0) for one geometry.

        Keeps only parts of the original geometry type; a result of another
        type (e.g. a polygon collapsed to a line) becomes a null geometry, as
        the algorithm leaves it.
        """
        geometry_type = geometry.type()
        if geometry_type == Qgis.GeometryType.Point:
            return geometry  # Points have nothing to repair.
        fixed = geometry.makeValid(Qgis.MakeValidMethod.Linework, False)
        if fixed.isNull():
            return QgsGeometry()
        if (fixed.wkbType() == Qgis.WkbType.Unknown
                or QgsWkbTypes.flatType(fixed.wkbType()) == Qgis.WkbType.GeometryCollection):
            fixed = QgsGeometry.collectGeometry(
                [part for part in fixed.asGeometryCollection() if part.type() == geometry_type]
            )
        if fixed.type() != geometry_type:
            return QgsGeometry()
        return fixed

    # -------------------------------------------------------------------
    # Phase 3 — parallel rule export (file → file)
    # -------------------------------------------------------------------
    def _export_rules_parallel(
        self,
        rule_groups: List[_RuleGroupSnapshot],
        base_layers: Dict[str, str],
    ) -> Dict[str, Optional[str]]:
        """Export the rule groups in parallel, one pass per base layer and
        filter: groups that read the same features share the read and the
        expressions they have in common."""
        outputs: Dict[str, Optional[str]] = {}
        passes: Dict[Tuple[str, Optional[str], bool], List[_RuleGroupSnapshot]] = {}
        for grp in rule_groups:
            src_path = base_layers.get(grp.layer_id)
            if not src_path or not exists(src_path):
                outputs[grp.output_dataset] = None
                continue
            key = (grp.layer_id, grp.filter_expression, grp.keep_biggest_part)
            passes.setdefault(key, []).append(grp)
        tasks: List[Tuple[str, List[_RuleGroupSnapshot]]] = []
        for (layer_id, _, _), groups in passes.items():
            for i in range(0, len(groups), _MAX_GROUPS_PER_PASS):
                tasks.append((base_layers[layer_id], groups[i:i + _MAX_GROUPS_PER_PASS]))
        if not tasks:
            return outputs
        # Largest first, so that no big pass starts last and holds up the end.
        tasks.sort(key=lambda task: self._file_size(task[0]) * len(task[1]), reverse=True)

        max_workers = self._compute_pool_size(len(tasks))

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rules-export",
            initializer=_lower_thread_priority,
        ) as pool:
            futures: Dict[Future, List[_RuleGroupSnapshot]] = {
                pool.submit(self._export_rule_pass, groups, src_path): groups
                for src_path, groups in tasks
            }
            for fut in self._iter_completed(futures):
                groups = futures[fut]
                try:
                    outputs.update(fut.result(timeout=_PER_ALG_TIMEOUT_S))
                except _Cancelled:
                    self.feedback.pushInfo("Rule export cancelled.")
                    for pending_groups in futures.values():
                        for grp in pending_groups:
                            outputs.setdefault(grp.output_dataset, None)
                    return outputs
                except Exception as e:  # noqa: BLE001
                    for grp in groups:
                        self.feedback.pushWarning(
                            f"A rule in the {grp.label} was skipped: {e}"
                        )
                        outputs[grp.output_dataset] = None
                    self.feedback.pushDebugInfo(
                        f"Rule export failed for "
                        f"{', '.join(grp.output_dataset for grp in groups)}:\n"
                        f"{traceback.format_exc()}"
                    )
        return outputs

    def validate_expression(self, grp, expr_str: str):
        warning_msg = f'The expression "{expr_str}" within the {grp.label}'

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

    def _export_rule_pass(
        self, groups: List[_RuleGroupSnapshot], source_path: str
    ) -> Dict[str, Optional[str]]:
        """Worker: export rule groups that share a base layer and a filter in
        a single streaming pass.

        Per group, equivalent to the chain extractbyexpression →
        refactorfields → [dissolve → keepnbiggestparts] →
        geometrybyexpression → clip to the extent → removenullgeometries →
        multiparttosingleparts, but every feature is read once for all the
        groups and only the final outputs are written to disk.
        """
        self._check_cancel()
        layer = QgsVectorLayer(source_path, "base", "ogr")
        try:
            return self._export_rule_pass_from(groups, layer)
        finally:
            self._dispose_layer(layer)

    def _export_rule_pass_from(
        self, groups: List[_RuleGroupSnapshot], layer: QgsVectorLayer
    ) -> Dict[str, Optional[str]]:
        """_export_rule_pass, once the base layer is open."""
        results: Dict[str, Optional[str]] = {grp.output_dataset: None for grp in groups}
        if not layer.isValid():
            return results
        src_fields = layer.fields()

        context = QgsProject.instance().createExpressionContext()
        context.appendScope(QgsExpressionContextUtils.layerScope(layer))
        context.setFields(src_fields)
        # Planar measurements, as with a default QgsProcessingContext.
        distance_area = QgsDistanceArea()
        distance_area.setSourceCrs(layer.crs(), self._transform_context)

        # One prepared expression per distinct string, shared by the groups,
        # so that per feature each is evaluated once.
        prepared: Dict[str, QgsExpression] = {}

        def prepare(expr_str: str) -> QgsExpression:
            expr = prepared.get(expr_str)
            if expr is None:
                expr = prepared[expr_str] = self._prepare_expression(
                    expr_str, context, distance_area
                )
            return expr

        outputs: List[_RuleOutput] = []
        completed = False
        try:
            for grp in groups:
                output_path = join(self.utils_dir, f"{grp.output_dataset}.{_TEMP_RULE_FORMAT}")
                if exists(output_path):
                    results[grp.output_dataset] = output_path
                    continue
                output = self._open_rule_output(grp, output_path, src_fields, layer.crs(), prepare)
                if output is not None:
                    outputs.append(output)
            if not outputs:
                return results

            # Every group of a pass has the same filter and biggest-part mode.
            request = QgsFeatureRequest()
            request.setInvalidGeometryCheck(
                QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck
            )
            if groups[0].filter_expression:
                request.setFilterExpression(groups[0].filter_expression)
                request.setExpressionContext(context)
            if groups[0].keep_biggest_part:
                biggest = self._biggest_part_ids(layer, request)
                if biggest is not None:
                    request = QgsFeatureRequest()
                    request.setInvalidGeometryCheck(
                        QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck
                    )
                    request.setFilterFids(biggest)

            for n, feature in enumerate(layer.getFeatures(request)):
                if n % _WRITE_BATCH_SIZE == 0:
                    self._check_cancel()
                context.setFeature(feature)
                self._export_rule_feature(feature, context, outputs)
            for output in outputs:
                self._flush_rule_output(output)
            completed = True
        finally:
            for output in outputs:
                output.writer = None  # Closes the file.
                if not completed or output.failed or output.written == 0:
                    self._remove_file(output.path)

        for output in outputs:
            for what, (count, error) in output.problems.items():
                self.feedback.pushWarning(
                    f"In the {output.grp.label}: {count} {what}. First error: {error.strip()}"
                )
            if not output.failed and output.written:
                results[output.grp.output_dataset] = output.path
        return results

    def _open_rule_output(
        self,
        grp: _RuleGroupSnapshot,
        output_path: str,
        src_fields: QgsFields,
        crs: QgsCoordinateReferenceSystem,
        prepare,
    ) -> Optional[_RuleOutput]:
        """Plan one group's output fields and geometry and open its writer;
        None if the group can't be exported."""
        if grp.filter_expression and not self.validate_expression(grp, grp.filter_expression):
            return None
        if not self.validate_expression(grp, grp.geometry_expression):
            return None
        # layer.geometryType() returns 0 for point and 2 for polygon but
        # geometrybyexpression codes 0 as polygon and 2 as point, so flip it.
        out_wkb = _RULE_OUTPUT_WKB.get(abs(self._enum_int(grp.geometry_target) - 2))
        if out_wkb is None:
            return None

        output = _RuleOutput(grp, output_path, out_wkb)
        # Per output field: how to compute its value, cheapest kind first —
        # a constant, a straight copy of a source attribute, or an expression.
        for m in self._build_field_mapping(grp, src_fields):
            field = QgsField(m["name"], QMetaType.Type(m["type"]))
            if not output.fields.append(field):
                continue  # Duplicate name: first definition wins.
            try:
                plan = self._plan_field(field, m["expression"], src_fields, prepare)
            except RuntimeError as e:
                output.note(f'field "{field.name()}" (exported as NULL)', str(e))
                plan = (_FIELD_CONSTANT, field, None, False)
            output.plan.append(plan)
        try:
            output.geometry = prepare(grp.geometry_expression)
            output.writer = self._create_writer(output_path, output.fields, out_wkb, crs)
        except RuntimeError as e:
            self.feedback.pushWarning(f"A rule in the {grp.label} was skipped: {e}")
            return None
        return output

    def _export_rule_feature(
        self, feature: QgsFeature, context: QgsExpressionContext, outputs: List[_RuleOutput]
    ) -> None:
        """Add one base-layer feature to every output it belongs in."""
        # Per expression object: its result for this feature.
        geometries: Dict[int, Tuple[Optional[List[QgsGeometry]], Optional[Tuple[str, str]]]] = {}
        values: Dict[int, Tuple[Any, Optional[str]]] = {}
        src_values = None
        for output in outputs:
            if output.failed:
                continue
            key = id(output.geometry)
            evaluated = geometries.get(key)
            if evaluated is None:
                evaluated = geometries[key] = self._evaluate_geometry(output.geometry, context)
            parts, problem = evaluated
            if problem is not None:
                output.note(*problem)
            if not parts:
                continue
            parts = self._parts_of_type(parts, output)
            if not parts:
                continue

            if src_values is None:
                src_values = feature.attributes()
            attributes = []
            for kind, field, source, convert in output.plan:
                if kind == _FIELD_CONSTANT:
                    attributes.append(source)
                    continue
                if kind == _FIELD_COPY:
                    value = src_values[source]
                else:
                    cached = values.get(id(source), _MISSING)
                    if cached is _MISSING:
                        value = source.evaluate(context)
                        error = source.evalErrorString() if source.hasEvalError() else None
                        cached = values[id(source)] = (value, error)
                    value, error = cached
                    if error is not None:
                        output.note(f'values of field "{field.name()}" (exported as NULL)', error)
                        attributes.append(None)
                        continue
                if convert:
                    try:
                        value = self._convert_value(field, value)
                    except RuntimeError as e:
                        output.note(f'values of field "{field.name()}" (exported as NULL)', str(e))
                        value = None
                attributes.append(value)

            for part in parts:
                out = QgsFeature(output.fields)
                out.setAttributes(attributes)
                out.setGeometry(part)
                output.batch.append(out)
            if len(output.batch) >= _WRITE_BATCH_SIZE:
                self._flush_rule_output(output)

    def _evaluate_geometry(
        self, expr: QgsExpression, context: QgsExpressionContext
    ) -> Tuple[Optional[List[QgsGeometry]], Optional[Tuple[str, str]]]:
        """(single parts of the clipped geometry or None, problem or None)."""
        geometry = expr.evaluate(context)
        if expr.hasEvalError():
            return None, ("features (skipped: geometry could not be computed)", expr.evalErrorString())
        if geometry is None:
            return None, None
        if not isinstance(geometry, QgsGeometry):
            return None, ("features (skipped: geometry expression did not return a geometry)",
                          f"got {geometry!r}")
        if geometry.isNull() or geometry.isEmpty():
            return None, None
        geometry = self._clip_to_extent(geometry)
        if geometry is None:
            return None, None
        return (geometry.asGeometryCollection() if geometry.isMultipart() else [geometry]), None

    def _clip_to_extent(self, geometry: QgsGeometry) -> Optional[QgsGeometry]:
        """The part of ``geometry`` inside the export extent; None if none.

        Most features lie wholly inside the extent and are returned as they
        are; the rest are cut with GEOS's rectangle clip, much cheaper than a
        general intersection with the extent polygon.
        """
        bbox = geometry.boundingBox()
        if self.extent.contains(bbox):
            return geometry
        if not self.extent.intersects(bbox):
            return None
        clipped = geometry.clipped(self.extent)
        if clipped.isNull() or clipped.isEmpty():
            return None
        return clipped

    @staticmethod
    def _parts_of_type(parts: List[QgsGeometry], output: _RuleOutput) -> List[QgsGeometry]:
        """``parts`` as the output's WKB type; parts of another geometry type
        (e.g. lines from a polygon geometry generator) are skipped."""
        if all(part.wkbType() == output.wkb_type for part in parts):
            return parts
        matching: List[QgsGeometry] = []
        for part in parts:
            if part.wkbType() == output.wkb_type:
                matching.append(part)
            elif part.type() == output.geometry_type:
                # Same kind with Z/M or curves: drop them as the file needs.
                matching.extend(part.coerceToType(output.wkb_type))
            else:
                output.note(
                    "geometry parts of another type (skipped)",
                    f"got {QgsWkbTypes.displayString(part.wkbType())}",
                )
        return matching

    def _flush_rule_output(self, output: _RuleOutput) -> None:
        """Write a group's buffered features; a failure drops that group only."""
        if output.failed:
            return
        try:
            output.written += self._write_batch(output.writer, output.batch)
        except RuntimeError as e:
            output.failed = True
            output.batch.clear()
            self.feedback.pushWarning(f"A rule in the {output.grp.label} was skipped: {e}")

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
        prepare,
    ) -> Tuple[int, QgsField, Any, bool]:
        """(kind, field, source, convert) for one output field.

        Constants are evaluated and converted once; plain references to a
        source field are copied by index (and only converted when the types
        differ); anything else is evaluated per feature. ``prepare`` turns
        an expression string into a prepared QgsExpression.
        """
        if not isinstance(expr_str, str) or not expr_str.strip():
            # As in refactorfields, an empty expression yields NULL. The DDP
            # fetcher emits these for field-based properties.
            return (_FIELD_CONSTANT, field, None, False)
        expr = prepare(expr_str)
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
        # The result is clipped to the extent after evaluation, in
        # _clip_to_extent.
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
    def _file_size(path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    @staticmethod
    def _remove_file(path: str) -> None:
        try:
            if exists(path):
                os.remove(path)
        except OSError:
            pass

    @staticmethod
    def _dispose_layer(layer: QgsVectorLayer) -> None:
        """Delete a worker-created layer now, in the thread that owns it.

        Left to Python's garbage collector, the wrapper may be released from
        another thread; sip then defers the deletion to this (pool) thread,
        and Qt runs it while the thread exits, which corrupts the heap and
        crashes QGIS.
        """
        if not sip.isdeleted(layer):
            sip.delete(layer)

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
        if driver == "FlatGeobuf":
            # GDAL's default index reorders features; keep them in order.
            options.layerOptions = options.layerOptions + ["SPATIAL_INDEX=NO"]
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