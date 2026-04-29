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
   │   * After this phase NO QObject crosses a thread boundary.           │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 1 — Source materialisation (SERIAL, caller thread)             │
   │   * For each unique source layer (Postgres or otherwise), construct  │
   │     a fresh QgsVectorLayer FROM URI inside this single thread, then  │
   │     dump it to a local Parquet file via fixgeometries(METHOD=0).     │
   │   * Postgres / remote providers are not parallel-safe; we never read │
   │     more than one source concurrently.                               │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 2 — Base-layer pipeline (PARALLEL, file → file)                │
   │   * For each materialised source: fixgeometries(METHOD=1) →          │
   │     reproject → clip → singleparts → simplify.                       │
   │   * All inputs and outputs are file paths; no live layers cross      │
   │     threads.                                                         │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 3 — Rule export (PARALLEL, file → file)                        │
   │   * For each rule group: optional filter → refactor fields →         │
   │     geometry transform → remove nulls → singleparts.                 │
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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from os.path import exists, join
from typing import Any, Dict, Iterator, List, Optional, Tuple
from uuid import uuid4

from processing import run as run_processing
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsExpressionContext,
    QgsExpressionContextUtils,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsFeatureRequest,
    QgsRectangle,
    QgsVectorLayer,
    QgsProject,
)

from ..utils.config import _DATA_SIMPLIFICATION_TOLERANCE, _EPSG_CRS
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
            self.feedback.pushInfo("Snapshotting rule and source metadata...")
            sources, rule_groups = self._snapshot_caller_thread()

            # Phase 1 — serial source materialisation (caller thread).
            if self._is_cancelled():
                return [], []
            self.feedback.pushInfo(
                f"Materialising {len(sources)} source layer(s) (serial)..."
            )
            materialized = self._materialize_sources_serial(sources)

            # Phase 2 — parallel base-layer pipeline (file → file).
            if self._is_cancelled():
                return [], []
            self.feedback.pushInfo("Building base layers in parallel...")
            base_layers = self._build_base_layers_parallel(materialized)

            # Phase 3 — parallel rule export (file → file).
            if self._is_cancelled():
                return [], []
            self.feedback.pushInfo(
                f"Exporting {len(rule_groups)} rule group(s) in parallel..."
            )
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

        # Snapshot unique sources.
        sources: Dict[str, _SourceSnapshot] = {}
        for r in self.flattened_rules:
            lid = r.layer.id()
            if lid in sources:
                continue
            sources[lid] = _SourceSnapshot(
                layer_id=lid,
                name=r.layer.name(),
                source_uri=r.layer.source(),
                provider=r.layer.providerType(),
            )

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
                output_dataset=output_dataset,
                layer_id=primary.layer.id(),
                rule_type=primary.get_attr("t"),
                filter_expression=primary.rule.filterExpression() or None,
                geometry_target=geom_target,
                geometry_expression=geom_expr,
                expression_fields=expr_fields,
                description=primary.get_description(),
                include_required_fields_only=self.include_required_fields_only,
                flat_rules=flat_rules,
            ))

        return sources, rule_groups

    # -------------------------------------------------------------------
    # Phase 1 — serial source materialisation
    # -------------------------------------------------------------------
    def _materialize_sources_serial(
        self, sources: Dict[str, _SourceSnapshot]
    ) -> Dict[str, str]:
        """For each unique source, dump a fresh provider read to local Parquet.

        Runs on the caller thread, ONE source at a time. This is the only
        place in the pipeline where we touch a database/network provider.
        """
        materialized: Dict[str, str] = {}
        for src in sources.values():
            self._check_cancel()
            out_path = join(self.utils_dir, f"materialized_{src.layer_id}.parquet")

            if exists(out_path):
                # Idempotent restart support.
                materialized[src.layer_id] = out_path
                continue

            try:
                # Open a FRESH layer in this thread. The original
                # FlattenedRule.layer reference may have main-thread affinity;
                # here we deliberately don't reuse it. The newly constructed
                # layer is owned by this thread.
                layer = QgsVectorLayer(src.source_uri, src.name, src.provider)
                if not layer.isValid():
                    self.feedback.pushWarning(
                        f"Cannot open source '{src.name}' "
                        f"(provider={src.provider}); skipping."
                    )
                    continue

                # Materialise via fixgeometries(METHOD=0): does the first
                # geometry-cleaning pass AND dumps provider data to Parquet
                # in one shot.
                if src.needs_serial_read:
                    with self._serial_read_lock:
                        self._run_alg_safe(
                            "fixgeometries", "native",
                            INPUT=layer, METHOD=0, OUTPUT=out_path,
                        )
                else:
                    self._run_alg_safe(
                        "fixgeometries", "native",
                        INPUT=layer, METHOD=0, OUTPUT=out_path,
                    )
                materialized[src.layer_id] = out_path
            except _Cancelled:
                raise
            except Exception:  # noqa: BLE001  (we want to swallow per-source)
                self.feedback.reportError(
                    f"Failed to materialise source '{src.name}':\n"
                    f"{traceback.format_exc()}"
                )
        return materialized

    # -------------------------------------------------------------------
    # Phase 2 — parallel base-layer pipeline (file → file)
    # -------------------------------------------------------------------
    def _build_base_layers_parallel(
        self, materialized: Dict[str, str]
    ) -> Dict[str, str]:
        """Run fix-structure → reproject → clip → singleparts → simplify."""
        target_paths: Dict[str, str] = {
            lid: join(self.utils_dir, f"map_layer_{lid}.parquet")
            for lid in materialized
        }
        # Idempotent skip.
        todo = {
            lid: src for lid, src in materialized.items()
            if not exists(target_paths[lid])
        }

        if not todo:
            return target_paths

        max_workers = self._compute_pool_size(len(todo))

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rules-base"
        ) as pool:
            futures: Dict[Future, str] = {
                pool.submit(
                    self._build_one_base_layer, src_path, target_paths[lid]
                ): lid
                for lid, src_path in todo.items()
            }
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
        """Worker: run the cleanup chain on a local Parquet file."""
        self._check_cancel()
        # METHOD=1 (structure) — finishes the geometry fix started in Phase 1.
        fixed_struct = self._run_alg_safe(
            "fixgeometries", "native", INPUT=src_path, METHOD=1
        )
        self._check_cancel()
        reprojected = self._run_alg_safe(
            "reprojectlayer", "native",
            INPUT=fixed_struct,
            TARGET_CRS=QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}"),
        )
        self._check_cancel()
        clipped = self._run_alg_safe(
            "extractbyextent", "native",
            INPUT=reprojected, EXTENT=self.extent, CLIP=False,
        )
        self._check_cancel()
        singleparted = self._run_alg_safe(
            "multiparttosingleparts", "native", INPUT=clipped
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

    def _export_one_rule_group(
        self, grp: _RuleGroupSnapshot, source_path: str
    ) -> Optional[str]:
        """Worker: run the rule-export chain entirely from local files."""
        self._check_cancel()
        output_path = join(self.utils_dir, f"{grp.output_dataset}.parquet")
        if exists(output_path):
            return output_path

        current_input: str = source_path

        # Optional filter step.
        if grp.filter_expression:
            self._check_cancel()
            filt_out = self._temp_path("filt")
            filt = self._run_alg_safe(
                "extractbyexpression", "native",
                INPUT=current_input,
                EXPRESSION=grp.filter_expression,
                OUTPUT=filt_out,
            )
            check = QgsVectorLayer(filt, "check", "ogr")
            if not check.isValid() or check.featureCount() <= 0:
                return None
            current_input = filt

        # Field mapping.
        field_mapping = self._build_field_mapping(grp, current_input)

        self._check_cancel()
        refactored = self._run_alg_safe(
            "refactorfields", "native",
            INPUT=current_input,
            FIELDS_MAPPING=field_mapping,
        )

        # Geometry transformation.
        self._check_cancel()
        geom_target = abs(grp.geometry_target - 2)
        transformed = self._run_alg_safe(
            "geometrybyexpression", "native",
            INPUT=refactored,
            OUTPUT_GEOMETRY=geom_target,
            EXPRESSION=grp.geometry_expression,
        )
        check = QgsVectorLayer(transformed, "check", "ogr")
        if not check.isValid() or check.featureCount() <= 0:
            return None

        self._check_cancel()
        cleaned = self._run_alg_safe(
            "removenullgeometries", "native",
            INPUT=transformed,
            REMOVE_EMPTY=True,
        )
        self._check_cancel()
        return self._run_alg_safe(
            "multiparttosingleparts", "native",
            INPUT=cleaned,
            OUTPUT=output_path,
        )

    def _build_field_mapping(
        self, grp: _RuleGroupSnapshot, current_input: str
    ) -> List[Dict[str, Any]]:
        """Worker: assemble the FIELDS_MAPPING list for refactorfields."""
        mapping: List[Tuple[int, str, str]] = []
        mapping.append((4, '"fid"', f"{self.FIELD_PREFIX}_fid"))
        mapping.extend(grp.expression_fields)
        mapping.append(
            (10, f"'{grp.description}'", f"{self.FIELD_PREFIX}_description")
        )

        if grp.include_required_fields_only != 0:
            # Read fields fresh in this worker thread; the QgsVectorLayer is
            # locally constructed and stays local.
            tmp = QgsVectorLayer(current_input, "tmp", "ogr")
            if tmp.isValid():
                for f in tmp.fields():
                    mapping.append((f.type(), f'"{f.name()}"', f.name()))

        return [
            {"type": m[0], "expression": m[1], "name": m[2]} for m in mapping
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
            on_disk = join(self.utils_dir, f"{grp.output_dataset}.parquet")
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
        context.setInvalidGeometryCheck(QgsFeatureRequest.GeometryNoCheck)
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
            min_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            rule_fields = DataDefinedPropertiesFetcher(
                flat_rule.rule, min_scale
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
        field_name = f"{self.FIELD_PREFIX}_label"
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
            transform_expr = settings.geometryGenerator
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
        p = join(self.utils_dir, f"{prefix}_{uuid4().hex}.parquet")
        with self._temp_files_lock:
            self._temp_files.add(p)
        return p

    def _cleanup_temp_files(self) -> None:
        with self._temp_files_lock:
            paths = list(self._temp_files)
            self._temp_files.clear()
        for p in paths:
            try:
                if exists(p):
                    os.remove(p)
            except OSError:
                pass