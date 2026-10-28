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
   │   * Local files GDAL reads as QGIS does skip Phases 1 + 2: GDAL      │
   │     builds their base layers in C++, one task each, in parallel      │
   │     with the reads above (see _gdal_base_plan).                      │
   ├──────────────────────────────────────────────────────────────────────┤
   │ Phase 3 — Rule export (PARALLEL, file → file)                        │
   │   * Rules that would write the same rows (the symbol layers of one   │
   │     rule) share one dataset, and rules that differ only by filter    │
   │     (categories, labels split by category) share one dataset too,    │
   │     each rule's rows tagged with its q2vt_rule number.               │
   │   * Datasets whose filters and fields translate to OGR SQL, and      │
   │     whose geometry is the feature's own, are written by GDAL in C++  │
   │     (no Python per feature, so they run truly in parallel).          │
   │   * The others: datasets of one base layer are exported in ONE       │
   │     streaming pass: filter → geometry transform → clip to the        │
   │     extent → drop null / empty → explode multiparts → field          │
   │     expressions, written straight to each dataset's output file.     │
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

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import re
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field as dataclass_field
from os.path import basename, exists, join, splitext
from typing import Any, Deque, Dict, Iterator, List, Optional, Set, Tuple
from uuid import uuid4
from xml.sax.saxutils import escape as xml_escape
from osgeo import gdal, ogr
from qgis.PyQt import sip
from qgis.PyQt.QtCore import QMetaType
from processing import run as run_processing
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCsException,
    QgsDistanceArea,
    QgsExpressionContext,
    QgsExpressionContextScope,
    QgsExpression,
    QgsExpressionContextUtils,
    QgsExpressionNode,
    QgsExpressionNodeBinaryOperator,
    QgsExpressionNodeUnaryOperator,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPalLayerSettings,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsFeatureRequest,
    QgsRectangle,
    QgsCoordinateTransform,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsVectorLayerFeatureSource,
    QgsProject,
    QgsSymbolLayerUtils,
    QgsWkbTypes,
)

from ..utils.config import (
    _DATA_SIMPLIFICATION_TOLERANCE, _EPSG_CRS, _FIELD_PREFIX, _FIELDS_ALL, _FIELDS_VISIBLE,
    _TILES_MAX_ZOOM,
)
from ..utils.flattened_rule import (
    FlattenedRule, MARKER_ALL_VERTICES, MARKER_CENTRAL_POINT, MARKER_FIRST_VERTEX,
    MARKER_INNER_VERTICES, MARKER_LAST_VERTEX, MARKER_SEGMENT_CENTERS, cluster_settings,
    displacement_label_settings, displacement_settings, marker_angle_field, pinned_field,
)
from ..utils.zoom_levels import ZoomLevels
from .base_layer_cache import BaseLayerCache
from .ddp_fetcher import COLOR_RAW_EXPRESSIONS, DataDefinedPropertiesFetcher
# Registers the q2vt_* expression functions the rules' geometry generators call.
from . import symbol_geometry  # noqa: F401  pylint: disable=unused-import


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

# Providers whose data lives in the project layer itself, so a layer opened
# from the URI is empty: these are read from a snapshot of the live layer.
_LIVE_PROVIDERS = frozenset({"memory"})

# Base layers and rule outputs. FlatGeobuf, not SQLite/SpatiaLite/GeoPackage:
# many threads create and read these files at once, and SpatiaLite's
# per-connection setup isn't thread-safe (it calls setlocale), which corrupted
# the heap. Written without a spatial index (nothing reads them by area, and
# the index would reorder features), so writing is a plain append.
_TEMP_LAYER_FORMAT = 'fgb'
_TEMP_RULE_FORMAT = 'fgb'

# Unicode direction marks, embeddings and isolates (in a regex character
# class): LRM, RLM, LRE..RLO, LRI..PDI.
_BIDI_MARKS = "‎‏‪-‮⁦-⁩"

# Features buffered per writer.addFeatures() call, features per base-layer
# batch handed to a worker, and how often streaming loops poll for
# cancellation.
_WRITE_BATCH_SIZE = 1000

# Most rule outputs one pass writes at once, as each holds an open file and a
# write buffer; groups beyond this go to further passes.
_MAX_GROUPS_PER_PASS = 64

# Python passes run in other processes (pass_worker.py), which use several
# cores where this process's threads share one interpreter, when their base
# layers have at least _PROCESS_MIN_FEATURES features in all (starting QGIS
# in a process takes a second or two); a pass is split into parts of about
# _PROCESS_PART_FEATURES features. Q2VT_NO_PASS_PROCESSES=1 keeps them here.
_PASS_PROCESSES = os.environ.get("Q2VT_NO_PASS_PROCESSES") != "1"
_PROCESS_MIN_FEATURES = 200000
_PROCESS_PART_FEATURES = 100000
# Smaller passes are batched: a few processes, each running its share of
# them one after another (QGIS starts once per process), when there are at
# least _BATCH_MIN_PASSES of them and _BATCH_MIN_FEATURES features in all.
_BATCH_MIN_PASSES = 4
_BATCH_MIN_FEATURES = 20000
# Units of work per process (a long pass split in ranges evens them out),
# and the fewest features a range of a pass has.
_BATCH_UNITS_PER_PROCESS = 1
_BATCH_MIN_PART_FEATURES = 5000
_PASS_WORKER = join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "processing", "pass_worker.py")

# Output WKB type per geometrybyexpression OUTPUT_GEOMETRY code
# (0 = polygon, 1 = line, 2 = point), already exploded to single parts.
_RULE_OUTPUT_WKB = {
    0: Qgis.WkbType.Polygon,
    1: Qgis.WkbType.LineString,
    2: Qgis.WkbType.Point,
}

# How an output field's value is produced in the rule-export pass.
_FIELD_CONSTANT, _FIELD_COPY, _FIELD_EXPRESSION = range(3)

# A field expression reading this variable is evaluated per part written
# (a marker on each vertex turned with the line): the part's geometry.
_PART_VARIABLE = "q2vt_part"

# Expression fragments that read attributes dynamically, so their field
# usage can't be determined statically by referencedColumns().
_DYNAMIC_ATTRIBUTE_TOKENS = ("@feature", "$currentfeature")

# Base-layer field flagging the largest part of each original feature (1),
# for rules that show only that part.
_BIGGEST_FIELD = f"{_FIELD_PREFIX}_biggest"

# Dataset field holding, in a dataset shared by several rules that differ by
# filter, the number of the rule each row belongs to; styles filter on it.
RULE_FIELD = f"{_FIELD_PREFIX}_rule"

# Most rules one shared dataset holds.
_MAX_MEMBERS = 250

# Most distinct raw colour values of a field the style maps to colours;
# beyond it the field is rewritten as #RRGGBBAA values.
_MAX_COLOR_VALUES = 1024

# Set Q2VT_NO_GDAL_EXPORT=1 to export every dataset with the Python pass, and
# Q2VT_NO_GDAL_BASE=1 to build every base layer by reading it with QGIS (for
# comparing the two paths).
_GDAL_EXPORT = os.environ.get("Q2VT_NO_GDAL_EXPORT") != "1"
_GDAL_BASE = os.environ.get("Q2VT_NO_GDAL_BASE") != "1"

# Intermediate GDAL files are kept in memory (/vsimem/) when their input is
# at most this large, and written next to the other temporary files if not.
_GDAL_MEMORY_INPUT_MAX = 256 * 1024 * 1024

# Geometry family (geometrybyexpression OUTPUT_GEOMETRY code, as in
# _OGR_OUTPUT) of an OGR layer geometry type, linear and single.
_OGR_FAMILY = {ogr.wkbPolygon: 0, ogr.wkbLineString: 1, ogr.wkbPoint: 2}

# GDAL drivers without SQL of their own: attribute filters (QGIS subset
# filters) are OGR SQL conditions in them (see _gdal_open_source).
_OGR_SQL_FILTER_DRIVERS = frozenset({
    "ESRI Shapefile", "FlatGeobuf", "GeoJSON", "GeoJSONSeq", "MapInfo File", "CSV",
    "GML", "KML", "LIBKML", "GPX", "DXF",
})
_QGIS_FAMILY = {
    Qgis.GeometryType.Polygon: 0, Qgis.GeometryType.Line: 1, Qgis.GeometryType.Point: 2,
}

# Output geometry per geometrybyexpression OUTPUT_GEOMETRY code, as OGR names:
# the geometry type GDAL writes and the source geometries it keeps.
_OGR_OUTPUT = {
    0: ("POLYGON", ("POLYGON", "MULTIPOLYGON", "CURVEPOLYGON", "MULTISURFACE",
                    "GEOMETRYCOLLECTION")),
    1: ("LINESTRING", ("LINESTRING", "MULTILINESTRING", "CIRCULARSTRING", "COMPOUNDCURVE",
                       "MULTICURVE", "GEOMETRYCOLLECTION")),
    2: ("POINT", ("POINT", "MULTIPOINT", "GEOMETRYCOLLECTION")),
}

# QMetaType codes of the field types the GDAL export writes, and their OGR SQL
# CAST types: int, long long, double, string.
_OGR_CAST_TYPES = {2: "integer", 4: "integer64", 6: "float", 10: "character"}
# Casts GDAL makes exactly as QGIS's field conversion does: integers to text
# or to decimals.
_OGR_SAFE_CASTS = {(2, 10), (4, 10), (2, 6), (4, 6), (2, 4)}

# Field types whose values a filter like "field" = literal can be looked up
# by (QMetaType codes): a dictionary matches exactly what the expression
# engine does for them. 2-5: int, uint, long long, ulong long; 10: string.
_LOOKUP_INT_TYPES = frozenset({2, 3, 4, 5})
_LOOKUP_STRING_TYPE = 10

# A line's marker points per marker line placement, in GDAL's SQLite dialect.
_MARKER_POINT_SQL = {
    MARKER_FIRST_VERTEX: "ST_StartPoint(GEOMETRY)",
    MARKER_LAST_VERTEX: "ST_EndPoint(GEOMETRY)",
    MARKER_CENTRAL_POINT: "ST_Line_Interpolate_Point(GEOMETRY, 0.5)",
}

# Polygon "using perimeter" placements (QgsPalLayerSettings.placement codes)
# and the line placement each becomes on the polygon's boundary.
_PERIMETER_PLACEMENTS = {
    2: Qgis.LabelPlacement.Line,             # Using perimeter
    7: Qgis.LabelPlacement.Curved,           # Using perimeter (curved)
}

# Polygon placements that look for room inside the polygon rather than
# around its centroid: Horizontal (4) and Free (5).
_POLYGON_INTERIOR_PLACEMENTS = frozenset({4, 5})

# The expression function applying a text format's type case
# (Qgis.Capitalization) to the label text: all uppercase, all lowercase,
# small caps, title case, all small caps.
_TYPE_CASE_FUNCTIONS = {1: "upper", 2: "lower", 4: "upper", 1004: "title", 1006: "upper"}

# Ground size of one pixel at scale 1:1 (OGC's 0.28 mm rendering pixel).
_METRES_PER_PIXEL_AT_SCALE_1 = 0.00028


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
class _LiveSource:
    """A memory layer's features, fields, geometry type and CRS (as WKT),
    taken from the project layer. QgsVectorLayerFeatureSource is QGIS's
    snapshot for reading a layer from another thread."""
    source: QgsVectorLayerFeatureSource
    fields: QgsFields
    wkb_type: Any
    crs_wkt: str

    @classmethod
    def of(cls, layer: QgsVectorLayer) -> "_LiveSource":
        return cls(
            QgsVectorLayerFeatureSource(layer), QgsFields(layer.fields()), layer.wkbType(),
            layer.crs().toWkt(Qgis.CrsWktVariant.Preferred),
        )


@dataclass(frozen=True)
class _SourceSnapshot:
    """Everything a worker needs to read a base layer; carries no QObjects."""
    layer_id: str
    name: str
    source_uri: str
    provider: str
    # Export extent in the source CRS as (xmin, ymin, xmax, ymax); None reads
    # the whole source (e.g. when the extent can't be transformed).
    extent: Optional[Tuple[float, float, float, float]] = None
    # Source field names referenced by any rule expression; None keeps all.
    required_fields: Optional[Tuple[str, ...]] = None
    # Set for _LIVE_PROVIDERS, which are read from it instead of the URI.
    live: Optional[_LiveSource] = None
    # The layer's CRS in QGIS (which may differ from the file's), as WKT.
    crs_wkt: str = ""
    # Whether a rule reads q2vt_biggest (the largest part of each feature):
    # otherwise GDAL-built base layers leave it at 1 for every part.
    needs_biggest: bool = True
    # The layer's geometry type in QGIS, as an _OGR_FAMILY value; None if
    # it has none.
    family: Optional[int] = None

    @property
    def needs_serial_read(self) -> bool:
        return self.provider in _SERIAL_READ_PROVIDERS


@dataclass
class _MemberSnapshot:
    """The rows one rule (or several rules writing the same rows) adds to a
    dataset: its filter, its calculated fields and its description."""
    filter_expression: Optional[str]
    expression_fields: List[Tuple[int, str, str]]
    description: str
    # Kept ONLY to drive the success/failure return value of export(); workers
    # MUST NOT read any live state from these.
    flat_rules: List[FlattenedRule]


@dataclass
class _RuleGroupSnapshot:
    """Everything a worker needs to export one output dataset.

    A dataset holds the rows of one or more members, which share the base
    layer, geometry transform and zoom range; with several, each row carries
    its member's number in q2vt_rule.

    Every field is a primitive type, string, list or dataclass — no
    QObjects, no QgsVectorLayer references. Safe to consume from any thread.
    """
    output_dataset: str
    layer_id: str
    rule_type: int
    geometry_target: int
    geometry_expression: str
    # Source fields copied into the output: None = all of them.
    copied_fields: Optional[Tuple[str, ...]]
    members: List[_MemberSnapshot] = dataclass_field(default_factory=list)
    # Export only the largest part of each original feature (labels without
    # "label every part", centroid fills without "point on all parts").
    keep_biggest_part: bool = False
    # Source layer name, for user-facing messages.
    layer_name: str = ""
    # Whether rules differing by filter may join this dataset.
    shareable: bool = True
    # Line labels with "merge connected lines": after the export, touching
    # lines with the same label (and other values) become one line.
    merge_lines: bool = False
    # geometry_expression in GDAL's SQLite dialect; None if it has none.
    geometry_sql: Optional[str] = None
    # The layer's "control feature rendering order": (expression, ascending,
    # nulls first) clauses its rows are written in; empty for source order.
    order_by: Tuple[Tuple[str, bool, bool], ...] = ()
    # The whole zoom levels whose tiles hold the dataset.
    zoom_range: Tuple[int, int] = (0, 22)

    @property
    def tagged(self) -> bool:
        """Whether rows carry their member's number (several members)."""
        return len(self.members) > 1

    @property
    def flat_rules(self) -> List[FlattenedRule]:
        return [rule for member in self.members for rule in member.flat_rules]

    @property
    def label(self) -> str:
        rule_type = 'labeling' if self.rule_type == 1 else 'symbology'
        return f'{rule_type} of the "{self.layer_name or self.layer_id}" layer'


@dataclass
class _Entry:
    """The rules writing one output dataset's rows, while the snapshot
    groups them into shared datasets (see _plan_bands)."""
    flat_rules: List[FlattenedRule]
    geom_target: Any
    geom_expr: str
    fields: List[Tuple[int, str, str]] = dataclass_field(default_factory=list)
    # The whole zoom levels the dataset holds the rows for: the rule's own,
    # or the zoom bands' it shares a dataset with.
    zoom_range: Optional[Tuple[int, int]] = None

    @property
    def primary(self) -> FlattenedRule:
        return self.flat_rules[0]

    @property
    def zooms(self) -> Tuple[int, int]:
        return self.zoom_range or (self.primary.get_attr("o"), self.primary.get_attr("i"))


def _band_zoom(flat_rule: FlattenedRule) -> int:
    """The zoom level naming a zoom band's own fields (_zNN): the one its
    @map_scale expressions were evaluated at, unique among its bands."""
    return flat_rule.scale_zoom if flat_rule.scale_zoom is not None else flat_rule.get_attr("o")


def color_hex(raw: str) -> Optional[str]:
    """A raw colour value (any form QGIS reads: #RRGGBB, names, "r,g,b,a",
    rgb()...) as #RRGGBBAA, as the export's hex conversion writes it;
    None if QGIS reads no colour from it."""
    color = QgsSymbolLayerUtils.decodeColor(raw)
    if not color.isValid():
        return None
    return f"#{color.red():02X}{color.green():02X}{color.blue():02X}{color.alpha():02X}"


def group_to_dict(grp: _RuleGroupSnapshot) -> dict:
    """A rule group as JSON data, for a pass run in another process
    (pass_worker.py); its members' rules stay here."""
    return {
        "output_dataset": grp.output_dataset,
        "layer_id": grp.layer_id,
        "rule_type": grp.rule_type,
        "geometry_target": RulesExporter._enum_int(grp.geometry_target),
        "geometry_expression": grp.geometry_expression,
        "copied_fields": None if grp.copied_fields is None else list(grp.copied_fields),
        "members": [
            {
                "filter_expression": m.filter_expression,
                "expression_fields": [
                    [None if t is None else RulesExporter._enum_int(t), expr, name]
                    for t, expr, name in m.expression_fields
                ],
                "description": m.description,
            }
            for m in grp.members
        ],
        "keep_biggest_part": grp.keep_biggest_part,
        "layer_name": grp.layer_name,
        "shareable": grp.shareable,
        "merge_lines": grp.merge_lines,
        "geometry_sql": grp.geometry_sql,
        "order_by": [list(clause) for clause in grp.order_by],
    }


def group_from_dict(data: dict) -> _RuleGroupSnapshot:
    """group_to_dict's group back (without its members' rules)."""
    members = [
        _MemberSnapshot(
            filter_expression=m["filter_expression"],
            expression_fields=[tuple(f) for f in m["expression_fields"]],
            description=m["description"],
            flat_rules=[],
        )
        for m in data["members"]
    ]
    copied = data["copied_fields"]
    return _RuleGroupSnapshot(
        output_dataset=data["output_dataset"], layer_id=data["layer_id"],
        rule_type=data["rule_type"], geometry_target=data["geometry_target"],
        geometry_expression=data["geometry_expression"],
        copied_fields=None if copied is None else tuple(copied), members=members,
        keep_biggest_part=data["keep_biggest_part"], layer_name=data["layer_name"],
        shareable=data["shareable"], merge_lines=data["merge_lines"],
        geometry_sql=data["geometry_sql"],
        order_by=tuple(tuple(clause) for clause in data.get("order_by", ())),
    )


def group_expressions(grp: _RuleGroupSnapshot) -> List[str]:
    """Every QGIS expression a rule group's pass evaluates."""
    expressions = [grp.geometry_expression] + [clause[0] for clause in grp.order_by]
    for member in grp.members:
        if member.filter_expression:
            expressions.append(member.filter_expression)
        expressions.extend(expr for _, expr, _ in member.expression_fields
                           if isinstance(expr, str) and expr.strip())
    return expressions


class _BaseBuild:
    """One base layer being written by the caller thread (Phases 1 + 2)."""

    def __init__(self, src: _SourceSnapshot, path: str):
        self.src = src
        self.path = path
        self.writer: Optional[QgsVectorFileWriter] = None
        self.orig_idx = -1
        self.biggest_idx = -1
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
        # Per member: how each output field's value is produced.
        self.plans: List[List[Tuple[int, QgsField, Any, bool]]] = []
        self.geometry: Optional[QgsExpression] = None
        self.writer: Optional[QgsVectorFileWriter] = None
        self.batch: List[QgsFeature] = []
        self.written = 0
        self.failed = False
        # Problems that don't stop the export, reported once per group as
        # {what: [occurrences, first error]}.
        self.problems: Dict[str, List[Any]] = {}
        # Per member: the indexes of its plan's fields evaluated per part
        # (see _PART_VARIABLE), found on first use.
        self.part_fields: Dict[int, List[int]] = {}

    def note(self, what: str, error: str) -> None:
        entry = self.problems.setdefault(what, [0, error])
        entry[0] += 1


class _Cancelled(Exception):
    """Raised inside workers when the caller has signalled cancellation."""


# Marks a per-feature cache entry not computed yet.
_MISSING = object()

# Serialises each rule-export pass's setup (see _export_rule_pass_from).
_PASS_SETUP_LOCK = threading.Lock()


class _FilterMatcher:
    """Tells which of a pass's rule filters a feature passes.

    A filter "field" = literal or "field" IN (literals), as categorized
    renderers produce, alone or ANDed with other conditions, as for labels
    split by such categories, is found by looking the feature's value up in
    a dictionary instead of being evaluated; only its other conditions are
    evaluated. A feature then costs one lookup per such field rather than
    one evaluation per filter. Other filters are evaluated as they are.
    Results are those of QgsFeatureRequest.setFilterExpression().
    """

    _Node = QgsExpressionNode.NodeType
    _Op = QgsExpressionNodeBinaryOperator.BinaryOperator

    def __init__(self, filters: List[Optional[str]], fields: QgsFields, prepare):
        self._always: Set[Optional[str]] = set()
        self._evaluated: List[Tuple[str, QgsExpression]] = []
        # Per field index: {value: [(filter, its other conditions or None)]}.
        self._lookups: Dict[int, Dict[Any, List[Tuple[str, Optional[QgsExpression]]]]] = {}
        for filter_expression in filters:
            if not filter_expression:
                self._always.add(filter_expression)
                continue
            lookup = self._lookup_of(filter_expression, fields)
            if lookup is None:
                self._evaluated.append((filter_expression, prepare(filter_expression)))
                continue
            field_idx, values, remainder = lookup
            condition = prepare(remainder) if remainder else None
            table = self._lookups.setdefault(field_idx, {})
            for value in set(values):
                table.setdefault(value, []).append((filter_expression, condition))

    def passed(self, attributes: List[Any], context: QgsExpressionContext) -> Set[Optional[str]]:
        """The filters the feature in ``context`` passes."""
        passed = set(self._always)
        for field_idx, table in self._lookups.items():
            try:
                candidates = table.get(attributes[field_idx], ())
            except TypeError:  # An unhashable value equals no literal.
                continue
            for filter_expression, condition in candidates:
                if condition is None or self._is_true(condition.evaluate(context)):
                    passed.add(filter_expression)
        for filter_expression, expr in self._evaluated:
            if self._is_true(expr.evaluate(context)):
                passed.add(filter_expression)
        return passed

    @staticmethod
    def _is_true(value) -> bool:
        """QVariant::toBool(), which feature requests apply to filters."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.lower() not in ("", "0", "false")
        try:
            return bool(value)  # NULL is False.
        except (TypeError, ValueError):
            return False

    @classmethod
    def _lookup_of(
        cls, filter_expression: str, fields: QgsFields
    ) -> Optional[Tuple[int, List[Any], Optional[str]]]:
        """(field index, values, other conditions) of a filter that can be
        looked up; None if it must be evaluated."""
        expr = QgsExpression(filter_expression)
        root = expr.rootNode()
        if expr.hasParserError() or root is None:
            return None
        conjuncts = list(cls._conjuncts(root))
        for i, node in enumerate(conjuncts):
            found = cls._field_values(node, fields)
            if found is None:
                continue
            rest = conjuncts[:i] + conjuncts[i + 1:]
            remainder = " AND ".join(f"({n.dump()})" for n in rest) or None
            return found[0], found[1], remainder
        return None

    @classmethod
    def _conjuncts(cls, node: QgsExpressionNode) -> Iterator[QgsExpressionNode]:
        """The conditions ANDed together in ``node``."""
        if node.nodeType() == cls._Node.ntBinaryOperator and node.op() == cls._Op.boAnd:
            yield from cls._conjuncts(node.opLeft())
            yield from cls._conjuncts(node.opRight())
        else:
            yield node

    @classmethod
    def _field_values(
        cls, node: QgsExpressionNode, fields: QgsFields
    ) -> Optional[Tuple[int, List[Any]]]:
        """(field index, values) of "field" = literal or "field" IN
        (literals), for fields and literals a dictionary compares exactly as
        the expression engine does: both strings, or both integers."""
        node_type = node.nodeType()
        if node_type == cls._Node.ntBinaryOperator and node.op() == cls._Op.boEQ:
            column, literal = node.opLeft(), node.opRight()
            if column.nodeType() != cls._Node.ntColumnRef:
                column, literal = literal, column
            literals = [literal]
        elif node_type == cls._Node.ntInOperator and not node.isNotIn():
            column, literals = node.node(), node.list().list()
        else:
            return None
        if column.nodeType() != cls._Node.ntColumnRef or not literals or any(
            literal.nodeType() != cls._Node.ntLiteral for literal in literals
        ):
            return None
        field_idx = fields.lookupField(column.name())
        if field_idx < 0:
            return None
        field_type = int(getattr(fields.at(field_idx).type(), "value", fields.at(field_idx).type()))
        values = [literal.value() for literal in literals]
        if field_type == _LOOKUP_STRING_TYPE:
            ok = all(isinstance(value, str) for value in values)
        elif field_type in _LOOKUP_INT_TYPES:
            ok = all(type(value) is int for value in values)  # Not bool or float.
        else:
            ok = False
        return (field_idx, values) if ok else None


class _NotCompilable(Exception):
    """An expression OGR SQL can't compute exactly as QGIS does."""


def _sql_name(name: str) -> str:
    """A name for inside an OGR SQL double-quoted identifier."""
    return name.replace('"', '""')


def _sql_string(value: str) -> str:
    """An OGR SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def _sql_number(value) -> str:
    """An OGR SQL numeric literal."""
    if isinstance(value, int):
        return str(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise _NotCompilable(repr(value))
    text = repr(float(value))
    return text if "e" not in text else f"{value:.17f}"


class _OgrSqlCompiler:
    """Translates a QGIS filter expression to an OGR SQL WHERE clause that
    selects exactly the features QGIS's filter selects.

    Only constructs whose OGR SQL results match QGIS's are accepted:
    comparisons of a field with a literal of its own kind (=, <, <=, >, >=
    for numbers; = for text), IN lists, IS [NOT] NULL, AND and OR. Without
    NOT, a comparison with a NULL field is false in both (QGIS's NULL result
    counts as false), so AND / OR combine them alike. Anything else raises
    _NotCompilable.

    OGR SQL compares text with = ignoring case, unlike QGIS, so text
    equality becomes LIKE (case-sensitive since GDAL 3.1) with its
    wildcards escaped.
    """

    _Node = QgsExpressionNode.NodeType
    _Op = QgsExpressionNodeBinaryOperator.BinaryOperator
    # QMetaType codes: int, uint, long long, ulong long, double; string.
    _NUMERIC = frozenset({2, 3, 4, 5, 6})
    _STRING = 10

    def __init__(self, fields: QgsFields):
        self.fields = fields
        self._comparisons = {
            self._Op.boEQ: "=", self._Op.boLT: "<", self._Op.boLE: "<=",
            self._Op.boGT: ">", self._Op.boGE: ">=",
        }
        # The operator with its operands swapped (literal first).
        self._swapped = {
            self._Op.boEQ: self._Op.boEQ, self._Op.boLT: self._Op.boGT,
            self._Op.boLE: self._Op.boGE, self._Op.boGT: self._Op.boLT,
            self._Op.boGE: self._Op.boLE,
        }

    def compile(self, expression: str) -> str:
        expr = QgsExpression(expression)
        if expr.hasParserError() or expr.rootNode() is None:
            raise _NotCompilable(expression)
        return self._node(expr.rootNode())

    def _node(self, node) -> str:
        node_type = node.nodeType()
        if node_type == self._Node.ntBinaryOperator:
            op = node.op()
            if op in (self._Op.boAnd, self._Op.boOr):
                joiner = "AND" if op == self._Op.boAnd else "OR"
                return f"({self._node(node.opLeft())}) {joiner} ({self._node(node.opRight())})"
            if op in (self._Op.boIs, self._Op.boIsNot):
                column, literal = node.opLeft(), node.opRight()
                if (literal.nodeType() != self._Node.ntLiteral
                        or literal.value() is not None
                        or column.nodeType() != self._Node.ntColumnRef):
                    raise _NotCompilable(node.dump())
                name, _ = self._column(column)
                return f"{name} IS {'NOT ' if op == self._Op.boIsNot else ''}NULL"
            if op in self._comparisons:
                column, literal = node.opLeft(), node.opRight()
                if column.nodeType() != self._Node.ntColumnRef:
                    column, literal, op = literal, column, self._swapped[op]
                name, kind = self._column(column)
                if kind == self._STRING:
                    if op != self._Op.boEQ:
                        raise _NotCompilable(node.dump())  # Text order differs.
                    return self._text_equals(name, literal)
                return f"{name} {self._comparisons[op]} {self._literal(literal, kind)}"
            raise _NotCompilable(node.dump())
        if node_type == self._Node.ntInOperator and not node.isNotIn():
            name, kind = self._column(node.node())
            items = node.list().list()
            if not items:
                raise _NotCompilable(node.dump())
            if kind == self._STRING:
                return " OR ".join(f"({self._text_equals(name, item)})" for item in items)
            return f"{name} IN ({', '.join(self._literal(item, kind) for item in items)})"
        raise _NotCompilable(node.dump())

    def _column(self, node) -> Tuple[str, int]:
        """(quoted name, kind) of a field reference to a number or text field."""
        if node.nodeType() != self._Node.ntColumnRef:
            raise _NotCompilable(node.dump())
        idx = self.fields.lookupField(node.name())
        if idx < 0:
            raise _NotCompilable(node.name())
        field = self.fields.at(idx)
        field_type = int(getattr(field.type(), "value", field.type()))
        if field_type in self._NUMERIC:
            kind = 6
        elif field_type == self._STRING:
            kind = self._STRING
        else:
            raise _NotCompilable(node.name())
        return f'"{_sql_name(field.name())}"', kind

    def _text_equals(self, name: str, literal) -> str:
        """Case-sensitive equality of a text field with a text literal."""
        self._literal(literal, self._STRING)  # Checks it is a text literal.
        pattern = literal.value()
        for char in ("\\", "%", "_"):
            pattern = pattern.replace(char, "\\" + char)
        return f"{name} LIKE {_sql_string(pattern)} ESCAPE '\\'"

    def _literal(self, node, kind: int) -> str:
        """A literal of the field's kind, in OGR SQL."""
        if node.nodeType() != self._Node.ntLiteral:
            raise _NotCompilable(node.dump())
        value = node.value()
        if kind == self._STRING and isinstance(value, str):
            return _sql_string(value)
        if kind != self._STRING and isinstance(value, (int, float)) and not isinstance(value, bool):
            return _sql_number(value)
        raise _NotCompilable(node.dump())


# GDAL's SQLite SQL dialect, and SpatiaLite's geometry functions in it:
# (dialect, spatialite) availability, probed once (see _sqlite_capabilities).
_SQLITE_CAPS: Optional[Tuple[bool, bool]] = None
_SQLITE_CAPS_LOCK = threading.Lock()


# SQL in GDAL's SQLite dialect never runs in this process: each statement
# opens a SQLite connection with SpatiaLite, whose setup calls setlocale,
# which races with every other thread's GDAL work (access violations, heap
# corruption). It runs in GDAL's command-line tools instead (ogr2ogr,
# ogrinfo), one process per statement, still in C++ and in parallel.
_CREATE_NO_WINDOW = 0x08000000


def _gdal_tool(arguments: List[str], cancelled=None, timeout: float = None) -> str:
    """Run a GDAL command-line tool; returns its output. Raises
    RuntimeError if it fails (or _Cancelled once ``cancelled()``)."""
    kwargs = {"creationflags": _CREATE_NO_WINDOW} if os.name == "nt" else {}
    with open(os.devnull, "rb") as stdin:
        process = subprocess.Popen(
            arguments, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    output: List[bytes] = []
    reader = threading.Thread(target=lambda: output.append(process.stdout.read()), daemon=True)
    reader.start()
    waited = 0.0
    try:
        while process.poll() is None:
            if cancelled is not None and cancelled():
                raise _Cancelled()
            if timeout is not None and waited > timeout:
                raise RuntimeError(f"{arguments[0]} timed out")
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                waited += 0.25
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join()
        process.stdout.close()
    text = b"".join(output).decode("utf-8", "replace")
    if process.returncode != 0:
        raise RuntimeError(text.strip()[-500:] or f"{arguments[0]} failed ({process.returncode})")
    return text


def _sqlite_capabilities() -> Tuple[bool, bool]:
    """Whether GDAL's tools run SQL in the SQLite dialect, and with
    SpatiaLite's geometry functions (ST_Centroid...)."""
    global _SQLITE_CAPS  # pylint: disable=global-statement
    with _SQLITE_CAPS_LOCK:
        if _SQLITE_CAPS is not None:
            return _SQLITE_CAPS
        caps = [False, False]
        path = os.path.join(tempfile.gettempdir(), f"q2vt_probe_{uuid4().hex}.fgb")
        try:
            ds = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(path)
            layer = ds.CreateLayer("probe", None, ogr.wkbPoint)
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(ogr.CreateGeometryFromWkt("POINT (0 0)"))
            layer.CreateFeature(feature)
            ds = None
            probes = ("SELECT 1234567 AS x FROM probe",
                      "SELECT ST_Area(ST_Buffer(GEOMETRY, 1000)) > 3000000 AS x FROM probe")
            for i, sql in enumerate(probes):
                try:
                    out = _gdal_tool(["ogrinfo", "-ro", "-q", path, "-dialect", "SQLITE", "-sql", sql],
                                     timeout=60)
                except (RuntimeError, OSError):
                    continue
                caps[i] = ("1234567" in out) if i == 0 else ("x (Integer) = 1" in out)
        except (RuntimeError, AttributeError, OSError):
            pass
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        _SQLITE_CAPS = (caps[0], caps[0] and caps[1])
        return _SQLITE_CAPS


class _SqliteCompiler:
    """Translates QGIS expressions to SQL in GDAL's SQLite dialect that
    computes exactly what QGIS computes, for the export's GDAL path.

    Only constructs whose SQLite results match QGIS's are accepted, typed
    as they go (int, real, str, bool, null): field references and
    literals; AND, OR, NOT; comparisons of numbers, = and <> of texts;
    IS [NOT]; IN lists; + - * of numbers, / (as real), % of integers;
    || of texts and integers; CASE WHEN; coalesce(), if(), upper(),
    lower(), abs(), round(x, n). Text compares byte-wise and case-sensitively in both;
    NULL propagates alike, a field the layer lacks being NULL. Anything
    else raises _NotCompilable.
    """

    _Node = QgsExpressionNode.NodeType
    _Op = QgsExpressionNodeBinaryOperator.BinaryOperator
    _INT, _REAL, _STR, _BOOL, _NULL = "int", "real", "str", "bool", "null"
    # QMetaType codes: bool; int, uint, long long, ulong long; double; string.
    _FIELD_KINDS = {1: "bool", 2: "int", 3: "int", 4: "int", 5: "int", 6: "real", 10: "str"}
    # Output field type (QMetaType code) -> SQLite CAST type, and the
    # expression kinds QGIS converts to it exactly as the CAST does.
    _OUTPUT = {
        1: ("INTEGER", {"bool", "int", "null"}),
        2: ("INTEGER", {"int", "bool", "null"}),
        4: ("INTEGER", {"int", "bool", "null"}),
        6: ("REAL", {"int", "real", "null"}),
        10: ("TEXT", {"str", "int", "null"}),
    }

    def __init__(self, fields: QgsFields):
        self.fields = fields
        self._expressions: List[QgsExpression] = []
        ops = self._Op
        self._comparisons = {ops.boEQ: "=", ops.boNE: "<>", ops.boLT: "<", ops.boLE: "<=",
                             ops.boGT: ">", ops.boGE: ">="}
        self._arithmetic = {ops.boPlus: "+", ops.boMinus: "-", ops.boMul: "*"}

    # --- entry points ---

    def condition(self, expression: str) -> str:
        """A filter: true exactly where QGIS's filter selects a feature."""
        sql, kind = self._node(self._root(expression))
        if kind not in (self._BOOL, self._NULL):
            raise _NotCompilable(expression)  # QGIS's truthiness of other values differs.
        return sql

    def value(self, expression: str, field_type: int, src_fields: QgsFields) -> str:
        """An output field's value, cast to the field's type as QGIS's
        field conversion does."""
        expr = QgsExpression(expression)
        if expr.isField():
            idx = src_fields.lookupField(next(iter(expr.referencedColumns())))
            if idx >= 0 and RulesExporter._enum_int(src_fields.at(idx).type()) == field_type:
                return f'"{_sql_name(src_fields.at(idx).name())}"'  # A copy, any type.
        cast, kinds = self._OUTPUT.get(field_type, (None, ()))
        sql, kind = self._node(self._root(expression))
        if cast is None or kind not in kinds:
            raise _NotCompilable(expression)
        return f"CAST(({sql}) AS {cast})"

    def order_key(self, expression: str) -> str:
        """A feature rendering order's expression, to sort rows by (text
        sorts byte-wise, as QGIS's OGR provider sorts it)."""
        sql, _ = self._node(self._root(expression))
        return sql

    def _root(self, expression: str):
        expr = QgsExpression(expression)
        if expr.hasParserError() or expr.rootNode() is None or expr.referencedVariables():
            raise _NotCompilable(expression)
        # The nodes belong to the expression: it must outlive their use.
        self._expressions.append(expr)
        return expr.rootNode()

    # --- nodes ---

    def _node(self, node) -> Tuple[str, str]:
        node_type = node.nodeType()
        if node_type == self._Node.ntLiteral:
            return self._literal(node.value())
        if node_type == self._Node.ntColumnRef:
            idx = self.fields.lookupField(node.name())
            if idx < 0:
                # A field the layer doesn't have is NULL in QGIS (a style
                # shared by layers of different fields).
                return "NULL", self._NULL
            field = self.fields.at(idx)
            kind = self._FIELD_KINDS.get(RulesExporter._enum_int(field.type()))
            if kind is None:
                raise _NotCompilable(node.name())
            return f'"{_sql_name(field.name())}"', kind
        if node_type == self._Node.ntUnaryOperator:
            sql, kind = self._node(node.operand())
            if node.op() == QgsExpressionNodeUnaryOperator.UnaryOperator.uoNot and kind in (self._BOOL, self._NULL):
                return f"NOT ({sql})", self._BOOL
            if node.op() == QgsExpressionNodeUnaryOperator.UnaryOperator.uoMinus and kind in (self._INT, self._REAL):
                return f"-({sql})", kind
            raise _NotCompilable(node.dump())
        if node_type == self._Node.ntBinaryOperator:
            return self._binary(node)
        if node_type == self._Node.ntInOperator:
            sql, kind = self._node(node.node())
            items = [self._node(item) for item in node.list().list()]
            if not items or any(not self._comparable(kind, k) for _, k in items):
                raise _NotCompilable(node.dump())
            keyword = "NOT IN" if node.isNotIn() else "IN"
            return f"({sql}) {keyword} ({', '.join(s for s, _ in items)})", self._BOOL
        if node_type == self._Node.ntCondition:
            parts, kinds = [], []
            for when_then in node.conditions():
                condition, ckind = self._node(when_then.whenExp())
                if ckind not in (self._BOOL, self._NULL):
                    raise _NotCompilable(node.dump())
                result, rkind = self._node(when_then.thenExp())
                parts.append(f"WHEN {condition} THEN {result}")
                kinds.append(rkind)
            otherwise = ""
            if node.elseExp() is not None:
                result, rkind = self._node(node.elseExp())
                otherwise = f" ELSE {result}"
                kinds.append(rkind)
            return f"(CASE {' '.join(parts)}{otherwise} END)", self._common(kinds, node)
        if node_type == self._Node.ntFunction:
            return self._function(node)
        raise _NotCompilable(node.dump())

    def _binary(self, node) -> Tuple[str, str]:
        op = node.op()
        left, lkind = self._node(node.opLeft())
        right, rkind = self._node(node.opRight())
        if op in (self._Op.boAnd, self._Op.boOr):
            if {lkind, rkind} <= {self._BOOL, self._NULL}:
                joiner = "AND" if op == self._Op.boAnd else "OR"
                return f"({left}) {joiner} ({right})", self._BOOL
            raise _NotCompilable(node.dump())
        if op in self._comparisons or op in (self._Op.boIs, self._Op.boIsNot):
            if not self._comparable(lkind, rkind):
                raise _NotCompilable(node.dump())
            text = self._STR in (lkind, rkind)
            if text and op not in (self._Op.boEQ, self._Op.boNE, self._Op.boIs, self._Op.boIsNot):
                raise _NotCompilable(node.dump())  # Text order differs.
            sql_op = {self._Op.boIs: "IS", self._Op.boIsNot: "IS NOT"}.get(op) or self._comparisons[op]
            return f"({left}) {sql_op} ({right})", self._BOOL
        numeric = {self._INT, self._REAL, self._NULL}
        if op in self._arithmetic and {lkind, rkind} <= numeric:
            kind = self._REAL if self._REAL in (lkind, rkind) else self._INT
            return f"(({left}) {self._arithmetic[op]} ({right}))", kind
        if op == self._Op.boDiv and {lkind, rkind} <= numeric:
            return f"(CAST(({left}) AS REAL) / ({right}))", self._REAL
        if op == self._Op.boMod and {lkind, rkind} <= {self._INT, self._NULL}:
            return f"(({left}) % ({right}))", self._INT
        if op == self._Op.boConcat and {lkind, rkind} <= {self._STR, self._INT, self._NULL}:
            return f"(({left}) || ({right}))", self._STR
        raise _NotCompilable(node.dump())

    def _function(self, node) -> Tuple[str, str]:
        name = QgsExpression.Functions()[node.fnIndex()].name().lower()
        args = [self._node(arg) for arg in (node.args().list() if node.args() else [])]
        kinds = [kind for _, kind in args]
        if name == "coalesce" and args:
            return f"COALESCE({', '.join(s for s, _ in args)})", self._common(kinds, node)
        if name in ("upper", "lower") and len(args) == 1 and kinds[0] in (self._STR, self._NULL):
            return f"{name.upper()}({args[0][0]})", self._STR
        if name == "abs" and len(args) == 1 and kinds[0] in (self._INT, self._REAL, self._NULL):
            return f"ABS({args[0][0]})", kinds[0]
        if (name == "round" and len(args) == 2 and kinds[0] in (self._INT, self._REAL, self._NULL)
                and kinds[1] == self._INT and node.args().list()[1].nodeType() == self._Node.ntLiteral):
            # Both round half away from zero.
            return f"ROUND({args[0][0]}, {args[1][0]})", self._REAL
        if name == "if" and len(args) == 3 and kinds[0] in (self._BOOL, self._NULL):
            return (f"(CASE WHEN {args[0][0]} THEN {args[1][0]} ELSE {args[2][0]} END)",
                    self._common(kinds[1:], node))
        raise _NotCompilable(node.dump())

    def _literal(self, value) -> Tuple[str, str]:
        if value is None or (hasattr(value, "isNull") and value.isNull()):
            return "NULL", self._NULL
        if isinstance(value, bool):
            return ("1" if value else "0"), self._BOOL
        if isinstance(value, int):
            return str(value), self._INT
        if isinstance(value, float):
            return _sql_number(value), self._REAL
        if isinstance(value, str):
            return _sql_string(value), self._STR
        raise _NotCompilable(repr(value))

    def _comparable(self, a: str, b: str) -> bool:
        if self._NULL in (a, b):
            return True
        numbers = (self._INT, self._REAL)
        # A boolean compares as 1 / 0 in both (an ELSE rule's "(...) IS 1").
        bool_int = {a, b} == {self._BOOL, self._INT}
        return (a in numbers and b in numbers) or (a == b == self._STR) or (a == b == self._BOOL) or bool_int

    def _common(self, kinds: List[str], node) -> str:
        """The one kind of several results (NULLs fit any)."""
        found = {k for k in kinds if k != self._NULL}
        if not found:
            return self._NULL
        if found == {self._INT, self._REAL}:
            return self._REAL
        if len(found) != 1:
            raise _NotCompilable(node.dump())
        return found.pop()


# OGR field type (and subtype) the GDAL path's virtual layers declare per
# output field type (QMetaType code), so every dataset keeps its schema
# whatever SQLite's values look like (an all-NULL column, say).
_VRT_FIELD_TYPES = {1: ("Integer", "Boolean"), 2: ("Integer", None), 4: ("Integer64", None),
                    6: ("Real", None), 10: ("String", None)}


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
        included_fields: int,
        max_zoom,
        utils_dir: str,
        feedback: QgsProcessingFeedback,
        cpu_percent: int = 100,
    ):
        self.flattened_rules = flattened_rules
        # QgsRectangle is a value type — safe to share across threads.
        self.extent = extent
        self.included_fields = included_fields
        self.max_zoom = max_zoom
        self.utils_dir = utils_dir
        self.feedback = feedback
        self.cpu_percent = cpu_percent

        self.processed_layers: List[QgsVectorLayer] = []
        # Per dataset: its colour fields holding raw colour values (the style
        # maps each value to its colour); others hold #RRGGBBAA values.
        self.raw_color_fields: Dict[str, Set[str]] = {}

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

        # Per geometry expression the export made: the same geometry in
        # GDAL's SQLite dialect (see _sql_for_geometry).
        self._geometry_sql: Dict[str, str] = {}

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
            self._merge_label_lines(rule_groups, rule_outputs)
            self._collect_parts(rule_groups, rule_outputs)
            self._finish_raw_colors(rule_outputs)
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

        # Snapshot rule groups, then gather them into shared datasets.
        datasets: Dict[tuple, List[_RuleGroupSnapshot]] = {}
        rule_groups: List[_RuleGroupSnapshot] = []
        # Per renderer rule (by id): the dataset and member holding its rows.
        placed: Dict[int, Tuple[_RuleGroupSnapshot, _MemberSnapshot]] = {}

        # Each dataset's geometry, computed once. Before the expression
        # fields: they replace data-defined properties (e.g. the label's
        # "centroid of") with references to fields the base layer lacks.
        entries: List[_Entry] = []
        for output_dataset, flat_rules in rules_by_dataset.items():
            transformation = self._get_geometry_transformation(flat_rules[0])
            if transformation is None:
                continue  # No geometry transformation: cannot be exported.
            entries.append(_Entry(flat_rules, *transformation))

        # Symbology first: labels may be drawn from its rows.
        renderer_entries = [e for e in entries if e.primary.get_attr("t") == 0]
        for entry in renderer_entries:
            entry.fields = self._create_expression_fields(entry.flat_rules)
            entry.fields.extend(self._marker_angle_fields(entry.primary))
        self._plan_bands(renderer_entries)
        for entry in renderer_entries:
            group, holder = self._group_entry(entry, datasets, rule_groups)
            for flat_rule in entry.flat_rules:
                placed[id(flat_rule)] = (group, holder)

        label_entries = []
        for entry in entries:
            if entry.primary.get_attr("t") != 1:
                continue
            if len(entry.flat_rules) == 1 and self._attach_label(
                    entry.primary, entry.geom_target, entry.geom_expr, placed):
                continue
            entry.fields = self._add_label_expression_field(
                entry.primary, self._create_expression_fields(entry.flat_rules))
            label_entries.append(entry)
        self._plan_bands(label_entries)
        for entry in label_entries:
            self._group_entry(entry, datasets, rule_groups)

        # Point every rule at its dataset (and, if shared, its member number).
        for group in rule_groups:
            for index, member in enumerate(group.members):
                for flat_rule in member.flat_rules:
                    flat_rule.output_dataset = group.output_dataset
                    flat_rule.member = index if group.tagged else None

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
                live=_LiveSource.of(layer) if layer.providerType() in _LIVE_PROVIDERS else None,
                crs_wkt=layer.crs().toWkt(Qgis.CrsWktVariant.Preferred) if layer.crs().isValid() else "",
                family=_QGIS_FAMILY.get(layer.geometryType()),
                needs_biggest=any(
                    grp.keep_biggest_part or any(
                        _BIGGEST_FIELD in expr for m in grp.members for _, expr, _ in m.expression_fields
                        if isinstance(expr, str))
                    for grp in groups
                ),
            )

        return sources, rule_groups

    def _group_entry(
        self, entry: _Entry, datasets: Dict[tuple, List[_RuleGroupSnapshot]],
        rule_groups: List[_RuleGroupSnapshot],
    ) -> Tuple[_RuleGroupSnapshot, _MemberSnapshot]:
        """Add an entry's rows to a dataset it can share, or a new one;
        returns the dataset and the member holding its rules."""
        primary = entry.primary
        member = _MemberSnapshot(
            filter_expression=primary.rule.filterExpression() or None,
            expression_fields=entry.fields,
            description=primary.get_description(),
            flat_rules=entry.flat_rules,
        )
        rule_type = primary.get_attr("t")
        merge_lines = self._merges_lines(primary, entry.geom_target)
        # Merged lines are made of all the parts of every feature.
        keep_biggest = self._keeps_biggest_part(primary) and not merge_lines
        low, high = entry.zooms
        key = (
            primary.layer.id(), rule_type, self._enum_int(entry.geom_target), entry.geom_expr,
            keep_biggest, low, high, merge_lines,
        )
        # Rules drawn by symbol levels keep their own datasets: QGIS draws
        # them level by level, not feature by feature.
        shareable = rule_type == 1 or not self._uses_symbol_levels(primary.layer)
        if rule_type == 0 and (cluster_settings(primary.rule.symbol()) is not None
                               or displacement_settings(primary.rule.symbol()) is not None):
            shareable = False  # Its points become clusters (group centres) in the tiles, alone.
        if rule_type == 1 and displacement_label_settings(primary.rule.settings()) is not None:
            shareable = False  # Its labels move beside their points in the tiles, alone.
        group, holder = self._dataset_for(datasets.setdefault(key, []), member, shareable)
        if group is None:
            holder = member
            group = _RuleGroupSnapshot(
                # Named for the zoom levels it holds (tiles_generator reads them).
                output_dataset=re.sub(r"o\d\di\d\d", f"o{low:02d}i{high:02d}", primary.output_dataset, count=1),
                layer_id=primary.layer.id(),
                rule_type=rule_type,
                geometry_target=entry.geom_target,
                geometry_expression=entry.geom_expr,
                copied_fields=self._copied_fields(primary.layer),
                members=[member],
                keep_biggest_part=keep_biggest,
                layer_name=primary.layer.name(),
                shareable=shareable,
                merge_lines=merge_lines,
                geometry_sql=self._sql_for_geometry(entry.geom_expr),
                order_by=self._order_by(primary.layer) if rule_type == 0 else (),
                zoom_range=(low, high),
            )
            datasets[key].append(group)
            rule_groups.append(group)
        return group, holder

    def _plan_bands(self, entries: List[_Entry]) -> None:
        """Let the zoom bands of a rule (split by its @map_scale
        expressions) share one dataset where that stores less.

        Bands drawing the same rows (the same filter and geometry) with the
        same fields write the same data: one dataset holds it for all their
        zoom levels, and the style draws the bands from it (merged into one
        layer where it can, see merge_zoom_bands). Bands whose field values
        differ keep their own datasets below the highest tile zoom (each
        tile holds one band's rows anyway); the bands reaching it, whose
        rows all go to its tiles, share one dataset there, each band's
        differing values in fields of its own (named _zNN): the tiles hold
        each feature once instead of once per band.
        """
        bands: Dict[tuple, List[_Entry]] = {}
        for entry in entries:
            primary = entry.primary
            if primary.band_group is None:
                continue
            key = (primary.band_group, entry.geom_expr, primary.rule.filterExpression(),
                   primary.label_filter)
            bands.setdefault(key, []).append(entry)
        for group in bands.values():
            group.sort(key=lambda e: e.primary.get_attr("o"))
            for run in self._contiguous_runs(group):
                differing = self._differing_fields(run)
                if differing:
                    run = [e for e in run if e.primary.get_attr("i") >= _TILES_MAX_ZOOM]
                    if len(run) < 2:
                        continue
                    differing = self._differing_fields(run)
                low = run[0].primary.get_attr("o")
                high = max(e.primary.get_attr("i") for e in run)
                for entry in run:
                    entry.zoom_range = (low, high)
                    if differing:
                        self._rename_band_fields(entry, differing)

    @staticmethod
    def _contiguous_runs(group: List[_Entry]) -> List[List[_Entry]]:
        """Bands (in zoom order) as runs without a zoom level between them."""
        runs: List[List[_Entry]] = []
        for entry in group:
            # A band starting at its exact switch shares the zoom level the
            # band before ends in.
            if runs and entry.primary.get_attr("o") <= runs[-1][-1].primary.get_attr("i") + 1:
                runs[-1].append(entry)
            else:
                runs.append([entry])
        return [run for run in runs if len(run) > 1]

    @staticmethod
    def _differing_fields(run: List[_Entry]) -> Set[str]:
        """Names of the fields whose expression differs between bands."""
        maps = [{name: (field_type, expr) for field_type, expr, name in e.fields} for e in run]
        names = set().union(*maps)
        return {name for name in names if len({repr(m.get(name)) for m in maps}) > 1}

    @staticmethod
    def _rename_band_fields(entry: _Entry, differing: Set[str]) -> None:
        """Give a band its own names (_zNN) for the fields that differ
        between bands, in its fields and in the properties reading them."""
        suffix = f"_z{_band_zoom(entry.primary):02d}"
        renames = {name: f"{name}{suffix}" for name in differing}
        entry.fields = [(t, expr, renames.get(name, name)) for t, expr, name in entry.fields]
        for flat_rule in entry.flat_rules:
            DataDefinedPropertiesFetcher.rename_fields(flat_rule.rule, renames)
            settings = flat_rule.rule.settings() if flat_rule.get_attr("t") == 1 else None
            if settings is not None and settings.fieldName in renames:
                settings.fieldName = renames[settings.fieldName]

    @staticmethod
    def _dataset_for(
        candidates: List[_RuleGroupSnapshot], member: _MemberSnapshot, shareable: bool
    ) -> Tuple[Optional[_RuleGroupSnapshot], Optional[_MemberSnapshot]]:
        """Add ``member`` to one of the datasets it can share; return that
        dataset and the member now holding its rules, or (None, None) if it
        needs its own.

        A member writing the same rows as an existing one (the symbol layers
        of one rule: same filter and description) joins it, adding its
        fields; one differing by filter becomes another member if both sides
        allow sharing and their field types agree.
        """
        for group in candidates:
            for existing in group.members:
                if (existing.filter_expression == member.filter_expression
                        and existing.description == member.description):
                    names = {name for _, _, name in existing.expression_fields}
                    existing.expression_fields.extend(
                        f for f in member.expression_fields if f[2] not in names
                    )
                    existing.flat_rules.extend(member.flat_rules)
                    return group, existing
        if not shareable:
            return None, None
        for group in candidates:
            if not group.shareable or len(group.members) >= _MAX_MEMBERS:
                continue
            types = {
                name: field_type for m in group.members
                for field_type, _, name in m.expression_fields
            }
            if all(types.get(name, field_type) == field_type
                   for field_type, _, name in member.expression_fields):
                group.members.append(member)
                return group, member
        return None, None

    def _attach_label(
        self,
        label: FlattenedRule,
        geom_target,
        geom_expr: str,
        placed: Dict[int, Tuple[_RuleGroupSnapshot, _MemberSnapshot]],
    ) -> bool:
        """Draw a label rule from its renderer rule's rows, without a
        dataset of its own: its text (and data-defined values) become fields
        of those rows, NULL where the label's own filter or "label every
        part" leaves a feature unlabelled. Returns False (nothing changed)
        when the label needs other rows: another geometry (polygon labels,
        geometry generators, merged lines) or fields that would clash.

        A point or line layer's labels would otherwise store every feature's
        geometry in the tiles twice.
        """
        entry = placed.get(id(label.renderer)) if label.renderer is not None else None
        if entry is None or label.label_filter is None:
            return False
        if displacement_label_settings(label.rule.settings()) is not None:
            return False  # Placed beside its points by the tiles, in rows of its own.
        group, member = entry
        if self._merge_mode(group) != "parts":
            return False  # Its rows are merged or inverted (labels are per feature).
        if (geom_expr != "@geometry" or group.geometry_expression != "@geometry"
                or self._enum_int(geom_target) != self._enum_int(group.geometry_target)
                or self._merges_lines(label, geom_target)):
            return False
        # Its rows must be in the tiles of every zoom level the label shows at.
        low, high = group.zoom_range
        if label.get_attr("o") < low or label.get_attr("i") > high:
            return False
        settings = label.rule.settings()
        if not settings or not settings.getLabelExpression().expression():
            return False
        # The fields the label adds, worked out on a copy first: the real
        # fetch rewrites the label's properties to read them.
        trial = FlattenedRule(label.rule.clone(), label.layer)
        trial.rule.setDescription(label.rule.description())
        new_fields = self._create_expression_fields([trial])
        name = f"{_FIELD_PREFIX}_label_{label.get_attr('d'):02d}{label.get_attr('r'):02d}"
        conditions = []
        if label.label_filter:
            conditions.append(f"({label.label_filter})")
        if self._keeps_biggest_part(label):
            conditions.append(f'"{_BIGGEST_FIELD}" = 1')
        text_field = self._add_label_expression_field(trial, [], name, " AND ".join(conditions))
        existing = {field_name: expr for _, expr, field_name in member.expression_fields}
        if name in existing and (not text_field or existing[name] != text_field[0][1]):
            return False  # Another label's text (or this one's, other band) is there.
        # A data-defined value taken by another rule's: a zoom band of this
        # label gets its own field (_zNN), others need rows of their own.
        renames = {}
        for _, expr, field_name in new_fields:
            if existing.get(field_name, expr) != expr:
                if label.band_group is None:
                    return False
                renamed = f"{field_name}_z{_band_zoom(label):02d}"
                if existing.get(renamed, expr) != expr:
                    return False
                renames[field_name] = renamed
        fields = self._create_expression_fields([label])
        if renames:
            fields = [(t, expr, renames.get(field_name, field_name)) for t, expr, field_name in fields]
            DataDefinedPropertiesFetcher.rename_fields(label.rule, renames)
        fields = self._add_label_expression_field(label, fields, name, " AND ".join(conditions))
        member.expression_fields.extend(f for f in fields if f[2] not in existing)
        member.flat_rules.append(label)
        return True

    @staticmethod
    def _order_by(layer: QgsVectorLayer) -> Tuple[Tuple[str, bool, bool], ...]:
        """A layer's feature rendering order, if "control feature rendering
        order" is on: (expression, ascending, nulls first) per clause."""
        try:
            renderer = layer.renderer()
            if not renderer or not renderer.orderByEnabled():
                return ()
            return tuple(
                (clause.expression().expression(), bool(clause.ascending()), bool(clause.nullsFirst()))
                for clause in renderer.orderBy().list()
                if clause.expression().expression().strip()
            )
        except (AttributeError, RuntimeError, TypeError):
            return ()

    @staticmethod
    def _uses_symbol_levels(layer: QgsVectorLayer) -> bool:
        try:
            return bool(layer.renderer() and layer.renderer().usingSymbolLevels())
        except (AttributeError, RuntimeError):
            return False

    @classmethod
    def _merges_lines(cls, flat_rule: FlattenedRule, geom_target) -> bool:
        """Whether a label rule labels its lines merged ("merge connected
        lines"), as QGIS does before placing them."""
        if flat_rule.get_attr("t") != 1 or cls._enum_int(geom_target) != 1:
            return False
        try:
            return bool(flat_rule.rule.settings().lineSettings().mergeLines())
        except (AttributeError, RuntimeError):
            return False

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

    def _copied_fields(self, layer: QgsVectorLayer) -> Optional[Tuple[str, ...]]:
        """Source fields the Included Fields setting copies into the output;
        None = all of them."""
        if self.included_fields == _FIELDS_ALL:
            return None
        if self.included_fields == _FIELDS_VISIBLE:
            return self._visible_fields(layer)
        return ()

    @staticmethod
    def _visible_fields(layer: QgsVectorLayer) -> Tuple[str, ...]:
        """Fields shown in QGIS: not "Hidden" in the attribute form and not a
        hidden column of the attribute table."""
        hidden_columns = {
            column.name for column in layer.attributeTableConfig().columns()
            if column.hidden
        }
        fields = layer.fields()
        return tuple(
            fields.at(i).name() for i in range(fields.count())
            if layer.editorWidgetSetup(i).type() != "Hidden"
            and fields.at(i).name() not in hidden_columns
        )

    def _required_source_fields(
        self, groups: List[_RuleGroupSnapshot]
    ) -> Optional[Tuple[str, ...]]:
        """Names of source fields any of ``groups`` evaluates or copies;
        None = all."""
        if groups[0].copied_fields is None:
            return None  # All source fields are exported.
        names = set(groups[0].copied_fields)
        for grp in groups:
            expressions = [grp.geometry_expression] + [clause[0] for clause in grp.order_by]
            for member in grp.members:
                expressions.append(member.filter_expression)
                expressions.extend(expr for _, expr, _ in member.expression_fields)
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
            # Sources GDAL builds in C++, each one task, while this thread
            # reads the others with QGIS.
            gdal_builds: Dict[str, Future] = {}
            try:
                for lid, src in todo.items():
                    # Cached entries are built under a temporary name.
                    path = cache.temp_path_for(target_paths[lid]) if cached[lid] else target_paths[lid]
                    builds[lid] = _BaseBuild(src, path)
                    plan = self._gdal_base_plan(src) if _GDAL_BASE else None
                    if plan is not None:
                        gdal_builds[lid] = pool.submit(self._build_base_gdal, src, plan, path)
                for lid, build in builds.items():
                    if lid not in gdal_builds:
                        self._read_into_build(build, pool, pending, max_workers)
                self._write_ready_batches(pending, keep=0)
                for lid, fut in gdal_builds.items():
                    build = builds[lid]
                    if self._wait_for(fut):
                        build.finished = True
                    else:
                        # Read it with QGIS instead.
                        self._remove_file(build.path)
                        self._read_into_build(build, pool, pending, max_workers)
                self._write_ready_batches(pending, keep=0)
            except _Cancelled:
                self.feedback.pushInfo("Base-layer build cancelled.")
                for _, fut in pending:
                    fut.cancel()
                for fut in gdal_builds.values():
                    fut.cancel()
                for build in builds.values():
                    if not build.finished:
                        self._close_base_build(build, remove=True)

        # Base layers finished before a cancellation are still worth keeping.
        for lid, build in builds.items():
            if build.finished and cached[lid]:
                target_paths[lid] = cache.commit(build.path, target_paths[lid])
        return target_paths

    def _read_into_build(
        self,
        build: _BaseBuild,
        pool: ThreadPoolExecutor,
        pending: "Deque[Tuple[_BaseBuild, Future]]",
        max_workers: int,
    ) -> None:
        """Read a source with QGIS into its base layer (see _read_source)."""
        try:
            if build.src.needs_serial_read:
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

    def _wait_for(self, fut: Future) -> bool:
        """Wait for a GDAL task, watching for cancellation; whether it
        succeeded."""
        while True:
            done, _ = wait([fut], timeout=0.5)
            self._check_cancel()
            if done:
                break
        try:
            fut.result()
            return True
        except _Cancelled:
            raise
        except Exception:  # noqa: BLE001  (the caller falls back to QGIS)
            return False

    # -------------------------------------------------------------------
    # Phases 1 + 2, GDAL path — base layers built by GDAL in C++
    # -------------------------------------------------------------------
    def _gdal_base_plan(self, src: _SourceSnapshot) -> Optional[dict]:
        """How GDAL reads a source into its base layer, or None if QGIS must
        read it.

        GDAL reads local files whose URI has no options besides the layer
        and a subset filter (a geometry type split falls back to QGIS), in
        the layer's QGIS CRS, when the project sets no particular datum
        transformation for it and GDAL sees the geometry type QGIS does.
        Caller thread: the source is opened here, one at a time, as QGIS
        reads them.
        """
        if src.live is not None or not src.crs_wkt:
            return None
        source_crs = QgsCoordinateReferenceSystem.fromWkt(src.crs_wkt)
        target_crs = QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}")
        if not source_crs.isValid() or self._transform_context.hasTransform(source_crs, target_crs):
            return None
        try:
            opened = self._gdal_open_source(src)
        except RuntimeError:
            return None
        if opened is None:
            return None
        ds, layer, size, subset = opened
        family = _OGR_FAMILY.get(ogr.GT_GetSingle(ogr.GT_GetLinear(ogr.GT_Flatten(layer.GetGeomType()))))
        select = self._gdal_base_columns(layer, src.required_fields)
        if family is None or select is None:
            return None
        if src.family is not None and family != src.family:
            return None
        # OGR SQL over a project source; every name in it is quoted and
        # escaped (_sql_name). The layer's subset filter is its own OGR SQL
        # condition, as QGIS gives it to GDAL (see _gdal_open_source).
        sql = f'SELECT {select} FROM "{_sql_name(layer.GetName())}"'  # nosec B608
        if subset:
            sql += f" WHERE ({subset})"
        return {"dataset": ds, "sql": sql, "family": family, "size": size}

    def _gdal_open_source(self, src: _SourceSnapshot):
        """(dataset, layer, size in bytes, subset filter or "") of a source
        GDAL can read as QGIS does (see _gdal_base_plan), or None. Raises
        RuntimeError if GDAL fails to open it.

        QGIS's OGR provider gives a subset filter to GDAL as an attribute
        filter, which formats without SQL of their own (_OGR_SQL_FILTER_DRIVERS)
        evaluate as OGR SQL: GDAL applies the same condition in a WHERE. A
        filter that is a whole SELECT, or a format with its own SQL
        (GeoPackage, SpatiaLite...), is left to QGIS."""
        if src.provider == "ogr":
            uri, _, subset = src.source_uri.partition("|subset=")
            subset = subset.strip()
            if subset.lower().startswith("select"):
                return None
            path, *options = uri.split("|")
            layer_name, layer_index = None, 0
            for option in options:
                key, _, value = option.partition("=")
                if key == "layername":
                    layer_name = value
                elif key == "layerid" and value.isdigit():
                    layer_index = int(value)
                else:
                    return None
            if not os.path.isfile(path):
                return None
            ds = gdal.OpenEx(path, gdal.OF_VECTOR | gdal.OF_READONLY)
            if ds is None:
                return None
            if subset and ds.GetDriver().ShortName not in _OGR_SQL_FILTER_DRIVERS:
                return None
            layer = ds.GetLayerByName(layer_name) if layer_name else ds.GetLayer(layer_index)
            return None if layer is None else (ds, layer, self._file_size(path), subset)
        # Databases are read by their QGIS provider, which every connection
        # setting and expression of QGIS applies to.
        return None

    @staticmethod
    def _gdal_base_columns(layer, required_fields: Optional[Tuple[str, ...]]) -> Optional[str]:
        """The OGR SQL select list of a base layer: the fields QGIS would
        read (the FID column too, which QGIS shows as a field), orig_id
        numbered by FID and q2vt_biggest set to 1; None if GDAL can't
        select them as QGIS does."""
        defn = layer.GetLayerDefn()
        names = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
        fid_column = layer.GetFIDColumn()
        available = ([fid_column] if fid_column else []) + names
        if any(name in (f"{_FIELD_PREFIX}_orig_id", _BIGGEST_FIELD) for name in available):
            return None
        if required_fields is None:
            wanted = available
        else:
            # QGIS finds a field regardless of case, and skips unknown names.
            by_lower = {}
            for name in available:
                by_lower.setdefault(name.lower(), name)
            wanted = []
            for name in required_fields:
                found = name if name in available else by_lower.get(name.lower())
                if found is not None and found not in wanted:
                    wanted.append(found)
        items = [
            f'CAST(FID AS integer64) AS "{_sql_name(name)}"' if name == fid_column
            else f'"{_sql_name(name)}"'
            for name in wanted
        ]
        items.append(f'CAST(FID AS float) AS "{_FIELD_PREFIX}_orig_id"')
        items.append(f'CAST(1 AS integer) AS "{_BIGGEST_FIELD}"')
        return ", ".join(items)

    def _build_base_gdal(self, src: _SourceSnapshot, plan: dict, path: str) -> None:
        """Worker: build a base layer with GDAL — what _prepare_base_batch
        does, in C++ without the Python interpreter lock.

        A first copy reads the in-extent features (bounding-box test, as
        QGIS's filter rectangle), repairs them (GEOS make-valid, keeping
        the original dimension) and reprojects them; a second explodes
        multi-parts, keeps the layer's geometry type and simplifies each
        part. Parts of multi-part features then get their q2vt_biggest flags
        (see _flag_biggest_parts). Raises if GDAL fails.
        """
        self._check_cancel()
        family = plan["family"]
        geometry_type, kept_types = _OGR_OUTPUT[family]
        stage = self._gdal_stage_path(f"base_{uuid4().hex}", plan["size"])
        callback = self._gdal_progress
        try:
            self._gdal_translate(stage, plan["dataset"], gdal.VectorTranslateOptions(
                format="FlatGeobuf", layerName="base", layerCreationOptions=["SPATIAL_INDEX=NO"],
                SQLStatement=plan["sql"], SQLDialect="OGRSQL",
                spatFilter=list(src.extent) if src.extent is not None else None,
                srcSRS=src.crs_wkt, dstSRS=f"EPSG:{_EPSG_CRS}", makeValid=family != 2,
                geometryType="GEOMETRY", dim="XY", skipFailures=True, callback=callback,
            ))
            self._check_cancel()
            kept = ", ".join(f"'{t}'" for t in kept_types)
            self._gdal_translate(path, stage, gdal.VectorTranslateOptions(
                format="FlatGeobuf", layerName=splitext(basename(path))[0],
                layerCreationOptions=["SPATIAL_INDEX=NO"], explodeCollections=True,
                geometryType=geometry_type, dim="XY", where=f"OGR_GEOMETRY IN ({kept})",
                simplifyTolerance=None if family == 2 else _DATA_SIMPLIFICATION_TOLERANCE,
                skipFailures=True, callback=callback,
            ))
        except BaseException:
            self._remove_file(path)
            raise
        finally:
            self._remove_file(stage)
            plan["dataset"] = None  # Closes the source.
        self._check_cancel()
        if src.needs_biggest:
            self._flag_biggest_parts(path, family)

    def _flag_biggest_parts(self, path: str, family: int) -> None:
        """Clear q2vt_biggest on all but the largest part of each multi-part
        feature (parts share their orig_id and are consecutive).

        Most layers have none; that is checked on the orig_id column alone.
        Otherwise the layer is copied with the flags set, the first of equal
        parts counting as the largest, as in _prepare_base_batch.
        """
        orig_name = f"{_FIELD_PREFIX}_orig_id"
        if not self._has_repeated_values(path, orig_name):
            return
        temp_path = join(self.utils_dir, f"flags_{uuid4().hex}.{_TEMP_LAYER_FORMAT}")
        if self._flag_biggest_parts_sqlite(path, family, temp_path):
            os.replace(temp_path, path)
            return
        src_ds = ogr.Open(path)
        out_ds = None
        try:
            src_layer = src_ds.GetLayer(0)
            defn = src_layer.GetLayerDefn()
            orig_idx = defn.GetFieldIndex(orig_name)
            biggest_idx = defn.GetFieldIndex(_BIGGEST_FIELD)
            out_ds = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(temp_path)
            out_layer = out_ds.CreateLayer(
                src_layer.GetName(), src_layer.GetSpatialRef(), src_layer.GetGeomType(),
                ["SPATIAL_INDEX=NO"],
            )
            for i in range(defn.GetFieldCount()):
                out_layer.CreateField(defn.GetFieldDefn(i))
            out_defn = out_layer.GetLayerDefn()

            def size(feature) -> float:
                geometry = feature.GetGeometryRef()
                if geometry is None:
                    return -1.0
                return geometry.GetArea() if family == 0 else geometry.Length() if family == 1 else 0.0

            def write(group) -> None:
                if len(group) > 1:
                    biggest = max(group, key=size)
                    for feature in group:
                        feature.SetField(biggest_idx, 1 if feature is biggest else 0)
                for feature in group:
                    out = ogr.Feature(out_defn)
                    out.SetFrom(feature)
                    out_layer.CreateFeature(out)

            group, group_id = [], None
            for n, feature in enumerate(src_layer):
                if n % _WRITE_BATCH_SIZE == 0:
                    self._check_cancel()
                feature_id = feature.GetField(orig_idx)
                if group and feature_id != group_id:
                    write(group)
                    group = []
                group_id = feature_id
                group.append(feature)
            write(group)
        except BaseException:
            out_ds = None
            self._remove_file(temp_path)
            raise
        finally:
            src_ds = None
        out_ds = None  # Closes and flushes the file.
        os.replace(temp_path, path)

    def _flag_biggest_parts_sqlite(self, path: str, family: int, temp_path: str) -> bool:
        """_flag_biggest_parts in C++: one SQLite statement ranks each
        feature's parts by area (length for lines) with a window function,
        the first of equal parts first, keeping the layer's order and schema.
        Returns False (nothing written) if GDAL can't run it."""
        sqlite, spatialite = _sqlite_capabilities()
        if not sqlite or (family != 2 and not spatialite):
            return False
        ds = ogr.Open(path)
        if ds is None:
            return False
        layer = ds.GetLayer(0)
        defn = layer.GetLayerDefn()
        fields, columns = [], []
        for i in range(defn.GetFieldCount()):
            field = defn.GetFieldDefn(i)
            name = field.GetName()
            subtype = field.GetSubType()
            fields.append(
                f'<Field name="{xml_escape(name)}" type="{ogr.GetFieldTypeName(field.GetType())}"'
                + (f' subtype="{ogr.GetFieldSubTypeName(subtype)}"' if subtype != ogr.OFSTNone else "")
                + "/>")
            if name != _BIGGEST_FIELD:
                columns.append(f'"{_sql_name(name)}"')
        layer_name = layer.GetName()
        ds = None
        size = {0: "ST_Area(GEOMETRY)", 1: "ST_Length(GEOMETRY)"}.get(family, "0")
        orig = f'"{_FIELD_PREFIX}_orig_id"'
        # SQL over the plugin's own base layer; every name is quoted.
        sql = (
            f"SELECT {', '.join(columns)}, CASE WHEN ROW_NUMBER() OVER (PARTITION BY {orig} "
            f"ORDER BY {size} DESC, ROWID) = 1 THEN 1 ELSE 0 END AS \"{_BIGGEST_FIELD}\", "
            f'GEOMETRY FROM "{_sql_name(layer_name)}" ORDER BY ROWID'  # nosec B608
        )
        vrt_path = f"{temp_path}.vrt"
        try:
            with open(vrt_path, "w", encoding="utf-8") as f:
                f.write(
                    f'<OGRVRTDataSource><OGRVRTLayer name="{xml_escape(layer_name)}">'
                    f"<SrcDataSource>{xml_escape(path)}</SrcDataSource>"
                    f'<SrcSQL dialect="SQLITE">{xml_escape(sql)}</SrcSQL>'
                    f"<LayerSRS>EPSG:{_EPSG_CRS}</LayerSRS>{''.join(fields)}"
                    "</OGRVRTLayer></OGRVRTDataSource>"
                )
            self._ogr2ogr(temp_path, vrt_path, ["-nln", layer_name, "-nlt", _OGR_OUTPUT[family][0]])
            return True
        except (RuntimeError, OSError) as e:
            self._remove_file(temp_path)
            self.feedback.pushDebugInfo(f"Largest parts flagged by Python ({path}): {e}")
            return False
        finally:
            self._remove_file(vrt_path)

    def _ogr2ogr(self, dest: str, src: str, arguments: List[str]) -> None:
        """Write ``src`` (a VRT running SQLite-dialect SQL) to the
        FlatGeobuf ``dest`` with ogr2ogr, in its own process (see
        _gdal_tool); raises RuntimeError if it fails."""
        self._check_cancel()
        _gdal_tool(["ogr2ogr", "-f", "FlatGeobuf", "-lco", "SPATIAL_INDEX=NO", *arguments, dest, src],
                   cancelled=self._is_cancelled)

    @staticmethod
    def _has_repeated_values(path: str, field_name: str) -> bool:
        """Whether a field of a dataset holds any value twice; read as one
        column (Arrow), True if that isn't possible."""
        try:
            import numpy  # pylint: disable=import-outside-toplevel
            ds = gdal.OpenEx(path, gdal.OF_VECTOR)
            layer = ds.GetLayer(0)
            defn = layer.GetLayerDefn()
            layer.SetIgnoredFields([
                defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())
                if defn.GetFieldDefn(i).GetName() != field_name
            ] + ["OGR_GEOMETRY"])
            batches = [batch[field_name] for batch in layer.GetArrowStreamAsNumPy()]
            ds = None
        except Exception:  # noqa: BLE001  (then copy the layer to find out)
            return True
        if not batches:
            return False
        values = numpy.concatenate(batches)
        return len(numpy.unique(values)) != len(values)

    def _gdal_stage_path(self, name: str, input_size: int) -> str:
        """An intermediate FlatGeobuf file: in memory for inputs up to
        _GDAL_MEMORY_INPUT_MAX, else with the temporary files."""
        if input_size <= _GDAL_MEMORY_INPUT_MAX:
            return f"/vsimem/q2vt_{name}.{_TEMP_LAYER_FORMAT}"
        return join(self.utils_dir, f"{name}.{_TEMP_LAYER_FORMAT}")

    def _gdal_progress(self, *_args) -> int:
        """GDAL progress callback: 0 stops GDAL once the run is cancelled."""
        return 0 if self._is_cancelled() else 1

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
        # Memory layers can't be reopened (their data is in the project
        # layer) and are read from their snapshot instead.
        layer = None if src.live else QgsVectorLayer(src.source_uri, src.name, src.provider)
        try:
            if src.live:
                reader, fields, wkb_type = src.live.source, src.live.fields, src.live.wkb_type
                source_crs = src.live.crs_wkt
            elif not layer.isValid():
                build.failed = True
                self.feedback.pushWarning(
                    f"Cannot open source '{src.name}' "
                    f"(provider={src.provider}); skipping."
                )
                return
            else:
                reader, fields, wkb_type = layer, layer.fields(), layer.wkbType()
                # As WKT, not the CRS object: workers must build their own
                # (see _prepare_base_batch).
                source_crs = layer.crs().toWkt(Qgis.CrsWktVariant.Preferred)

            if src.required_fields is None:
                indices = list(range(fields.count()))
            else:
                indices = sorted(
                    {fields.lookupField(name) for name in src.required_fields} - {-1}
                )
            subset = len(indices) != fields.count()

            # Output schema: the read fields + q2vt_orig_id, typed as
            # fieldcalculator's FIELD_TYPE=0 (decimal, length 10, precision 3),
            # + q2vt_biggest.
            out_fields = QgsFields()
            for idx in indices:
                out_fields.append(fields.at(idx))
            orig_name = f"{_FIELD_PREFIX}_orig_id"
            build.orig_idx = out_fields.lookupField(orig_name)
            if build.orig_idx < 0:
                out_fields.append(QgsField(orig_name, QMetaType.Type.Double, "", 10, 3))
                build.orig_idx = out_fields.count() - 1
            build.biggest_idx = out_fields.lookupField(_BIGGEST_FIELD)
            if build.biggest_idx < 0:
                out_fields.append(QgsField(_BIGGEST_FIELD, QMetaType.Type.Int))
                build.biggest_idx = out_fields.count() - 1

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
                build.path, out_fields, QgsWkbTypes.singleType(wkb_type),
                QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}"),
            )
            batch: List[Tuple[List[Any], QgsGeometry]] = []
            for feature in reader.getFeatures(request):
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
            if layer is not None:
                self._dispose_layer(layer)

    def _submit_base_batch(
        self, build, batch, source_crs, out_fields, pool, pending, max_workers
    ) -> None:
        """Hand a batch to the pool, then write whatever batches are ready."""
        pending.append((build, pool.submit(
            self._prepare_base_batch, batch, source_crs, out_fields, build.biggest_idx
        )))
        build.pending += 1
        # Bounds the features held in memory while workers catch up.
        self._write_ready_batches(pending, keep=2 * max_workers)

    def _prepare_base_batch(
        self,
        batch: List[Tuple[List[Any], QgsGeometry]],
        source_crs: str,
        out_fields: QgsFields,
        biggest_idx: int,
    ) -> List[Optional[List[QgsFeature]]]:
        """Worker: fix → reproject → singleparts → simplify one batch.

        Returns, per feature that survives, its output parts, the largest
        one flagged in q2vt_biggest; q2vt_orig_id is filled in by the
        writer, which knows the feature's position.

        The CRS is rebuilt here from WKT, as each worker did when it read the
        CRS from a materialised file: a CRS object from another thread may
        keep this thread's PROJ objects after the thread exits, where a later
        thread can pick them up (heap corruption). Built from WKT it comes
        from QGIS's CRS cache, which frees them when the thread ends.
        """
        self._check_cancel()
        transform = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem.fromWkt(source_crs),
            QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}"),
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
                # Points have nothing to simplify.
                out.setGeometry(
                    part if part.isNull() or part.type() == Qgis.GeometryType.Point
                    else part.simplify(_DATA_SIMPLIFICATION_TOLERANCE)
                )
                out.setAttribute(biggest_idx, 0)
                features.append(out)
            biggest = max(features, key=lambda f: self._part_size(f.geometry()))
            biggest.setAttribute(biggest_idx, 1)
            results.append(features)
        return results

    @staticmethod
    def _part_size(geometry: QgsGeometry) -> float:
        """Area of a polygon, length of a line, 0 for a point, -1 for none;
        the first of equal parts counts as the biggest."""
        if geometry.isNull():
            return -1.0
        geometry_type = geometry.type()
        if geometry_type == Qgis.GeometryType.Polygon:
            return geometry.area()
        if geometry_type == Qgis.GeometryType.Line:
            return geometry.length()
        return 0.0

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
        """Export the rule groups in parallel.

        Datasets GDAL can write (see _gdal_job) are each one GDAL task, run
        in C++ without the Python interpreter lock, so they use every worker
        fully. The rest go through Python passes, one per base layer: each
        feature is read once for all the layer's groups, which share the
        filters and other expressions they have in common. A GDAL task that
        fails is redone by a Python pass.
        """
        outputs: Dict[str, Optional[str]] = {}
        by_layer: Dict[str, List[_RuleGroupSnapshot]] = {}
        for grp in rule_groups:
            src_path = base_layers.get(grp.layer_id)
            if not src_path or not exists(src_path):
                outputs[grp.output_dataset] = None
                continue
            by_layer.setdefault(grp.layer_id, []).append(grp)

        gdal_tasks: List[Tuple[str, _RuleGroupSnapshot, dict]] = []
        pass_tasks: List[Tuple[str, List[_RuleGroupSnapshot]]] = []
        for layer_id, groups in by_layer.items():
            src_path = base_layers[layer_id]
            jobs = self._gdal_jobs(src_path, groups) if _GDAL_EXPORT else {}
            for grp in groups:
                if grp.output_dataset in jobs:
                    gdal_tasks.append((src_path, grp, jobs[grp.output_dataset]))
            pass_tasks.extend(self._pass_tasks(
                src_path, [grp for grp in groups if grp.output_dataset not in jobs]
            ))
        if not gdal_tasks and not pass_tasks:
            return outputs
        self.feedback.pushDebugInfo(
            f"Rule export: {len(gdal_tasks)} datasets by GDAL "
            f"({sum(1 for _, _, job in gdal_tasks if job.get('dialect') == 'SQLITE')} in SQLite), "
            f"{sum(len(groups) for _, groups in pass_tasks)} in {len(pass_tasks)} Python passes."
        )
        # Largest first, so that no big task starts last and holds up the end.
        gdal_tasks.sort(key=lambda task: self._file_size(task[0]), reverse=True)
        pass_tasks.sort(key=lambda task: self._file_size(task[0]) * len(task[1]), reverse=True)

        max_workers = self._compute_pool_size(len(gdal_tasks) + len(pass_tasks))

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rules-export",
            initializer=_lower_thread_priority,
        ) as pool:
            # (kind, base layer path, groups) per running task; for a batch
            # of passes, its passes instead of a path.
            futures: Dict[Future, Tuple[str, str, List[_RuleGroupSnapshot]]] = {}
            # Small passes another process can run, batched once all are known.
            batchable: List[Tuple[str, List[_RuleGroupSnapshot], dict]] = []

            def submit_pass(src_path: str, groups: List[_RuleGroupSnapshot],
                            in_process: bool = True, batch: bool = False) -> None:
                # Each pass's expression context (global and project scopes)
                # is made here, on the caller thread: the project must not be
                # read from the pass threads, where it corrupted the heap.
                context = QgsProject.instance().createExpressionContext()
                # A pass over a large base layer runs in other processes
                # (see _run_pass_processes), split in this many parts.
                process_parts = self._process_parts(src_path, len(pass_tasks)) if in_process else 0
                if any(grp.order_by for grp in groups):
                    process_parts = min(process_parts, 1)  # Parts would each be sorted alone.
                if process_parts:
                    variables = self._portable_variables(context, groups)
                    if variables is not None:
                        fut = pool.submit(self._run_pass_processes, groups, src_path,
                                          variables, process_parts)
                        futures[fut] = ("process", src_path, groups)
                        return
                if batch and in_process and self._process_parts(src_path, 1, batch=True):
                    variables = self._portable_variables(context, groups)
                    if variables is not None:
                        batchable.append((src_path, groups, variables))
                        return
                fut = pool.submit(self._export_rule_pass, groups, src_path, context)
                futures[fut] = ("pass", src_path, groups)

            # Per GDAL-written dataset: its fields holding raw colour values.
            gdal_raw_colors = {grp.output_dataset: job.get("raw_colors") for _, grp, job in gdal_tasks}
            for src_path, grp, job in gdal_tasks:
                futures[pool.submit(self._export_dataset_gdal, grp, src_path, job)] = (
                    "gdal", src_path, [grp])
            for src_path, groups in pass_tasks:
                submit_pass(src_path, groups, batch=True)
            batch_pool = None
            bins = self._pass_batches(batchable)
            if not bins:  # Too few to batch: run here.
                for src_path, groups, _ in batchable:
                    submit_pass(src_path, groups)
            else:
                # One task starts and waits for the batch's processes.
                batch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rules-batch")
                fut = batch_pool.submit(self._run_pass_batches, batchable, bins)
                futures[fut] = ("batch", batchable, [grp for _, groups, _ in batchable for grp in groups])

            while futures:
                done, _ = wait(list(futures), timeout=1.0, return_when=FIRST_COMPLETED)
                if self._is_cancelled():
                    self.feedback.pushInfo("Rule export cancelled.")
                    for fut, (_, _, groups) in futures.items():
                        fut.cancel()
                        for grp in groups:
                            outputs.setdefault(grp.output_dataset, None)
                    if batch_pool is not None:
                        batch_pool.shutdown(wait=False, cancel_futures=True)
                    return outputs
                for fut in done:
                    kind, src_path, groups = futures.pop(fut)
                    try:
                        result = fut.result(timeout=_PER_ALG_TIMEOUT_S)
                    except _Cancelled:
                        continue  # Reported by the cancellation check above.
                    except Exception as e:  # noqa: BLE001
                        if kind == "batch":  # Redo its passes here.
                            self.feedback.pushDebugInfo(f"Pass batch failed, run here: {e}")
                            for batch_src, batch_groups, _ in src_path:
                                submit_pass(batch_src, batch_groups, in_process=False)
                            continue
                        if kind == "gdal":
                            # Redo it the general way.
                            submit_pass(src_path, groups)
                            continue
                        if kind == "process":
                            # Redo it in this process.
                            self.feedback.pushDebugInfo(f"Pass process failed, run here: {e}")
                            submit_pass(src_path, groups, in_process=False)
                            continue
                        for grp in groups:
                            self.feedback.pushWarning(
                                f"A rule in the {grp.label} was skipped: {e}"
                            )
                            outputs[grp.output_dataset] = None
                        continue
                    if kind == "gdal":
                        outputs[groups[0].output_dataset] = result
                        raw = gdal_raw_colors.get(groups[0].output_dataset)
                        if result and raw:
                            self.raw_color_fields[groups[0].output_dataset] = set(raw)
                    elif kind == "batch":
                        done_outputs, failed = result
                        outputs.update(done_outputs)
                        for failed_src, failed_groups in failed:  # Redo them here.
                            submit_pass(failed_src, failed_groups, in_process=False)
                    else:
                        outputs.update(result)
        if batch_pool is not None:
            batch_pool.shutdown(wait=True)
        return outputs

    # -------------------------------------------------------------------
    # Phase 3, Python passes in other processes
    # -------------------------------------------------------------------
    def _process_parts(self, src_path: str, pass_count: int, batch: bool = False) -> int:
        """How many processes a Python pass over ``src_path`` is split over;
        0 to run it in this process: a small pass, where starting QGIS in
        another process costs more than it saves, or no interpreter to
        start. The workers are shared by the ``pass_count`` passes. With
        ``batch``: 1 if the pass may share a process with others (see
        _pass_batches), else 0."""
        if not _PASS_PROCESSES:
            return 0
        from ..processing.subprocess_runner import python_executable  # pylint: disable=import-outside-toplevel
        if python_executable() is None:
            return 0
        if batch:
            return 1
        features = self._feature_count(src_path)
        if features < _PROCESS_MIN_FEATURES:
            return 0
        workers = max(1, int((os.cpu_count() or 1) * self.cpu_percent / 100))
        return max(1, min(workers // max(1, pass_count), -(-features // _PROCESS_PART_FEATURES)))

    def _pass_batches(self, passes: list) -> List[list]:
        """The smaller passes as work for a few processes: per process, its
        units, each a pass or a range of a pass's features ((pass index,
        orig_id range or None)). A pass with more than a unit's share of the
        work (features times datasets) is split by q2vt_orig_id, unless its
        features are drawn in a set order; units go largest first to the
        least loaded process. [] when too few to be worth starting QGIS for.
        """
        if len(passes) < _BATCH_MIN_PASSES:
            return []
        features = [self._feature_count(src_path) for src_path, _, _ in passes]
        if sum(features) < _BATCH_MIN_FEATURES:
            return []
        workers = max(1, int((os.cpu_count() or 1) * self.cpu_percent / 100))
        work = [max(1, f) * len(groups) for f, (_, groups, _) in zip(features, passes)]
        share = sum(work) / (workers * _BATCH_UNITS_PER_PROCESS)
        units = []  # (work, pass index, orig_id range or None)
        for index, (src_path, groups, _) in enumerate(passes):
            parts = min(math.ceil(work[index] / share), features[index] // _BATCH_MIN_PART_FEATURES)
            if parts < 2 or any(grp.order_by for grp in groups):
                units.append((work[index], index, None))
                continue
            low, high = self._orig_id_range(src_path)
            step = (high - low) / parts
            bounds = [low + i * step for i in range(parts)] + [high + 1]
            units += [(work[index] / parts, index, (bounds[i], bounds[i + 1])) for i in range(parts)]
        bins: List[list] = [[] for _ in range(min(workers, len(units)))]
        loads = [0.0] * len(bins)
        for unit_work, index, fid_range in sorted(units, key=lambda u: -u[0]):
            i = loads.index(min(loads))
            bins[i].append((index, fid_range))
            loads[i] += unit_work
        return bins

    def _run_pass_batches(self, passes: list, bins: List[list]) -> Tuple[Dict[str, Optional[str]], list]:
        """Worker: the passes run by pass_worker.py processes, one per bin
        of units (see _pass_batches), each running its units one after
        another; a split pass's parts are then joined in order. Returns the
        outputs of the passes that completed and the (base layer, groups)
        of those that didn't (a unit failed), which the caller runs itself."""
        from ..processing.subprocess_runner import python_executable  # pylint: disable=import-outside-toplevel
        self._check_cancel()
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tag = uuid4().hex
        cancel_file = join(self.utils_dir, f"cancel_{tag}")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        env["PYTHONIOENCODING"] = "utf-8"
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000 | 0x00004000  # No window, below normal.
        split = {index for units in bins for index, fid_range in units if fid_range is not None}
        # Per pass: its units' (orig_id range start, result file).
        unit_results: Dict[int, List[Tuple[float, str]]] = {}
        processes, files = [], [cancel_file]
        try:
            for b, units in enumerate(bins):
                job_units = []
                for u, (index, fid_range) in enumerate(units):
                    src_path, groups, _ = passes[index]
                    result_path = join(self.utils_dir, f"pass_{tag}_{b}_{u}.json")
                    unit_results.setdefault(index, []).append(
                        (fid_range[0] if fid_range else 0.0, result_path))
                    job_units.append({
                        "base_path": src_path,
                        "groups": [group_to_dict(grp) for grp in groups],
                        "fid_range": list(fid_range) if fid_range else None,
                        # A split pass's parts are joined into its datasets.
                        "suffix": f"_part{tag}_{b}_{u}" if index in split else "",
                        "result_path": result_path,
                    })
                    files.append(result_path)
                job = {
                    "plugins_dir": os.path.dirname(package_dir),
                    "package": os.path.basename(package_dir),
                    "prefix_path": QgsApplication.prefixPath(),
                    "profile_dir": QgsApplication.qgisSettingsDirPath(),
                    "extent": [self.extent.xMinimum(), self.extent.yMinimum(),
                               self.extent.xMaximum(), self.extent.yMaximum()],
                    "utils_dir": self.utils_dir,
                    # The variables are the project's, the same for every pass.
                    "variables": passes[0][2],
                    "cancel_file": cancel_file,
                    "passes": job_units,
                }
                job_path = join(self.utils_dir, f"pass_{tag}_{b}.job.json")
                files += [job_path, f"{job_path}.log"]
                with open(job_path, "w", encoding="utf-8") as f:
                    json.dump(job, f)
                with open(f"{job_path}.log", "w", encoding="utf-8") as log:
                    processes.append(subprocess.Popen(
                        [python_executable(), "-u", _PASS_WORKER, job_path],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
                        env=env, cwd=self.utils_dir, **kwargs,
                    ))
            while any(process.poll() is None for process in processes):
                if self._is_cancelled():
                    open(cancel_file, "w").close()
                    for process in processes:
                        try:
                            process.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    raise _Cancelled()
                time.sleep(0.25)
        except OSError:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            unit_results = {}
        outputs: Dict[str, Optional[str]] = {}
        failed = []
        for index, (src_path, groups, _) in enumerate(passes):
            results, complete = [], index in unit_results
            for _, result_path in sorted(unit_results.get(index, ())):
                try:
                    with open(result_path, encoding="utf-8") as f:
                        results.append(json.load(f))
                except (OSError, ValueError):
                    complete = False
            if not complete:
                # Redone by the caller: its parts that were written go.
                for result in results:
                    for path in result["outputs"].values():
                        if path:
                            self._remove_file(path)
                failed.append((src_path, groups))
                continue
            for result in results:
                for warning in result["warnings"]:
                    self.feedback.pushWarning(warning)
            for grp in groups:
                paths = [r["outputs"].get(grp.output_dataset) for r in results]
                paths = [path for path in paths if path]
                outputs[grp.output_dataset] = (
                    self._join_parts(grp.output_dataset, paths) if index in split
                    else (paths[0] if paths else None))
        if failed:
            self.feedback.pushDebugInfo(
                f"{len(failed)} of {len(passes)} batched passes failed, run here.")
        for path in files:
            self._remove_file(path)
        return outputs, failed

    @staticmethod
    def _feature_count(path: str) -> int:
        ds = ogr.Open(path)
        try:
            return max(0, ds.GetLayer(0).GetFeatureCount()) if ds is not None else 0
        finally:
            ds = None

    @staticmethod
    def _portable_variables(context: QgsExpressionContext, groups) -> Optional[dict]:
        """The context's variables as JSON, for a pass in another process;
        None if an expression of the groups reads one that isn't plain data
        (e.g. @layers), which only this process can give it."""
        variables, others = {}, set()
        for name in context.variableNames():
            value = context.variable(name)
            try:
                json.dumps(value)
                variables[name] = value
            except (TypeError, ValueError):
                others.add(name)
        for grp in groups:
            for expression in group_expressions(grp):
                if others & set(QgsExpression(expression).referencedVariables()):
                    return None
        return variables

    def _run_pass_processes(
        self, groups: List[_RuleGroupSnapshot], src_path: str, variables: dict, parts: int,
    ) -> Dict[str, Optional[str]]:
        """Worker: a Python pass run by pass_worker.py processes, each over
        one range of the base layer's features (by q2vt_orig_id, so that
        a feature's parts stay together); the parts' outputs are then
        joined in order. Raises if a process fails (the caller then runs
        the pass here)."""
        from ..processing.subprocess_runner import python_executable  # pylint: disable=import-outside-toplevel
        self._check_cancel()
        low, high = self._orig_id_range(src_path)
        step = (high - low) / parts if high > low else 1.0
        bounds = [low + i * step for i in range(parts)] + [high + 1]
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tag = uuid4().hex
        cancel_file = join(self.utils_dir, f"cancel_{tag}")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        env["PYTHONIOENCODING"] = "utf-8"
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000 | 0x00004000  # No window, below normal.
        processes = []
        try:
            for i in range(parts):
                job = {
                    "plugins_dir": os.path.dirname(package_dir),
                    "package": os.path.basename(package_dir),
                    "prefix_path": QgsApplication.prefixPath(),
                    "profile_dir": QgsApplication.qgisSettingsDirPath(),
                    "base_path": src_path,
                    "groups": [group_to_dict(grp) for grp in groups],
                    "extent": [self.extent.xMinimum(), self.extent.yMinimum(),
                               self.extent.xMaximum(), self.extent.yMaximum()],
                    "utils_dir": self.utils_dir,
                    "fid_range": [bounds[i], bounds[i + 1]],
                    # One part writes the datasets themselves (a dataset's
                    # layer is named as its file: tiles_generator).
                    "suffix": f"_part{tag}_{i}" if parts > 1 else "",
                    "variables": variables,
                    "cancel_file": cancel_file,
                    "result_path": join(self.utils_dir, f"pass_{tag}_{i}.json"),
                }
                job_path = join(self.utils_dir, f"pass_{tag}_{i}.job.json")
                with open(job_path, "w", encoding="utf-8") as f:
                    json.dump(job, f)
                # Its error output goes to a file (a pipe nobody reads while
                # waiting could fill and stall it).
                with open(f"{job_path}.log", "w", encoding="utf-8") as log:
                    processes.append((job, subprocess.Popen(
                        [python_executable(), "-u", _PASS_WORKER, job_path],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
                        env=env, cwd=self.utils_dir, **kwargs,
                    )))
            results = []
            for job, process in processes:
                while process.poll() is None:
                    if self._is_cancelled():
                        open(cancel_file, "w").close()
                        process.wait(timeout=60)
                        raise _Cancelled()
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode != 0 or not exists(job["result_path"]):
                    log_path = job["result_path"].replace(".json", ".job.json.log")
                    with open(log_path, encoding="utf-8", errors="replace") as log:
                        raise RuntimeError(f"exit code {process.returncode}: {log.read()[-500:]}")
                with open(job["result_path"], encoding="utf-8") as f:
                    results.append(json.load(f))
        except BaseException:
            for _, process in processes:
                if process.poll() is None:
                    process.kill()
            self._remove_parts(processes)
            raise
        for result in results:
            for warning in result["warnings"]:
                self.feedback.pushWarning(warning)
        outputs = {}
        for grp in groups:
            paths = [r["outputs"].get(grp.output_dataset) for r in results]
            outputs[grp.output_dataset] = self._join_parts(
                grp.output_dataset, [p for p in paths if p])
        self._remove_parts(processes, succeeded=True)
        return outputs

    @staticmethod
    def _orig_id_range(path: str) -> Tuple[float, float]:
        """The lowest and highest q2vt_orig_id of a base layer."""
        ds = ogr.Open(path)
        try:
            layer_name = ds.GetLayer(0).GetName()
            orig = f'"{_FIELD_PREFIX}_orig_id"'
            # OGR SQL over the plugin's own base layer; the name is quoted.
            result = ds.ExecuteSQL(
                f'SELECT MIN({orig}), MAX({orig}) FROM "{_sql_name(layer_name)}"')  # nosec B608
            feature = result.GetNextFeature()
            low, high = feature.GetFieldAsDouble(0), feature.GetFieldAsDouble(1)
            ds.ReleaseResultSet(result)
            return low, high
        finally:
            ds = None

    def _join_parts(self, name: str, parts: List[str]) -> Optional[str]:
        """One dataset's part files (in order) as its output file."""
        output_path = join(self.utils_dir, f"{name}.{_TEMP_RULE_FORMAT}")
        if not parts:
            return None
        if parts == [output_path]:
            return output_path  # Written under the dataset's own name.
        vrt_path = join(self.utils_dir, f"{name}.parts.vrt")
        layers = ""
        for i, part in enumerate(parts):
            ds = ogr.Open(part)
            layer_name = ds.GetLayer(0).GetName()
            ds = None
            layers += (f'<OGRVRTLayer name="p{i}"><SrcDataSource>{xml_escape(part)}</SrcDataSource>'
                       f"<SrcLayer>{xml_escape(layer_name)}</SrcLayer></OGRVRTLayer>")
        try:
            with open(vrt_path, "w", encoding="utf-8") as f:
                f.write(f'<OGRVRTDataSource><OGRVRTUnionLayer name="{xml_escape(name)}">'
                        f"<FieldStrategy>FirstLayer</FieldStrategy>{layers}"
                        "</OGRVRTUnionLayer></OGRVRTDataSource>")
            self._gdal_translate(output_path, vrt_path, gdal.VectorTranslateOptions(
                format="FlatGeobuf", layerName=name, layerCreationOptions=["SPATIAL_INDEX=NO"],
            ))
        finally:
            self._remove_file(vrt_path)
            for p in parts:
                self._remove_file(p)
        return output_path

    def _remove_parts(self, processes, succeeded: bool = False) -> None:
        """A failed or finished multi-process pass's job, result and part
        files (a finished single part's files are its datasets: kept)."""
        for job, _ in processes:
            self._remove_file(job["result_path"])
            self._remove_file(job["result_path"].replace(".json", ".job.json"))
            self._remove_file(job["result_path"].replace(".json", ".job.json.log"))
            if succeeded and not job["suffix"]:
                continue
            for grp in job["groups"]:
                self._remove_file(join(
                    self.utils_dir, f"{grp['output_dataset']}{job['suffix']}.{_TEMP_RULE_FORMAT}"))

    @staticmethod
    def _pass_tasks(
        src_path: str, groups: List[_RuleGroupSnapshot]
    ) -> List[Tuple[str, List[_RuleGroupSnapshot]]]:
        """Python passes over one base layer, at most _MAX_GROUPS_PER_PASS
        datasets each."""
        return [
            (src_path, groups[i:i + _MAX_GROUPS_PER_PASS])
            for i in range(0, len(groups), _MAX_GROUPS_PER_PASS)
        ]

    # -------------------------------------------------------------------
    # Phase 3, GDAL path — datasets written by GDAL in C++
    # -------------------------------------------------------------------
    def _gdal_jobs(self, src_path: str, groups: List[_RuleGroupSnapshot]) -> Dict[str, dict]:
        """The GDAL jobs of the datasets GDAL can write, by dataset name.

        Caller thread: parses expressions and reads the base layer's schema.
        """
        layer = QgsVectorLayer(src_path, "base", "ogr")
        try:
            if not layer.isValid():
                return {}
            src_fields = layer.fields()
        finally:
            self._dispose_layer(layer)
        ds = ogr.Open(src_path)
        if ds is None or ds.GetLayerCount() != 1:
            return {}
        layer_name = ds.GetLayer(0).GetName()
        defn = ds.GetLayer(0).GetLayerDefn()
        # Each source field's OGR type and subtype, declared for copies.
        ogr_types = {}
        for i in range(defn.GetFieldCount()):
            field = defn.GetFieldDefn(i)
            subtype = field.GetSubType()
            ogr_types[field.GetName()] = (
                ogr.GetFieldTypeName(field.GetType()),
                ogr.GetFieldSubTypeName(subtype) if subtype != ogr.OFSTNone else None,
            )
        ds = None
        sqlite, spatialite = _sqlite_capabilities()
        jobs = {}
        for grp in groups:
            job = None
            attempts = []
            if sqlite and len(grp.members) > 1:
                # One scan for all the rules of a categorized layer.
                attempts.append(lambda g=grp: self._sqlite_job(
                    g, src_fields, layer_name, ogr_types, spatialite, single_scan=True))
            attempts.append(lambda g=grp: self._gdal_job(g, src_fields, layer_name))
            if sqlite:
                attempts.append(lambda g=grp: self._sqlite_job(
                    g, src_fields, layer_name, ogr_types, spatialite, single_scan=False))
            reason = ""
            for attempt in attempts:
                try:
                    job = attempt()
                except (_NotCompilable, RuntimeError, ValueError, TypeError, AttributeError) as e:
                    job, reason = None, str(e)
                if job is not None:
                    break
            if job is not None:
                jobs[grp.output_dataset] = job
            else:
                self.feedback.pushDebugInfo(
                    f"The {grp.label} ({grp.output_dataset}) is exported by Python: {reason[:200]}")
        return jobs

    def _sqlite_job(
        self, grp: _RuleGroupSnapshot, src_fields: QgsFields, layer_name: str,
        ogr_types: Dict[str, Tuple[str, Optional[str]]], spatialite: bool, single_scan: bool,
    ) -> Optional[dict]:
        """A dataset's GDAL job in GDAL's SQLite dialect (_SqliteCompiler),
        for filters, fields and geometries (SpatiaLite) that OGR SQL lacks;
        None if ``single_scan`` is asked and the rules' filters don't pick
        disjoint values of one field.

        Rules that pick disjoint values of one field (the categories of a
        categorized renderer) are written by one statement, one scan of the
        base layer, each rule's values chosen by CASE; others by one
        statement per rule. The declared fields keep the dataset's schema.
        """
        geometry = grp.geometry_sql
        if geometry is None or (geometry != "GEOMETRY" and not spatialite):
            raise _NotCompilable(grp.geometry_expression)
        output = _OGR_OUTPUT.get(abs(self._enum_int(grp.geometry_target) - 2))
        if output is None:
            raise _NotCompilable(grp.output_dataset)
        compiler = _SqliteCompiler(src_fields)
        columns, expressions = self._dataset_fields(grp, src_fields)
        declared = []
        for name, field_type in columns:
            ogr_type = _VRT_FIELD_TYPES.get(field_type) or ogr_types.get(name)
            if ogr_type is None:
                raise _NotCompilable(name)
            declared.append((name, *ogr_type))
        conditions = [
            compiler.condition(m.filter_expression) if m.filter_expression else None
            for m in grp.members
        ]
        # Colour fields hold their raw colour value (the style maps each to
        # its colour, see _finish_raw_colors), as the hex conversion has no
        # SQL equivalent.
        raw_colors = set()

        def value_of(expression, name: str, field_type: int) -> str:
            if not isinstance(expression, str) or not expression.strip():
                return "NULL"
            raw = COLOR_RAW_EXPRESSIONS.get(expression)
            if raw is not None:
                raw_colors.add(name)
                return compiler.value(raw, field_type, src_fields)
            return compiler.value(expression, field_type, src_fields)

        values = [
            [value_of(member_expressions.get(name), name, field_type) for name, field_type in columns]
            for member_expressions in expressions
        ]
        extra = []
        if grp.keep_biggest_part and src_fields.lookupField(_BIGGEST_FIELD) >= 0:
            extra.append(f'"{_BIGGEST_FIELD}" = 1')
        source = f'"{_sql_name(layer_name)}"'
        geometry_column = f"{geometry} AS GEOMETRY"
        # The layer's feature rendering order, ties in source order.
        order_keys = [(compiler.order_key(expr), ascending, nulls_first)
                      for expr, ascending, nulls_first in grp.order_by]

        def direction(ascending: bool, nulls_first: bool) -> str:
            return f"{'ASC' if ascending else 'DESC'} NULLS {'FIRST' if nulls_first else 'LAST'}"

        if self._disjoint_lookups(grp, src_fields):
            selected = []
            for c, (name, _) in enumerate(columns):
                column = [member_values[c] for member_values in values]
                if all(v == column[0] for v in column):
                    selected.append(f'{column[0]} AS "{_sql_name(name)}"')
                else:
                    cases = " ".join(f"WHEN {cond} THEN {v}" for cond, v in zip(conditions, column))
                    selected.append(f'(CASE {cases} END) AS "{_sql_name(name)}"')
            where = " AND ".join([f"({' OR '.join(f'({c})' for c in conditions)})"] + extra)
            order = ""
            if order_keys:
                order = " ORDER BY " + ", ".join(
                    f"{key} {direction(asc, nf)}" for key, asc, nf in order_keys) + ", ROWID"
            # SQL over the plugin's own temporary base layer; every name and
            # literal in it is quoted and escaped (_sql_name, _sql_string).
            statements = [f"SELECT {', '.join(selected)}, {geometry_column} FROM {source} WHERE {where}{order}"]  # nosec B608
        elif single_scan:
            return None
        else:
            selects = []
            for member_index, (condition, member_values) in enumerate(zip(conditions, values)):
                selected = ", ".join(
                    f'{v} AS "{_sql_name(name)}"' for v, (name, _) in zip(member_values, columns))
                if order_keys:
                    # Sort columns (the VRT's declared fields leave them out).
                    selected += "".join(f', {key} AS "q2vt_order_{k}"' for k, (key, _, _) in enumerate(order_keys))
                    selected += f', ROWID AS "q2vt_order_fid", {member_index} AS "q2vt_order_member"'
                where = " AND ".join(([f"({condition})"] if condition else []) + extra)
                where = f" WHERE {where}" if where else ""
                selects.append(f"SELECT {selected}, {geometry_column} FROM {source}{where}")  # nosec B608
            # One statement, one VRT layer for ogr2ogr: the rules' rows one
            # after another, or in the layer's rendering order.
            statement = " UNION ALL ".join(selects)
            if order_keys:
                statement += " ORDER BY " + ", ".join(
                    f'"q2vt_order_{k}" {direction(asc, nf)}' for k, (_, asc, nf) in enumerate(order_keys)
                ) + ', "q2vt_order_fid", "q2vt_order_member"'
            statements = [statement]
        return {"statements": statements, "geometry": output, "dialect": "SQLITE", "fields": declared,
                "raw_colors": sorted(raw_colors)}

    @staticmethod
    def _disjoint_lookups(grp: _RuleGroupSnapshot, src_fields: QgsFields) -> bool:
        """Whether a dataset's rules each pick values of one same field
        ("field" = literal / IN), no value picked by two: each feature
        then belongs to one rule at most."""
        if len(grp.members) < 2:
            return False
        field_idx, seen = None, set()
        for member in grp.members:
            if not member.filter_expression:
                return False
            lookup = _FilterMatcher._lookup_of(member.filter_expression, src_fields)
            if lookup is None or lookup[2] is not None:
                return False
            if field_idx is None:
                field_idx = lookup[0]
            values = set(lookup[1])
            if lookup[0] != field_idx or values & seen:
                return False
            seen |= values
        return True

    def _gdal_job(
        self, grp: _RuleGroupSnapshot, src_fields: QgsFields, layer_name: str
    ) -> Optional[dict]:
        """One OGR SQL statement per member, or None if GDAL can't write the
        dataset exactly as the Python pass would.

        That takes the feature's own geometry and, per member, a filter and
        fields that OGR SQL computes as QGIS would (see _OgrSqlCompiler and
        _gdal_column). The result is clipped to the extent, exploded to
        single parts and kept to the output geometry type, as in
        _evaluate_geometry and _parts_of_type.
        """
        if grp.geometry_expression.strip() != "@geometry":
            return None
        if grp.order_by:
            # OGR SQL can't order a shared dataset's rows across its rules.
            raise _NotCompilable("feature rendering order")
        output = _OGR_OUTPUT.get(abs(self._enum_int(grp.geometry_target) - 2))
        if output is None:
            return None
        biggest_idx = src_fields.lookupField(_BIGGEST_FIELD)
        columns, expressions = self._dataset_fields(grp, src_fields)
        compiler = _OgrSqlCompiler(src_fields)
        statements = []
        for member_expressions, member in zip(expressions, grp.members):
            conditions = []
            if member.filter_expression:
                conditions.append(compiler.compile(member.filter_expression))
            if grp.keep_biggest_part and biggest_idx >= 0:
                conditions.append(f'"{_BIGGEST_FIELD}" = 1')
            selected = ", ".join(
                self._gdal_column(name, field_type, member_expressions.get(name), src_fields)
                for name, field_type in columns
            )
            where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            # OGR SQL over the plugin's own temporary base layer; every name
            # and literal in it is quoted and escaped (_sql_name, _sql_string,
            # _sql_number) by _OgrSqlCompiler and _gdal_column.
            sql = f'SELECT {selected} FROM "{_sql_name(layer_name)}"{where}'  # nosec B608
            statements.append(sql)
        return {"statements": statements, "geometry": output}

    @staticmethod
    def _gdal_column(name: str, field_type: int, expr_str: Optional[str],
                     src_fields: QgsFields) -> str:
        """The OGR SQL select item computing one output field as _plan_field
        and _convert_value would; raises _NotCompilable otherwise."""
        alias = f'"{_sql_name(name)}"'
        cast = _OGR_CAST_TYPES.get(field_type)
        if not isinstance(expr_str, str) or not expr_str.strip():
            if cast is None:
                raise _NotCompilable(name)
            return f"CAST(NULL AS {cast}) AS {alias}"
        expr = QgsExpression(expr_str)
        if expr.hasParserError():
            raise _NotCompilable(expr_str)
        if expr.isField():
            idx = src_fields.lookupField(next(iter(expr.referencedColumns())))
            if idx < 0:
                raise _NotCompilable(expr_str)
            src_name = f'"{_sql_name(src_fields.at(idx).name())}"'
            src_type = RulesExporter._enum_int(src_fields.at(idx).type())
            if src_type == field_type:
                return f"{src_name} AS {alias}"
            if (src_type, field_type) in _OGR_SAFE_CASTS:
                return f"CAST({src_name} AS {cast}) AS {alias}"
            raise _NotCompilable(expr_str)
        expr.prepare(QgsExpressionContext())
        root = expr.rootNode()
        if cast is None or root is None or not root.hasCachedStaticValue():
            raise _NotCompilable(expr_str)
        value = QgsField(name, QMetaType.Type(field_type)).convertCompatible(
            root.cachedStaticValue()
        )
        if value is None or (hasattr(value, "isNull") and value.isNull()):
            return f"CAST(NULL AS {cast}) AS {alias}"
        if isinstance(value, bool):
            raise _NotCompilable(expr_str)
        if field_type == 10 and isinstance(value, str):
            return f"CAST({_sql_string(value)} AS character) AS {alias}"
        if field_type != 10 and isinstance(value, (int, float)):
            return f"CAST({_sql_number(value)} AS {cast}) AS {alias}"
        raise _NotCompilable(expr_str)

    def _export_dataset_gdal(self, grp: _RuleGroupSnapshot, src_path: str, job: dict) -> Optional[str]:
        """Worker: write one dataset with GDAL; returns its path, None if it
        has no features. Raises if GDAL fails (the caller then uses the
        Python pass).

        GDAL runs each member's SQL over the base layer (one VRT union
        layer), clipping to the extent; a second copy explodes the clipped
        multi-part results, keeps the output geometry type and makes curves
        linear. The base layer's parts are single, so points, which
        clipping keeps or drops whole, are written in one copy.
        """
        self._check_cancel()
        name = grp.output_dataset
        output_path = join(self.utils_dir, f"{name}.{_TEMP_RULE_FORMAT}")
        if exists(output_path):
            return output_path
        vrt_path = join(self.utils_dir, f"{name}.gdal.vrt")
        dialect = job.get("dialect", "OGRSQL")
        if dialect == "SQLITE":
            # Written by ogr2ogr in its own process (see _gdal_tool): on disk.
            clipped_path = join(self.utils_dir, f"{name}_{uuid4().hex}.{_TEMP_LAYER_FORMAT}")
        else:
            clipped_path = self._gdal_stage_path(f"{name}_{uuid4().hex}", self._file_size(src_path))
        geometry_type, kept_types = job["geometry"]
        # Declared fields (the SQLite dialect's): the dataset's types,
        # whatever the values look like.
        fields = "".join(
            f'<Field name="{xml_escape(field_name)}" type="{ogr_type}"'
            + (f' subtype="{subtype}"' if subtype else "") + "/>"
            for field_name, ogr_type, subtype in job.get("fields", ())
        )
        layers = [
            f'<SrcDataSource>{xml_escape(src_path)}</SrcDataSource>'
            f'<SrcSQL dialect="{dialect}">{xml_escape(sql)}</SrcSQL>{fields}'
            for sql in job["statements"]
        ]
        e = self.extent
        bbox = [e.xMinimum(), e.yMinimum(), e.xMaximum(), e.yMaximum()]
        completed = False
        try:
            with open(vrt_path, "w", encoding="utf-8") as f:
                if dialect == "SQLITE":
                    # One statement, one layer (see _sqlite_job).
                    f.write(f'<OGRVRTDataSource><OGRVRTLayer name="{xml_escape(name)}">{layers[0]}'
                            "</OGRVRTLayer></OGRVRTDataSource>")
                else:
                    members = "".join(f'<OGRVRTLayer name="m{i}">{layer}</OGRVRTLayer>'
                                      for i, layer in enumerate(layers))
                    f.write(
                        f'<OGRVRTDataSource><OGRVRTUnionLayer name="{xml_escape(name)}">'
                        f"<FieldStrategy>FirstLayer</FieldStrategy>{members}"
                        "</OGRVRTUnionLayer></OGRVRTDataSource>"
                    )
            kept = ", ".join(f"'{t}'" for t in kept_types)
            if geometry_type == "POINT" and dialect == "OGRSQL":
                # A point's bounding box is the point: the spatial filter
                # keeps exactly the points clipping would. (An SQLite
                # statement can't take the OGR_GEOMETRY filter: its points
                # go through the two copies below.)
                self._gdal_translate(output_path, vrt_path, gdal.VectorTranslateOptions(
                    format="FlatGeobuf", layerName=name, layerCreationOptions=["SPATIAL_INDEX=NO"],
                    spatFilter=bbox, explodeCollections=True, geometryType=geometry_type,
                    dim="XY", where=f"OGR_GEOMETRY IN ({kept})", skipFailures=True,
                    callback=self._gdal_progress,
                ))
            else:
                if dialect == "SQLITE":
                    box = [repr(v) for v in bbox]
                    self._ogr2ogr(clipped_path, vrt_path, [
                        "-nln", "clipped", "-spat", *box, "-clipdst", *box,
                        "-nlt", "GEOMETRY", "-dim", "XY",
                    ])
                else:
                    self._gdal_translate(clipped_path, vrt_path, gdal.VectorTranslateOptions(
                        format="FlatGeobuf", layerName="clipped",
                        layerCreationOptions=["SPATIAL_INDEX=NO"], spatFilter=bbox, clipDst=bbox,
                        # Any geometry type: clipping makes multi-part geometries.
                        geometryType="GEOMETRY", dim="XY", callback=self._gdal_progress,
                    ))
                self._check_cancel()
                # Clipping can leave parts of another type (a line touching
                # the extent's edge leaves a point); like _parts_of_type,
                # they are skipped (FlatGeobuf has no transactions, so this
                # costs nothing).
                self._gdal_translate(output_path, clipped_path, gdal.VectorTranslateOptions(
                    format="FlatGeobuf", layerName=name, layerCreationOptions=["SPATIAL_INDEX=NO"],
                    explodeCollections=True, geometryType=geometry_type, dim="XY",
                    where=f"OGR_GEOMETRY IN ({kept})", skipFailures=True,
                    callback=self._gdal_progress,
                ))
            ds = ogr.Open(output_path)
            count = ds.GetLayer(0).GetFeatureCount() if ds is not None else -1
            ds = None
            if count < 0:
                raise RuntimeError(f"cannot read {output_path}")
            completed = True
            if count == 0:
                self._remove_file(output_path)
                return None
            return output_path
        finally:
            if not completed:
                self._remove_file(output_path)
            self._remove_file(clipped_path)
            self._remove_file(vrt_path)

    @staticmethod
    def _gdal_translate(dest: str, src, options) -> None:
        """gdal.VectorTranslate; raises RuntimeError on failure, whether GDAL
        reports it by exception or by result."""
        # GDAL's messages (this thread's only) are not shown: a failure
        # raises, with the last one.
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        try:
            ds = gdal.VectorTranslate(dest, src, options=options)
            if ds is None:
                raise RuntimeError(gdal.GetLastErrorMsg() or f"GDAL could not write {dest}")
            ds = None  # Closes and flushes the file.
        finally:
            gdal.PopErrorHandler()

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
        self, groups: List[_RuleGroupSnapshot], source_path: str, context: QgsExpressionContext,
        fid_range: Optional[Tuple[float, float]] = None, suffix: str = "",
    ) -> Dict[str, Optional[str]]:
        """Worker: export rule groups that share a base layer in a single
        streaming pass.

        Per group, equivalent to the chain extractbyexpression →
        refactorfields → [dissolve → keepnbiggestparts] →
        geometrybyexpression → clip to the extent → removenullgeometries →
        multiparttosingleparts, but every feature is read once for all the
        groups and only the final outputs are written to disk.

        With ``fid_range`` (low, high), only the base features whose
        q2vt_orig_id is in [low, high) are read, and the outputs are named
        with ``suffix``: one part of a pass split over several processes.
        """
        self._check_cancel()
        layer = QgsVectorLayer(source_path, "base", "ogr")
        try:
            return self._export_rule_pass_from(groups, layer, context, fid_range, suffix)
        finally:
            self._dispose_layer(layer)

    def _export_rule_pass_from(
        self, groups: List[_RuleGroupSnapshot], layer: QgsVectorLayer, context: QgsExpressionContext,
        fid_range: Optional[Tuple[float, float]] = None, suffix: str = "",
    ) -> Dict[str, Optional[str]]:
        """_export_rule_pass, once the base layer is open; ``context`` is this
        pass's own expression context, made on the caller thread."""
        results: Dict[str, Optional[str]] = {grp.output_dataset: None for grp in groups}
        if not layer.isValid():
            return results
        src_fields = layer.fields()

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
            # One pass at a time parses its expressions and creates its
            # output files: done by several passes at once, this corrupted
            # the heap (Windows 0xC0000374) on projects with thousands of
            # rules. Reading and writing features stays parallel.
            with _PASS_SETUP_LOCK:
                for grp in groups:
                    output_path = join(self.utils_dir, f"{grp.output_dataset}{suffix}.{_TEMP_RULE_FORMAT}")
                    if exists(output_path):
                        results[grp.output_dataset] = output_path
                        continue
                    output = self._open_rule_output(grp, output_path, src_fields, layer.crs(), prepare)
                    if output is not None:
                        outputs.append(output)
                if not outputs:
                    return results

                # Per filter: the (output, member) pairs it selects rows for.
                by_filter: Dict[Optional[str], List[Tuple[_RuleOutput, int]]] = {}
                for output in outputs:
                    for index, member in enumerate(output.grp.members):
                        if output.plans[index] is not None:
                            by_filter.setdefault(member.filter_expression, []).append(
                                (output, index))
                matcher = _FilterMatcher(list(by_filter), src_fields, prepare)
            biggest_idx = src_fields.lookupField(_BIGGEST_FIELD)
            request = QgsFeatureRequest()
            request.setInvalidGeometryCheck(
                QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck
            )
            if fid_range is not None:
                # The provider filters it (OGR SQL), before any feature is built.
                orig = f"{_FIELD_PREFIX}_orig_id"
                request.setFilterExpression(
                    f'"{orig}" >= {fid_range[0]!r} AND "{orig}" < {fid_range[1]!r}')
            order_by = next((grp.order_by for grp in groups if grp.order_by), ())
            if order_by:
                # The layer's feature rendering order (its groups share it).
                request.setExpressionContext(context)
                request.setOrderBy(QgsFeatureRequest.OrderBy([
                    QgsFeatureRequest.OrderByClause(expr, ascending, nulls_first)
                    for expr, ascending, nulls_first in order_by
                ]))
            for n, feature in enumerate(layer.getFeatures(request)):
                if n % _WRITE_BATCH_SIZE == 0:
                    self._check_cancel()
                context.setFeature(feature)
                attributes = feature.attributes()
                passed = matcher.passed(attributes, context)
                if not passed:
                    continue
                is_biggest = biggest_idx < 0 or attributes[biggest_idx] == 1
                selected = [
                    selection for filter_expression in passed
                    for selection in by_filter[filter_expression]
                    if is_biggest or not selection[0].grp.keep_biggest_part
                ]
                if selected:
                    self._export_rule_feature(feature, context, selected)
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
        """Plan one dataset's output fields and geometry and open its writer;
        None if the dataset can't be exported. A member whose filter is
        invalid is left out (its plan is None)."""
        if not self.validate_expression(grp, grp.geometry_expression):
            return None
        # layer.geometryType() returns 0 for point and 2 for polygon but
        # geometrybyexpression codes 0 as polygon and 2 as point, so flip it.
        out_wkb = _RULE_OUTPUT_WKB.get(abs(self._enum_int(grp.geometry_target) - 2))
        if out_wkb is None:
            return None

        output = _RuleOutput(grp, output_path, out_wkb)
        columns, expressions = self._dataset_fields(grp, src_fields)
        fields = [QgsField(name, QMetaType.Type(field_type)) for name, field_type in columns]
        for field in fields:
            output.fields.append(field)
        for member, member_expressions in zip(grp.members, expressions):
            if member.filter_expression and not self.validate_expression(
                    grp, member.filter_expression):
                output.plans.append(None)
                continue
            # Per output field: how to compute its value, cheapest kind first
            # — a constant, a straight copy of a source attribute, or an
            # expression.
            plan = []
            for field in fields:
                try:
                    plan.append(self._plan_field(
                        field, member_expressions.get(field.name()), src_fields, prepare
                    ))
                except RuntimeError as e:
                    output.note(f'field "{field.name()}" (exported as NULL)', str(e))
                    plan.append((_FIELD_CONSTANT, field, None, False))
            output.plans.append(plan)
        if all(plan is None for plan in output.plans):
            return None
        try:
            output.geometry = prepare(grp.geometry_expression)
            output.writer = self._create_writer(output_path, output.fields, out_wkb, crs)
        except RuntimeError as e:
            self.feedback.pushWarning(f"A rule in the {grp.label} was skipped: {e}")
            return None
        return output

    def _export_rule_feature(
        self,
        feature: QgsFeature,
        context: QgsExpressionContext,
        selections: List[Tuple[_RuleOutput, int]],
    ) -> None:
        """Add one base-layer feature to every (output, member) it belongs in."""
        # Per expression object: its result for this feature.
        geometries: Dict[int, Tuple[Optional[List[QgsGeometry]], Optional[Tuple[str, str]]]] = {}
        values: Dict[int, Tuple[Any, Optional[str]]] = {}
        src_values = None
        for output, member_index in selections:
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
            for kind, field, source, convert in output.plans[member_index]:
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

            part_fields = output.part_fields.get(member_index)
            if part_fields is None:
                part_fields = output.part_fields[member_index] = [
                    i for i, (kind, _, source, _) in enumerate(output.plans[member_index])
                    if kind == _FIELD_EXPRESSION and _PART_VARIABLE in source.referencedVariables()
                ]
            for part in parts:
                out = QgsFeature(output.fields)
                if part_fields:
                    attributes = list(attributes)
                    scope = QgsExpressionContextScope()
                    scope.setVariable(_PART_VARIABLE, part)
                    context.appendScope(scope)
                    try:
                        for i in part_fields:
                            _, field, source, convert = output.plans[member_index][i]
                            value = source.evaluate(context)
                            attributes[i] = self._convert_value(field, value) if convert else value
                    except RuntimeError:
                        pass
                    finally:
                        context.popScope()
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

    def _dataset_fields(
        self, grp: _RuleGroupSnapshot, src_fields: QgsFields
    ) -> Tuple[List[Tuple[str, int]], List[Dict[str, str]]]:
        """The dataset's output fields as (name, type), and per member the
        expression computing each field it has (others are NULL).

        Fields: the description, the member number (shared datasets), the
        members' calculated fields, the copied source fields and orig_id.
        The first definition of a name wins.
        """
        columns: List[Tuple[str, int]] = []
        names: Set[str] = set()

        def add(name: str, field_type) -> None:
            if name not in names:
                names.add(name)
                # Untyped data-defined properties are exported as strings.
                columns.append((name, 10 if field_type is None else self._enum_int(field_type)))

        expressions: List[Dict[str, str]] = [{} for _ in grp.members]
        add(f"{_FIELD_PREFIX}_description", 10)
        if grp.tagged:
            add(RULE_FIELD, 2)
        for index, member in enumerate(grp.members):
            member_expressions = expressions[index]
            member_expressions[f"{_FIELD_PREFIX}_description"] = member.description
            if grp.tagged:
                member_expressions[RULE_FIELD] = str(index)
            for field_type, expr, name in member.expression_fields:
                add(name, field_type)
                member_expressions.setdefault(name, expr)
        copied = []
        for f in src_fields:
            name = f.name()
            if 'ogc_fid' in name.lower() or name == _BIGGEST_FIELD:
                continue
            if grp.copied_fields is None or name in grp.copied_fields:
                add(name, f.type())
                copied.append(name)
        orig_id = f"{_FIELD_PREFIX}_orig_id"
        add(orig_id, 6)
        for member_expressions in expressions:
            for name in copied:
                member_expressions.setdefault(name, f'"{name}"')
            member_expressions.setdefault(orig_id, f'"{orig_id}"')
        return columns, expressions

    def _finish_raw_colors(self, outputs: Dict[str, Optional[str]]) -> None:
        """Raw colour fields with more distinct values than the style maps
        (_MAX_COLOR_VALUES) are rewritten as #RRGGBBAA values, each raw
        value decoded once as QGIS decodes it. Caller thread."""
        for name, fields in list(self.raw_color_fields.items()):
            path = outputs.get(name)
            if not path:
                self.raw_color_fields.pop(name)
                continue
            many = {f for f in fields if self._distinct_count(path, f) > _MAX_COLOR_VALUES}
            if not many:
                continue
            try:
                self._rewrite_colors_hex(path, many)
                fields -= many
            except (RuntimeError, OSError, AttributeError, TypeError) as e:
                self.feedback.pushWarning(f"Colours of {name} could not be converted: {e}")
            if not fields:
                self.raw_color_fields.pop(name)

    @staticmethod
    def _distinct_count(path: str, field: str) -> int:
        ds = ogr.Open(path)
        try:
            layer = ds.GetLayer(0)
            # OGR SQL over the plugin's own dataset; every name is quoted.
            result = ds.ExecuteSQL(
                f'SELECT COUNT(DISTINCT "{_sql_name(field)}") FROM "{_sql_name(layer.GetName())}"')  # nosec B608
            count = result.GetNextFeature().GetField(0)
            ds.ReleaseResultSet(result)
            return int(count or 0)
        finally:
            ds = None

    def _rewrite_colors_hex(self, path: str, fields: Set[str]) -> None:
        """Rewrite ``fields`` of a dataset from raw colours to #RRGGBBAA."""
        temp_path = f"{path}.hex.{_TEMP_RULE_FORMAT}"
        src_ds = ogr.Open(path)
        src_layer = src_ds.GetLayer(0)
        defn = src_layer.GetLayerDefn()
        indices = [defn.GetFieldIndex(f) for f in fields]
        out_ds = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(temp_path)
        out_layer = out_ds.CreateLayer(src_layer.GetName(), src_layer.GetSpatialRef(),
                                       src_layer.GetGeomType(), ["SPATIAL_INDEX=NO"])
        for i in range(defn.GetFieldCount()):
            out_layer.CreateField(defn.GetFieldDefn(i))
        out_defn = out_layer.GetLayerDefn()
        decoded: Dict[str, Optional[str]] = {}
        for n, feature in enumerate(src_layer):
            if n % _WRITE_BATCH_SIZE == 0:
                self._check_cancel()
            out = ogr.Feature(out_defn)
            out.SetFrom(feature)
            for i in indices:
                if not feature.IsFieldSetAndNotNull(i):
                    continue
                raw = feature.GetFieldAsString(i)
                if raw not in decoded:
                    decoded[raw] = color_hex(raw)
                if decoded[raw] is None:
                    out.SetFieldNull(i)
                else:
                    out.SetField(i, decoded[raw])
            out_layer.CreateFeature(out)
        src_ds = out_ds = None
        os.replace(temp_path, path)

    def _collect_parts(
        self, rule_groups: List[_RuleGroupSnapshot], outputs: Dict[str, Optional[str]]
    ) -> None:
        """Rejoin the parts of each multi-part feature in symbology datasets.

        Base layers hold single parts (rules pick the largest, geometry
        recipes work per part), so a feature of 30 islands was 30 features
        in the tiles, each with its own copy of the attributes. Rows of one
        feature with the same values (rule, data-defined values...) become
        one multi-part row again, in the dataset's order. Datasets with
        label text (labels are placed per part, or on the largest) keep
        their parts. In C++ (SQLite and SpatiaLite), when GDAL has them:
        one ogr2ogr process per dataset, several at once.

        Layers drawn with the merged features renderer have the rows of one
        rule and style values merged into one, as QGIS merges its features
        per symbol; with the inverted polygon renderer, they become the
        export extent outside them (see _MERGE_MODES).
        """
        sqlite, spatialite = _sqlite_capabilities()
        if not (sqlite and spatialite):
            return
        todo = []
        for grp in rule_groups:
            path = outputs.get(grp.output_dataset)
            if grp.rule_type != 0 or not path:
                continue
            if any(name.startswith(f"{_FIELD_PREFIX}_label")
                   for m in grp.members for _, _, name in m.expression_fields):
                continue
            mode = self._merge_mode(grp)
            todo.append((grp, path, mode))
        if not todo:
            return
        workers = max(1, int((os.cpu_count() or 1) * self.cpu_percent / 100))
        with ThreadPoolExecutor(max_workers=min(workers, len(todo)),
                                thread_name_prefix="rules-parts") as pool:
            for future in [pool.submit(self._collect_parts_of, grp, path, mode)
                           for grp, path, mode in todo]:
                try:
                    future.result()
                except _Cancelled:
                    return

    @staticmethod
    def _merge_mode(grp: _RuleGroupSnapshot) -> str:
        """How a symbology dataset's rows are joined (see _collect_parts):
        "parts", "merge" or "invert". Caller thread (reads the layer)."""
        try:
            kind = grp.flat_rules[0].layer.renderer().type()
        except (AttributeError, IndexError, RuntimeError):
            return "parts"
        polygons = abs(RulesExporter._enum_int(grp.geometry_target) - 2) == 0
        lines = abs(RulesExporter._enum_int(grp.geometry_target) - 2) == 1
        if kind == "mergedFeatureRenderer" and (polygons or lines):
            return "merge"
        if kind == "invertedPolygonRenderer" and polygons:
            return "invert"
        return "parts"

    def _collect_parts_of(self, grp: _RuleGroupSnapshot, path: str, mode: str = "parts") -> None:
        """Rejoin one dataset's parts, or merge or invert its features (see
        _collect_parts). Worker thread."""
        if self._is_cancelled():
            raise _Cancelled()
        if mode == "parts" and not self._has_repeated_values(path, f"{_FIELD_PREFIX}_orig_id"):
            return
        multi = {0: "MULTIPOLYGON", 1: "MULTILINESTRING", 2: "MULTIPOINT"}.get(
            abs(self._enum_int(grp.geometry_target) - 2))
        if multi is None:
            return
        temp_path = join(self.utils_dir, f"{grp.output_dataset}_collected.{_TEMP_RULE_FORMAT}")
        vrt_path = f"{temp_path}.vrt"
        ds = ogr.Open(path)
        layer = ds.GetLayer(0)
        defn = layer.GetLayerDefn()
        columns, fields = [], []
        for i in range(defn.GetFieldCount()):
            field = defn.GetFieldDefn(i)
            columns.append(f'"{_sql_name(field.GetName())}"')
            subtype = field.GetSubType()
            fields.append(
                f'<Field name="{xml_escape(field.GetName())}" type="{ogr.GetFieldTypeName(field.GetType())}"'
                + (f' subtype="{ogr.GetFieldSubTypeName(subtype)}"' if subtype != ogr.OFSTNone else "")
                + "/>")
        layer_name = layer.GetName()
        ds = None
        orig = f'"{_sql_name(f"{_FIELD_PREFIX}_orig_id")}"'
        keys = columns if mode == "parts" else [c for c in columns if c != orig]
        selected = [c if c in keys else f"MIN({c}) AS {c}" for c in columns]
        if mode == "merge":
            geometry = ("ST_LineMerge(ST_Union(GEOMETRY))" if multi == "MULTILINESTRING"
                        else "ST_Union(GEOMETRY)")
        elif mode == "invert":
            e = self.extent
            pad_x, pad_y = e.width() * 0.05, e.height() * 0.05
            box = (f"BuildMbr({e.xMinimum() - pad_x!r}, {e.yMinimum() - pad_y!r}, "
                   f"{e.xMaximum() + pad_x!r}, {e.yMaximum() + pad_y!r}, ST_SRID(ST_Union(GEOMETRY)))")
            geometry = f"ST_Difference({box}, ST_Union(GEOMETRY))"
        else:
            geometry = "ST_Collect(GEOMETRY)"
        # SQL over the plugin's own rule dataset; every name is quoted.
        sql = (
            f"SELECT {', '.join(selected)}, {geometry} AS GEOMETRY "
            f'FROM "{_sql_name(layer_name)}" '
            + (f'GROUP BY {", ".join(keys)} ' if keys else "")
            + "ORDER BY MIN(ROWID)"  # nosec B608
        )
        try:
            with open(vrt_path, "w", encoding="utf-8") as f:
                f.write(
                    f'<OGRVRTDataSource><OGRVRTLayer name="{xml_escape(layer_name)}">'
                    f"<SrcDataSource>{xml_escape(path)}</SrcDataSource>"
                    f'<SrcSQL dialect="SQLITE">{xml_escape(sql)}</SrcSQL>'
                    f"<LayerSRS>EPSG:{_EPSG_CRS}</LayerSRS>{''.join(fields)}"
                    "</OGRVRTLayer></OGRVRTDataSource>"
                )
            self._ogr2ogr(temp_path, vrt_path, ["-nln", layer_name, "-nlt", multi])
            os.replace(temp_path, path)
        except (RuntimeError, OSError) as e:
            self._remove_file(temp_path)
            self.feedback.pushDebugInfo(f"Parts of {grp.output_dataset} kept apart: {e}")
        finally:
            self._remove_file(vrt_path)

    def _merge_label_lines(
        self, rule_groups: List[_RuleGroupSnapshot], outputs: Dict[str, Optional[str]]
    ) -> None:
        """Merge the touching lines of "merge connected lines" label
        datasets, as QGIS merges them before placing their labels: lines
        with the same values (label text, rule, data-defined properties)
        become one line per connected set. Caller thread."""
        orig_name = f"{_FIELD_PREFIX}_orig_id"
        for grp in rule_groups:
            path = outputs.get(grp.output_dataset)
            if not grp.merge_lines or not path or self._is_cancelled():
                continue
            merged_path = join(self.utils_dir, f"{grp.output_dataset}_merged.{_TEMP_RULE_FORMAT}")
            # Read and written with OGR: a QGIS layer would keep the file
            # open (its connection pool), and it is replaced here.
            src_ds = out_ds = None
            try:
                src_ds = ogr.Open(path)
                src_layer = src_ds.GetLayer(0)
                defn = src_layer.GetLayerDefn()
                orig_idx = defn.GetFieldIndex(orig_name)
                count = defn.GetFieldCount()
                groups: Dict[tuple, List[Any]] = {}
                for feature in src_layer:
                    values = [feature.GetField(i) if feature.IsFieldSetAndNotNull(i) else None
                              for i in range(count)]
                    key = tuple(repr(v) for i, v in enumerate(values) if i != orig_idx)
                    entry = groups.setdefault(key, [values, []])
                    if orig_idx >= 0 and values[orig_idx] is not None:
                        current = entry[0][orig_idx]
                        if current is None or values[orig_idx] < current:
                            entry[0][orig_idx] = values[orig_idx]
                    geometry = feature.GetGeometryRef()
                    if geometry is not None:
                        line = QgsGeometry()
                        line.fromWkb(bytes(geometry.ExportToIsoWkb()))
                        entry[1].append(line)
                out_ds = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(merged_path)
                out_layer = out_ds.CreateLayer(
                    src_layer.GetName(), src_layer.GetSpatialRef(), ogr.wkbLineString,
                    ["SPATIAL_INDEX=NO"],
                )
                for i in range(count):
                    out_layer.CreateField(defn.GetFieldDefn(i))
                out_defn = out_layer.GetLayerDefn()
                src_ds = None
                for values, geometries in groups.values():
                    merged = QgsGeometry.unaryUnion(geometries).mergeLines()
                    parts = merged.asGeometryCollection() if merged.isMultipart() else [merged]
                    for part in parts:
                        if part.isNull() or part.type() != Qgis.GeometryType.Line:
                            continue
                        out = ogr.Feature(out_defn)
                        for i, value in enumerate(values):
                            if value is None:
                                out.SetFieldNull(i)
                            else:
                                out.SetField(i, value)
                        out.SetGeometry(ogr.CreateGeometryFromWkb(bytes(part.asWkb())))
                        out_layer.CreateFeature(out)
                out_ds = None  # Closes and flushes the file.
                os.replace(merged_path, path)
            except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as e:
                src_ds = out_ds = None
                self._remove_file(merged_path)
                self.feedback.pushWarning(
                    f"The lines of the {grp.label} could not be merged: {e}"
                )

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
            zoom_scale = flat_rule.map_scale()
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
            min_scale = flat_rule.map_scale()
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
        field_name: str = f"{_FIELD_PREFIX}_label",
        condition: str = "",
    ) -> List[Tuple[int, str, str]]:
        """Add the label text field (NULL where ``condition`` is not met)."""
        if not flat_rule.rule.settings():
            return fields
        label_exp = flat_rule.rule.settings().getLabelExpression().expression()
        if not label_exp:
            return fields
        settings = flat_rule.rule.settings()
        filter_exp = f'"{label_exp}"' if not settings.isExpression else label_exp
        text = self._with_type_case(settings, filter_exp)
        if settings.wrapChar:
            # QGIS breaks the line at (and drops) the wrap character; the
            # web map breaks lines at line breaks.
            text = f"replace(({text}), {QgsExpression.quotedString(settings.wrapChar)}, '\\n')"
            settings.wrapChar = ""
        # Spaces starting or ending a line, which QGIS draws and the web map
        # trims (any space): kept behind a word joiner (no width, no break),
        # past any direction marks there (the web map's right-to-left plugin
        # removes those before trimming).
        lead = QgsExpression.quotedString(f"(?m)^([{_BIDI_MARKS}]*)(?= )")
        trail = QgsExpression.quotedString(f"(?m)(?<= )([{_BIDI_MARKS}]*)$")
        text = (f"regexp_replace(regexp_replace(to_string({text}), {lead}, '\\\\1' || char(8288)), "
                f"{trail}, char(8288) || '\\\\1')")
        position = flat_rule.label_position
        if position is not None:
            # The labels placed at their position (see
            # _get_labeling_transformation), which QGIS aligns there by
            # their own corner (the style converter's text-anchor).
            pinned = f"({position[0]}) IS NOT NULL AND ({position[1]}) IS NOT NULL"
            if condition:
                pinned = f"({condition}) AND {pinned}"
            fields.append((2, f"CASE WHEN {pinned} THEN 1 END", pinned_field(field_name)))
        if condition:
            text = f"CASE WHEN {condition} THEN ({text}) END"
        fields.append((10, text, field_name))
        settings.isExpression = False
        settings.fieldName = field_name
        return fields

    @classmethod
    def _with_type_case(cls, settings: QgsPalLayerSettings, label_exp: str) -> str:
        """The label expression with the text format's type case applied.

        The web map then shows the text as QGIS draws it, and its glyphs
        (made from the exported text) include every letter it shows.
        Title case becomes QGIS's title(); small capitals become capitals;
        "first letter capital" and upper camel case are left as written.
        The format is then set to mixed case.
        """
        text_format = settings.format()
        function = _TYPE_CASE_FUNCTIONS.get(cls._enum_int(text_format.capitalization()))
        if function is None:
            return label_exp
        text_format.setCapitalization(Qgis.Capitalization.MixedCase)
        settings.setFormat(text_format)
        return f"{function}(({label_exp}))"

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
        if settings and target_geom == 2 and not settings.geometryGeneratorEnabled:
            self._perimeter_to_geometry_generator(settings)
        if settings and settings.geometryGeneratorEnabled:
            target_geom = settings.geometryGeneratorType
           
            generator_exp = settings.geometryGenerator
            layer_crs = flat_rule.layer.crs().authid()
            generator_exp = generator_exp.replace('@geometry', f"transform(@geometry, 'EPSG:3857', '{layer_crs}')")
            transform_expr =  f"transform({generator_exp}, '{layer_crs}',  'EPSG:3857')"
            settings.geometryGeneratorEnabled = False
            flat_rule.set_attr("c", target_geom)
        elif target_geom == 2:
            transform_expr = self._polygon_label_point_expression(flat_rule)
            flat_rule.set_attr("c", 0)
            target_geom = 0
        position = self._label_position(settings, flat_rule.layer)
        if position is not None and target_geom == 0:
            # A data-defined position (X, Y in the layer's CRS) where the
            # feature has one, as QGIS places it; else the label's own point.
            flat_rule.label_position = position
            x, y = position
            layer_crs = flat_rule.layer.crs().authid()
            transform_expr = (
                f"if(({x}) IS NULL OR ({y}) IS NULL, {transform_expr}, "
                f"transform(make_point(to_real({x}), to_real({y})), '{layer_crs}', 'EPSG:3857'))")
        return [target_geom, transform_expr]

    @staticmethod
    def _label_position(settings, layer) -> Optional[Tuple[str, str]]:
        """A label's data-defined X and Y position, as expressions; None
        unless both are set."""
        if not settings:
            return None
        props = settings.dataDefinedProperties()
        x = props.property(QgsPalLayerSettings.Property.PositionX)
        y = props.property(QgsPalLayerSettings.Property.PositionY)
        if not (x and y and x.isActive() and y.isActive()):
            return None
        return x.asExpression(), y.asExpression()

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
            placement = flat_rule.get_attr("p")
            if placement and symbol_layer.layerType() == "MarkerLine":
                transform_expr = self._marker_points_expression(source_geom, placement)
                if source_geom == 1 and placement in _MARKER_POINT_SQL:
                    self._geometry_sql[transform_expr] = _MARKER_POINT_SQL[placement]
            elif source_geom != target_geom:
                if target_geom == 0:
                    transform_expr = self._centroid_fill_expression(symbol_layer)
                elif target_geom == 1:
                    transform_expr = "boundary(@geometry)"
        return [target_geom, transform_expr]

    def _sql_for_geometry(self, geometry_expression: str) -> Optional[str]:
        """The SQLite (SpatiaLite) geometry computing what a rule's geometry
        expression computes in QGIS, if the export made it from a known
        recipe; None for others (geometry generators...)."""
        expression = geometry_expression.strip()
        if expression == "@geometry":
            return "GEOMETRY"
        if expression == "boundary(@geometry)":
            return "ST_Boundary(GEOMETRY)"
        return self._geometry_sql.get(geometry_expression)

    @staticmethod
    def _marker_line_base(source_geom: int) -> str:
        """The line a marker line follows: the feature, or a polygon's outline."""
        return "@geometry" if source_geom == 1 else "boundary(@geometry)"

    @classmethod
    def _marker_points_expression(cls, source_geom: int, placement: int) -> str:
        """The points a marker line with this single placement draws at."""
        line = cls._marker_line_base(source_geom)
        if placement == MARKER_FIRST_VERTEX:
            return f"start_point({line})"
        if placement == MARKER_LAST_VERTEX:
            return f"end_point({line})"
        if placement == MARKER_CENTRAL_POINT:
            return f"line_interpolate_point({line}, length({line}) / 2)"
        if placement == MARKER_SEGMENT_CENTERS:
            segment = "make_line(point_n(@line, @element), point_n(@line, @element + 1))"
            return (
                f"with_variable('line', {line}, if(num_points(@line) > 1, collect_geometries("
                f"array_foreach(generate_series(1, num_points(@line) - 1), "
                f"line_interpolate_point({segment}, length({segment}) / 2))), NULL))"
            )
        if placement == MARKER_INNER_VERTICES:
            return (
                f"with_variable('line', {line}, if(num_points(@line) > 2, collect_geometries("
                f"array_foreach(generate_series(2, num_points(@line) - 1), point_n(@line, @element))"
                f"), NULL))"
            )
        return f"nodes_to_points({line}, true)"  # Every vertex.

    def _marker_angle_fields(self, flat_rule: FlattenedRule) -> List[Tuple[int, str, str]]:
        """A rotation field for a marker line drawn as points, when its
        markers follow the line.

        QGIS turns such markers with the line: by the line's direction,
        clockwise from east (line_interpolate_angle's bearing from north,
        less 90°), plus the marker's own angle. The sprite is drawn
        unrotated, so the field holds the whole rotation. Markers on every
        (or every inner) vertex each get their own: the direction where the
        point lies on the line (at a vertex, between its two segments, as
        QGIS turns them), evaluated per point (see _PART_VARIABLE).
        """
        if flat_rule.get_attr("t") != 0 or not flat_rule.rule.symbol():
            return []
        placement = flat_rule.get_attr("p")
        if placement not in (MARKER_FIRST_VERTEX, MARKER_LAST_VERTEX, MARKER_CENTRAL_POINT,
                             MARKER_ALL_VERTICES, MARKER_INNER_VERTICES, MARKER_SEGMENT_CENTERS):
            return []
        symbol_layer = flat_rule.rule.symbol().symbolLayers()[0]
        if symbol_layer.layerType() != "MarkerLine" or not symbol_layer.rotateSymbols():
            return []
        line = self._marker_line_base(flat_rule.get_attr("g"))
        distance = {
            MARKER_FIRST_VERTEX: "0",
            MARKER_LAST_VERTEX: f"length({line})",
            MARKER_CENTRAL_POINT: f"length({line}) / 2",
        }.get(placement, f"line_locate_point({line}, @{_PART_VARIABLE})")
        own_angle = 0.0
        sub_symbol = symbol_layer.subSymbol()
        if sub_symbol is not None and sub_symbol.symbolLayerCount():
            own_angle = sub_symbol.symbolLayer(0).angle()
        expression = f"line_interpolate_angle({line}, {distance}) - 90 + {own_angle!r}"
        return [(6, expression, marker_angle_field(flat_rule.get_attr("s") or 0))]

    @classmethod
    def _perimeter_to_geometry_generator(cls, settings: QgsPalLayerSettings) -> None:
        """Turn a polygon "using perimeter" label into a line label of the
        polygon's boundary, as a geometry generator.

        The generator branch then exports the boundary as lines; the label's
        line settings (above / below / on line, offset, repeat) apply as they
        are. "Using perimeter (curved)" has no line placement of that name,
        so it becomes "Curved".
        """
        placement = cls._enum_int(settings.placement)
        if placement not in _PERIMETER_PLACEMENTS:
            return
        settings.geometryGeneratorEnabled = True
        settings.geometryGenerator = "boundary(@geometry)"
        settings.geometryGeneratorType = Qgis.GeometryType.Line
        settings.placement = _PERIMETER_PLACEMENTS[placement]

    def _polygon_label_point_expression(self, flat_rule: FlattenedRule) -> str:
        """The point a polygon label is placed around, as QGIS picks it.

        "Visible" polygons are clipped to the export extent: the area the
        tiles show. Horizontal / free placements search the visible polygon
        for room, which the pole of inaccessibility approximates; centroid
        placements use the whole or visible polygon per "Centroid of"
        (centroidWhole), which may be data-defined.
        """
        settings = flat_rule.rule.settings()
        visible = self._visible_polygons_expression()
        if self._enum_int(settings.placement) in _POLYGON_INTERIOR_PLACEMENTS:
            # About one pixel at the rule's largest zoom, in metres.
            scale = ZoomLevels.zoom_to_scale(flat_rule.get_attr("i")) or 1
            tolerance = scale * _METRES_PER_PIXEL_AT_SCALE_1
            return self._with_polygons(
                visible, f"pole_of_inaccessibility(@source, {tolerance:.6g})"
            )

        whole_point = self._centroid_expression("@geometry", settings.centroidInside)
        visible_point = self._centroid_expression(visible, settings.centroidInside)
        prop = settings.dataDefinedProperties().property(
            QgsPalLayerSettings.Property.CentroidWhole
        )
        if prop and prop.isActive() and prop.asExpression():
            zoom_scale = flat_rule.map_scale()
            centroid_whole = prop.asExpression().replace("@map_scale", zoom_scale)
            default = "true" if settings.centroidWhole else "false"
            return (
                f"if(coalesce(({centroid_whole}), {default}), "
                f"{whole_point}, {visible_point})"
            )
        if settings.centroidWhole:
            self._geometry_sql[whole_point] = self._with_polygons_sql(
                "GEOMETRY", self._centroid_sql("GEOMETRY", settings.centroidInside))
            return whole_point
        visible_sql = self._visible_polygons_sql()
        self._geometry_sql[visible_point] = self._with_polygons_sql(
            visible_sql, self._centroid_sql(visible_sql, settings.centroidInside))
        return visible_point

    def _centroid_fill_expression(self, symbol_layer) -> str:
        """The marker point of a centroid fill, as QGIS draws it: on the
        polygon clipped to the extent shown (here, the export extent)."""
        visible = self._visible_polygons_expression()
        visible_sql = self._visible_polygons_sql()
        if symbol_layer.layerType() != "CentroidFill":
            expression = self._centroid_expression(visible, True)
            self._geometry_sql[expression] = self._with_polygons_sql(
                visible_sql, self._centroid_sql(visible_sql, True))
            return expression
        point = (
            "point_on_surface(@source)" if symbol_layer.pointOnSurface()
            else "centroid(@source)"
        )
        point_sql = (f"ST_PointOnSurface({visible_sql})" if symbol_layer.pointOnSurface()
                     else f"ST_Centroid({visible_sql})")
        if symbol_layer.clipPoints():
            # Markers outside the polygon are not drawn.
            point = (
                f"with_variable('point', {point}, "
                f"if(intersects(@point, @source), @point, NULL))"
            )
            point_sql = f"(CASE WHEN ST_Intersects({point_sql}, {visible_sql}) THEN {point_sql} END)"
        expression = self._with_polygons(visible, point)
        self._geometry_sql[expression] = self._with_polygons_sql(visible_sql, point_sql)
        return expression

    # The same recipes in GDAL's SQLite dialect (SpatiaLite, i.e. GEOS, as
    # QGIS's own geometry functions), for the export's GDAL path.

    def _visible_polygons_sql(self) -> str:
        e = self.extent
        box = (f"BuildMbr({e.xMinimum()!r}, {e.yMinimum()!r}, {e.xMaximum()!r}, "
               f"{e.yMaximum()!r}, SRID(GEOMETRY))")
        return (f"(CASE WHEN MbrWithin(GEOMETRY, {box}) THEN GEOMETRY "
                f"ELSE ST_Intersection(GEOMETRY, {box}) END)")

    @staticmethod
    def _centroid_sql(polygons: str, force_inside: bool) -> str:
        if not force_inside:
            return f"ST_Centroid({polygons})"
        return (f"(CASE WHEN ST_Intersects(ST_Centroid({polygons}), {polygons}) "
                f"THEN ST_Centroid({polygons}) ELSE ST_PointOnSurface({polygons}) END)")

    @staticmethod
    def _with_polygons_sql(polygons: str, point: str) -> str:
        return f"(CASE WHEN {polygons} IS NULL OR ST_IsEmpty({polygons}) THEN NULL ELSE {point} END)"

    def _visible_polygons_expression(self) -> str:
        """The polygon clipped to the export extent; most lie wholly inside
        it, and only those that cross it pay for the intersection."""
        e = self.extent
        return (
            f"with_variable('box', bounds(@geometry), "
            f"if(x_min(@box) >= {e.xMinimum()!r} and y_min(@box) >= {e.yMinimum()!r} "
            f"and x_max(@box) <= {e.xMaximum()!r} and y_max(@box) <= {e.yMaximum()!r}, "
            f"@geometry, intersection(@geometry, geom_from_wkt('{e.asWktPolygon()}'))))"
        )

    @classmethod
    def _centroid_expression(cls, polygons: str, force_inside: bool) -> str:
        """Centroid of ``polygons``; with ``force_inside``, a point on the
        surface instead when the centroid falls outside it."""
        point = "centroid(@source)"
        if force_inside:
            # The centroid is computed once and reused.
            point = (
                "with_variable('centroid', centroid(@source), "
                "if(intersects(@centroid, @source), @centroid, point_on_surface(@source)))"
            )
        return cls._with_polygons(polygons, point)

    @staticmethod
    def _with_polygons(polygons: str, point: str) -> str:
        """``point`` evaluated with @source set to ``polygons``; NULL where
        nothing is left of them (e.g. outside the extent)."""
        return (
            f"with_variable('source', {polygons}, "
            f"if(is_empty_or_null(@source), NULL, {point}))"
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
        if path.startswith("/vsimem/"):
            if gdal.VSIStatL(path) is not None:
                gdal.Unlink(path)
            return
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