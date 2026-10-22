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

import os
import threading
import platform
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field as dataclass_field
from os.path import basename, exists, join, splitext
from typing import Any, Deque, Dict, Iterator, List, Optional, Set, Tuple
from uuid import uuid4
from xml.sax.saxutils import escape as xml_escape
from osgeo import gdal, ogr
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
    QgsExpressionNode,
    QgsExpressionNodeBinaryOperator,
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
    QgsWkbTypes,
)

from ..utils.config import (
    _DATA_SIMPLIFICATION_TOLERANCE, _EPSG_CRS, _FIELD_PREFIX, _FIELDS_ALL, _FIELDS_VISIBLE,
)
from ..utils.flattened_rule import (
    FlattenedRule, MARKER_CENTRAL_POINT, MARKER_FIRST_VERTEX, MARKER_INNER_VERTICES,
    MARKER_LAST_VERTEX, marker_angle_field,
)
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

# Base-layer field flagging the largest part of each original feature (1),
# for rules that show only that part.
_BIGGEST_FIELD = f"{_FIELD_PREFIX}_biggest"

# Dataset field holding, in a dataset shared by several rules that differ by
# filter, the number of the rule each row belongs to; styles filter on it.
RULE_FIELD = f"{_FIELD_PREFIX}_rule"

# Most rules one shared dataset holds.
_MAX_MEMBERS = 250

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
            self._merge_label_lines(rule_groups, rule_outputs)
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
        for output_dataset, flat_rules in rules_by_dataset.items():
            primary = flat_rules[0]

            # Compute geometry transformation tuple. Before the expression
            # fields: they replace data-defined properties (e.g. the label's
            # "centroid of") with references to fields the base layer lacks.
            transformation = self._get_geometry_transformation(primary)
            if transformation is None:
                # No geometry transformation → cannot be exported.
                continue
            geom_target, geom_expr = transformation
            if (primary.get_attr("t") == 1 and len(flat_rules) == 1
                    and self._attach_label(primary, geom_target, geom_expr, placed)):
                continue

            # Compute expression fields (data-defined properties, optional
            # label field) — these read from the rule symbol/settings.
            expr_fields = self._create_expression_fields(flat_rules)
            if primary.get_attr("t") == 1:
                expr_fields = self._add_label_expression_field(
                    primary, expr_fields
                )
            expr_fields.extend(self._marker_angle_fields(primary))

            member = _MemberSnapshot(
                filter_expression=primary.rule.filterExpression() or None,
                expression_fields=expr_fields,
                description=primary.get_description(),
                flat_rules=flat_rules,
            )
            rule_type = primary.get_attr("t")
            merge_lines = self._merges_lines(primary, geom_target)
            # Merged lines are made of all the parts of every feature.
            keep_biggest = self._keeps_biggest_part(primary) and not merge_lines
            key = (
                primary.layer.id(), rule_type, self._enum_int(geom_target), geom_expr,
                keep_biggest, primary.get_attr("o"), primary.get_attr("i"), merge_lines,
            )
            # Rules drawn by symbol levels keep their own datasets: QGIS draws
            # them level by level, not feature by feature.
            shareable = rule_type == 1 or not self._uses_symbol_levels(primary.layer)
            group, holder = self._dataset_for(datasets.setdefault(key, []), member, shareable)
            if group is None:
                holder = member
                group = _RuleGroupSnapshot(
                    output_dataset=primary.output_dataset,
                    layer_id=primary.layer.id(),
                    rule_type=rule_type,
                    geometry_target=geom_target,
                    geometry_expression=geom_expr,
                    copied_fields=self._copied_fields(primary.layer),
                    members=[member],
                    keep_biggest_part=keep_biggest,
                    layer_name=primary.layer.name(),
                    shareable=shareable,
                    merge_lines=merge_lines,
                )
                datasets[key].append(group)
                rule_groups.append(group)
            if rule_type == 0:
                for flat_rule in flat_rules:
                    placed[id(flat_rule)] = (group, holder)

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
            )

        return sources, rule_groups

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
        group, member = entry
        if (geom_expr != "@geometry" or group.geometry_expression != "@geometry"
                or self._enum_int(geom_target) != self._enum_int(group.geometry_target)
                or self._merges_lines(label, geom_target)):
            return False
        renderer = member.flat_rules[0]
        if (label.get_attr("o") < renderer.get_attr("o")
                or label.get_attr("i") > renderer.get_attr("i")):
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
        existing = {field_name: expr for _, expr, field_name in member.expression_fields}
        if name in existing or any(
                existing.get(field_name, expr) != expr for _, expr, field_name in new_fields):
            return False
        conditions = []
        if label.label_filter:
            conditions.append(f"({label.label_filter})")
        if self._keeps_biggest_part(label):
            conditions.append(f'"{_BIGGEST_FIELD}" = 1')
        fields = self._create_expression_fields([label])
        fields = self._add_label_expression_field(label, fields, name, " AND ".join(conditions))
        member.expression_fields.extend(f for f in fields if f[2] not in existing)
        member.flat_rules.append(label)
        return True

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
            expressions = [grp.geometry_expression]
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
        (a subset filter or a geometry type split falls back to QGIS), in
        the layer's QGIS CRS, when the project sets no particular datum
        transformation for it. Caller thread: the source is opened here, one
        at a time, as QGIS reads them.
        """
        if src.live is not None or src.provider != "ogr" or not src.crs_wkt:
            return None
        path, *options = src.source_uri.split("|")
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
        source_crs = QgsCoordinateReferenceSystem.fromWkt(src.crs_wkt)
        target_crs = QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}")
        if not source_crs.isValid() or self._transform_context.hasTransform(source_crs, target_crs):
            return None
        try:
            ds = gdal.OpenEx(path, gdal.OF_VECTOR | gdal.OF_READONLY)
        except RuntimeError:
            return None
        if ds is None:
            return None
        layer = ds.GetLayerByName(layer_name) if layer_name else ds.GetLayer(layer_index)
        if layer is None:
            return None
        family = _OGR_FAMILY.get(ogr.GT_GetSingle(ogr.GT_GetLinear(ogr.GT_Flatten(layer.GetGeomType()))))
        select = self._gdal_base_columns(layer, src.required_fields)
        if family is None or select is None:
            return None
        # OGR SQL over a project source; every name in it is quoted and
        # escaped (_sql_name).
        sql = f'SELECT {select} FROM "{_sql_name(layer.GetName())}"'  # nosec B608
        return {"dataset": ds, "sql": sql, "family": family, "size": self._file_size(path)}

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
        # Largest first, so that no big task starts last and holds up the end.
        gdal_tasks.sort(key=lambda task: self._file_size(task[0]), reverse=True)
        pass_tasks.sort(key=lambda task: self._file_size(task[0]) * len(task[1]), reverse=True)

        max_workers = self._compute_pool_size(len(gdal_tasks) + len(pass_tasks))

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rules-export",
            initializer=_lower_thread_priority,
        ) as pool:
            # (kind, base layer path, groups) per running task.
            futures: Dict[Future, Tuple[str, str, List[_RuleGroupSnapshot]]] = {}

            def submit_pass(src_path: str, groups: List[_RuleGroupSnapshot]) -> None:
                # Each pass's expression context (global and project scopes)
                # is made here, on the caller thread: the project must not be
                # read from the pass threads, where it corrupted the heap.
                fut = pool.submit(
                    self._export_rule_pass, groups, src_path,
                    QgsProject.instance().createExpressionContext(),
                )
                futures[fut] = ("pass", src_path, groups)

            for src_path, grp, job in gdal_tasks:
                futures[pool.submit(self._export_dataset_gdal, grp, src_path, job)] = (
                    "gdal", src_path, [grp])
            for src_path, groups in pass_tasks:
                submit_pass(src_path, groups)

            while futures:
                done, _ = wait(list(futures), timeout=1.0, return_when=FIRST_COMPLETED)
                if self._is_cancelled():
                    self.feedback.pushInfo("Rule export cancelled.")
                    for fut, (_, _, groups) in futures.items():
                        fut.cancel()
                        for grp in groups:
                            outputs.setdefault(grp.output_dataset, None)
                    return outputs
                for fut in done:
                    kind, src_path, groups = futures.pop(fut)
                    try:
                        result = fut.result(timeout=_PER_ALG_TIMEOUT_S)
                    except _Cancelled:
                        continue  # Reported by the cancellation check above.
                    except Exception as e:  # noqa: BLE001
                        if kind == "gdal":
                            # Redo it the general way.
                            submit_pass(src_path, groups)
                            continue
                        for grp in groups:
                            self.feedback.pushWarning(
                                f"A rule in the {grp.label} was skipped: {e}"
                            )
                            outputs[grp.output_dataset] = None
                        continue
                    if kind == "gdal":
                        outputs[groups[0].output_dataset] = result
                    else:
                        outputs.update(result)
        return outputs

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
        ds = None
        jobs = {}
        for grp in groups:
            try:
                job = self._gdal_job(grp, src_fields, layer_name)
            except (_NotCompilable, RuntimeError, ValueError, TypeError):
                job = None
            if job is not None:
                jobs[grp.output_dataset] = job
        return jobs

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
        clipped_path = self._gdal_stage_path(f"{name}_{uuid4().hex}", self._file_size(src_path))
        geometry_type, kept_types = job["geometry"]
        layers = "".join(
            f'<OGRVRTLayer name="m{i}"><SrcDataSource>{xml_escape(src_path)}</SrcDataSource>'
            f'<SrcSQL dialect="OGRSQL">{xml_escape(sql)}</SrcSQL></OGRVRTLayer>'
            for i, sql in enumerate(job["statements"])
        )
        e = self.extent
        bbox = [e.xMinimum(), e.yMinimum(), e.xMaximum(), e.yMaximum()]
        completed = False
        try:
            with open(vrt_path, "w", encoding="utf-8") as f:
                f.write(
                    f'<OGRVRTDataSource><OGRVRTUnionLayer name="{xml_escape(name)}">'
                    f"<FieldStrategy>FirstLayer</FieldStrategy>{layers}"
                    "</OGRVRTUnionLayer></OGRVRTDataSource>"
                )
            kept = ", ".join(f"'{t}'" for t in kept_types)
            if geometry_type == "POINT":
                # A point's bounding box is the point: the spatial filter
                # keeps exactly the points clipping would.
                self._gdal_translate(output_path, vrt_path, gdal.VectorTranslateOptions(
                    format="FlatGeobuf", layerName=name, layerCreationOptions=["SPATIAL_INDEX=NO"],
                    spatFilter=bbox, explodeCollections=True, geometryType=geometry_type,
                    dim="XY", where=f"OGR_GEOMETRY IN ({kept})", skipFailures=True,
                    callback=self._gdal_progress,
                ))
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
    def _gdal_translate(dest: str, src: str, options) -> None:
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
        self, groups: List[_RuleGroupSnapshot], source_path: str, context: QgsExpressionContext
    ) -> Dict[str, Optional[str]]:
        """Worker: export rule groups that share a base layer in a single
        streaming pass.

        Per group, equivalent to the chain extractbyexpression →
        refactorfields → [dissolve → keepnbiggestparts] →
        geometrybyexpression → clip to the extent → removenullgeometries →
        multiparttosingleparts, but every feature is read once for all the
        groups and only the final outputs are written to disk.
        """
        self._check_cancel()
        layer = QgsVectorLayer(source_path, "base", "ogr")
        try:
            return self._export_rule_pass_from(groups, layer, context)
        finally:
            self._dispose_layer(layer)

    def _export_rule_pass_from(
        self, groups: List[_RuleGroupSnapshot], layer: QgsVectorLayer, context: QgsExpressionContext
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
                    output_path = join(self.utils_dir, f"{grp.output_dataset}.{_TEMP_RULE_FORMAT}")
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
            placement = flat_rule.get_attr("p")
            if placement and symbol_layer.layerType() == "MarkerLine":
                transform_expr = self._marker_points_expression(source_geom, placement)
            elif source_geom != target_geom:
                if target_geom == 0:
                    transform_expr = self._centroid_fill_expression(symbol_layer)
                elif target_geom == 1:
                    transform_expr = "boundary(@geometry)"
        return [target_geom, transform_expr]

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
        if placement == MARKER_INNER_VERTICES:
            return (
                f"with_variable('line', {line}, if(num_points(@line) > 2, collect_geometries("
                f"array_foreach(generate_series(2, num_points(@line) - 1), point_n(@line, @element))"
                f"), NULL))"
            )
        return f"nodes_to_points({line}, true)"  # Every vertex.

    def _marker_angle_fields(self, flat_rule: FlattenedRule) -> List[Tuple[int, str, str]]:
        """A rotation field for a marker line drawn as points at one vertex
        or the middle, when its markers follow the line.

        QGIS turns such markers with the line: by the line's direction,
        clockwise from east (line_interpolate_angle's bearing from north,
        less 90°), plus the marker's own angle. The sprite is drawn
        unrotated, so the field holds the whole rotation.
        """
        if flat_rule.get_attr("t") != 0 or not flat_rule.rule.symbol():
            return []
        placement = flat_rule.get_attr("p")
        if placement not in (MARKER_FIRST_VERTEX, MARKER_LAST_VERTEX, MARKER_CENTRAL_POINT):
            return []
        symbol_layer = flat_rule.rule.symbol().symbolLayers()[0]
        if symbol_layer.layerType() != "MarkerLine" or not symbol_layer.rotateSymbols():
            return []
        line = self._marker_line_base(flat_rule.get_attr("g"))
        distance = {
            MARKER_FIRST_VERTEX: "0",
            MARKER_LAST_VERTEX: f"length({line})",
            MARKER_CENTRAL_POINT: f"length({line}) / 2",
        }[placement]
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
            zoom_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            centroid_whole = prop.asExpression().replace("@map_scale", zoom_scale)
            default = "true" if settings.centroidWhole else "false"
            return (
                f"if(coalesce(({centroid_whole}), {default}), "
                f"{whole_point}, {visible_point})"
            )
        return whole_point if settings.centroidWhole else visible_point

    def _centroid_fill_expression(self, symbol_layer) -> str:
        """The marker point of a centroid fill, as QGIS draws it: on the
        polygon clipped to the extent shown (here, the export extent)."""
        visible = self._visible_polygons_expression()
        if symbol_layer.layerType() != "CentroidFill":
            return self._centroid_expression(visible, True)
        point = (
            "point_on_surface(@source)" if symbol_layer.pointOnSurface()
            else "centroid(@source)"
        )
        if symbol_layer.clipPoints():
            # Markers outside the polygon are not drawn.
            point = (
                f"with_variable('point', {point}, "
                f"if(intersects(@point, @source), @point, NULL))"
            )
        return self._with_polygons(visible, point)

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