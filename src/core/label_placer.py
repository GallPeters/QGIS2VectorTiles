"""
label_placer.py

Point labels whose balloon callout QGIS moves around their point ("Around
point", or several cartographic positions) placed as QGIS places them.

MapLibre picks such a label's side itself, as it draws, and its balloon is
one image: its wedge can't turn towards the point from the side MapLibre
picks. So QGIS lays these labels out instead, at each half zoom level they
show at (its labeling engine, every visible layer taking part as in QGIS,
the symbols not drawn), and each label's place is written to the tiles: per
band, the number of its place (side and distance from its point, 0 where
QGIS leaves it out) in a field of its dataset. The style then draws each
label on its side, with a balloon pointing at its point from there, and
only the labels QGIS shows, all of them (QGIS has already kept them apart).

The layout starts in other processes as soon as the rules are known (start),
while the export goes on, and each band's layout is kept between exports
(_LayoutCache): an export of an unchanged project lays nothing out again.

Depends on: maplibre_converter.TextPropertyExtractor (which labels).
"""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from os.path import basename, join
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from osgeo import ogr
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QImage, QPainter
from qgis.PyQt.QtXml import QDomDocument
from qgis.core import (
    Qgis, QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsCsException, QgsExpressionContext,
    QgsExpressionContextUtils, QgsFeatureRequest, QgsMapRendererCustomPainterJob, QgsMapSettings,
    QgsPalLayerSettings, QgsProject, QgsProperty, QgsProviderRegistry, QgsReadWriteContext, QgsRectangle,
    QgsVectorLayer,
)

from ..utils.config import _CACHE_DIR_NAME, _CACHE_MAX_AGE_DAYS, _CACHE_MAX_MB, _FIELD_PREFIX
from ..utils.flattened_rule import FlattenedRule
from ..utils.zoom_levels import ZoomLevels

# The label settings' property carrying their QGIS rule (layer id and rule
# key), as JSON, inactive (a polygon label's "outside" flag, which a point
# label never reads): kept through every copy of the settings.
RULE_PROPERTY = QgsPalLayerSettings.Property.PolygonLabelOutside
# The rule key standing for a layer's single labeling (not rule-based).
SINGLE_LABELING = "*"
# EPSG:3857 metres per web map pixel at zoom 0 (512 px tiles).
_M_PER_PX_Z0 = 40075016.68557849 / 512
# Pixels per labeling render (a side), and around it the labels placed in
# it but kept from the next one (theirs there). Nothing is drawn (see
# _render_labels): large, few renders.
_CHUNK_PIXELS = 20000
_CHUNK_MARGIN = 400
# Positions within this many pixels of each other are one place.
_PLACE_ROUNDING = 0.5
# Pixels within which a label's box touches its point's side.
_SIDE_TOLERANCE = 1.0


# Zoom bands laid out per zoom level: from each band's start (where QGIS
# lays the labels out, the most crowded) to the next. MapLibre reads zoom in
# a symbol layer's filter and layout at whole zooms only, so each band is a
# layer of its own (its zoom range).
_BANDS_PER_ZOOM = 2
# How far into its band (zoom levels) a band is laid out.
_BAND_NUDGE = BAND_NUDGE = 0.002

# Bands are laid out in other processes (place_worker.py), several at once,
# when there are at least this many; each process lays out its share.
_PARALLEL_MIN_BANDS = 3
_PLACE_WORKER = join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "processing", "place_worker.py")

# Bands' layouts kept between exports (_LayoutCache): bump when the layout
# (_layout, _render_labels) changes.
_LAYOUT_CACHE_VERSION = "1"
# Providers whose data is in files (their changes seen by the files' times
# and sizes); a project with other vector layers isn't cached.
_FILE_PROVIDERS = {"ogr", "delimitedtext", "spatialite", "gpx"}
# From this many nested with_variable calls a label expression is slow to
# parse (the time doubles with each: 11 took 0.1 s), and QGIS parses it in
# every layout: the export says so (with_variable_depth).
_DEEP_WITH_VARIABLE = 9


def placement_field(number: int, zoom: float) -> str:
    """The dataset field holding a placed label rule's places from a zoom
    band's start."""
    return f"{_FIELD_PREFIX}_lp{number:02d}_{int(round(zoom * 100)):04d}"


def zoom_bands(low: int, high: int, start: float) -> List[float]:
    """The starts of the bands from ``start`` (a rule's first exact zoom,
    within zoom level ``low``) to zoom level ``high``'s last band."""
    bands = []
    for zoom in range(low, high + 1):
        for step in range(_BANDS_PER_ZOOM):
            band = zoom + step / _BANDS_PER_ZOOM
            if band + 1 / _BANDS_PER_ZOOM > start + 1e-6:
                bands.append(round(max(band, start), 4))
    return sorted(set(bands))


def mark_rule(settings: QgsPalLayerSettings, layer_id: str, rule_key: str) -> None:
    """Record a label rule's QGIS rule in its settings (RULE_PROPERTY)."""
    props = settings.dataDefinedProperties()
    props.setProperty(RULE_PROPERTY, QgsProperty.fromValue(
        json.dumps({"layer": layer_id, "rule": rule_key}), False))
    settings.setDataDefinedProperties(props)


def rule_mark(settings: QgsPalLayerSettings) -> Optional[Tuple[str, str]]:
    """(layer id, rule key) recorded by mark_rule, or None."""
    try:
        prop = settings.dataDefinedProperties().property(RULE_PROPERTY)
        if prop is None or prop.isActive() or not prop.staticValue():
            return None
        data = json.loads(prop.staticValue())
        return data["layer"], data["rule"]
    except (AttributeError, RuntimeError, ValueError, TypeError, KeyError):
        return None


class Placement:
    """A QGIS label rule's places: ``places`` numbers its distinct places
    (from 1) as (text-anchor, x, y[, width, height]) (_place): the pixels
    from the point to the corner (or side's middle, or centre) of QGIS's
    label box the anchor names, y down. ``fields`` lists, by dataset, the zoom bands' starts it
    has a placement field of (placement_field)."""

    def __init__(self, number: int):
        self.number = number
        self.places: Dict[tuple, int] = {}
        self.fields: Dict[str, List[float]] = {}

    def code(self, place: tuple) -> int:
        if place not in self.places:
            self.places[place] = len(self.places) + 1
        return self.places[place]

    def by_code(self) -> List[Tuple[int, tuple]]:
        return sorted((code, place) for place, code in self.places.items())


def _wedge_pixels(settings: QgsPalLayerSettings) -> Optional[float]:
    """A label's balloon wedge width in pixels (None in map units)."""
    try:
        callout = settings.callout()
        unit = int(callout.wedgeWidthUnit().value if hasattr(callout.wedgeWidthUnit(), "value")
                   else callout.wedgeWidthUnit())
        per_unit = {0: 96 / 25.4, 2: 1.0, 4: 96 / 72, 5: 96.0}.get(unit)
        return callout.wedgeWidth() * per_unit if per_unit else None
    except (AttributeError, RuntimeError, TypeError):
        return None


def _place(box: Tuple[float, float, float, float], wedge: Optional[float] = None) -> tuple:
    """The place (text-anchor, x, y) of a label box (left, top, right,
    bottom: pixels from its point, y down): the box's side facing the point.
    With a balloon wedge as wide as the box's shorter side (QGIS narrows it
    to the side it is on, spanning it whole), the box's width and height
    too: its balloon is drawn at that size (BalloonShape)."""
    left, top, right, bottom = box
    if left >= -_SIDE_TOLERANCE:
        horizontal, x = "left", left
    elif right <= _SIDE_TOLERANCE:
        horizontal, x = "right", right
    else:
        horizontal, x = "", (left + right) / 2
    if bottom <= _SIDE_TOLERANCE:
        vertical, y = "bottom", bottom
    elif top >= -_SIDE_TOLERANCE:
        vertical, y = "top", top
    else:
        vertical, y = "", (top + bottom) / 2
    anchor = "-".join(part for part in (vertical, horizontal) if part) or "center"
    rounded = lambda v: round(round(v / _PLACE_ROUNDING) * _PLACE_ROUNDING, 2)  # noqa: E731
    width, height = right - left, bottom - top
    if wedge is not None and wedge >= min(width, height) - _SIDE_TOLERANCE:
        return anchor, rounded(x), rounded(y), round(width), round(height)
    return anchor, rounded(x), rounded(y)


def with_variable_depth(expression: str) -> int:
    """How many with_variable calls ``expression`` nests at most (strings,
    quoted names and comments skipped)."""
    deepest, depth, stack = 0, 0, []
    i, n = 0, len(expression or "")
    while i < n:
        ch = expression[i]
        if ch in "'\"":
            end = i + 1
            while end < n and not (expression[end] == ch and expression[end - 1] != "\\"):
                end += 1
            i = end
        elif expression.startswith("--", i):
            end = expression.find("\n", i)
            i = n if end < 0 else end
        elif ch == "(":
            word = re.search(r"([A-Za-z_]\w*)\s*$", expression[max(0, i - 40):i])
            nested = bool(word) and word.group(1).lower() == "with_variable"
            stack.append(nested)
            depth += nested
            deepest = max(deepest, depth)
        elif ch == ")" and stack:
            depth -= stack.pop()
        i += 1
    return deepest


def _to_rows(boxes: Dict[Tuple[str, str], Dict[int, tuple]]) -> list:
    """A band's label boxes as JSON rows: [layer id, provider, fid, box...]."""
    return [[layer_id, provider, fid, *box] for (layer_id, provider), found in boxes.items()
            for fid, box in found.items()]


def _from_rows(rows) -> Dict[Tuple[str, str], Dict[int, tuple]]:
    """A band's label boxes from JSON rows (_to_rows); rows not of that shape
    are left out."""
    found: Dict[Tuple[str, str], Dict[int, tuple]] = defaultdict(dict)
    for row in rows if isinstance(rows, list) else []:
        if (isinstance(row, list) and len(row) == 7 and isinstance(row[0], str) and isinstance(row[1], str)
                and all(isinstance(v, (int, float)) for v in row[2:])):
            found[(row[0], row[1])][int(row[2])] = tuple(float(v) for v in row[3:])
    return found


def _file_stamp(path: str) -> Optional[list]:
    """Times and sizes of a data file (or of the files of a folder, such as a
    File Geodatabase); None if it isn't there."""
    try:
        if os.path.isfile(path):
            stat = os.stat(path)
            return [stat.st_mtime_ns, stat.st_size]
        if os.path.isdir(path):
            stamps = []
            for root, _, files in os.walk(path):
                for name in sorted(files):
                    stat = os.stat(join(root, name))
                    stamps.append([name, stat.st_mtime_ns, stat.st_size])
            return stamps
    except OSError:
        pass
    return None


def _layout_key(project: QgsProject, extent: QgsRectangle) -> Optional[str]:
    """What the bands' layouts depend on, as a digest: every vector layer as
    the project saves it (labels, scales, sources) and its data files' times
    and sizes, which layers are shown, the project file, its variables and
    labeling settings, the extent, QGIS and the installed fonts. None where
    a change can't be seen (data not in files)."""
    from .glyphs_generator import qfontdb  # pylint: disable=import-outside-toplevel  (Qt 5 and 6)
    parts: list = [_LAYOUT_CACHE_VERSION, Qgis.version(),
                   [round(v, 3) for v in (extent.xMinimum(), extent.yMinimum(),
                                          extent.xMaximum(), extent.yMaximum())],
                   project.crs().toWkt(), project.ellipsoid()]
    path = project.fileName()
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            parts.append(hashlib.sha1(f.read(), usedforsecurity=False).hexdigest())
    engine = project.labelingEngineSettings()
    parts.append([str(getattr(engine, name)()) for name in (
        "flags", "maximumLineCandidatesPerCm", "maximumPolygonCandidatesPerCmSquared",
        "placementVersion", "defaultTextRenderFormat") if hasattr(engine, name)])
    scope = QgsExpressionContextUtils.projectScope(project)
    # Plain values only: objects (the transform context) print their address.
    parts.append(sorted([name, scope.variable(name)] for name in scope.variableNames()
                        if isinstance(scope.variable(name), (str, int, float, bool, list, type(None)))))
    parts.append([[node.layerId(), node.isVisible()] for node in project.layerTreeRoot().findLayers()])
    context = QgsReadWriteContext()
    for layer_id, layer in sorted(project.mapLayers().items()):
        if not isinstance(layer, QgsVectorLayer):
            continue
        if layer.providerType() not in _FILE_PROVIDERS:
            return None
        source = QgsProviderRegistry.instance().decodeUri(layer.providerType(), layer.source()).get("path")
        stamp = _file_stamp(source) if source else None
        if stamp is None:
            return None
        document = QDomDocument()
        element = document.createElement("maplayer")
        document.appendChild(element)
        layer.writeLayerXml(element, document, context)
        parts.append([layer_id, stamp, document.toString()])
    parts.append(hashlib.sha1("|".join(sorted(qfontdb.families())).encode("utf-8"),
                              usedforsecurity=False).hexdigest())
    text = json.dumps(parts, default=str)
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


class _LayoutCache:
    """Bands' layouts of an unchanged project (_layout_key), kept between
    exports in the plugin's cache folder; JSON only (read, never run)."""

    def __init__(self, project: QgsProject, extent: QgsRectangle):
        self.path = None
        if _CACHE_MAX_MB <= 0:
            return
        try:
            key = _layout_key(project, extent)
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
            key = None
        if key is None:
            return
        directory = join(tempfile.gettempdir(), _CACHE_DIR_NAME, "placements")
        try:
            os.makedirs(directory, exist_ok=True)
            limit = time.time() - _CACHE_MAX_AGE_DAYS * 86400
            for entry in os.scandir(directory):
                if entry.is_file() and entry.stat().st_mtime < limit:
                    os.remove(entry.path)
        except OSError:
            return
        self.path = join(directory, f"{key}.json")

    def read(self) -> Dict[float, dict]:
        """{band zoom: label boxes} kept for this project."""
        if not self.path:
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            os.utime(self.path)
        except (OSError, ValueError):
            return {}
        result = {}
        for zoom, rows in (data.items() if isinstance(data, dict) else []):
            try:
                result[float(zoom)] = _from_rows(rows)
            except ValueError:
                continue
        return result

    def write(self, layouts: Dict[float, dict]) -> None:
        if not self.path or not layouts:
            return
        known = self.read()
        known.update(layouts)
        temp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(temp, "w", encoding="utf-8") as f:
                json.dump({repr(zoom): _to_rows(boxes) for zoom, boxes in known.items()}, f,
                          separators=(",", ":"))
            os.replace(temp, self.path)
        except (OSError, TypeError, ValueError):
            try:
                os.remove(temp)
            except OSError:
                pass


class LabelPlacer:
    """Places the labels QGIS must place (see the module) and writes their
    places to their datasets."""

    def __init__(self, rules: List[FlattenedRule], datasets: List[QgsVectorLayer], extent: QgsRectangle,
                 max_zoom: int, work_dir: str, feedback=None):
        self.rules = rules
        self.datasets = {basename(layer.source().split("|")[0]).split(".")[0]: layer for layer in datasets}
        self.extent = extent  # EPSG:3857
        self.max_zoom = max_zoom
        self.work_dir = work_dir
        self.feedback = feedback
        self.project = QgsProject.instance()
        # Set by start: the layer points, the cached layouts and the bands
        # being laid out in other processes (_launch).
        self._coordinates: Dict[str, dict] = {}
        self._cache: Optional[_LayoutCache] = None
        self._cached: Dict[float, dict] = {}
        self._launched: Optional[dict] = None
        self._started = False

    @staticmethod
    def placed_rules(rules: List[FlattenedRule]) -> List[Tuple[FlattenedRule, Tuple[str, str]]]:
        """The label rules QGIS places, with their QGIS rule. A live
        (PostGIS) layer's labels are left to MapLibre: its live tiles have
        no places."""
        from .maplibre_converter import TextPropertyExtractor  # pylint: disable=import-outside-toplevel
        found = []
        for rule in rules:
            if rule.get_attr("t") != 1 or not rule.rule.settings():
                continue
            if rule.layer.providerType() == "postgres":  # Its live tiles (live_sql) have no places.
                continue
            settings = rule.rule.settings()
            mark = rule_mark(settings)
            if mark and TextPropertyExtractor.placed_by_qgis(settings, rule.layer.geometryType()):
                found.append((rule, mark))
        return found

    def _targets(self, with_datasets: bool = True) -> list:
        """(rule, QGIS rule, dataset, band zooms) of the rules to place; with
        ``with_datasets``, those whose dataset was written only."""
        targets = []
        for rule, mark in self.placed_rules(self.rules):
            dataset = self.datasets.get(rule.output_dataset)
            if with_datasets and (dataset is None or not os.path.isfile(dataset.source().split("|")[0])):
                continue
            low = rule.get_attr("o")
            high = min(rule.get_attr("i"), max(self.max_zoom, low))
            start = ZoomLevels.fractional_zoom(rule.rule.minimumScale()) if rule.rule.minimumScale() else low
            targets.append((rule, mark, dataset, zoom_bands(low, high, max(start, low))))
        return targets

    def start(self) -> None:
        """Start laying the bands out, in other processes, and return: the
        caller goes on (the rules' export) and then calls place(), which
        takes them. Reads the project's own layers only, so ``rules`` may
        be the flattened rules, before their export. Bands an earlier export
        laid out for the same project (_LayoutCache) are not laid out again."""
        if self._started:
            return
        self._started = True
        targets = self._targets(with_datasets=False)
        if not targets:
            return
        self._warn_slow_expressions(targets)
        self._cache = _LayoutCache(self.project, self.extent)
        self._cached = self._cache.read()
        layers = {mark[0] for _, mark, _, _ in targets}
        self._coordinates = {layer_id: self._points(layer_id) for layer_id in layers}
        missing = sorted({zoom for *_, zooms in targets for zoom in zooms} - set(self._cached))
        self._launched = self._launch(missing, self._coordinates)

    def abort(self) -> None:
        """Stop the bands started (the export failed or was cancelled)."""
        if self._launched:
            launched, self._launched = self._launched, None
            open(launched["cancel_file"], "w").close()
            self._finish(launched, wait=True)

    def place(self, rules: Optional[List[FlattenedRule]] = None,
              datasets: Optional[List[QgsVectorLayer]] = None) -> Dict[Tuple[str, str], Placement]:
        """Lay the labels out (or take the layouts start began) and write
        their places; by QGIS rule. ``rules`` and ``datasets``: the rules as
        exported and their datasets, when known only now."""
        if rules is not None:
            self.rules = rules
        if datasets is not None:
            self.datasets = {basename(layer.source().split("|")[0]).split(".")[0]: layer for layer in datasets}
        self.start()
        try:
            targets = self._targets()
        except BaseException:
            self.abort()
            raise
        if not targets:
            self.abort()
            return {}
        layers = {mark[0] for _, mark, _, _ in targets}
        evaluated = sorted({zoom for *_, zooms in targets for zoom in zooms})
        coordinates = {layer_id: self._coordinates.get(layer_id) or self._points(layer_id) for layer_id in layers}
        points = {layer_id: {fid: _key(*xy) for fid, xy in found.items()}
                  for layer_id, found in coordinates.items()}
        # Per zoom evaluated: {(layer id, provider): {feature id: label box}}.
        # From one zoom on, QGIS places every label as at the zoom before
        # (room enough): the higher zooms then keep that band's places.
        laid_out = dict(self._cached)
        fresh = self._collect()  # Laid out by this export (kept for the next).
        laid_out.update(fresh)
        if self.feedback is not None and self._cached:
            self.feedback.pushDebugInfo(
                f"Label placement: {len(set(evaluated) & set(self._cached))} of {len(evaluated)} zoom bands "
                f"as laid out by an earlier export of the same project.")
        boxes, previous = {}, None
        for zoom in evaluated:
            if self._cancelled():
                raise RuntimeError("Label placement cancelled.")
            if zoom not in laid_out:
                # A hair within the band: at a rule's exact first zoom QGIS's
                # own scale can fall just outside its scale range.
                laid_out[zoom] = fresh[zoom] = self._layout(zoom + _BAND_NUDGE, coordinates)
            boxes[zoom] = laid_out[zoom]
            places = {key: {fid: _place(box) for fid, box in found.items()} for key, found in boxes[zoom].items()}
            if places == previous:
                del boxes[zoom]
                break
            previous = places
        if self._cache is not None:
            self._cache.write(fresh)
        targets = [(rule, mark, dataset, [zoom for zoom in zooms if zoom in boxes])
                   for rule, mark, dataset, zooms in targets]
        targets = [target for target in targets if target[3]]

        placements: Dict[Tuple[str, str], Placement] = {}
        values: Dict[str, Dict[str, Dict[tuple, int]]] = defaultdict(dict)
        for rule, mark, dataset, zooms in targets:
            placement = placements.setdefault(mark, Placement(len(placements) + 1))
            wedge = _wedge_pixels(rule.rule.settings())
            name = rule.output_dataset
            fields = placement.fields.setdefault(name, [])
            for zoom in zooms:
                field = placement_field(placement.number, zoom)
                codes = values[name].setdefault(field, {})
                if zoom not in fields:
                    fields.append(zoom)
                for (layer_id, provider), by_feature in boxes[zoom].items():
                    if layer_id != mark[0] or (mark[1] != SINGLE_LABELING and provider != mark[1]):
                        continue
                    for fid, box in by_feature.items():
                        point = points[layer_id].get(fid)
                        if point is not None:
                            codes[point] = placement.code(_place(box, wedge))
        for name, fields in values.items():
            self._write_fields(self.datasets[name], fields)
        return placements

    # --- QGIS's labeling ---

    def _warn_slow_expressions(self, targets: list) -> None:
        """Say which placed labels' expressions are slow to parse (deeply
        nested with_variable): QGIS parses them in every layout."""
        seen = set()
        for rule, _, _, _ in targets:
            settings = rule.rule.settings()
            if not settings.isExpression or (rule.layer.id(), settings.fieldName) in seen:
                continue
            seen.add((rule.layer.id(), settings.fieldName))
            depth = with_variable_depth(settings.fieldName)
            if depth >= _DEEP_WITH_VARIABLE and self.feedback is not None:
                self.feedback.pushWarning(
                    f'The label expression of "{rule.layer.name()}" nests {depth} with_variable calls: '
                    f"QGIS takes long to read it (twice as long with each one), in every layout of its "
                    f"labels. One with_variable holding a map() of the values (@v['name']) is read at once.")

    def _launch(self, zooms: List[float], coordinates: Dict[str, dict]) -> Optional[dict]:
        """Start place_worker.py processes laying the bands out, several at
        once (each starts QGIS and reads the project: worth it from
        _PARALLEL_MIN_BANDS bands), and return them (for _collect); None
        when they can't be (no project file, a memory layer, no Python to
        start): then laid out here. The bands a serial layout would skip
        (see place) are laid out too."""
        from ..processing.subprocess_runner import python_executable  # pylint: disable=import-outside-toplevel
        path = self.project.fileName()
        if (len(zooms) < _PARALLEL_MIN_BANDS or not path or not os.path.isfile(path)
                or python_executable() is None
                or any(layer.providerType() == "memory" for layer in self.project.mapLayers().values()
                       if isinstance(layer, QgsVectorLayer))):
            return None
        processes_count = max(1, min(len(zooms), (os.cpu_count() or 2) // 2))
        if processes_count < 2:
            return None
        # Bands spread over the processes in turn (the lowest, most crowded,
        # are the slowest).
        shares = [zooms[i::processes_count] for i in range(processes_count)]
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tag = uuid4().hex
        cancel_file = join(self.work_dir, f"place_cancel_{tag}")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        env["PYTHONIOENCODING"] = "utf-8"
        kwargs = {"creationflags": 0x08000000 | 0x00004000} if os.name == "nt" else {}
        points = {layer_id: [[fid, x, y] for fid, (x, y) in found.items()]
                  for layer_id, found in coordinates.items()}
        launched = {"zooms": zooms, "jobs": [], "processes": [], "files": [cancel_file],
                    "cancel_file": cancel_file, "started": time.perf_counter()}
        try:
            for i, share in enumerate(shares):
                job = {
                    "plugins_dir": os.path.dirname(package_dir), "package": os.path.basename(package_dir),
                    "prefix_path": QgsApplication.prefixPath(), "profile_dir": QgsApplication.qgisSettingsDirPath(),
                    "project_path": path, "work_dir": self.work_dir, "cancel_file": cancel_file,
                    "extent": [self.extent.xMinimum(), self.extent.yMinimum(),
                               self.extent.xMaximum(), self.extent.yMaximum()],
                    "zooms": share, "coordinates": points,
                    "result_path": join(self.work_dir, f"place_{tag}_{i}.json"),
                }
                job_path = join(self.work_dir, f"place_{tag}_{i}.job.json")
                launched["files"] += [job_path, job["result_path"], f"{job_path}.log"]
                with open(job_path, "w", encoding="utf-8") as f:
                    json.dump(job, f)
                with open(f"{job_path}.log", "w", encoding="utf-8") as log:
                    launched["processes"].append(subprocess.Popen(
                        [python_executable(), "-u", _PLACE_WORKER, job_path],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
                        env=env, cwd=self.work_dir, **kwargs))
                launched["jobs"].append(job)
        except OSError:
            self._finish(launched, wait=False)
            return None
        return launched

    def _collect(self) -> Dict[float, dict]:
        """The layouts of the bands _launch started, once their processes
        end: {zoom: label boxes}; bands that failed are missing (laid out
        here)."""
        launched, self._launched = self._launched, None
        if not launched:
            return {}
        processes = launched["processes"]
        try:
            waited = time.perf_counter()
            while any(process.poll() is None for process in processes):
                if self._cancelled():
                    open(launched["cancel_file"], "w").close()
                    self._finish(launched, wait=True)
                    raise RuntimeError("Label placement cancelled.")
                time.sleep(0.25)
            result: Dict[float, dict] = {}
            if self.feedback is not None:
                now = time.perf_counter()
                self.feedback.pushDebugInfo(
                    f"Label placement: {len(launched['zooms'])} zoom bands laid out in {len(processes)} "
                    f"processes; {sum(p.returncode != 0 for p in processes)} failed (laid out here); "
                    f"{now - launched['started']:.1f} s, of which {now - waited:.1f} s after the rules' export.")
            for job, process in zip(launched["jobs"], processes):
                if process.returncode != 0 or not os.path.isfile(job["result_path"]):
                    continue  # Its bands are laid out here.
                with open(job["result_path"], encoding="utf-8") as f:
                    data = json.load(f)
                for zoom in job["zooms"]:
                    result[zoom] = _from_rows(data.get(repr(zoom), []))
            return result
        except (OSError, ValueError):
            return {}
        finally:
            self._finish(launched, wait=False)

    @staticmethod
    def _finish(launched: dict, wait: bool) -> None:
        """End the processes _launch started (waiting a while for those told
        to stop, with ``wait``) and remove their files."""
        for process in launched["processes"]:
            if process.poll() is None and wait:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                process.kill()
        for file_path in launched["files"]:
            try:
                os.remove(file_path)
            except OSError:
                pass

    def _layout(self, zoom: float, point_cache: Dict[str, dict]) -> Dict[Tuple[str, str], Dict[int, tuple]]:
        """QGIS's labels of ``layer_ids`` at a zoom over the export extent:
        each one's box, in pixels from its point (left, top, right, bottom,
        y down). Every visible vector layer takes part, its symbols not
        drawn; the extent is laid out in parts, each label kept from the
        part its point is in."""
        resolution = _M_PER_PX_Z0 / 2 ** zoom
        width = math.ceil(self.extent.width() / resolution)
        height = math.ceil(self.extent.height() / resolution)
        columns, rows = max(1, math.ceil(width / _CHUNK_PIXELS)), max(1, math.ceil(height / _CHUNK_PIXELS))
        root = self.project.layerTreeRoot()
        layers = [node.layer() for node in root.findLayers()
                  if node.isVisible() and isinstance(node.layer(), QgsVectorLayer)]
        crs = self.project.crs() if self.project.crs().isValid() else QgsCoordinateReferenceSystem("EPSG:3857")
        to_project = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:3857"), crs, self.project)
        to_3857 = QgsCoordinateTransform(crs, QgsCoordinateReferenceSystem("EPSG:3857"), self.project)
        layer_ids = set(point_cache)
        all_points = [xy for found in point_cache.values() for xy in found.values()]
        result: Dict[Tuple[str, str], Dict[int, tuple]] = defaultdict(dict)
        for row in range(rows):
            for column in range(columns):
                if self._cancelled():
                    raise RuntimeError("Label placement cancelled.")
                # The block, within the export extent on every side.
                core = QgsRectangle(
                    self.extent.xMinimum() + column * _CHUNK_PIXELS * resolution,
                    max(self.extent.yMinimum(), self.extent.yMaximum() - (row + 1) * _CHUNK_PIXELS * resolution),
                    min(self.extent.xMaximum(), self.extent.xMinimum() + (column + 1) * _CHUNK_PIXELS * resolution),
                    self.extent.yMaximum() - row * _CHUNK_PIXELS * resolution)
                if not any(core.contains(x, y) for x, y in all_points):
                    continue  # No label of ours here.
                margin = _CHUNK_MARGIN * resolution
                view = QgsRectangle(core.xMinimum() - margin, core.yMinimum() - margin,
                                    core.xMaximum() + margin, core.yMaximum() + margin)
                labels = self._render_labels(view, resolution, layers, crs, to_project)
                for label in labels:
                    if label.layerID not in layer_ids or label.isUnplaced:
                        continue
                    point = point_cache[label.layerID].get(label.featureId)
                    if point is None or not core.contains(point[0], point[1]):
                        continue
                    try:
                        rect = to_3857.transformBoundingBox(label.labelRect)
                    except QgsCsException:
                        continue
                    result[(label.layerID, label.providerID)][label.featureId] = (
                        (rect.xMinimum() - point[0]) / resolution, (point[1] - rect.yMaximum()) / resolution,
                        (rect.xMaximum() - point[0]) / resolution, (point[1] - rect.yMinimum()) / resolution)
        return result

    def _render_labels(self, view: QgsRectangle, resolution: float, layers, crs, to_project) -> list:
        """QGIS's labels over ``view`` (EPSG:3857) at ``resolution``."""
        size = QSize(max(1, round(view.width() / resolution)), max(1, round(view.height() / resolution)))
        settings = QgsMapSettings()
        settings.setDestinationCrs(crs)
        settings.setOutputSize(size)
        settings.setOutputDpi(96)
        try:
            settings.setExtent(to_project.transformBoundingBox(view))
        except QgsCsException:
            return []
        settings.setLayers(layers)
        settings.setLabelingEngineSettings(self.project.labelingEngineSettings())
        settings.setTransformContext(self.project.transformContext())
        settings.setEllipsoid(self.project.ellipsoid())
        if hasattr(Qgis.MapSettingsFlag, "SkipSymbolRendering"):
            settings.setFlag(Qgis.MapSettingsFlag.SkipSymbolRendering, True)
        settings.setFlag(Qgis.MapSettingsFlag.Antialiasing, False)
        context = QgsExpressionContext()
        context.appendScopes([QgsExpressionContextUtils.globalScope(),
                              QgsExpressionContextUtils.projectScope(self.project),
                              QgsExpressionContextUtils.mapSettingsScope(settings)])
        settings.setExpressionContext(context)
        # Only the labels' places are kept: drawn on a pixel (clipped), the
        # map can be laid out large without an image its size.
        image = QImage(1, 1, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        job = QgsMapRendererCustomPainterJob(settings, painter)
        job.renderSynchronously()
        painter.end()
        results = job.takeLabelingResults()
        return list(results.allLabels()) if results is not None else []

    def _points(self, layer_id: str) -> dict:
        """A layer's points (EPSG:3857 coordinates) by feature id."""
        layer = self.project.mapLayer(layer_id)
        if not isinstance(layer, QgsVectorLayer):
            return {}
        transform = QgsCoordinateTransform(layer.crs(), QgsCoordinateReferenceSystem("EPSG:3857"), self.project)
        request = QgsFeatureRequest().setNoAttributes()
        found = {}
        for feature in layer.getFeatures(request):
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue
            try:
                geometry.transform(transform)
            except QgsCsException:
                continue
            vertex = next(geometry.vertices(), None)
            if vertex is None:
                continue
            found[feature.id()] = (vertex.x(), vertex.y())
        return found

    # --- Datasets ---

    def _write_fields(self, dataset: QgsVectorLayer, fields: Dict[str, Dict[tuple, int]]) -> None:
        """Copy a dataset with ``fields`` added (each feature's value by its
        point, 0 when none), under the same name in a folder of its own, and
        point its layer there."""
        source = dataset.source().split("|")[0]
        name = basename(source).split(".")[0]
        src = ogr.Open(source)
        if src is None:
            return
        src_layer = src.GetLayer(0)
        folder = join(self.work_dir, "placed")
        os.makedirs(folder, exist_ok=True)
        path = join(folder, f"{name}.fgb")
        if os.path.exists(path):
            os.remove(path)
        out = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(path)
        out_layer = out.CreateLayer(src_layer.GetName() or name, src_layer.GetSpatialRef(),
                                    src_layer.GetGeomType(), ["SPATIAL_INDEX=NO"])
        defn = src_layer.GetLayerDefn()
        for i in range(defn.GetFieldCount()):
            out_layer.CreateField(defn.GetFieldDefn(i))
        names = sorted(fields)
        for field in names:
            out_layer.CreateField(ogr.FieldDefn(field, ogr.OFTInteger))
        out_defn = out_layer.GetLayerDefn()
        for feature in src_layer:
            copy = ogr.Feature(out_defn)
            copy.SetFrom(feature)
            geometry = feature.GetGeometryRef()
            key = None
            if geometry is not None and not geometry.IsEmpty():
                first = geometry.GetGeometryRef(0) if geometry.GetGeometryCount() else geometry
                key = _key(first.GetX(), first.GetY())
            for field in names:
                copy.SetField(field, fields[field].get(key, 0) if key else 0)
            out_layer.CreateFeature(copy)
        out = src = None
        dataset.setDataSource(path, dataset.name(), "ogr")

    def _cancelled(self) -> bool:
        return self.feedback is not None and self.feedback.isCanceled()


def _key(x: float, y: float) -> Tuple[float, float]:
    """A point's key (EPSG:3857, to 10 cm)."""
    return round(x, 1), round(y, 1)
