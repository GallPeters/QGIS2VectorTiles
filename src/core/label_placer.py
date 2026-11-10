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

Depends on: maplibre_converter.TextPropertyExtractor (which labels).
"""

import json
import math
import os
from collections import defaultdict
from os.path import basename, join
from typing import Dict, List, Optional, Tuple

from osgeo import ogr
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QImage, QPainter
from qgis.core import (
    Qgis, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsCsException, QgsExpressionContext,
    QgsExpressionContextUtils, QgsFeatureRequest, QgsMapRendererCustomPainterJob, QgsMapSettings,
    QgsPalLayerSettings, QgsProject, QgsProperty, QgsRectangle, QgsVectorLayer,
)

from ..utils.config import _FIELD_PREFIX
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
_BAND_NUDGE = 0.002


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

    def place(self) -> Dict[Tuple[str, str], Placement]:
        """Lay the labels out and write their places; by QGIS rule."""
        targets = []
        for rule, mark in self.placed_rules(self.rules):
            dataset = self.datasets.get(rule.output_dataset)
            if dataset is None or not os.path.isfile(dataset.source().split("|")[0]):
                continue
            low = rule.get_attr("o")
            high = min(rule.get_attr("i"), max(self.max_zoom, low))
            start = ZoomLevels.fractional_zoom(rule.rule.minimumScale()) if rule.rule.minimumScale() else low
            targets.append((rule, mark, dataset, zoom_bands(low, high, max(start, low))))
        if not targets:
            return {}
        layers = {mark[0] for _, mark, _, _ in targets}
        evaluated = sorted({zoom for *_, zooms in targets for zoom in zooms})
        last = math.floor(evaluated[-1])
        self._log(". Computing labels with balloon and wedge placement...")
        coordinates = {layer_id: self._points(layer_id) for layer_id in layers}
        points = {layer_id: {fid: _key(*xy) for fid, xy in found.items()}
                  for layer_id, found in coordinates.items()}
        # Per zoom evaluated: {(layer id, provider): {feature id: label box}}.
        # From one zoom on, QGIS places every label as at the zoom before
        # (room enough): the higher zooms then keep that band's places.
        boxes, previous = {}, None
        for index, zoom in enumerate(evaluated):
            if self._cancelled():
                raise RuntimeError("Label placement cancelled.")
            # A hair within the band: at a rule's exact first zoom QGIS's own
            # scale can fall just outside its scale range.
            boxes[zoom] = self._layout(zoom + _BAND_NUDGE, coordinates)
            places = {key: {fid: _place(box) for fid, box in found.items()} for key, found in boxes[zoom].items()}
            level = math.floor(zoom)
            if places == previous:
                del boxes[zoom]
                self._log(f"... Zoom level {level} finished." if level == last
                          else f"... Zoom levels {level} - {last} finished (placement unchanged).")
                break
            following = evaluated[index + 1] if index + 1 < len(evaluated) else None
            if following is None or math.floor(following) != level:
                self._log(f"... Zoom level {level} finished.")
            previous = places
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

    def _log(self, message: str) -> None:
        if self.feedback is not None:
            self.feedback.pushInfo(message)


def _key(x: float, y: float) -> Tuple[float, float]:
    """A point's key (EPSG:3857, to 10 cm)."""
    return round(x, 1), round(y, 1)
