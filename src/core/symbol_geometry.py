"""
symbol_geometry.py

Expression functions computing, on the ground, shapes QGIS draws on screen
from a feature, so the web map draws them exactly as polygons or lines in
the tiles (RulesFlattener writes geometry generators calling them):

    q2vt_arrow(...)    an arrow symbol layer's arrows (QgsArrowSymbolLayer)
    q2vt_offset(...)   a geometry moved by a data-defined x/y offset
    q2vt_callout(...)  a label's callout line to its data-defined position

Lengths come in EPSG:3857 metres (the exported base layers' unit), with
the length of a screen pixel at the rule's zoom where shapes follow the
screen. Registered on import (the rules export and its pass processes
import it, see rules_exporter).
"""

import math
from typing import List, Optional

from qgis.PyQt.QtCore import QPointF, QRectF
from qgis.PyQt.QtGui import QFont, QFontMetricsF, QPainterPath, QPolygonF
from qgis.core import Qgis, QgsExpression, QgsGeometry, QgsPointXY, QgsRectangle, qgsfunction

_GROUP = "QGIS2VectorTiles"

# QgsArrowSymbolLayer's head and arrow types.
_HEAD_SINGLE, _HEAD_REVERSED, _HEAD_DOUBLE = 0, 1, 2
_ARROW_PLAIN, _ARROW_LEFT_HALF, _ARROW_RIGHT_HALF = 0, 1, 2


# --- Arrows: QGIS's own drawing (qgsarrowsymbollayer.cpp), in screen
# pixels (y down), a feature's line given in them ---

def _clamp_angle(a: float) -> float:
    if a > 2 * math.pi:
        return a - 2 * math.pi
    if a < 0.0:
        return a + 2 * math.pi
    return a


def _distance(a: QPointF, b: QPointF) -> float:
    return math.hypot(b.x() - a.x(), b.y() - a.y())


def _points_to_circle(a: QPointF, b: QPointF, c: QPointF):
    """The circle through three points, as (centre, radius); None if they
    are aligned."""
    ab, bc = b - a, c - b
    ab2, bc2 = (a + b) / 2.0, (b + c) / 2.0
    if abs(ab.x() * bc.y() - ab.y() * bc.x()) < 0.001:
        return None
    if ab.y() == 0:
        cx = ab2.x()
        cy = bc2.y() - (cx - bc2.x()) * bc.x() / bc.y()
    elif bc.y() == 0:
        cx = bc2.x()
        cy = ab2.y() - (cx - ab2.x()) * ab.x() / ab.y()
    else:
        cx = ((bc2.y() - ab2.y() + bc.x() * bc2.x() / bc.y() - ab.x() * ab2.x() / ab.y())
              / (bc.x() / bc.y() - ab.x() / ab.y()))
        cy = bc2.y() - (cx - bc2.x()) * bc.x() / bc.y()
    return QPointF(cx, cy), math.hypot(a.x() - cx, a.y() - cy)


def _circle_point(center: QPointF, radius: float, angle: float) -> QPointF:
    return QPointF(math.cos(-angle) * radius + center.x(), math.sin(-angle) * radius + center.y())


def _path_arc_to(path: QPainterPath, center: QPointF, radius: float,
                 angle_o: float, angle_d: float, direction: int) -> None:
    rect = QRectF(center - QPointF(radius, radius), center + QPointF(radius, radius))
    start = angle_o / math.pi * 180.0
    if direction == 1:
        sweep = ((angle_d - angle_o) if angle_o < angle_d
                 else 2 * math.pi - (angle_o - angle_d)) / math.pi * 180.0
    else:
        sweep = (-(2 * math.pi - (angle_d - angle_o)) if angle_o < angle_d
                 else (angle_d - angle_o)) / math.pi * 180.0
    path.arcTo(rect, start, sweep)


def _spiral_arc_to(path: QPainterPath, center: QPointF, start_angle: float, start_radius: float,
                   end_angle: float, end_radius: float, direction: int) -> None:
    a = _circle_point(center, start_radius, start_angle)
    b = _circle_point(center, end_radius, end_angle)
    delta = end_angle - start_angle
    if direction * delta < 0.0:
        delta += direction * 2 * math.pi
    i1 = _circle_point(center, 0.75 * start_radius + 0.25 * end_radius, start_angle + 0.25 * delta)
    i2 = _circle_point(center, 0.50 * start_radius + 0.50 * end_radius, start_angle + 0.50 * delta)
    i3 = _circle_point(center, 0.25 * start_radius + 0.75 * end_radius, start_angle + 0.75 * delta)
    for p, q, r, end in ((a, i1, i2, i2), (i2, i3, b, b)):
        circle = _points_to_circle(p, q, r)
        if circle is None:
            path.lineTo(end)
            continue
        c, radius = circle
        a1 = math.atan2(c.y() - p.y(), p.x() - c.x())
        a2 = math.atan2(c.y() - r.y(), r.x() - c.x())
        _path_arc_to(path, c, radius, a1, a2, direction)


def _straight_arrow(po: QPointF, pd: QPointF, start_width: float, width: float, head_width: float,
                    head_height: float, head_type: int, arrow_type: int, offset: float) -> List[QPointF]:
    length = _distance(po, pd)
    if length == 0:
        return []
    if head_type == _HEAD_SINGLE and length < head_width:
        po = pd - (pd - po) / length * head_width
        length = head_width
    elif head_type == _HEAD_REVERSED and length < head_width:
        pd = po + (pd - po) / length * head_width
        length = head_width
    elif head_type == _HEAD_DOUBLE and length < 2 * head_width:
        v = (pd - po) / length * head_width
        po, pd = (po + pd) / 2.0 - v, (po + pd) / 2.0 + v
        length = 2 * head_width
    body = length - head_width
    unit = (pd - po) / length
    perp = QPointF(-unit.y(), unit.x())
    po, pd = po + perp * offset, pd + perp * offset
    right = arrow_type in (_ARROW_PLAIN, _ARROW_RIGHT_HALF)
    left = arrow_type in (_ARROW_PLAIN, _ARROW_LEFT_HALF)
    polygon = []
    if head_type == _HEAD_DOUBLE:
        polygon.append(po)
        if right:
            polygon += [po + unit * head_width + perp * head_height,
                        po + unit * head_width + perp * (width * 0.5),
                        po + unit * body + perp * (width * 0.5),
                        po + unit * body + perp * head_height]
        polygon.append(pd)
        if left:
            polygon += [po + unit * body - perp * head_height,
                        po + unit * body - perp * (width * 0.5),
                        po + unit * head_width - perp * (width * 0.5),
                        po + unit * head_width - perp * head_height]
    elif head_type == _HEAD_SINGLE:
        if right:
            polygon += [po + perp * (start_width * 0.5),
                        po + unit * body + perp * (width * 0.5),
                        po + unit * body + perp * head_height]
        else:
            polygon.append(po)
        polygon.append(pd)
        if left:
            polygon += [po + unit * body - perp * head_height,
                        po + unit * body - perp * (width * 0.5),
                        po - perp * (start_width * 0.5)]
        else:
            polygon.append(po)
    elif head_type == _HEAD_REVERSED:
        polygon.append(po)
        if right:
            polygon += [po + unit * head_width + perp * head_height,
                        po + unit * head_width + perp * (width * 0.5),
                        pd + perp * (start_width * 0.5)]
        else:
            polygon.append(pd)
        if left:
            polygon += [pd - perp * (start_width * 0.5),
                        po + unit * head_width - perp * (width * 0.5),
                        po + unit * head_width - perp * head_height]
        else:
            polygon.append(pd)
    if polygon:
        polygon.append(polygon[0])
    return polygon


def _curved_arrow(po: QPointF, pm: QPointF, pd: QPointF, start_width: float, width: float,
                  head_width: float, head_height: float, head_type: int, arrow_type: int,
                  offset: float) -> List[QPointF]:
    circle = _points_to_circle(po, pm, pd)
    if circle is None:
        return _straight_arrow(po, pd, start_width, width, head_width, head_height,
                               head_type, arrow_type, offset)
    center, radius = circle
    angle_o = _clamp_angle(math.atan2(center.y() - po.y(), po.x() - center.x()))
    angle_m = _clamp_angle(math.atan2(center.y() - pm.y(), pm.x() - center.x()))
    angle_d = _clamp_angle(math.atan2(center.y() - pd.y(), pd.x() - center.x()))
    direction = 1 if _clamp_angle(angle_m - angle_o) < _clamp_angle(angle_m - angle_d) else -1
    a_type = 0
    if arrow_type == _ARROW_RIGHT_HALF:
        a_type = direction
    elif arrow_type == _ARROW_LEFT_HALF:
        a_type = -direction
    delta = angle_d - angle_o
    if direction * delta < 0.0:
        delta += direction * 2 * math.pi
    length = _distance(po, pd)
    if abs(delta) < math.pi and (
            (head_type in (_HEAD_SINGLE, _HEAD_REVERSED) and length < head_width)
            or (head_type == _HEAD_DOUBLE and length < 2 * head_width)):
        return _straight_arrow(po, pd, start_width, width, head_width, head_height,
                               head_type, arrow_type, offset)
    radius += offset
    po = _circle_point(center, radius, angle_o)
    pd = _circle_point(center, radius, angle_d)
    head_angle = direction * math.atan(head_width / radius)
    cp = lambda r, a: _circle_point(center, r, a)  # noqa: E731
    path = QPainterPath()
    if head_type == _HEAD_DOUBLE:
        path.moveTo(po)
        if a_type <= 0:
            path.lineTo(cp(radius + direction * head_height, angle_o + head_angle))
            _path_arc_to(path, center, radius + direction * width / 2,
                         angle_o + head_angle, angle_d - head_angle, direction)
            path.lineTo(cp(radius + direction * head_height, angle_d - head_angle))
            path.lineTo(pd)
        else:
            _path_arc_to(path, center, radius, angle_o, angle_d, direction)
        if a_type >= 0:
            path.lineTo(cp(radius - direction * head_height, angle_d - head_angle))
            _path_arc_to(path, center, radius - direction * width / 2,
                         angle_d - head_angle, angle_o + head_angle, -direction)
            path.lineTo(cp(radius - direction * head_height, angle_o + head_angle))
            path.lineTo(po)
        else:
            _path_arc_to(path, center, radius, angle_d, angle_o, -direction)
    elif head_type == _HEAD_SINGLE:
        if a_type <= 0:
            path.moveTo(cp(radius + direction * start_width / 2, angle_o))
            _spiral_arc_to(path, center, angle_o, radius + direction * start_width / 2,
                           angle_d - head_angle, radius + direction * width / 2, direction)
            path.lineTo(cp(radius + direction * head_height, angle_d - head_angle))
            path.lineTo(pd)
        else:
            path.moveTo(po)
            _path_arc_to(path, center, radius, angle_o, angle_d, direction)
        if a_type >= 0:
            path.lineTo(cp(radius - direction * head_height, angle_d - head_angle))
            _spiral_arc_to(path, center, angle_d - head_angle, radius - direction * width / 2,
                           angle_o, radius - direction * start_width / 2, -direction)
            path.lineTo(cp(radius + direction * start_width / 2, angle_o))
        else:
            _path_arc_to(path, center, radius, angle_d, angle_o, -direction)
            path.lineTo(cp(radius + direction * start_width / 2, angle_o))
    elif head_type == _HEAD_REVERSED:
        path.moveTo(po)
        if a_type <= 0:
            path.lineTo(cp(radius + direction * head_height, angle_o + head_angle))
            path.lineTo(cp(radius + direction * width / 2, angle_o + head_angle))
            _spiral_arc_to(path, center, angle_o + head_angle, radius + direction * width / 2,
                           angle_d, radius + direction * start_width / 2, direction)
        else:
            _path_arc_to(path, center, radius, angle_o, angle_d, direction)
        if a_type >= 0:
            path.lineTo(cp(radius - direction * start_width / 2, angle_d))
            _spiral_arc_to(path, center, angle_d, radius - direction * start_width / 2,
                           angle_o + head_angle, radius - direction * width / 2, -direction)
            path.lineTo(cp(radius - direction * head_height, angle_o + head_angle))
            path.lineTo(po)
        else:
            path.lineTo(pd)
            _path_arc_to(path, center, radius, angle_d, angle_o, -direction)
    polygons = path.toSubpathPolygons()
    if not polygons:
        return []
    # Copies: the points of the list's polygon go with it.
    return [QPointF(p.x(), p.y()) for p in polygons[0]]


def _arrows_of_line(points: List[QPointF], curved: bool, repeated: bool, args: tuple) -> List[list]:
    """The arrow polygons QGIS draws along one line (renderPolyline)."""
    arrows = []
    if curved:
        if not repeated:
            if len(points) >= 3:
                arrows.append(_curved_arrow(points[0], points[len(points) // 2], points[-1], *args))
            elif len(points) == 2:
                arrows.append(_straight_arrow(points[0], points[1], *args))
        else:
            for i in range(0, len(points) - 1, 2):
                if len(points) - i >= 3:
                    arrows.append(_curved_arrow(points[i], points[i + 1], points[i + 2], *args))
                elif len(points) - i == 2:
                    arrows.append(_straight_arrow(points[i], points[i + 1], *args))
    elif not repeated:
        if points:
            arrows.append(_straight_arrow(points[0], points[-1], *args))
    else:
        for i in range(len(points) - 1):
            arrows.append(_straight_arrow(points[i], points[i + 1], *args))
    return [arrow for arrow in arrows if len(arrow) >= 4]


@qgsfunction(group=_GROUP, referenced_columns=[], handlesnull=True)
def q2vt_arrow(geometry, start_width, width, head_length, head_thickness, offset,
               head_type, arrow_type, curved, repeated, pixel, feature, parent):
    """The arrows QGIS's arrow symbol layer draws along a line geometry
    (EPSG:3857), as a multipolygon: lengths in metres, ``pixel`` the length
    of a screen pixel, in which QGIS draws (and flattens curves)."""
    if geometry is None or geometry.isEmpty() or not pixel or pixel <= 0:
        return None
    centre = geometry.boundingBox().center()
    # Screen pixels, y down (QGIS's drawing), around the geometry's centre.
    to_screen = lambda p: QPointF((p.x() - centre.x()) / pixel, (centre.y() - p.y()) / pixel)  # noqa: E731
    args = (start_width / pixel, width / pixel, head_length / pixel, head_thickness / pixel,
            int(head_type), int(arrow_type), offset / pixel)
    arrows = []
    for part in geometry.asGeometryCollection():
        points = [to_screen(p) for p in part.asPolyline()]
        for arrow in _arrows_of_line(points, bool(curved), bool(repeated), args):
            ring = [QgsPointXY(centre.x() + p.x() * pixel, centre.y() - p.y() * pixel) for p in arrow]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            # QGIS's rings may cross themselves near a tip, and arrows of
            # one line overlap: valid polygons, merged, for the tiles.
            arrows.append(QgsGeometry.fromPolygonXY([ring]).makeValid(Qgis.MakeValidMethod.Structure))
    if not arrows:
        return None
    return QgsGeometry.unaryUnion(arrows)


# --- Data-defined offsets ---

def _offset_pair(value) -> Optional[tuple]:
    """A QGIS x/y offset value ("x,y", a list or a point) as floats."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        items = value[:2]
    elif hasattr(value, "x") and hasattr(value, "y"):
        items = (value.x(), value.y())
    else:
        items = str(value).split(",")
        if len(items) != 2:
            return None
    try:
        return float(items[0]), float(items[1])
    except (TypeError, ValueError):
        return None


@qgsfunction(group=_GROUP, referenced_columns=[], handlesnull=True)
def q2vt_offset(geometry, value, static_x, static_y, unit_length, feature, parent):
    """A geometry (EPSG:3857) moved by an x/y offset as QGIS reads it (x
    right, y down on screen), ``value`` else (static_x, static_y), times
    ``unit_length`` (metres per offset unit)."""
    if geometry is None:
        return None
    pair = _offset_pair(value) or (static_x, static_y)
    moved = QgsGeometry(geometry)
    moved.translate(pair[0] * unit_length, -pair[1] * unit_length)
    return moved


# --- Callouts ---

# MapLibre text anchors: the label's box corner (or middle) at its point,
# as fractions of its width and height from its top left.
_ANCHOR_FRACTIONS = {
    "center": (0.5, 0.5), "left": (0, 0.5), "right": (1, 0.5), "top": (0.5, 0),
    "bottom": (0.5, 1), "top-left": (0, 0), "top-right": (1, 0),
    "bottom-left": (0, 1), "bottom-right": (1, 1),
}

# QgsCallout.LabelAnchorPoint: the callout's end on the label's box, as
# fractions of its width and height from its top left; None: the box's
# point closest to the feature (LabelPointOnExterior).
_LABEL_ANCHORS = {
    0: None, 1: (0.5, 0.5), 2: (0, 0), 3: (0.5, 0), 4: (1, 0), 5: (0, 0.5),
    6: (1, 0.5), 7: (0, 1), 8: (0.5, 1), 9: (1, 1),
}

# QgsCallout.AnchorPoint
_POLE, _EXTERIOR, _SURFACE, _CENTROID = range(4)

_metrics_cache = {}


def _text_size(text: str, family: str, pixel_size: float, line_height: float) -> tuple:
    """A label's text box (width, height) in screen pixels, as MapLibre
    lays it out: its lines' widths, line_height ems a line."""
    key = (family, round(pixel_size * 4))
    metrics = _metrics_cache.get(key)
    if metrics is None:
        font = QFont(family)
        font.setPixelSize(max(1, round(pixel_size)))
        metrics = _metrics_cache[key] = (QFontMetricsF(font), pixel_size / max(1, round(pixel_size)))
    fm, scale = metrics
    lines = str(text).split("\n")
    width = max(fm.horizontalAdvance(line) for line in lines) * scale
    return width, len(lines) * line_height * pixel_size


def _feature_anchor(geometry: QgsGeometry, kind: int, toward: QgsPointXY) -> Optional[QgsGeometry]:
    """The point of a feature a callout starts from (QgsCallout.AnchorPoint)."""
    geometry_type = int(getattr(geometry.type(), "value", geometry.type()))
    if geometry_type == 0:
        return geometry.centroid() if geometry.isMultipart() else geometry
    if geometry_type == 1:
        if kind == _EXTERIOR:
            return geometry.nearestPoint(QgsGeometry.fromPointXY(toward))
        return geometry.interpolate(geometry.length() / 2)
    if kind == _EXTERIOR:
        return geometry.convertToType(Qgis.GeometryType.Line, True).nearestPoint(QgsGeometry.fromPointXY(toward))
    if kind == _SURFACE:
        return geometry.pointOnSurface()
    if kind == _CENTROID:
        return geometry.centroid()
    pole = geometry.poleOfInaccessibility(max(geometry.boundingBox().width(), 1) / 1000)
    return pole[0] if isinstance(pole, tuple) else pole


@qgsfunction(group=_GROUP, referenced_columns=[], handlesnull=True)
def q2vt_callout(geometry, label_point, text, family, text_px, line_height, anchor,
                 offset_x, offset_y, feature_anchor, label_anchor, manhattan,
                 min_length, from_anchor, from_label, pixel, feature, parent):
    """A label's callout (EPSG:3857 line): from the feature's anchor point
    to its label's box drawn at ``label_point`` (text box laid out as
    MapLibre lays it: ``anchor`` and ``offset_x``/``offset_y`` ems), as
    QgsSimpleLineCallout (or QgsManhattanLineCallout) draws it. Lengths in
    metres; ``pixel`` the length of a screen pixel."""
    if geometry is None or label_point is None or text in (None, "") or not pixel or pixel <= 0:
        return None
    point = label_point.asPoint()
    width, height = _text_size(text, family, text_px, line_height)
    fx, fy = _ANCHOR_FRACTIONS.get(anchor, (0.5, 0.5))
    # The box's top left, on the ground (y up).
    left = point.x() + (offset_x * text_px - fx * width) * pixel
    top = point.y() - (offset_y * text_px - fy * height) * pixel
    w, h = width * pixel, height * pixel
    box = QgsGeometry.fromRect(QgsRectangle(left, top - h, left + w, top))
    start = _feature_anchor(geometry, int(feature_anchor), QgsPointXY(left + w / 2, top - h / 2))
    if start is None or start.isEmpty() or box.contains(start):
        return None
    fractions = _LABEL_ANCHORS.get(int(label_anchor))
    if fractions is None:
        end = box.convertToType(Qgis.GeometryType.Line, False).nearestPoint(start)
    else:
        end = QgsGeometry.fromPointXY(QgsPointXY(left + fractions[0] * w, top - fractions[1] * h))
    a, b = start.asPoint(), end.asPoint()
    length = math.hypot(b.x() - a.x(), b.y() - a.y())
    if length <= 0 or length < min_length:
        return None
    # Shortened by the offsets from the feature and from the label.
    if from_anchor + from_label >= length:
        return None
    ux, uy = (b.x() - a.x()) / length, (b.y() - a.y()) / length
    a = QgsPointXY(a.x() + ux * from_anchor, a.y() + uy * from_anchor)
    b = QgsPointXY(b.x() - ux * from_label, b.y() - uy * from_label)
    points = [a, QgsPointXY(b.x(), a.y()), b] if manhattan else [a, b]
    return QgsGeometry.fromPolylineXY(points)


def registered() -> bool:
    """Whether the functions are registered (they are on import)."""
    return QgsExpression.isFunctionName("q2vt_arrow")
