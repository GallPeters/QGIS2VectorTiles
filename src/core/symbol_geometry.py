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

from qgis.PyQt.QtCore import QLineF, QPointF, QRectF
from qgis.PyQt.QtGui import QFont, QFontMetricsF, QPainterPath, QTransform
from qgis.core import (
    Qgis, QgsCircularString, QgsExpression, QgsExpressionContext, QgsExpressionContextScope, QgsGeometry,
    QgsLineString, QgsPoint, QgsPointXY, QgsRectangle, QgsSymbolLayerUtils, qgsfunction,
)

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
               head_type, arrow_type, curved, repeated, pixel, static_head_type,
               static_arrow_type, feature, parent):
    """The arrows QGIS's arrow symbol layer draws along a line geometry
    (EPSG:3857), as a multipolygon: lengths in metres, ``pixel`` the length
    of a screen pixel, in which QGIS draws (and flattens curves). The head
    and arrow types are read as QGIS reads them (a number or a name; the
    static type where the value is empty or unknown)."""
    if geometry is None or geometry.isEmpty() or not pixel or pixel <= 0:
        return None
    decoded, ok = QgsSymbolLayerUtils.decodeArrowHeadType(head_type)
    head_type = int(decoded) if ok and head_type is not None else int(static_head_type)
    decoded, ok = QgsSymbolLayerUtils.decodeArrowType(arrow_type)
    arrow_type = int(decoded) if ok and arrow_type is not None else int(static_arrow_type)
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

# Callout kinds (RulesFlattener): QgsSimpleLineCallout, QgsManhattanLineCallout,
# QgsCurvedLineCallout, QgsBalloonCallout.
CALLOUT_SIMPLE, CALLOUT_MANHATTAN, CALLOUT_CURVED, CALLOUT_BALLOON = range(4)

# QgsCurvedLineCallout.Orientation
_AUTOMATIC, _CLOCKWISE, _COUNTER_CLOCKWISE = range(3)

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


def _near(a: float, b: float, epsilon: float = 1e-8) -> bool:
    return abs(a - b) <= epsilon


def _closest_side_of_rectangle(right, bottom, left, top, x, y) -> int:
    """QgsGeometryUtilsBase::closestSideOfRectangle: 1 top, 2 top right,
    3 right, 4 bottom right, 5 bottom, 6 bottom left, 7 left, 8 top left
    (y up; callers pass a screen rectangle's sides as QGIS does)."""
    if x <= left and y <= bottom:
        dx, dy = left - x, bottom - y
        return 6 if _near(dx, dy) else (5 if dx < dy else 7)
    if x >= right and y >= top:
        dx, dy = x - right, y - top
        return 2 if _near(dx, dy) else (1 if dx < dy else 3)
    if x >= right and y <= bottom:
        dx, dy = x - right, bottom - y
        return 4 if _near(dx, dy) else (5 if dx < dy else 3)
    if x <= left and y >= top:
        dx, dy = left - x, y - top
        return 8 if _near(dx, dy) else (1 if dx < dy else 7)
    if x <= left:
        return 7
    if x >= right:
        return 3
    if y <= bottom:
        return 5
    if y >= top:
        return 1
    smallest_x = min(right - x, x - left)
    smallest_y = min(top - y, y - bottom)
    if smallest_x < smallest_y:
        return 3 if right - x < x - left else 7
    return 1 if top - y < y - bottom else 5


def _automatic_orientation(rect: QRectF, start: QPointF, end: QPointF) -> int:
    """QgsCurvedLineCallout's automatic orientation (screen coordinates)."""
    side = _closest_side_of_rectangle(rect.right(), rect.bottom(), rect.left(), rect.top(),
                                      start.x(), start.y())
    sx, sy, ex, ey = start.x(), start.y(), end.x(), end.y()
    mid_x, mid_y = rect.left() + 0.5 * rect.width(), rect.top() + 0.5 * rect.height()
    cw, ccw = _CLOCKWISE, _COUNTER_CLOCKWISE
    if side == 1:
        if _near(ex, sx):
            return ccw if sx < mid_x else cw
        return ccw if ex > sx else cw
    if side == 2:
        if ex < sx:
            return cw
        if ey < sy:
            return ccw
        return cw if (ex - sx) < (ey - sy) else ccw
    if side == 3:
        if _near(ey, sy):
            return cw if sy < mid_y else ccw
        return ccw if ey < sy else cw
    if side == 4:
        if ex < sx:
            return ccw
        if ey > sy:
            return cw
        return ccw if (ex - sx) < (sy - ey) else cw
    if side == 5:
        if _near(ex, sx):
            return cw if sx < mid_x else ccw
        return ccw if ex < sx else cw
    if side == 6:
        if ex > sx:
            return cw
        if ey > sy:
            return ccw
        return cw if (sx - ex) < (sy - ey) else ccw
    if side == 7:
        if _near(ey, sy):
            return ccw if sy < mid_y else cw
        return ccw if ey > sy else cw
    if side == 8:
        if ex > sx:
            return ccw
        if ey < sy:
            return cw
        return ccw if (sx - ex) < (ey - sy) else cw
    return _AUTOMATIC


def _curved_line(rect: QRectF, start: QPointF, end: QPointF, curvature: float, orientation: int):
    """QgsCurvedLineCallout::createCalloutLine: a circular arc from start to
    end through their middle moved aside by curvature × their distance."""
    if orientation == _AUTOMATIC:
        orientation = _automatic_orientation(rect, start, end)
    sign = 1 if orientation == _CLOCKWISE else -1
    dx, dy = end.x() - start.x(), end.y() - start.y()
    length = math.hypot(dx, dy)
    if length == 0:
        return QgsLineString([QgsPoint(start.x(), start.y()), QgsPoint(end.x(), end.y())])
    offset = sign * length * curvature
    mid_x = start.x() + 0.5 * dx + offset * dy / length
    mid_y = start.y() + 0.5 * dy - offset * dx / length
    return QgsCircularString(QgsPoint(start.x(), start.y()), QgsPoint(mid_x, mid_y),
                             QgsPoint(end.x(), end.y()))


def _balloon_side(i: int, rect: QRectF, r: float) -> QLineF:
    """QgsShapeGenerator's balloon side i (top, right, bottom, left), inset
    by the corner radius."""
    ym = -1 if rect.height() < 0 else 1
    if i == 0:
        return QLineF(rect.left() + r, rect.top(), rect.right() - r, rect.top())
    if i == 1:
        return QLineF(rect.right(), rect.top() + ym * r, rect.right(), rect.bottom() - ym * r)
    if i == 2:
        return QLineF(rect.right() - r, rect.bottom(), rect.left() + r, rect.bottom())
    return QLineF(rect.left(), rect.bottom() - ym * r, rect.left(), rect.top() + ym * r)


def _closest_on_segment(line: QLineF, p: QPointF) -> QPointF:
    dx, dy = line.dx(), line.dy()
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p.x() - line.x1()) * dx + (p.y() - line.y1()) * dy) / length2))
    return QPointF(line.x1() + t * dx, line.y1() + t * dy)


def _balloon(origin: QPointF, rect: QRectF, wedge_width: float, r: float) -> List[QPointF]:
    """QgsShapeGenerator::createBalloon: the label's rectangle, its corners
    rounded, with a wedge pointing at origin from its nearest side."""
    inverted = rect.height() < 0
    r = min(r, min(abs(rect.height()), rect.width()) / 2)
    balloon_segment = -1
    p1 = p2 = QPointF()
    if not rect.contains(origin):
        best_d, best = float("inf"), -1
        best_p, best_side = QPointF(), QLineF()
        for i in range(4):
            side = _balloon_side(i, rect, r)
            p = _closest_on_segment(side, origin)
            d = (p.x() - origin.x()) ** 2 + (p.y() - origin.y()) ** 2
            prefer = False
            if _near(d, best_d):
                theta = (math.degrees(math.atan2(p.x() - origin.x(), p.y() - origin.y())) + 360) % 360
                prefer = ((theta < 45 or theta > 315) and i == 0) or (45 <= theta < 135 and i == 3) \
                    or (135 <= theta < 225 and i == 2) or (225 <= theta <= 315 and i == 1)
            elif d < best_d:
                prefer = True
            if prefer:
                best, best_d, best_p, best_side = i, d, p, side
        balloon_segment = best
        length = best_side.length()
        w = max(0.0, min(wedge_width, length))
        a, b = best_side.p1(), best_side.p2()
        p1 = best_p
        if math.hypot(p1.x() - b.x(), p1.y() - b.y()) < w:
            unit = (a - b) / max(length, 1e-12)
            p1 = b + unit * w
        to_b = b - p1
        distance = math.hypot(to_b.x(), to_b.y())
        p2 = p1 + (to_b / distance * w if distance else QPointF())
    path = QPainterPath()
    first = last = QPointF()
    for i in range(4):
        side = _balloon_side(i, rect, r)
        if i == 0:
            path.moveTo(side.p1())
            first = side.p1()
        else:
            corner = QRectF(min(last.x(), side.p1().x()), min(last.y(), side.p1().y()), r, r)
            start = {1: -90, 2: 0, 3: 90}[i] if inverted else {1: 90, 2: 0, 3: -90}[i]
            path.arcTo(corner, start, 90 if inverted else -90)
        if i == balloon_segment:
            path.lineTo(p1)
            path.lineTo(origin)
            path.lineTo(p2)
        last = side.p2()
        path.lineTo(last)
    corner = QRectF(min(last.x(), first.x()), min(last.y(), first.y()), r, r)
    path.arcTo(corner, 180 if inverted else -180, 90 if inverted else -90)
    polygon = path.toFillPolygon()
    # Copies: the points of the polygon go with it.
    return [QPointF(p.x(), p.y()) for p in polygon]


def _part_anchor(part: QgsGeometry, kind: int, box: QgsGeometry) -> Optional[QgsGeometry]:
    """The point of a feature's part a callout runs to (QgsCallout's
    calloutLineToPart): a point itself, a line's point nearest the label,
    a polygon's anchor point; None if the label is on the polygon."""
    part_type = int(getattr(part.type(), "value", part.type()))
    if part_type == 0:
        return part
    if part_type == 1:
        line = box.shortestLine(part)
        return QgsGeometry(line.vertexAt(1)) if not line.isEmpty() else None
    if box.intersects(part):
        return None
    if kind == _EXTERIOR:
        line = box.shortestLine(part.convertToType(Qgis.GeometryType.Line, False))
        return QgsGeometry(line.vertexAt(1)) if not line.isEmpty() else None
    if kind == _SURFACE:
        return part.pointOnSurface()
    if kind == _CENTROID:
        return part.centroid()
    bbox = part.boundingBox()
    pole = part.poleOfInaccessibility(max(bbox.width(), bbox.height()) / 20)
    return pole[0] if isinstance(pole, tuple) else pole


# The variable a balloon's geometry generator reads its geometry from (its
# @geometry, renamed by RulesFlattener).
BALLOON_VARIABLE = "q2vt_balloon"


# Callout shapes in screen pixels by what makes them (see q2vt_callout).
_callout_cache: dict = {}
_CALLOUT_CACHE_SIZE = 200000
_MISSING = object()


def _generated(expression: str, geometry: QgsGeometry, feature) -> Optional[QgsGeometry]:
    """A geometry generator's expression evaluated on ``geometry`` (its
    BALLOON_VARIABLE) for the feature (its fields read as QGIS reads them)."""
    expr = QgsExpression(expression)
    context = QgsExpressionContext()
    if feature is not None:
        context.setFeature(feature)
        context.setFields(feature.fields())
    scope = QgsExpressionContextScope()
    scope.setVariable(BALLOON_VARIABLE, geometry)
    context.appendScope(scope)
    context.setGeometry(geometry)
    value = expr.evaluate(context)
    return QgsGeometry(value) if isinstance(value, QgsGeometry) and not value.isNull() else None


@qgsfunction(group=_GROUP, referenced_columns=[], handlesnull=True)
def q2vt_callout(geometry, label_point, text, family, text_px, line_height, anchor,
                 offset_x, offset_y, feature_anchor, label_anchor, kind,
                 min_length, from_anchor, from_label, pixel, curvature, orientation,
                 wedge_width, corner_radius, margin_left, margin_top, margin_right,
                 margin_bottom, generator, feature, parent):
    """A label's callout as QGIS draws it, on the ground (EPSG:3857): the
    line (simple, Manhattan or curved) from its label's box drawn at
    ``label_point`` (laid out as MapLibre lays it: ``anchor`` and
    ``offset_x``/``offset_y`` ems) to the feature's anchor point, or the
    balloon around the box pointing at it (a polygon), or what one of the
    balloon fill symbol's geometry generators (``generator``) makes of it.
    Lengths in metres; ``pixel`` the length of a screen pixel, in which QGIS
    draws."""
    if geometry is None or label_point is None or text in (None, "") or not pixel or pixel <= 0:
        return None
    centre = label_point.asPoint()
    # Screen pixels around the label's point, y down (QGIS's drawing).
    to_screen = QTransform(1 / pixel, 0, 0, -1 / pixel, -centre.x() / pixel, centre.y() / pixel)
    width, height = _text_size(text, family, text_px, line_height)
    fx, fy = _ANCHOR_FRACTIONS.get(anchor, (0.5, 0.5))
    rect = QRectF(offset_x * text_px - fx * width, offset_y * text_px - fy * height, width, height)
    box = QgsGeometry.fromRect(QgsRectangle(rect.left(), rect.top(), rect.right(), rect.bottom()))
    part = QgsGeometry(geometry)
    part.transform(to_screen)

    def _screen_shape():
        nonlocal kind, rect
        end = _part_anchor(part, int(feature_anchor), box)
        if end is None or end.isEmpty():
            return None
        fractions = _LABEL_ANCHORS.get(int(label_anchor))
        if fractions is None:
            start = box.convertToType(Qgis.GeometryType.Line, False).nearestPoint(end)
        else:
            start = QgsGeometry.fromPointXY(QgsPointXY(rect.left() + fractions[0] * width,
                                                       rect.top() + fractions[1] * height))
            if int(getattr(part.type(), "value", part.type())) == 1:
                end = QgsGeometry(start.shortestLine(part).vertexAt(1))
        s, e = start.asPoint(), end.asPoint()
        s, e = QPointF(s.x(), s.y()), QPointF(e.x(), e.y())
        if math.hypot(e.x() - s.x(), e.y() - s.y()) < min_length / pixel:
            return None
        kind = int(kind)
        if kind == CALLOUT_CURVED:
            line = _curved_line(rect, s, e, curvature, int(orientation))
        elif kind == CALLOUT_MANHATTAN:
            line = QgsLineString([QgsPoint(s.x(), s.y()), QgsPoint(s.x(), e.y()), QgsPoint(e.x(), e.y())])
        else:
            line = QgsLineString([QgsPoint(s.x(), s.y()), QgsPoint(e.x(), e.y())])
        length = line.length()
        if kind == CALLOUT_BALLOON:
            origin = e
            if from_anchor > 0 and length > 0:
                tip = line.interpolatePoint(max(0.0, length - from_anchor / pixel))
                origin = QPointF(tip.x(), tip.y()) if tip else e
            rect = QRectF(rect.left() - margin_left / pixel, rect.top() - margin_top / pixel,
                          width + (margin_left + margin_right) / pixel,
                          height + (margin_top + margin_bottom) / pixel)
            if rect.width() <= 0 or rect.height() <= 0:
                return None
            ring = [QgsPointXY(p.x(), p.y()) for p in _balloon(origin, rect, wedge_width / pixel,
                                                               corner_radius / pixel)]
            if len(ring) < 4:
                return None
            result = QgsGeometry.fromPolygonXY([ring]).makeValid(Qgis.MakeValidMethod.Structure)
            if generator:
                # A geometry generator of the balloon's fill symbol: QGIS gives it
                # the balloon in screen pixels, y down (its "top" is its y_min).
                result = _generated(generator, result, feature)
                if result is None:
                    return None
        else:
            # Shortened by the offsets from the label and from the feature.
            start_at, end_at = from_label / pixel, length - from_anchor / pixel
            if end_at <= start_at:
                return None
            trimmed = line.curveSubstring(start_at, end_at)
            result = QgsGeometry(trimmed.curveToLine() if hasattr(trimmed, "curveToLine") else trimmed)
        return result

    # The shape in screen pixels, the same at every zoom for a label at its
    # point (and for each of a balloon's rules): computed once.
    pixels = tuple(round(v / pixel, 6) for v in (min_length, from_anchor, from_label, wedge_width,
                                                 corner_radius, margin_left, margin_top,
                                                 margin_right, margin_bottom))
    key = (part.asWkt(3), text, family, text_px, line_height, anchor, offset_x, offset_y,
           int(feature_anchor), int(label_anchor), int(kind), curvature, int(orientation), pixels,
           generator, feature.id() if generator and feature is not None else None)
    shape = _callout_cache.get(key, _MISSING)
    if shape is _MISSING:
        shape = _callout_cache[key] = _screen_shape()
        if len(_callout_cache) > _CALLOUT_CACHE_SIZE:
            _callout_cache.clear()
    if shape is None:
        return None
    result = QgsGeometry(shape)
    to_ground, _ = to_screen.inverted()
    result.transform(to_ground)
    return result


def registered() -> bool:
    """Whether the functions are registered (they are on import)."""
    return QgsExpression.isFunctionName("q2vt_arrow")
