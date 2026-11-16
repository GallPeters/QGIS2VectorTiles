"""
source_geometry.py

The rules export evaluates expressions on its base layers: the features
reprojected to EPSG:3857 and split into their parts. A rule filter, label
or data-defined value reading the feature's geometry ($geometry, $area,
$length, $perimeter, $x, $y, $id) would then read the reprojected part,
not the feature QGIS reads (in its layer's CRS, whole, measured with the
project's ellipsoid and units). source_expression() rewrites those reads
into the functions below, which give back what QGIS reads:

    q2vt_source_geometry(part, whole, crs)
        the feature's geometry in its layer's CRS: ``whole`` (the
        feature's parts collected, EPSG:3857) if given, else ``part``
    q2vt_measure(kind, geometry, crs, ellipsoid, unit)
        $area / $length / $perimeter of a geometry in ``crs``, as QGIS
        measures them (QgsDistanceArea with the project's ellipsoid,
        converted to the project's units)

Registered on import (the rules export and its pass processes import it).
"""

import re
import threading
from typing import Dict, Optional, Tuple

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsCsException,
    QgsDistanceArea,
    QgsExpression,
    QgsGeometry,
    qgsfunction,
)

_GROUP = "QGIS2VectorTiles"

# The variable holding a multi-part feature's parts, collected (EPSG:3857).
WHOLE_VARIABLE = "q2vt_whole"
# The base layers' field holding the source feature's id.
ORIG_ID_FIELD = "q2vt_orig_id"

# Per thread: transforms and distance calculators by their settings (QGIS
# objects are not shared between threads).
_local = threading.local()


def _transform(crs: str) -> Optional[QgsCoordinateTransform]:
    cache: Dict[str, Optional[QgsCoordinateTransform]] = _local.__dict__.setdefault("transforms", {})
    if crs not in cache:
        target = _crs(crs)
        cache[crs] = None if not target.isValid() or target.authid() == "EPSG:3857" else QgsCoordinateTransform(
            QgsCoordinateReferenceSystem("EPSG:3857"), target, QgsCoordinateTransformContext())
    return cache[crs]


def _crs(text: str) -> QgsCoordinateReferenceSystem:
    crs = QgsCoordinateReferenceSystem(text)
    return crs if crs.isValid() else QgsCoordinateReferenceSystem.fromWkt(text)


def _distance_area(crs: str, ellipsoid: str) -> QgsDistanceArea:
    cache: Dict[Tuple[str, str], QgsDistanceArea] = _local.__dict__.setdefault("calculators", {})
    key = (crs, ellipsoid)
    if key not in cache:
        calculator = QgsDistanceArea()
        calculator.setSourceCrs(_crs(crs), QgsCoordinateTransformContext())
        if ellipsoid:
            calculator.setEllipsoid(ellipsoid)
        cache[key] = calculator
    return cache[key]


@qgsfunction(group=_GROUP, referenced_columns=[], handlesnull=True)
def q2vt_source_geometry(part, whole, crs, feature, parent):
    """The feature's geometry in its layer's CRS (``crs``): ``whole`` (its
    parts collected) if given, else ``part``; both in EPSG:3857."""
    geometry = whole if isinstance(whole, QgsGeometry) and not whole.isNull() else part
    if not isinstance(geometry, QgsGeometry) or geometry.isNull():
        return None
    transform = _transform(crs)
    if transform is None:
        return QgsGeometry(geometry)
    result = QgsGeometry(geometry)
    try:
        result.transform(transform)
    except QgsCsException:
        return None
    return result


@qgsfunction(group=_GROUP, referenced_columns=[], handlesnull=True)
def q2vt_measure(kind, geometry, crs, ellipsoid, unit, feature, parent):
    """$area ("area"), $length ("length") or $perimeter ("perimeter") of a
    geometry in ``crs``, as QGIS measures them: with ``ellipsoid`` (none:
    planar, in the CRS's units), in ``unit`` (the project's area or
    distance unit, as an integer)."""
    if not isinstance(geometry, QgsGeometry) or geometry.isNull():
        return None
    calculator = _distance_area(crs, ellipsoid or "")
    try:
        if kind == "area":
            if geometry.type() != Qgis.GeometryType.Polygon:
                return 0.0
            return calculator.convertAreaMeasurement(calculator.measureArea(geometry), Qgis.AreaUnit(int(unit)))
        if kind == "perimeter":
            if geometry.type() != Qgis.GeometryType.Polygon:
                return 0.0
            value = calculator.measurePerimeter(geometry)
        else:
            if geometry.type() != Qgis.GeometryType.Line:
                return 0.0
            value = calculator.measureLength(geometry)
        return calculator.convertLengthMeasurement(value, Qgis.DistanceUnit(int(unit)))
    except (RuntimeError, ValueError, TypeError):
        return None


# Expression text tokens: string literals, quoted names, $ functions.
_TOKENS = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|\$[A-Za-z_]\w*")


def _replace_dollars(expression: str, replacements: Dict[str, str]) -> str:
    """``expression`` with its $ functions in ``replacements`` replaced,
    outside string literals and quoted names."""
    def sub(match):
        token = match.group(0)
        return replacements.get(token.lower(), token)
    return _TOKENS.sub(sub, expression)


def reads_source_geometry(expression: Optional[str]) -> bool:
    """Whether an expression reads the feature's geometry or id through $
    functions (see source_expression)."""
    if not isinstance(expression, str) or "$" not in expression:
        return False
    return any(match.group(0).lower() in _DOLLARS for match in _TOKENS.finditer(expression)
               if match.group(0).startswith("$"))


_DOLLARS = frozenset({"$geometry", "$area", "$length", "$perimeter", "$x", "$y", "$id"})


def source_expression(expression: str, measure: Tuple[str, str, int, int]) -> str:
    """A rule expression (filter, label, data-defined value) reading the
    feature as QGIS reads it, though evaluated on a base layer's part in
    EPSG:3857: its $geometry, $area, $length, $perimeter, $x, $y and $id.
    ``measure`` is (layer CRS, project ellipsoid, project distance unit,
    project area unit). @geometry is left as it is: the export's own
    expressions read it in EPSG:3857."""
    if not reads_source_geometry(expression):
        return expression
    crs, ellipsoid, distance_unit, area_unit = measure
    crs_text = QgsExpression.quotedString(crs or "EPSG:3857")
    geometry = f"q2vt_source_geometry(@geometry, @{WHOLE_VARIABLE}, {crs_text})"
    ellipsoid_text = QgsExpression.quotedString(ellipsoid or "")
    return _replace_dollars(expression, {
        "$geometry": geometry,
        "$area": f"q2vt_measure('area', {geometry}, {crs_text}, {ellipsoid_text}, {int(area_unit)})",
        "$length": f"q2vt_measure('length', {geometry}, {crs_text}, {ellipsoid_text}, {int(distance_unit)})",
        "$perimeter": f"q2vt_measure('perimeter', {geometry}, {crs_text}, {ellipsoid_text}, {int(distance_unit)})",
        "$x": f"x({geometry})",
        "$y": f"y({geometry})",
        "$id": f'to_int("{ORIG_ID_FIELD}")',
    })


__all__ = ["ORIG_ID_FIELD", "WHOLE_VARIABLE", "reads_source_geometry", "source_expression"]
