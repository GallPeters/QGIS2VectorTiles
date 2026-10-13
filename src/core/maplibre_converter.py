"""Convert QGIS Vector Tile Layer styles to MapLibre GL JSON style format."""

import copy
import json
import math
import os
import re
from os.path import join
from typing import Any, Dict, List, Optional, Union

from qgis.PyQt.QtGui import QColor
from ..utils.config import Qt
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCsException,
    QgsUnitTypes,
    QgsVectorTileLayer,
    QgsVectorTileBasicRenderer,
    QgsVectorTileBasicLabeling,
    QgsSymbol,
    QgsPalLayerSettings,
    QgsSimpleLineSymbolLayer,
    QgsSimpleFillSymbolLayer,
    QgsProcessingUtils,
    QgsExpression,
    QgsProperty,
    QgsSymbolLayer,
    QgsTextFormat,
    QgsProject,
    QgsTextBackgroundSettings,
)
from qgis.utils import iface
from .glyphs_generator import GlyphGenerator
from .sprite_generator import FillPattern, HashMark, LinePattern, SpriteGenerator
from .rules_exporter import _TEMP_RULE_FORMAT, RULE_FIELD
from ..utils.config import _SPRITE_QUALITY, _TILES_MAX_ZOOM


def _mix_colors(*colors: QColor) -> QColor:
    """The mean of colours (with their alpha), e.g. a gradient's two ends."""
    valid = [c for c in colors if c is not None and c.isValid()]
    if not valid:
        return QColor(0, 0, 0, 0)
    n = len(valid)
    return QColor(
        round(sum(c.red() for c in valid) / n), round(sum(c.green() for c in valid) / n),
        round(sum(c.blue() for c in valid) / n), round(sum(c.alpha() for c in valid) / n),
    )


def _ramp_or_colors(symbol_layer, two_color_type: int) -> QColor:
    """A gradient or shapeburst fill's mean colour: of its two colours, or
    its colour ramp sampled along its length."""
    try:
        if PropertyExtractor.enum_int(symbol_layer.gradientColorType()
                                      if hasattr(symbol_layer, "gradientColorType")
                                      else symbol_layer.colorType()) == two_color_type:
            return _mix_colors(symbol_layer.color(), symbol_layer.color2())
        ramp = symbol_layer.colorRamp()
        if ramp is not None:
            return _mix_colors(*(ramp.color(i / 4) for i in range(5)))
    except (AttributeError, RuntimeError, TypeError):
        pass
    return symbol_layer.color()


# A style filter selecting one rule's rows of a shared dataset (TilesStyler).
_RULE_FILTER = re.compile(r'"' + re.escape(RULE_FIELD) + r'" = (\d+)')

# Layout properties MapLibre evaluates per feature, which a merged layer can
# give per rule; the other layout properties must be equal to merge.
_DATA_DRIVEN_LAYOUT = frozenset({
    "symbol-sort-key", "line-sort-key", "fill-sort-key", "icon-image", "icon-size",
    "icon-rotate", "icon-offset", "icon-anchor", "text-field", "text-size", "text-rotate",
    "text-offset", "text-letter-spacing", "text-max-width", "text-justify", "text-anchor",
    "text-transform", "text-radial-offset",
})
# Paint properties MapLibre can't evaluate per feature.
_CONSTANT_PAINT = frozenset({
    "fill-antialias", "fill-translate", "fill-translate-anchor", "fill-pattern",
    "line-translate", "line-translate-anchor", "line-dasharray", "line-pattern",
    "icon-translate", "icon-translate-anchor", "text-translate", "text-translate-anchor",
})


def _rule_of(layer_def: dict) -> Optional[int]:
    """The rule number a layer is filtered to, if it is a rule's layer."""
    flt = layer_def.get("filter")
    if (isinstance(flt, list) and len(flt) == 3 and flt[0] == "=="
            and flt[1] == ["get", RULE_FIELD] and isinstance(flt[2], int)):
        return flt[2]
    return None


def _mergeable(first: dict, other: dict) -> bool:
    """Whether two rule layers of one dataset can be drawn as one layer."""
    if _rule_of(first) is None or _rule_of(other) is None:
        return False
    for key in ("type", "source", "source-layer", "minzoom", "maxzoom"):
        if first.get(key) != other.get(key):
            return False
    for section, varying in (("layout", _DATA_DRIVEN_LAYOUT), ("paint", None)):
        a, b = first.get(section, {}), other.get(section, {})
        if a.keys() != b.keys():
            return False
        for key, value in a.items():
            if value == b[key]:
                continue
            allowed = key in varying if varying is not None else key not in _CONSTANT_PAINT
            if (not allowed or PropertyExtractor.uses_zoom(value)
                    or PropertyExtractor.uses_zoom(b[key])):
                return False
    return True


def _match_value(value):
    """A value as a ``match`` output: literal arrays need ``literal``."""
    if isinstance(value, list) and (not value or not isinstance(value[0], str)):
        return ["literal", value]
    return value


def merge_rule_layers(layers: List[dict]) -> List[dict]:
    """Merge consecutive layers of the rules sharing a dataset that differ
    only in properties MapLibre evaluates per feature (typically the
    categories of a categorized renderer, or labels split by category) into
    one layer, the varying properties as ``match`` expressions on the rule
    number. Fewer layers render and place labels faster; features are then
    drawn in their own order, as QGIS draws them (without symbol levels).
    """
    runs: List[List[dict]] = []
    for layer_def in layers:
        if runs and _mergeable(runs[-1][0], layer_def):
            runs[-1].append(layer_def)
        else:
            runs.append([layer_def])
    merged = []
    for run in runs:
        if len(run) == 1:
            merged.append(run[0])
            continue
        rules = [_rule_of(layer_def) for layer_def in run]
        layer_def = copy.deepcopy(run[0])
        layer_def["filter"] = ["match", ["get", RULE_FIELD], rules, True, False]
        for section in ("layout", "paint"):
            for key in layer_def.get(section, {}):
                values = [member[section][key] for member in run]
                if all(v == values[0] for v in values):
                    continue
                expression = ["match", ["get", RULE_FIELD]]
                for rule, value in zip(rules, values):
                    expression += [rule, _match_value(value)]
                expression.append(_match_value(values[-1]))
                layer_def[section][key] = expression
        merged.append(layer_def)
    return merged


class PropertyExtractor:
    """Utility class for extracting and converting PyQGIS properties to MapLibre format.

    Provides shared helpers used by every specialised extractor:
    - Resolution of static vs. data-defined values into MapLibre expressions.
    - Conversion of ``QColor`` instances into MapLibre ``rgba()`` strings.
    - Conversion of QGIS length units into pixels at a 96 DPI baseline.
    """

    # Conversion factors to pixels at 96 DPI, keyed by unit string keywords.
    _UNIT_KEYWORDS: dict = {
        "millimeter": 3.78, "mm": 3.78,
        "inch": 96.0, "in": 96.0,
        "point": 96.0 / 72.0, "pt": 96.0 / 72.0,
        "pixel": 1.0, "px": 1.0,
    }
    # QgsUnitTypes enum integer values → conversion factors.
    _UNIT_ENUMS: dict = {0: 3.78, 2: 1.0, 4: 96.0 / 72.0, 5: 96.0}
    # Map-based units: a length on the ground, so its pixel size depends on the zoom.
    _MAP_UNITS = 1
    _METERS_IN_MAP_UNITS = 7
    # EPSG:3857 metres per web map pixel at zoom 0 (MapLibre's 512 px tiles).
    _MERCATOR_M_PER_PX_Z0 = 40075016.68557849 / 512
    # EPSG:3857 metres per map unit / per ground metre at the data, set per export.
    map_unit_m = 1.0
    ground_m = 1.0

    @staticmethod
    def get_value_or_expression(value: Any, prop: QgsProperty) -> Union[Any, List]:
        """Return a static value or a MapLibre ``["get", field]`` expression.

        If the supplied ``QgsProperty`` is active and references a tile attribute
        (``q2vt*`` field naming convention), a MapLibre expression is returned so
        the property is data-driven at render time. Otherwise the original
        ``value`` is returned unchanged.

        Args:
            value: The static fallback value to use when no data-defined override applies.
            prop:  The data-defined property descriptor from QGIS.

        Returns:
            Either the original ``value`` or a MapLibre expression list.
        """
        if prop and prop.isActive():
            expression = prop.expressionString()
            qexpr = QgsExpression(expression)
            evaluation = qexpr.evaluate()
            if evaluation is not None and evaluation != "" and not qexpr.hasEvalError():
                return evaluation
            field_name = expression.replace('"', "")
            idx = field_name.find("q2vt")
            if field_name and idx != -1:
                extracted_name = [part for part in field_name.split(' ') if part.startswith('q2vt')][0]
                return ["get", extracted_name.replace(',', '')]
        return value

    @staticmethod
    def convert_qcolor_to_maplibre(color: QColor) -> str:
        """Convert a ``QColor`` into a MapLibre-compatible ``rgba()`` string.

        Args:
            color: The Qt color to convert.

        Returns:
            A string in the form ``"rgba(r, g, b, a)"`` where alpha is normalised
            to the ``[0, 1]`` range expected by MapLibre.
        """
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alphaF()})"

    @classmethod
    def convert_length_to_pixels(cls, value: float, unit_obj=None) -> float:
        """Convert a length value from a QGIS unit into pixels at 96 DPI.

        The unit can be supplied as either a textual hint (e.g. ``"MM"``,
        ``"Pixels"``) or as a ``QgsUnitTypes`` enum integer. When the unit is
        unknown, the value is treated as millimetres.

        Args:
            value:    The length to convert.
            unit_obj: A unit descriptor; either a string-like or enum integer.

        Returns:
            The equivalent pixel value (float).
        """
        if value is None:
            return value

        try:
            unit_str = str(unit_obj).lower() if unit_obj is not None else ""
        except (OSError, RuntimeError):
            unit_str = ""

        for keyword, factor in cls._UNIT_KEYWORDS.items():
            if keyword in unit_str:
                return value * factor

        try:
            factor = cls._UNIT_ENUMS.get(unit_obj)
            if factor is not None:
                return value * factor
        except (RuntimeError, AttributeError):
            pass

        return value * 3.78  # Default: treat as millimetres

    @staticmethod
    def enum_int(enum_obj) -> Optional[int]:
        """A QGIS/Qt enum value as an integer (Qt5 gives ints, Qt6 enums)."""
        try:
            return int(getattr(enum_obj, "value", enum_obj))
        except (TypeError, ValueError):
            return None

    @classmethod
    def is_map_unit(cls, unit_obj) -> bool:
        """Whether a length is on the ground (map units or metres) rather than on screen."""
        return cls.enum_int(unit_obj) in (cls._MAP_UNITS, cls._METERS_IN_MAP_UNITS)

    @classmethod
    def length_to_style_value(cls, value: float, unit_obj=None) -> Union[float, List]:
        """A length as a MapLibre value: pixels, or a zoom expression for ground lengths.

        A ground length doubles in pixels with every zoom level, which an
        exponential base-2 interpolation reproduces exactly.
        """
        if value is None or not cls.is_map_unit(unit_obj):
            return cls.convert_length_to_pixels(value, unit_obj)
        metres = cls.map_unit_m if cls.enum_int(unit_obj) == cls._MAP_UNITS else cls.ground_m
        px_z0 = value * metres / cls._MERCATOR_M_PER_PX_Z0
        return ["interpolate", ["exponential", 2], ["zoom"], 0, px_z0, 24, px_z0 * 2 ** 24]

    @classmethod
    def scale_length_expression(cls, expression: List, unit_obj=None) -> List:
        """A per-feature length (an expression giving a number in ``unit_obj``)
        as a MapLibre pixel value: times the unit's pixel size, or for
        ground lengths, doubled with every zoom level."""
        number = ["to-number", expression, 0]
        if cls.is_map_unit(unit_obj):
            metres = cls.map_unit_m if cls.enum_int(unit_obj) == cls._MAP_UNITS else cls.ground_m
            factor = metres / cls._MERCATOR_M_PER_PX_Z0
            return ["interpolate", ["exponential", 2], ["zoom"],
                    0, ["*", number, factor], 24, ["*", number, factor * 2 ** 24]]
        return ["*", number, cls.convert_length_to_pixels(1.0, unit_obj)]

    @classmethod
    def length_value_or_expression(cls, value: float, unit_obj, prop: QgsProperty) -> Union[float, List]:
        """A length that may be data-defined, as a MapLibre value.

        QGIS reads a data-defined length in the property's unit (mm, points,
        map units...), but MapLibre reads pixels, so the value is converted
        as the static length is: a constant directly, a field through an
        expression.
        """
        base = cls.length_to_style_value(value, unit_obj)
        resolved = cls.get_value_or_expression(None, prop)
        if resolved is None:
            return base
        if isinstance(resolved, list):
            return cls.scale_length_expression(resolved, unit_obj)
        try:
            return cls.length_to_style_value(float(resolved), unit_obj)
        except (TypeError, ValueError):
            return base

    @classmethod
    def opacity_value_or_expression(cls, base_opacity: float, prop: QgsProperty) -> Union[float, List]:
        """An opacity that may be data-defined: QGIS reads a data-defined
        opacity in percent (0-100), MapLibre in 0-1."""
        resolved = cls.get_value_or_expression(None, prop)
        if resolved is None:
            return base_opacity
        if isinstance(resolved, list):
            return ["/", ["to-number", resolved, 100], 100]
        try:
            return max(0.0, min(1.0, float(resolved) / 100.0))
        except (TypeError, ValueError):
            return base_opacity

    @staticmethod
    def uses_zoom(value) -> bool:
        """Whether a style value is (or contains) a zoom expression."""
        if isinstance(value, list):
            return value[:1] == ["zoom"] or any(PropertyExtractor.uses_zoom(v) for v in value)
        return False

    @classmethod
    def set_map_units(cls, crs, latitude: float) -> None:
        """Set how many EPSG:3857 metres a map unit and a ground metre are at the data.

        Web Mercator stretches ground lengths by 1/cos(latitude); a project
        in Web Mercator already draws its map units stretched that way.
        """
        stretch = 1.0 / max(math.cos(math.radians(latitude)), 0.01)
        cls.ground_m = stretch
        cls.map_unit_m = stretch
        try:
            if crs.isGeographic():
                cls.map_unit_m = 111319.49079327357  # 3857 metres per degree of longitude
            elif crs.authid() in ("EPSG:3857", "EPSG:900913", "EPSG:3785"):
                cls.map_unit_m = 1.0
            else:
                cls.map_unit_m = stretch * QgsUnitTypes.fromUnitToUnitFactor(
                    crs.mapUnits(), Qgis.DistanceUnit.Meters
                )
        except (AttributeError, RuntimeError, TypeError):
            pass

    @staticmethod
    def get_attribute(obj: Any, *names) -> Any:
        """Fetch the first available attribute from ``obj`` matching any of ``names``.

        Tries each attribute name in turn, returning the resolved value
        (calling it if callable). Returns ``None`` if no attribute exists or
        every access fails.

        Args:
            obj:    The object to inspect.
            *names: Attribute names to try, in order.

        Returns:
            The resolved value, or ``None`` if every name fails.
        """
        for name in names:
            if hasattr(obj, name):
                try:
                    attr = getattr(obj, name)
                    return attr() if callable(attr) else attr
                except (RuntimeError, AttributeError, OSError):
                    continue
        return None


class LinePropertyExtractor:
    """Extract line paint and layout properties from QGIS line symbol layers."""

    @staticmethod
    def get_line_color(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[str, List]:
        """Return ``line-color`` resolving any data-defined override."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.color())
        color_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.Property.PropertyStrokeColor)
        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_line_width(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[float, List]:
        """Return ``line-width`` in pixels, resolving any data-defined override."""
        width_unit = PropertyExtractor.get_attribute(
            symbol_layer, "widthUnit", "widthUnits", "strokeWidthUnit", "strokeWidthUnits"
        )
        width_prop = symbol_layer.dataDefinedProperties().property(
            QgsSymbolLayer.Property.PropertyStrokeWidth
        )
        return PropertyExtractor.length_value_or_expression(
            symbol_layer.width(), width_unit, width_prop
        )

    @staticmethod
    def get_line_opacity(
        symbol_layer: QgsSimpleLineSymbolLayer, symbol: "QgsSymbol" = None
    ) -> Union[float, List]:
        """Return ``line-opacity`` from the symbol's general Opacity setting.

        Independent of the stroke colour's alpha channel: that alpha is already
        embedded in the ``line-color`` rgba() string, and MapLibre multiplies
        ``line-opacity`` against it at render time. ``symbol.opacity()`` is
        QGIS's own general "Opacity" slider (top of the Symbol panel) - a
        separate multiplier from colour alpha. Honours a data-defined
        ``Property.PropertyOpacity`` override when active.
        """
        try:
            base_opacity = symbol.opacity() if symbol is not None else 1.0
        except (AttributeError, RuntimeError):
            base_opacity = 1.0
        try:
            opacity_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyOpacity
            )
            return PropertyExtractor.opacity_value_or_expression(base_opacity, opacity_prop)
        except (AttributeError, RuntimeError):
            return base_opacity

    @staticmethod
    def get_line_cap(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Return ``line-cap`` mapped from the Qt pen-cap style."""
        cap_map = {0: "butt", 16: "square", 32: "round"}
        return cap_map.get(PropertyExtractor.enum_int(symbol_layer.penCapStyle()), "round")

    @staticmethod
    def get_line_join(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Return ``line-join`` mapped from the Qt pen-join style."""
        # Qt's SvgMiterJoin (256) is a miter join too.
        join_map = {0: "miter", 64: "bevel", 128: "round", 256: "miter"}
        return join_map.get(PropertyExtractor.enum_int(symbol_layer.penJoinStyle()), "round")

    @staticmethod
    def get_line_miter_limit() -> float:
        """Return ``line-miter-limit`` (MapLibre default: 2.0)."""
        return 2.0

    @staticmethod
    def get_line_round_limit() -> float:
        """Return ``line-round-limit`` (MapLibre default: 1.05)."""
        return 1.05

    @staticmethod
    def get_line_dasharray(
        symbol_layer: QgsSimpleLineSymbolLayer, width_px: float
    ) -> Optional[List[float]]:
        """Return ``line-dasharray`` from a custom dash vector or pen-style preset.

        Custom dash patterns take precedence over Qt's pen-style enum presets.
        MapLibre dash lengths are multiples of the line width, so custom
        lengths are converted to pixels and divided by ``width_px``. When no
        dashing is configured, ``None`` is returned and ``line-dasharray``
        should be omitted.
        """
        dash_vector = None
        try:
            dash_vector = symbol_layer.customDashVector()
        except (RuntimeError, AttributeError):
            pass

        custom_dash_enabled = False
        try:
            for attr in ("useCustomDashPattern", "customDashEnabled", "isCustomDash"):
                if hasattr(symbol_layer, attr):
                    custom_dash_enabled = bool(getattr(symbol_layer, attr)())
                    break
        except (RuntimeError, AttributeError):
            pass

        pen_style = PropertyExtractor.get_attribute(
            symbol_layer, "penStyle", "strokeStyle", "pen_style"
        )

        if custom_dash_enabled and dash_vector:
            dash_unit = PropertyExtractor.get_attribute(
                symbol_layer, "customDashPatternUnit", "customDashVectorUnit"
            )
            width_unit = PropertyExtractor.get_attribute(
                symbol_layer, "widthUnit", "widthUnits", "strokeWidthUnit", "strokeWidthUnits"
            )
            unit = PropertyExtractor.enum_int(dash_unit or width_unit)
            if PropertyExtractor.is_map_unit(width_unit) and unit == PropertyExtractor.enum_int(width_unit):
                # Ground dashes on a ground width: the ratio holds at every zoom.
                width = symbol_layer.width() or 1.0
                return [d / width for d in dash_vector]
            width = width_px if width_px and width_px > 0 else 1.0
            return [
                PropertyExtractor.convert_length_to_pixels(d, dash_unit or width_unit) / width
                for d in dash_vector
            ]

        # Qt's own patterns, as [dash, gap, ...] multiples of the line width.
        fixed_presets = {
            2: [4, 2],             # DashLine
            3: [1, 2],             # DotLine
            4: [4, 2, 1, 2],       # DashDotLine
            5: [4, 2, 1, 2, 1, 2], # DashDotDotLine
        }
        return fixed_presets.get(PropertyExtractor.enum_int(pen_style))

    @staticmethod
    def get_line_offset(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[float, List]:
        """Return ``line-offset`` in pixels (positive values offset to the right).

        Honours a data-defined ``PropertyOffset`` override when active. Note
        that data-defined offsets are emitted in QGIS units (pixels for the
        static path); for field-based overrides the user must supply pixel
        values to match the MapLibre rendering convention.
        """
        try:
            offset = symbol_layer.offset()
            offset_unit = PropertyExtractor.get_attribute(symbol_layer, "offsetUnit", "offsetUnits")
            offset_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyOffset
            )
            return PropertyExtractor.length_value_or_expression(offset, offset_unit, offset_prop)
        except (AttributeError, RuntimeError, TypeError):
            return 0.0

    @staticmethod
    def get_line_blur() -> float:
        """Return ``line-blur`` in pixels (MapLibre default: 0)."""
        return 0

    @staticmethod
    def get_line_gap_width() -> float:
        """Return ``line-gap-width`` in pixels (MapLibre default: 0)."""
        return 0

    @staticmethod
    def get_line_translate() -> List[float]:
        """Return ``line-translate`` ``[x, y]`` offset in pixels (default ``[0, 0]``)."""
        return [0, 0]

    @staticmethod
    def get_line_translate_anchor() -> str:
        """Return ``line-translate-anchor`` (MapLibre default: ``"map"``)."""
        return "map"

    @staticmethod
    def get_line_sort_key(symbol_layer: QgsSymbolLayer) -> Union[float, List]:
        """Return ``line-sort-key`` honouring any data-defined override.

        The static default is ``0``; if a data-defined ``RenderingOrder`` (or
        equivalent) property is active on the layer, the corresponding
        MapLibre expression is emitted instead.
        """
        order_prop = None
        try:
            order_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyLayerEnabled
            )
        except (AttributeError, RuntimeError):
            pass
        return PropertyExtractor.get_value_or_expression(0, order_prop)

    @staticmethod
    def is_pattern_line(symbol_layer: QgsSymbolLayer) -> bool:
        """Return ``True`` if this line symbol layer is drawn from an image.

        Detected by class-name inspection so QGIS versions lacking the
        subclass remain compatible.
        """
        return type(symbol_layer).__name__ == "QgsRasterLineSymbolLayer"

    @staticmethod
    def get_line_pattern_path(symbol_layer: QgsSymbolLayer) -> Optional[str]:
        """Return the source image path for a pattern line, or ``None``.

        Used by the converter to register a pattern image with the sprite
        pipeline and to populate ``line-pattern``.
        """
        path = PropertyExtractor.get_attribute(symbol_layer, "path", "imagePath")
        if isinstance(path, str) and path:
            return path
        return None

    @staticmethod
    def is_marker_line(symbol_layer: QgsSymbolLayer) -> bool:
        """Return ``True`` if this line symbol layer decorates the line with markers.

        Detected by class-name inspection (matching the ``is_pattern_line``
        convention above) so QGIS versions lacking the symbol-layer class
        remain compatible.
        """
        return type(symbol_layer).__name__ == "QgsMarkerLineSymbolLayer"

    @staticmethod
    def is_hashed_line(symbol_layer: QgsSymbolLayer) -> bool:
        """Return ``True`` for a hashed line (short strokes across the line)."""
        return type(symbol_layer).__name__ == "QgsHashedLineSymbolLayer"

    @staticmethod
    def _placements(symbol_layer: QgsSymbolLayer):
        try:
            return symbol_layer.placements()
        except (AttributeError, RuntimeError):
            return None

    @staticmethod
    def get_marker_line_symbol_placement(symbol_layer: QgsSymbolLayer) -> str:
        """Map QGIS marker-line placements to MapLibre ``symbol-placement``.

        MapLibre only places evenly-spaced (``"line"``) or once-per-line
        (``"line-center"``) symbols. Vertex placements of marker lines are
        exported as points (see RulesFlattener._split_marker_line); what
        reaches here is interval placement (``"line"``), or segment centres
        and other single placements, approximated by one symbol in the
        middle of the line.
        """
        placements = LinePropertyExtractor._placements(symbol_layer)
        if placements is None or placements & Qgis.MarkerLinePlacement.Interval:
            return "line"
        return "line-center"

    @staticmethod
    def get_marker_line_spacing(symbol_layer: QgsSymbolLayer) -> float:
        """Return ``symbol-spacing`` in pixels from the marker-line interval,
        or MapLibre's default (250 px) when the line has none."""
        try:
            interval = float(symbol_layer.interval())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            interval = 0.0
        if interval > 0:
            unit = PropertyExtractor.get_attribute(
                symbol_layer, "intervalUnit", "intervalUnits"
            )
            return max(1.0, PropertyExtractor.convert_length_to_pixels(interval, unit))
        return 250.0

    @staticmethod
    def get_fallback_line(symbol_layer: QgsSymbolLayer) -> tuple:
        """(colour, width) approximating a line symbol layer MapLibre can't
        draw: its colour (the mean of a burst's two colours, an arrow's fill
        colour) and width, so it shows as a plain line of the right look
        rather than MapLibre's default black hairline."""
        kind = type(symbol_layer).__name__
        color = None
        width = None
        unit = None
        try:
            if kind == "QgsLineburstSymbolLayer":
                color = _mix_colors(symbol_layer.color(), symbol_layer.color2())
            elif kind == "QgsArrowSymbolLayer":
                sub = symbol_layer.subSymbol()
                color = sub.color() if sub is not None else symbol_layer.color()
                width, unit = symbol_layer.arrowWidth(), symbol_layer.arrowWidthUnit()
            else:
                color = symbol_layer.color()
        except (AttributeError, RuntimeError, TypeError):
            color = None
        if width is None:
            width = PropertyExtractor.get_attribute(symbol_layer, "width")
            unit = PropertyExtractor.get_attribute(symbol_layer, "widthUnit")
        if not isinstance(width, (int, float)) or width <= 0:
            width, unit = 0.26, None  # QGIS's default line width, in mm.
        if color is None or not color.isValid():
            color = QColor(0, 0, 0)
        return (PropertyExtractor.convert_qcolor_to_maplibre(color),
                PropertyExtractor.length_to_style_value(width, unit))

    @staticmethod
    def get_marker_line_rotate_symbols(symbol_layer: QgsSymbolLayer) -> bool:
        """Return whether markers should rotate to follow the line bearing.

        Mirrors QGIS ``rotateSymbols()``; defaults to ``True`` (QGIS's own
        default) if the property cannot be read.
        """
        try:
            return bool(symbol_layer.rotateSymbols())
        except (AttributeError, RuntimeError):
            return True

    @staticmethod
    def get_marker_line_offset(symbol_layer: QgsSymbolLayer) -> float:
        """Return the marker-line's perpendicular offset from the line, in pixels.

        Positive QGIS offsets shift markers to the left of the line
        direction; ``0`` is returned (and the caller should omit
        ``icon-offset``) when no offset is configured.
        """
        try:
            offset = float(symbol_layer.offset())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0
        if offset == 0:
            return 0.0
        unit = PropertyExtractor.get_attribute(symbol_layer, "offsetUnit", "offsetUnits")
        return PropertyExtractor.convert_length_to_pixels(offset, unit)


class FillPropertyExtractor:
    """Extract fill paint and layout properties from QGIS fill symbol layers."""

    @staticmethod
    def get_fill_color(symbol_layer: QgsSimpleFillSymbolLayer) -> Union[str, List]:
        """Return ``fill-color`` resolving any data-defined override."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.color())
        color_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.Property.PropertyFillColor)
        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_fill_opacity(
        symbol_layer: QgsSimpleFillSymbolLayer, symbol: "QgsSymbol" = None
    ) -> Union[float, List]:
        """Return ``fill-opacity`` from the symbol's general Opacity setting.

        Independent of the fill colour's alpha channel, for the same reason as
        ``get_line_opacity``: colour alpha lives in ``fill-color``, and
        ``symbol.opacity()`` is QGIS's separate general "Opacity" slider.
        Honours a data-defined ``Property.PropertyOpacity`` override when active.
        """
        try:
            base_opacity = symbol.opacity() if symbol is not None else 1.0
        except (AttributeError, RuntimeError):
            base_opacity = 1.0
        try:
            opacity_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyOpacity
            )
            return PropertyExtractor.opacity_value_or_expression(base_opacity, opacity_prop)
        except (AttributeError, RuntimeError):
            return base_opacity

    @staticmethod
    def get_fill_outline_color(
        symbol_layer: QgsSimpleFillSymbolLayer,
    ) -> Union[str, List, None]:
        """Return ``fill-outline-color`` if the polygon stroke is visible."""
        try:
            stroke_visible = symbol_layer.strokeWidth() > 0 and symbol_layer.strokeStyle() != 0
        except (AttributeError, RuntimeError):
            stroke_visible = False
        if stroke_visible:
            base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.strokeColor())
            color_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyStrokeColor
            )
            return PropertyExtractor.get_value_or_expression(base_color, color_prop)
        return None

    @staticmethod
    def get_fill_antialias() -> bool:
        """Return ``fill-antialias`` (MapLibre default: ``True``)."""
        return True

    @staticmethod
    def get_fill_translate() -> List[float]:
        """Return ``fill-translate`` ``[x, y]`` offset in pixels (default ``[0, 0]``)."""
        return [0, 0]

    @staticmethod
    def get_fill_translate_anchor() -> str:
        """Return ``fill-translate-anchor`` (MapLibre default: ``"map"``)."""
        return "map"

    @staticmethod
    def get_fill_sort_key(symbol_layer: QgsSymbolLayer) -> Union[float, List]:
        """Return ``fill-sort-key`` honouring any data-defined override."""
        order_prop = None
        try:
            order_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyLayerEnabled
            )
        except (AttributeError, RuntimeError):
            pass
        return PropertyExtractor.get_value_or_expression(0, order_prop)

    @staticmethod
    def is_pattern_fill(symbol_layer: QgsSymbolLayer) -> bool:
        """Return ``True`` for pattern-based fill layers (point/line/raster/SVG)."""
        pattern_class_names = {
            "QgsPointPatternFillSymbolLayer",
            "QgsLinePatternFillSymbolLayer",
            "QgsRasterFillSymbolLayer",
            "QgsSVGFillSymbolLayer",
            "QgsRandomMarkerFillSymbolLayer",
        }
        return type(symbol_layer).__name__ in pattern_class_names

    @staticmethod
    def get_fallback_fill_color(symbol_layer: QgsSymbolLayer) -> str:
        """A colour approximating a fill MapLibre can't draw as QGIS does.

        Gradient and shapeburst fills vary across each feature, which a
        MapLibre fill can't; their mean colour is the closest flat colour.
        Other fills use their own colour, never MapLibre's default black.
        """
        kind = type(symbol_layer).__name__
        if kind == "QgsGradientFillSymbolLayer":
            color = _ramp_or_colors(symbol_layer, 0)  # SimpleTwoColor
        elif kind == "QgsShapeburstFillSymbolLayer":
            color = _ramp_or_colors(symbol_layer, 0)
        else:
            try:
                color = symbol_layer.color()
            except (AttributeError, RuntimeError):
                color = QColor(0, 0, 0, 0)
        if color is None or not color.isValid():
            color = QColor(0, 0, 0, 0)
        return PropertyExtractor.convert_qcolor_to_maplibre(color)

    # Tile size of a non-repeating pattern (random markers), in pixels.
    _RANDOM_PATTERN_TILE = 128.0

    @staticmethod
    def get_pattern_tile_size(symbol_layer: QgsSymbolLayer) -> tuple:
        """The pattern's repeat (width, height) in pixels: one tile of ``fill-pattern``.

        A tile of exactly one repeat makes MapLibre's repeated image seamless.
        Point grids repeat every distance (twice that along a row
        displacement); parallel lines every distance / sin or cos of their
        angle; SVG and raster images every image width and its height.
        """
        px = PropertyExtractor.convert_length_to_pixels
        get = PropertyExtractor.get_attribute
        kind = type(symbol_layer).__name__
        try:
            if kind == "QgsPointPatternFillSymbolLayer":
                width = px(symbol_layer.distanceX(), get(symbol_layer, "distanceXUnit"))
                height = px(symbol_layer.distanceY(), get(symbol_layer, "distanceYUnit"))
                if symbol_layer.displacementX():
                    height *= 2
                if symbol_layer.displacementY():
                    width *= 2
                return width, height
            if kind == "QgsLinePatternFillSymbolLayer":
                distance = px(symbol_layer.distance(), get(symbol_layer, "distanceUnit"))
                angle = math.radians(symbol_layer.lineAngle())
                sin, cos = abs(math.sin(angle)), abs(math.cos(angle))
                # A tile edge parallel to the lines can be any length: use the distance.
                width = distance / sin if sin > 1e-3 else distance
                height = distance / cos if cos > 1e-3 else distance
                return width, height
            if kind == "QgsSVGFillSymbolLayer":
                from qgis.PyQt.QtSvg import QSvgRenderer
                width = px(symbol_layer.patternWidth(), get(symbol_layer, "patternWidthUnit"))
                box = QSvgRenderer(symbol_layer.svgFilePath()).viewBoxF()
                ratio = box.height() / box.width() if box.width() > 0 else 1.0
                return width, width * ratio
            if kind == "QgsRasterFillSymbolLayer":
                from qgis.PyQt.QtGui import QImageReader
                size = QImageReader(symbol_layer.imageFilePath()).size()
                width = symbol_layer.width()
                width = px(width, get(symbol_layer, "sizeUnit", "widthUnit")) if width > 0 else size.width()
                ratio = size.height() / size.width() if size.width() > 0 else 1.0
                return width, width * ratio
        except (AttributeError, RuntimeError, TypeError, ValueError, ImportError):
            pass
        return FillPropertyExtractor._RANDOM_PATTERN_TILE, FillPropertyExtractor._RANDOM_PATTERN_TILE

    @staticmethod
    def get_fill_pattern_path(symbol_layer: QgsSymbolLayer) -> Optional[str]:
        """Return the source image path for a raster/SVG fill pattern, or ``None``."""
        path = PropertyExtractor.get_attribute(symbol_layer, "imageFilePath", "svgFilePath", "path")
        if isinstance(path, str) and path:
            return path
        return None


class IconPropertyExtractor:
    """Extract icon paint and layout properties for MapLibre symbol layers."""

    @staticmethod
    def get_icon_image(marker_name: str) -> str:
        """Return the registered sprite name for ``icon-image``."""
        return marker_name

    @staticmethod
    def get_icon_size(
        symbol_layer: QgsSymbolLayer, default_size: float = 1.0
    ) -> Union[float, List]:
        """Return ``icon-size`` honouring any data-defined override.

        The sprite draws the marker at its own size (times _SPRITE_QUALITY).
        A data-defined size is a marker size, not a scale, so the icon is
        scaled by its ratio to the marker's own size: a constant becomes a
        number, a field a MapLibre expression.
        """
        size_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.Property.PropertySize)
        size_val = PropertyExtractor.get_value_or_expression(default_size, size_prop)
        if size_val is default_size:
            return default_size / _SPRITE_QUALITY
        try:
            base = float(symbol_layer.size())
        except (AttributeError, TypeError, ValueError):
            base = 0.0
        if base <= 0:
            return default_size / _SPRITE_QUALITY
        if isinstance(size_val, list):
            return ["/", ["to-number", size_val, base], base * _SPRITE_QUALITY]
        try:
            return float(size_val) / base / _SPRITE_QUALITY
        except (TypeError, ValueError):
            return default_size / _SPRITE_QUALITY

    @staticmethod
    def get_icon_rotate(
        symbol_layer: QgsSymbolLayer = None,
        background: QgsTextBackgroundSettings = None,
    ) -> Union[float, List]:
        """Return ``icon-rotate`` in degrees, honouring data-defined overrides.

        When given a marker symbol layer, the static rotation is taken from
        ``angle()`` and any active ``PropertyAngle`` data-defined override is
        applied. When given a label background, the background rotation is
        used instead.
        """
        if symbol_layer is not None:
            base_angle = 0.0
            try:
                base_angle = float(symbol_layer.angle())
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                angle_prop = symbol_layer.dataDefinedProperties().property(
                    QgsSymbolLayer.Property.PropertyAngle
                )
                return PropertyExtractor.get_value_or_expression(base_angle, angle_prop)
            except (AttributeError, RuntimeError):
                return base_angle
        if background is not None and background.enabled():
            try:
                return float(background.rotation())
            except (AttributeError, RuntimeError, TypeError):
                return 0.0
        return 0.0

    @staticmethod
    def get_icon_padding() -> float:
        """Return ``icon-padding`` in pixels (MapLibre default: 2)."""
        return 1

    @staticmethod
    def get_icon_rotation_alignment() -> str:
        """Return ``icon-rotation-alignment`` (default: ``"map"`` for QGIS markers)."""
        return "map"

    @staticmethod
    def get_icon_pitch_alignment() -> str:
        """Return ``icon-pitch-alignment`` (default: ``"viewport"``)."""
        return "viewport"

    @staticmethod
    def get_icon_anchor() -> str:
        """Return ``icon-anchor`` (default: ``"center"``)."""
        return "center"

    @staticmethod
    def get_icon_allow_overlap(allow_overlap: bool = False) -> bool:
        """Return ``icon-allow-overlap`` (MapLibre default: ``False``)."""
        return allow_overlap

    @staticmethod
    def get_icon_ignore_placement() -> bool:
        """Return ``icon-ignore-placement`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_icon_optional() -> bool:
        """Return ``icon-optional`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_icon_keep_upright() -> bool:
        """Return ``icon-keep-upright``.

        Defaults to ``True`` to match QGIS rendering for line-aligned markers.
        """
        return True

    @staticmethod
    def get_icon_text_fit(background: QgsTextBackgroundSettings) -> Optional[str]:
        """Return ``icon-text-fit`` if the background is sized as a text buffer."""
        if background.enabled() and background.sizeType() == 0:
            return "both"
        return None

    @staticmethod
    def get_icon_text_fit_padding(
        background: QgsTextBackgroundSettings,
    ) -> Optional[List[float]]:
        """Return ``icon-text-fit-padding`` ``[top, right, bottom, left]`` from buffer size."""
        if background.enabled() and background.sizeType() == 0:
            buf_px = PropertyExtractor.convert_length_to_pixels(
                background.size().width(), background.sizeUnit()
            )
            buf_px = max(buf_px, 3)  
            return [buf_px * 2, buf_px, buf_px * 2, buf_px]
        return None

    @staticmethod
    def get_icon_offset(background: QgsTextBackgroundSettings = None) -> List[float]:
        """Return ``icon-offset`` ``[x, y]`` in pixels from background offset settings."""
        if background and background.enabled():
            offset = background.offset()
            unit = background.offsetUnit()
            return [
                PropertyExtractor.convert_length_to_pixels(offset.x(), unit),
                PropertyExtractor.convert_length_to_pixels(offset.y(), unit),
            ]
        return [0, 0]

    @staticmethod
    def get_icon_opacity(background: QgsTextBackgroundSettings = None) -> float:
        """Return ``icon-opacity`` from background settings (default ``1.0``)."""
        if background and background.enabled():
            try:
                return background.opacity()
            except (OSError, RuntimeError):
                pass
        return 1.0

    @staticmethod
    def get_icon_color(background: QgsTextBackgroundSettings = None) -> str:
        """Return ``icon-color`` from background fill colour (default white)."""
        if background and background.enabled():
            try:
                return PropertyExtractor.convert_qcolor_to_maplibre(background.fillColor())
            except (OSError, RuntimeError):
                pass
        return "rgb(255, 255, 255)"

    @staticmethod
    def get_icon_halo_color(background: QgsTextBackgroundSettings = None) -> str:
        """Return ``icon-halo-color``.

        When a label background is provided and has a stroke, the stroke
        colour is used; otherwise black is returned to match the MapLibre
        default.
        """
        if background and background.enabled():
            try:
                return PropertyExtractor.convert_qcolor_to_maplibre(background.strokeColor())
            except (OSError, RuntimeError, AttributeError):
                pass
        return "rgb(0, 0, 0)"

    @staticmethod
    def get_icon_halo_width(background: QgsTextBackgroundSettings = None) -> float:
        """Return ``icon-halo-width`` in pixels from the background stroke width."""
        if background and background.enabled():
            try:
                return PropertyExtractor.convert_length_to_pixels(
                    background.strokeWidth(), background.strokeWidthUnit()
                )
            except (OSError, RuntimeError, AttributeError):
                pass
        return 0

    @staticmethod
    def get_icon_halo_blur() -> float:
        """Return ``icon-halo-blur`` in pixels (default 0)."""
        return 0

    @staticmethod
    def get_icon_translate() -> List[float]:
        """Return ``icon-translate`` ``[x, y]`` offset in pixels (default ``[0, 0]``)."""
        return [0, 0]

    @staticmethod
    def get_icon_translate_anchor() -> str:
        """Return ``icon-translate-anchor`` (MapLibre default: ``"map"``)."""
        return "map"

    @staticmethod
    def get_symbol_placement(label_settings: QgsPalLayerSettings = None) -> str:
        """Return ``symbol-placement`` (``"point"``, ``"line"``, or ``"line-center"``).

        Mapped from ``QgsPalLayerSettings.placement``: line/curved/perimeter
        placements become ``"line"``; everything else becomes ``"point"``.
        """
        if label_settings is None:
            return "point"
        try:
            placement = PropertyExtractor.enum_int(label_settings.placement)
        except AttributeError:
            return "point"
        # QgsPalLayerSettings placement enum: 2=Line, 3=Curved, 7=PerimeterCurved, 8=OutsidePolygons
        line_placements = (2, 3, 7, 8)
        if placement in line_placements:
            return "line"
        return "point"

    @staticmethod
    def get_symbol_spacing(label_settings: QgsPalLayerSettings = None) -> float:
        """Return ``symbol-spacing`` in pixels (MapLibre default: 250).

        Sourced from ``label_settings.repeatDistance`` when set; otherwise
        the MapLibre default of 250 px is returned.
        """
        if label_settings is None:
            return 250.0
        try:
            distance = label_settings.repeatDistance
            if distance and distance > 0:
                unit = PropertyExtractor.get_attribute(
                    label_settings, "repeatDistanceUnit", "distUnits"
                )
                return PropertyExtractor.convert_length_to_pixels(distance, unit)
        except (AttributeError, RuntimeError):
            pass
        return 250.0

    @staticmethod
    def get_symbol_avoid_edges() -> bool:
        """Return ``symbol-avoid-edges`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_symbol_sort_key() -> float:
        """Return ``symbol-sort-key`` (default 0; lower keys render first)."""
        return 0

    @staticmethod
    def get_symbol_z_order() -> str:
        """Return ``symbol-z-order`` (default ``"auto"``)."""
        return "auto"


class TextPropertyExtractor:
    """Extract text paint and layout properties from QGIS label settings."""

    # Widest halo MapLibre draws cleanly, in ems (see get_text_halo_width).
    _HALO_MAX_EM = 0.2

    @staticmethod
    def get_text_field(label_settings: QgsPalLayerSettings) -> Optional[List]:
        """Return ``text-field`` as a MapLibre ``["get", field]`` expression."""
        if label_settings.fieldName:
            return ["get", label_settings.fieldName]
        return None

    @staticmethod
    def get_text_font(text_format: QgsTextFormat) -> List[str]:
        """Return ``text-font`` as a single-element list of ``"family style"``."""
        font = text_format.font()
        font_str = f"{font.family()} {font.styleName()}"
        return font_str

    @staticmethod
    def get_text_size(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[float, List]:
        """Return ``text-size`` in pixels, honouring data-defined overrides."""
        size_prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Property.Size)
        return PropertyExtractor.length_value_or_expression(
            text_format.size(), text_format.sizeUnit(), size_prop
        )

    @staticmethod
    def get_text_size_px(text_format: QgsTextFormat) -> float:
        """The label's font size in pixels, to turn pixel lengths into ems.

        A size in map units has no single pixel size; 16 px stands in for it.
        """
        if PropertyExtractor.is_map_unit(text_format.sizeUnit()):
            return 16.0
        size = PropertyExtractor.convert_length_to_pixels(text_format.size(), text_format.sizeUnit())
        return size if size and size > 0 else 16.0

    @staticmethod
    def get_text_color(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[str, List]:
        """Return ``text-color`` honouring any data-defined override."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(text_format.color())
        color_prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Property.Color)
        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_text_opacity(
        text_format: QgsTextFormat,
        label_settings: QgsPalLayerSettings = None,
    ) -> Union[float, List]:
        """Return ``text-opacity`` from the text format's general Opacity setting.

        Independent of the font colour's alpha channel, same reasoning as the
        fill/line variants. ``text_format.opacity()`` is QGIS's own general
        text "Opacity" slider. Honours a data-defined ``FontOpacity`` override
        on ``label_settings`` when active. When the property does not exist on
        the running QGIS build, the static opacity value is returned.
        """
        try:
            base_opacity = text_format.opacity()
        except (AttributeError, RuntimeError):
            base_opacity = 1.0
        if label_settings is None:
            return base_opacity
        try:
            opacity_prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.Property.FontOpacity
            )
            return PropertyExtractor.opacity_value_or_expression(base_opacity, opacity_prop)
        except (AttributeError, RuntimeError):
            return base_opacity

    @staticmethod
    def get_text_halo_color(
        text_format: QgsTextFormat,
        label_settings: QgsPalLayerSettings = None,
    ) -> Union[str, List]:
        """Return ``text-halo-color`` from buffer settings.

        Honours a data-defined ``BufferColor`` override on ``label_settings``
        when active, so per-feature halo colour attributes are emitted as
        MapLibre ``["get", field]`` expressions rather than the hard-coded
        static buffer colour. Falls back to white when the buffer is
        disabled and no override is supplied.
        """
        buffer = text_format.buffer()
        if not buffer.enabled():
            return "rgb(255, 255, 255)"
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(buffer.color())
        if label_settings is None:
            return base_color
        try:
            color_prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.Property.BufferColor
            )
            return PropertyExtractor.get_value_or_expression(base_color, color_prop)
        except (AttributeError, RuntimeError):
            return base_color

    @staticmethod
    def get_text_halo_width(
        text_format: QgsTextFormat,
        label_settings: QgsPalLayerSettings = None,
    ) -> Union[float, List]:
        """Return ``text-halo-width`` in pixels from the buffer size.

        Honours a data-defined ``BufferSize`` override on ``label_settings``
        when active. Field-based overrides are returned as MapLibre
        expressions; the user is responsible for storing pixel-equivalent
        values in the source attribute since MapLibre expressions cannot
        express the QGIS unit conversion at evaluation time.
        """
        buffer = text_format.buffer()
        if not buffer.enabled():
            return 0
        # MapLibre draws halos up to about a fifth of the text size; wider
        # ones reach the edge of the glyphs' distance field and fill each
        # letter's whole box.
        cap = TextPropertyExtractor._HALO_MAX_EM * TextPropertyExtractor.get_text_size_px(text_format)
        size_prop = None
        if label_settings is not None:
            try:
                size_prop = label_settings.dataDefinedProperties().property(
                    QgsPalLayerSettings.Property.BufferSize
                )
            except (AttributeError, RuntimeError):
                size_prop = None
        # Both are the halo's width outside the letters.
        try:
            width = PropertyExtractor.length_value_or_expression(
                buffer.size(), buffer.sizeUnit(), size_prop
            )
        except (AttributeError, RuntimeError):
            width = buffer.size()
        if isinstance(width, (int, float)):
            return min(width, cap)
        if not PropertyExtractor.uses_zoom(width):
            return ["min", width, cap]
        return width

    @staticmethod
    def get_text_halo_blur(
        text_format: QgsTextFormat,
        label_settings: QgsPalLayerSettings = None,
    ) -> Union[float, List]:
        """Return ``text-halo-blur`` in pixels.

        QGIS exposes a separate buffer blur radius via ``blurRadius`` when
        available; otherwise half the buffer size is used as a sensible
        approximation. Honours the data-defined ``BufferBlurRadius``
        override on ``label_settings`` when the QGIS build supports it.
        """
        buffer = text_format.buffer()
        if not buffer.enabled():
            return 0
        try:
            blur = buffer.blurRadius()
            if blur is not None:
                base_blur = PropertyExtractor.convert_length_to_pixels(
                    blur, buffer.blurRadiusUnit()
                )
            else:
                base_blur = buffer.size() * 0.5
        except (AttributeError, RuntimeError):
            base_blur = buffer.size() * 0.5

        if label_settings is None:
            return base_blur
        try:
            prop_key = getattr(QgsPalLayerSettings, "BufferBlurRadius")
        except AttributeError:
            return base_blur
        try:
            blur_prop = label_settings.dataDefinedProperties().property(prop_key)
            return PropertyExtractor.get_value_or_expression(base_blur, blur_prop)
        except (AttributeError, RuntimeError):
            return base_blur

    @staticmethod
    def get_text_anchor(label_settings: QgsPalLayerSettings) -> str:
        """Return ``text-anchor`` mapped from the QGIS quadrant offset."""
        anchor_map = {
            0: "bottom-right", 1: "bottom",  2: "bottom-left",
            3: "right",        4: "center",  5: "left",
            6: "top-right",    7: "top",     8: "top-left",
        }
        return anchor_map.get(PropertyExtractor.enum_int(label_settings.quadOffset), "center")

    @staticmethod
    def get_text_justify(label_settings: QgsPalLayerSettings) -> Union[str, List]:
        """Return ``text-justify`` mapped from the multi-line alignment setting.

        Honours a data-defined ``MultiLineAlignment`` override when active.
        For field-based overrides the source attribute is expected to
        contain the literal MapLibre keyword (``"left"``, ``"center"``,
        ``"right"`` or ``"auto"``); QGIS-specific integer/string codes are
        not translated server-side.
        """
        try:
            justification = label_settings.multiLineAlignment
        except AttributeError:
            try:
                justification = label_settings.alignment
            except AttributeError:
                justification = 1
        justify_map = {0: "left", 1: "center", 2: "right", 3: "center"}
        base_justify = justify_map.get(PropertyExtractor.enum_int(justification), "left")
        try:
            justify_prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.Property.MultiLineAlignment
            )
            return PropertyExtractor.get_value_or_expression(base_justify, justify_prop)
        except (AttributeError, RuntimeError):
            return base_justify

    # QGIS's "Offset from point" placement, the only one that uses xOffset/yOffset.
    _OVER_POINT = 1

    # Line label placement flags (Qgis.LabelLinePlacementFlag).
    _ON_LINE, _ABOVE_LINE, _BELOW_LINE = 1, 2, 4
    # Half a line of text, in ems: from a line label's centre to its edge.
    _HALF_LINE_EM = 0.6
    # Placements of labels along lines (Line, Curved, Perimeter curved).
    _LINE_PLACEMENTS = (2, 3, 7)

    @staticmethod
    def get_text_offset(label_settings: QgsPalLayerSettings) -> List[float]:
        """Return ``text-offset`` ``[x, y]`` in ems (y down, as in QGIS).

        "Offset from point" labels use their X / Y offset. Line labels
        placed above or below (not on) the line are moved off it by half a
        line of text plus their distance, as QGIS places them.
        """
        em = TextPropertyExtractor.get_text_size_px(label_settings.format())
        placement = PropertyExtractor.enum_int(label_settings.placement)
        if placement == TextPropertyExtractor._OVER_POINT:
            x_offset = label_settings.xOffset
            y_offset = label_settings.yOffset
            if x_offset != 0 or y_offset != 0:
                offset_unit = PropertyExtractor.get_attribute(label_settings, "offsetUnits")
                return [
                    PropertyExtractor.convert_length_to_pixels(x_offset, offset_unit) / em,
                    PropertyExtractor.convert_length_to_pixels(y_offset, offset_unit) / em,
                ]
            return [0, 0]
        if placement in TextPropertyExtractor._LINE_PLACEMENTS:
            try:
                flags = PropertyExtractor.enum_int(label_settings.lineSettings().placementFlags())
            except (AttributeError, RuntimeError, TypeError):
                return [0, 0]
            if flags is None or flags & TextPropertyExtractor._ON_LINE:
                return [0, 0]
            shift = TextPropertyExtractor._HALF_LINE_EM + TextPropertyExtractor._distance_em(
                label_settings, em)
            if flags & TextPropertyExtractor._ABOVE_LINE:
                return [0, -shift]
            if flags & TextPropertyExtractor._BELOW_LINE:
                return [0, shift]
        return [0, 0]

    @staticmethod
    def _distance_em(label_settings: QgsPalLayerSettings, em: float) -> float:
        """The label's distance setting, in ems of its text."""
        try:
            distance = float(label_settings.dist)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        if not distance:
            return 0.0
        unit = PropertyExtractor.get_attribute(label_settings, "distUnits", "distUnit")
        return PropertyExtractor.convert_length_to_pixels(distance, unit) / em

    # Gap MapLibre leaves between a point and a label placed around it, in
    # ems: about the radius of a typical point marker, which QGIS keeps
    # labels clear of.
    _AROUND_POINT_EM = 0.7

    @staticmethod
    def get_text_radial_offset(label_settings: QgsPalLayerSettings) -> float:
        """Return ``text-radial-offset`` in ems: the gap around the point
        plus the label's distance setting."""
        em = TextPropertyExtractor.get_text_size_px(label_settings.format())
        return TextPropertyExtractor._AROUND_POINT_EM + TextPropertyExtractor._distance_em(
            label_settings, em)

    # QGIS's "around point" and "cartographic" (ordered positions) placements.
    _AROUND_POINT, _ORDERED_POSITIONS = 0, 6
    # QGIS's candidate order around a point: above right first.
    _AROUND_POINT_ANCHORS = [
        "bottom-left", "bottom-right", "top-left", "top-right", "left", "right", "bottom", "top",
    ]
    # Anchor (the side of the label at the point) per
    # Qgis.LabelPredefinedPointPosition: TopLeft, TopSlightlyLeft, TopMiddle,
    # TopSlightlyRight, TopRight, MiddleLeft, MiddleRight, BottomLeft,
    # BottomSlightlyLeft, BottomMiddle, BottomSlightlyRight, BottomRight,
    # OverPoint.
    _POSITION_ANCHORS = [
        "bottom-right", "bottom-right", "bottom", "bottom-left", "bottom-left", "right", "left",
        "top-right", "top-right", "top", "top-left", "top-left", "center",
    ]

    @staticmethod
    def get_text_variable_anchor(label_settings: QgsPalLayerSettings = None) -> Optional[List[str]]:
        """Return ``text-variable-anchor`` when QGIS chooses the label's side.

        "Around point" tries every side, above right first; "cartographic"
        tries its own position order. Other placements put the label at a
        fixed place (None: the static ``text-anchor``).
        """
        if label_settings is None:
            return None
        placement = PropertyExtractor.enum_int(label_settings.placement)
        if placement == TextPropertyExtractor._AROUND_POINT:
            return list(TextPropertyExtractor._AROUND_POINT_ANCHORS)
        if placement != TextPropertyExtractor._ORDERED_POSITIONS:
            return None
        anchors = []
        try:
            positions = label_settings.predefinedPositionOrder()
        except (AttributeError, RuntimeError):
            positions = []
        for position in positions:
            index = PropertyExtractor.enum_int(position)
            if index is not None and 0 <= index < len(TextPropertyExtractor._POSITION_ANCHORS):
                anchor = TextPropertyExtractor._POSITION_ANCHORS[index]
                if anchor not in anchors:
                    anchors.append(anchor)
        return anchors or list(TextPropertyExtractor._AROUND_POINT_ANCHORS)

    @staticmethod
    def radial_to_offset(anchor: str, radial: float) -> List[float]:
        """``text-offset`` [x, y] (ems, y down) placing a label with this
        anchor ``radial`` ems from its point, as MapLibre's radial offset
        does: along the axis for a side, split over both for a corner."""
        x = 1 if "left" in anchor else -1 if "right" in anchor else 0
        y = 1 if "top" in anchor else -1 if "bottom" in anchor else 0
        if x and y:
            radial /= math.sqrt(2)
        return [x * radial, y * radial]

    @staticmethod
    def get_text_sort_key(label_settings: QgsPalLayerSettings) -> Union[float, List]:
        """Return ``symbol-sort-key`` from the label priority.

        QGIS places higher priorities (0-10) first; MapLibre places lower
        sort keys first. A data-defined priority becomes an expression.
        """
        base = 10 - label_settings.priority
        try:
            prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.Property.Priority
            )
        except (AttributeError, RuntimeError):
            return base
        resolved = PropertyExtractor.get_value_or_expression(None, prop)
        if resolved is None:
            return base
        if isinstance(resolved, list):
            return ["-", 10, ["to-number", resolved, label_settings.priority]]
        try:
            return 10 - float(resolved)
        except (TypeError, ValueError):
            return base

    @staticmethod
    def get_text_max_angle(label_settings: QgsPalLayerSettings) -> float:
        """Return ``text-max-angle`` in degrees for curved-label placement.

        Sourced from ``maxCurvedCharAngleIn`` (with a fallback to
        ``maxCurvedCharAngleOut``); MapLibre defaults to 45° when omitted.
        """
        try:
            angle_in = label_settings.maxCurvedCharAngleIn
            if angle_in is not None:
                return abs(float(angle_in))
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            angle_out = label_settings.maxCurvedCharAngleOut
            if angle_out is not None:
                return abs(float(angle_out))
        except (AttributeError, RuntimeError, TypeError):
            pass
        return 45.0

    @staticmethod
    def get_text_allow_overlap(label_settings: QgsPalLayerSettings = None) -> bool:
        """Return ``text-allow-overlap``: QGIS's "Allow overlaps without penalty"."""
        try:
            handling = label_settings.placementSettings().overlapHandling()
            return PropertyExtractor.enum_int(handling) == PropertyExtractor.enum_int(
                Qgis.LabelOverlapHandling.AllowOverlapAtNoCost
            )
        except (AttributeError, RuntimeError):
            return False

    @staticmethod
    def get_text_ignore_placement() -> bool:
        """Return ``text-ignore-placement`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_text_optional() -> bool:
        """Return ``text-optional`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_text_padding() -> float:
        """Return ``text-padding`` in pixels (MapLibre default: 2).

        QGIS lets labels touch, so the default keeps about as many labels.
        """
        return 2

    @staticmethod
    def get_text_line_height(text_format: QgsTextFormat = None) -> float:
        """Return ``text-line-height`` in ems.

        QGIS's line height is a multiple of the font's own line spacing,
        about 1.2 em: MapLibre's default.
        """
        try:
            if PropertyExtractor.enum_int(text_format.lineHeightUnit()) == PropertyExtractor.enum_int(
                Qgis.RenderUnit.Percentage
            ):
                return 1.2 * text_format.lineHeight()
        except (AttributeError, RuntimeError, TypeError):
            pass
        return 1.2

    @staticmethod
    def get_text_letter_spacing(text_format: QgsTextFormat = None) -> float:
        """Return ``text-letter-spacing`` in ems, from the font's letter
        spacing (in the text size's unit, or a percentage of normal)."""
        if text_format is None:
            return 0
        try:
            font = text_format.font()
            spacing = font.letterSpacing()
            if PropertyExtractor.enum_int(font.letterSpacingType()) == 1:  # Absolute
                size = text_format.size()
                return spacing / size if size else 0
            return (spacing - 100) / 100 if spacing else 0  # Percentage
        except (AttributeError, RuntimeError, TypeError, ZeroDivisionError):
            return 0

    @staticmethod
    def get_text_transform(label_settings: QgsPalLayerSettings = None) -> str:
        """Return ``text-transform`` (``"none"``, ``"uppercase"``, or ``"lowercase"``).

        Maps from the QGIS capitalisation setting when available.
        """
        if label_settings is None:
            return "none"
        try:
            cap = PropertyExtractor.get_attribute(
                label_settings.format(), "capitalization"
            )
        except (AttributeError, RuntimeError):
            return "none"
        # QgsStringUtils::Capitalization: 1=AllUppercase, 2=AllLowercase
        if cap == 1:
            return "uppercase"
        if cap == 2:
            return "lowercase"
        return "none"

    @staticmethod
    def get_text_max_width(label_settings: QgsPalLayerSettings) -> float:
        """Return ``text-max-width`` in ems.

        QGIS's ``autoWrapLength`` counts characters; an average character is
        about half an em wide. Without it, a very large value effectively
        disables wrapping.
        """
        if label_settings.autoWrapLength > 0:
            return label_settings.autoWrapLength * 0.5
        return 999

    @staticmethod
    def get_text_keep_upright() -> bool:
        """Return ``text-keep-upright`` (default ``True``)."""
        return True

    @staticmethod
    def get_text_rotate(label_settings: QgsPalLayerSettings) -> Union[float, List]:
        """Return ``text-rotate`` in degrees, honouring data-defined overrides."""
        base_rotation = label_settings.angleOffset if label_settings.angleOffset != 0 else 0
        rotation_prop = label_settings.dataDefinedProperties().property(
            QgsPalLayerSettings.Property.LabelRotation
        )
        return PropertyExtractor.get_value_or_expression(base_rotation, rotation_prop)

    @staticmethod
    def get_text_rotation_alignment() -> str:
        """Return ``text-rotation-alignment`` (default ``"map"``)."""
        return "map"

    @staticmethod
    def get_text_pitch_alignment() -> str:
        """Return ``text-pitch-alignment`` (default ``"viewport"``)."""
        return "viewport"

    @staticmethod
    def get_text_translate() -> List[float]:
        """Return ``text-translate`` ``[x, y]`` offset in pixels (default ``[0, 0]``)."""
        return [0, 0]

    @staticmethod
    def get_text_translate_anchor() -> str:
        """Return ``text-translate-anchor`` (MapLibre default: ``"map"``)."""
        return "map"


class QgisMapLibreStyleExporter:
    """Export QGIS Vector Tile Layer styles to a MapLibre GL style JSON.

    Walks a ``QgsVectorTileLayer``'s renderer and labelling, converting each
    style entry into the appropriate MapLibre layer (``fill``, ``line``,
    ``symbol``, etc.). Marker symbols and label background markers are
    registered with an internal sprite registry so the companion
    ``SpriteGenerator`` can render them as a sprite sheet alongside the
    emitted ``style.json``.
    """

    def __init__(
        self,
        output_dir: str,
        utils_dir,
        layer: Optional[QgsVectorTileLayer] = None,
        minzoom: int = 0,
        maxzoom: int = 14,
        extent=None,
    ):
        """Initialise the exporter.

        Args:
            output_dir:      Directory where ``style.json`` and the sprite
                             folder will be written.
            layer:           The vector tile layer to export. If ``None``,
                             the active layer in the QGIS interface is used.
        """
        self.output_dir = output_dir
        self.utils_dir = utils_dir
        self.marker_symbols: dict = {}
        self.marker_counter = 0
        self.glyphs = {}
        self.minzoom = minzoom
        self.maxzoom = maxzoom
        # The export extent (EPSG:3857), whose latitude sizes ground lengths.
        self.extent = extent
        self.layer = self._resolve_layer(layer)
        self.source_name = "q2vt_tiles"
        self.style = self._build_style_skeleton()
        self.style["layers"].append(self._build_background_layer())

    def _resolve_layer(self, layer: Optional[QgsVectorTileLayer]) -> QgsVectorTileLayer:
        """Return ``layer`` or fall back to the active QGIS layer.

        Raises:
            ValueError: If no layer is supplied and no active layer exists,
                        or if the resolved layer is not a ``QgsVectorTileLayer``.
        """
        if layer is None:
            try:
                if iface and iface.activeLayer():
                    layer = iface.activeLayer()
                else:
                    raise ValueError("No active layer found and no layer provided")
            except (ImportError, ValueError) as e:
                raise e
        if not isinstance(layer, QgsVectorTileLayer):
            raise ValueError(f"Layer must be a QgsVectorTileLayer, got {type(layer).__name__}")
        return layer

    def _build_style_skeleton(self) -> dict:
        """Build the empty MapLibre style document with sources and sprite refs."""
        return {
            "version": 8,
            "name": f"{self.source_name}_style",
            "sources": {
                self.source_name: {
                    "type": "vector",
                    "tiles": ["http://localhost:9000/tiles/tiles/{z}/{x}/{y}.pbf?v=10031993"],
                    # The highest zoom with tiles; viewers enlarge its tiles
                    # beyond it rather than request tiles that don't exist.
                    "maxzoom": max(0, min(self.maxzoom, _TILES_MAX_ZOOM)),
                }
            },
            "glyphs": "http://localhost:9000/style/glyphs/{fontstack}/{range}.pbf",
            "sprite": "http://localhost:9000/style/sprite/sprite",
            "layers": [],
        }

    def _build_background_layer(self) -> dict:
        """The style's bottom layer: the project's background colour.

        The viewers show a basemap (OpenStreetMap, NASA Blue Marble) above
        it from their own basemap switcher.
        """
        bg_color = PropertyExtractor.convert_qcolor_to_maplibre(
            QgsProject.instance().backgroundColor()
        )
        return {
            "id": "background", "type": "background", "minzoom": 0, "maxzoom": 22,
            "paint": {"background-color": bg_color},
        }

    def export(self) -> Dict[str, Any]:
        """Convert all styles and labelling on the layer and write to disk.

        Returns:
            The complete MapLibre style dictionary.
        """
        self.convert()
        self.save_to_file()
        return self.style

    def convert(self) -> Dict[str, Any]:
        """Convert all styles and labelling on the layer, without writing
        anything (save_to_file() then writes the style, sprites and glyphs).

        Returns:
            The MapLibre style dictionary.
        """
        self.layer.source()
        self._set_map_units()

        renderer = self.layer.renderer()
        if isinstance(renderer, QgsVectorTileBasicRenderer):
            for style in renderer.styles():
                self._convert_renderer_style(style)

        labeling = self.layer.labeling()
        if isinstance(labeling, QgsVectorTileBasicLabeling):
            for style in labeling.styles():
                self._convert_labeling_style(style)

        self.style["layers"] = merge_rule_layers(self.style["layers"])
        return self.style

    def _set_map_units(self) -> None:
        """Size map-unit lengths for the project's CRS at the tiles' latitude."""
        latitude = 0.0
        try:
            # The tiles don't exist yet when the style is exported, so the
            # tiles layer has no extent: the export extent gives the latitude.
            extent, crs = self.extent, QgsCoordinateReferenceSystem("EPSG:3857")
            if extent is None or extent.isEmpty():
                extent, crs = self.layer.extent(), self.layer.crs()
            if not extent.isEmpty():
                to_wgs84 = QgsCoordinateTransform(
                    crs, QgsCoordinateReferenceSystem("EPSG:4326"), QgsProject.instance()
                )
                latitude = to_wgs84.transform(extent.center()).y()
        except (QgsCsException, RuntimeError, AttributeError):  # Equator sizes then.
            latitude = 0.0
        PropertyExtractor.set_map_units(QgsProject.instance().crs(), latitude)

    def _convert_renderer_style(self, style):
        """Convert a single ``QgsVectorTileBasicRendererStyle`` into MapLibre layer(s)."""
        if not style.isEnabled() or not style.symbol():
            return
        min_zoom = style.minZoomLevel()
        max_zoom = style.maxZoomLevel() + 1
        first = len(self.style["layers"])
        self._convert_symbol(
            style.symbol(), style.styleName(), style.layerName(),
            self.source_name, min_zoom, max_zoom,
        )
        self._apply_rule_filter(style, first)

    def _convert_labeling_style(self, style):
        """Convert a single ``QgsVectorTileBasicLabelingStyle`` into a MapLibre symbol layer."""
        if not style.isEnabled() or not style.labelSettings():
            return
        min_zoom = style.minZoomLevel()
        max_zoom = style.maxZoomLevel() + 1
        first = len(self.style["layers"])
        self._convert_label(
            style.labelSettings(), style.styleName(), style.layerName(),
            self.source_name, min_zoom, max_zoom,
        )
        self._apply_rule_filter(style, first)

    def _apply_rule_filter(self, style, first: int) -> None:
        """Filter the layers converted from ``style`` (from index ``first``)
        to its rule's rows, in a dataset shared by several rules."""
        match = _RULE_FILTER.fullmatch(style.filterExpression() or "")
        if not match:
            return
        for layer_def in self.style["layers"][first:]:
            layer_def["filter"] = ["==", ["get", RULE_FIELD], int(match.group(1))]

    def _convert_symbol(
        self,
        symbol: QgsSymbol,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Dispatch a QGIS symbol to the correct conversion routine by symbol type."""
        if symbol.symbolLayerCount() == 0:
            return
        symbol_layer = symbol.symbolLayer(0)
        symbol_type = symbol.type()

        if symbol_type == QgsSymbol.SymbolType.Marker:
            self._convert_marker_symbol(
                symbol_layer, symbol, style_name, source_layer_name,
                source_name, min_zoom, max_zoom,
            )
        elif symbol_type == QgsSymbol.SymbolType.Line:
            self._convert_line_symbol(
                symbol, style_name, source_layer_name, source_name, min_zoom, max_zoom
            )
        elif symbol_type == QgsSymbol.SymbolType.Fill:
            self._convert_fill_symbol(
                symbol_layer, symbol, style_name, source_layer_name, source_name, min_zoom, max_zoom
            )

    def _base_layer_def(
        self,
        layer_type: str,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int,
        max_zoom: int,
    ) -> dict:
        """Build a common MapLibre layer-definition skeleton with id/source/zoom range."""
        layer_def: dict = {
            "id": style_name,
            "type": layer_type,
            "source": source_name,
            "source-layer": source_layer_name,
            "paint": {},
            "layout": {},
        }
        if min_zoom >= 0:
            layer_def["minzoom"] = min_zoom
        if max_zoom >= 0:
            layer_def["maxzoom"] = max_zoom
        return layer_def

    def _register_pattern(self, symbol_or_layer) -> str:
        """Register a pattern image (or sub-symbol) in the sprite registry.

        Used by line/fill pattern conversion. Returns the unique pattern name
        the MapLibre layer should reference via ``line-pattern`` or
        ``fill-pattern``.
        """
        pattern_name = f"pattern_{self.marker_counter}"
        self.marker_counter += 1
        try:
            symbol_or_layer = symbol_or_layer.clone()
        except (AttributeError, RuntimeError):
            pass
        if FillPropertyExtractor.is_pattern_fill(symbol_or_layer):
            width, height = FillPropertyExtractor.get_pattern_tile_size(symbol_or_layer)
            symbol_or_layer = FillPattern(symbol_or_layer, width, height)
        self.marker_symbols[pattern_name] = symbol_or_layer
        return pattern_name

    def _convert_marker_symbol(
        self,
        symbol_layer: QgsSymbolLayer,
        symbol: QgsSymbol,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a QGIS marker symbol into a MapLibre ``symbol`` layer."""
        layer_def = self._base_layer_def(
            "symbol", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        marker_name = f"marker_{self.marker_counter}"
        self.marker_counter += 1
        sprite_symbol, ground_icon_size = self._sprite_marker(symbol)
        self.marker_symbols[marker_name] = sprite_symbol

        layer_def["layout"].update({
            "icon-image": IconPropertyExtractor.get_icon_image(marker_name),
            "icon-size": ground_icon_size or IconPropertyExtractor.get_icon_size(symbol_layer, 1.0),
            "icon-rotate": IconPropertyExtractor.get_icon_rotate(symbol_layer=symbol_layer),
            "icon-padding": IconPropertyExtractor.get_icon_padding(),
            "icon-rotation-alignment": IconPropertyExtractor.get_icon_rotation_alignment(),
            "icon-pitch-alignment": IconPropertyExtractor.get_icon_pitch_alignment(),
            "icon-anchor": IconPropertyExtractor.get_icon_anchor(),
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(True),
            "icon-ignore-placement": IconPropertyExtractor.get_icon_ignore_placement(),
            "icon-optional": IconPropertyExtractor.get_icon_optional(),
            "icon-keep-upright": IconPropertyExtractor.get_icon_keep_upright(),
            "symbol-placement": IconPropertyExtractor.get_symbol_placement(),
            "symbol-spacing": IconPropertyExtractor.get_symbol_spacing(),
            "symbol-avoid-edges": IconPropertyExtractor.get_symbol_avoid_edges(),
            "symbol-sort-key": IconPropertyExtractor.get_symbol_sort_key(),
            "symbol-z-order": IconPropertyExtractor.get_symbol_z_order(),
            "visibility": "visible",
        })
        layer_def["paint"].update({
            "icon-opacity": IconPropertyExtractor.get_icon_opacity(),
            "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(),
            "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(),
            "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
            "icon-translate": IconPropertyExtractor.get_icon_translate(),
            "icon-translate-anchor": IconPropertyExtractor.get_icon_translate_anchor(),
        })

        self.style["layers"].append(layer_def)

    # Size a marker sized on the ground is drawn at in the sprite, in mm;
    # the style scales it to its ground size at each zoom.
    _GROUND_MARKER_SPRITE_MM = 6.0

    @classmethod
    def _sprite_marker(cls, symbol: QgsSymbol) -> tuple:
        """(the symbol as the sprite draws it, its icon-size or None).

        The sprite is drawn unrotated: the style rotates it by the first
        symbol layer's angle (icon-rotate), which would otherwise apply
        twice; the other layers keep their angles relative to it.

        A marker sized on the ground (map units or metres) has no pixel
        size: it is drawn at a fixed size, and its icon-size grows with
        the zoom so that it covers its ground size, as in QGIS.
        """
        sprite_symbol = symbol.clone()
        try:
            if sprite_symbol.type() == QgsSymbol.SymbolType.Marker:
                sprite_symbol.setAngle(0)
        except (AttributeError, RuntimeError):
            pass
        try:
            unit = sprite_symbol.sizeUnit()
            size = sprite_symbol.size()
        except (AttributeError, RuntimeError):
            return sprite_symbol, None
        if not PropertyExtractor.is_map_unit(unit) or not size or size <= 0:
            return sprite_symbol, None
        ratio = cls._GROUND_MARKER_SPRITE_MM / size
        for layer in sprite_symbol.symbolLayers():
            width = PropertyExtractor.get_attribute(layer, "strokeWidth")
            stroke_unit = PropertyExtractor.get_attribute(layer, "strokeWidthUnit")
            if isinstance(width, (int, float)) and PropertyExtractor.is_map_unit(stroke_unit):
                layer.setStrokeWidth(width * ratio)
        sprite_symbol.setOutputUnit(Qgis.RenderUnit.Millimeters)
        sprite_symbol.setSize(cls._GROUND_MARKER_SPRITE_MM)
        sprite_px = PropertyExtractor.convert_length_to_pixels(cls._GROUND_MARKER_SPRITE_MM, "mm")
        ground = PropertyExtractor.length_to_style_value(size, unit)  # Pixels by zoom.
        if not isinstance(ground, list):
            return sprite_symbol, None
        icon_size = list(ground)
        icon_size[4] = ground[4] / sprite_px / _SPRITE_QUALITY
        icon_size[6] = ground[6] / sprite_px / _SPRITE_QUALITY
        return sprite_symbol, icon_size

    def _convert_line_symbol(
        self,
        symbol: QgsSymbol,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert every symbol layer of a QGIS line symbol into MapLibre layer(s).

        A QGIS line symbol can stack multiple symbol layers - typically a
        base ``QgsSimpleLineSymbolLayer`` stroke plus one or more
        ``QgsMarkerLineSymbolLayer`` marker decorations. Each sub-layer is
        converted independently and appended to ``self.style["layers"]`` in
        the same bottom-to-top order QGIS uses for its symbol layers, so
        marker decorations correctly render above or below the base stroke
        exactly as they do in QGIS.
        """
        for index in range(symbol.symbolLayerCount()):
            symbol_layer = symbol.symbolLayer(index)
            # The first sub-layer keeps the original style name (so existing
            # look-ups by id keep working); subsequent sub-layers get a
            # unique suffixed id since MapLibre layer ids must be unique.
            layer_id = style_name if index == 0 else f"{style_name}_layer{index}"

            if (LinePropertyExtractor.is_marker_line(symbol_layer)
                    or LinePropertyExtractor.is_hashed_line(symbol_layer)):
                self._convert_marker_line_symbol_layer(
                    symbol_layer, layer_id, source_layer_name, source_name,
                    min_zoom, max_zoom,
                )
            else:
                self._convert_simple_or_pattern_line_symbol_layer(
                    symbol_layer, symbol, layer_id, source_layer_name, source_name,
                    min_zoom, max_zoom,
                )

    def _convert_marker_line_symbol_layer(
        self,
        symbol_layer: QgsSymbolLayer,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a QGIS ``QgsMarkerLineSymbolLayer`` into a MapLibre ``symbol`` layer.

        The marker line's sub-symbol is rendered to a sprite (registered via
        ``self.marker_symbols`` exactly like a standalone marker symbol) and
        referenced through ``icon-image``. QGIS's interval/vertex placement
        is mapped to ``symbol-placement``/``symbol-spacing``,
        ``rotateSymbols()`` controls ``icon-rotation-alignment``, and the
        marker line's perpendicular ``offset()`` is reproduced via
        ``icon-offset`` so markers displaced from the line stay displaced
        the same way in MapLibre.
        """
        sub_symbol = None
        try:
            sub_symbol = symbol_layer.subSymbol()
        except (AttributeError, RuntimeError):
            pass
        if sub_symbol is None or sub_symbol.symbolLayerCount() == 0:
            return

        layer_def = self._base_layer_def(
            "symbol", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        marker_name = f"marker_{self.marker_counter}"
        self.marker_counter += 1
        icon_rotate = 0
        if LinePropertyExtractor.is_hashed_line(symbol_layer):
            # A hashed line: a short stroke of its line sub-symbol, drawn
            # across the line (at the hash angle) at every interval.
            length = PropertyExtractor.convert_length_to_pixels(
                symbol_layer.hashLength(),
                PropertyExtractor.get_attribute(symbol_layer, "hashLengthUnit"),
            )
            self.marker_symbols[marker_name] = HashMark(sub_symbol.clone(), max(length, 1.0))
            icon_rotate = symbol_layer.hashAngle()
            icon_size = 1.0 / _SPRITE_QUALITY
        else:
            self.marker_symbols[marker_name] = sub_symbol.clone()
            icon_size = IconPropertyExtractor.get_icon_size(sub_symbol.symbolLayer(0), 1.0)

        rotate_with_line = LinePropertyExtractor.get_marker_line_rotate_symbols(symbol_layer)
        offset_px = LinePropertyExtractor.get_marker_line_offset(symbol_layer)

        layer_def["layout"].update({
            "icon-image": IconPropertyExtractor.get_icon_image(marker_name),
            "icon-size": icon_size,
            "icon-rotate": icon_rotate,
            "icon-padding": IconPropertyExtractor.get_icon_padding(),
            # Following the line's bearing ("map") reproduces rotateSymbols()
            # == True; "viewport" keeps markers upright regardless of the
            # line's direction, matching rotateSymbols() == False.
            "icon-rotation-alignment": "map" if rotate_with_line else "viewport",
            "icon-pitch-alignment": IconPropertyExtractor.get_icon_pitch_alignment(),
            "icon-anchor": IconPropertyExtractor.get_icon_anchor(),
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(True),
            # QGIS marker lines stamp every marker unconditionally - there is
            # no collision system in QGIS to reproduce. icon-allow-overlap
            # alone only stops a marker from being hidden by collisions;
            # each repeat along the line is still registered in MapLibre's
            # collision index and will suppress its own later siblings (and
            # be suppressed by them) unless icon-ignore-placement is also
            # true. Both flags together are required to fully disable
            # collision-based hiding, matching QGIS's "always show" look.
            "icon-ignore-placement": True,
            "icon-optional": IconPropertyExtractor.get_icon_optional(),
            "icon-keep-upright": False,
            "symbol-placement": LinePropertyExtractor.get_marker_line_symbol_placement(symbol_layer),
            "symbol-spacing": LinePropertyExtractor.get_marker_line_spacing(symbol_layer),
            "symbol-avoid-edges": IconPropertyExtractor.get_symbol_avoid_edges(),
            "symbol-sort-key": IconPropertyExtractor.get_symbol_sort_key(),
            "symbol-z-order": IconPropertyExtractor.get_symbol_z_order(),
            "visibility": "visible",
        })
        if offset_px:
            # icon-offset is applied in the icon's own pixel space and
            # rotates together with icon-rotate/icon-rotation-alignment, so
            # when the icon follows the line bearing this reproduces QGIS's
            # perpendicular marker-to-line offset.
            icon_size_value = layer_def["layout"]["icon-size"]
            if isinstance(icon_size_value, (int, float)) and icon_size_value:
                compensated_offset = offset_px / icon_size_value
            else:
                # icon-size is data-defined (rare: only when the QGIS marker
                # sub-symbol itself has a per-feature size override). There is
                # no way to build a dynamic two-element icon-offset array in
                # the MapLibre expression language, so fall back to the
                # sprite-quality constant used for the non-data-defined case.
                compensated_offset = offset_px * _SPRITE_QUALITY
            layer_def["layout"]["icon-offset"] = [0, compensated_offset]

        layer_def["paint"].update({
            "icon-opacity": IconPropertyExtractor.get_icon_opacity(),
            "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(),
            "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(),
            "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
            "icon-translate": IconPropertyExtractor.get_icon_translate(),
            "icon-translate-anchor": IconPropertyExtractor.get_icon_translate_anchor(),
        })

        self.style["layers"].append(layer_def)

    def _convert_simple_or_pattern_line_symbol_layer(
        self,
        symbol_layer: QgsSymbolLayer,
        symbol: QgsSymbol,          # add this
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a plain-stroke or raster-pattern line symbol layer to MapLibre ``line``."""
        layer_def = self._base_layer_def(
            "line", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        if isinstance(symbol_layer, QgsSimpleLineSymbolLayer):
            layer_def["paint"].update({
                "line-color": LinePropertyExtractor.get_line_color(symbol_layer),
                "line-width": LinePropertyExtractor.get_line_width(symbol_layer),
                "line-opacity": LinePropertyExtractor.get_line_opacity(symbol_layer, symbol),
                "line-blur": LinePropertyExtractor.get_line_blur(),
                "line-gap-width": LinePropertyExtractor.get_line_gap_width(),
                "line-translate": LinePropertyExtractor.get_line_translate(),
                "line-translate-anchor": LinePropertyExtractor.get_line_translate_anchor(),
            })

            offset = LinePropertyExtractor.get_line_offset(symbol_layer)
            # Emit offset only when non-zero (preserve compact output) or when
            # the offset is data-defined (a list expression, never numerically zero).
            if isinstance(offset, list) or (isinstance(offset, (int, float)) and offset != 0):
                layer_def["paint"]["line-offset"] = offset

            width_value = layer_def["paint"]["line-width"]
            width_px = width_value if isinstance(width_value, (int, float)) else 1.0
            dasharray = LinePropertyExtractor.get_line_dasharray(symbol_layer, width_px)
            if dasharray:
                layer_def["paint"]["line-dasharray"] = dasharray

            layer_def["layout"].update({
                "line-cap": LinePropertyExtractor.get_line_cap(symbol_layer),
                "line-join": LinePropertyExtractor.get_line_join(symbol_layer),
                "line-miter-limit": LinePropertyExtractor.get_line_miter_limit(),
                "line-round-limit": LinePropertyExtractor.get_line_round_limit(),
                "line-sort-key": LinePropertyExtractor.get_line_sort_key(symbol_layer),
                "visibility": "visible",
            })
        elif (LinePropertyExtractor.is_pattern_line(symbol_layer)
              and LinePropertyExtractor.get_line_pattern_path(symbol_layer)):
            # Raster line: its image repeated along the line, at the line's
            # width (MapLibre draws a pattern at the line's width).
            width = LinePropertyExtractor.get_line_width(symbol_layer)
            height = width if isinstance(width, (int, float)) and width > 0 else 4.0
            pattern_name = f"pattern_{self.marker_counter}"
            self.marker_counter += 1
            self.marker_symbols[pattern_name] = LinePattern(
                LinePropertyExtractor.get_line_pattern_path(symbol_layer), height
            )
            layer_def["paint"].update({
                "line-pattern": pattern_name,
                "line-width": width,
                "line-opacity": LinePropertyExtractor.get_line_opacity(symbol_layer, symbol),
                "line-translate": LinePropertyExtractor.get_line_translate(),
                "line-translate-anchor": LinePropertyExtractor.get_line_translate_anchor(),
            })
            layer_def["layout"].update({
                "line-cap": "round",
                "line-join": "round",
                "line-miter-limit": LinePropertyExtractor.get_line_miter_limit(),
                "line-round-limit": LinePropertyExtractor.get_line_round_limit(),
                "visibility": "visible",
            })
        else:
            # A line MapLibre can't draw (burst, arrow, interpolated, ...):
            # a plain line of its colour and width.
            color, width = LinePropertyExtractor.get_fallback_line(symbol_layer)
            layer_def["paint"].update({
                "line-color": color,
                "line-width": width,
                "line-opacity": LinePropertyExtractor.get_line_opacity(symbol_layer, symbol),
            })
            layer_def["layout"].update({
                "line-cap": "round", "line-join": "round", "visibility": "visible",
            })

        self.style["layers"].append(layer_def)

    def _convert_fill_symbol(
        self,
        symbol_layer: QgsSymbolLayer,
        symbol: QgsSymbol,          # add this
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a QGIS fill symbol into a MapLibre ``fill`` layer."""
        layer_def = self._base_layer_def(
            "fill", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        if isinstance(symbol_layer, QgsSimpleFillSymbolLayer):
            if symbol_layer.brushStyle() != Qt.BrushStyle.NoBrush:
                layer_def["paint"].update({
                    "fill-color": FillPropertyExtractor.get_fill_color(symbol_layer),
                    "fill-opacity": FillPropertyExtractor.get_fill_opacity(symbol_layer, symbol),
                    "fill-antialias": FillPropertyExtractor.get_fill_antialias(),
                    "fill-translate": FillPropertyExtractor.get_fill_translate(),
                    "fill-translate-anchor": FillPropertyExtractor.get_fill_translate_anchor(),
                })

            else:
                layer_def["paint"].update({
                    "fill-color": "rgba(0, 0, 0, 0.0)"})
            if symbol_layer.strokeStyle() != Qt.PenStyle.NoPen:
                outline_color = FillPropertyExtractor.get_fill_outline_color(symbol_layer)
                if outline_color:
                    layer_def["paint"]["fill-outline-color"] = outline_color

                layer_def["layout"].update({
                    "fill-sort-key": FillPropertyExtractor.get_fill_sort_key(symbol_layer),
                    "visibility": "visible",
                })
        elif FillPropertyExtractor.is_pattern_fill(symbol_layer):
            # Pattern-based fill: emit fill-pattern; fill-color is ignored by MapLibre.
            pattern_name = self._register_pattern(symbol_layer)
            layer_def["paint"].update({
                "fill-pattern": pattern_name,
                "fill-opacity": FillPropertyExtractor.get_fill_opacity(symbol_layer, symbol),
                "fill-antialias": FillPropertyExtractor.get_fill_antialias(),
                "fill-translate": FillPropertyExtractor.get_fill_translate(),
                "fill-translate-anchor": FillPropertyExtractor.get_fill_translate_anchor(),
            })
            layer_def["layout"].update({
                "fill-sort-key": FillPropertyExtractor.get_fill_sort_key(symbol_layer),
                "visibility": "visible",
            })
        else:
            # A fill MapLibre can't draw (gradient, shapeburst, ...): a flat
            # fill of its mean colour.
            layer_def["paint"].update({
                "fill-color": FillPropertyExtractor.get_fallback_fill_color(symbol_layer),
                "fill-opacity": FillPropertyExtractor.get_fill_opacity(symbol_layer, symbol),
                "fill-antialias": FillPropertyExtractor.get_fill_antialias(),
            })
            layer_def["layout"]["visibility"] = "visible"

        self.style["layers"].append(layer_def)

    def _convert_label(
        self,
        label_settings: QgsPalLayerSettings,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a ``QgsPalLayerSettings`` into a MapLibre ``symbol`` layer."""
        layer_def = self._base_layer_def(
            "symbol", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        text_format = label_settings.format()

        text_field = TextPropertyExtractor.get_text_field(label_settings)
        if text_field:
            layer_def["layout"]["text-field"] = text_field
        font = TextPropertyExtractor.get_text_font(text_format)
        dataset = join(self.utils_dir, f'{source_layer_name}.{_TEMP_RULE_FORMAT}')
        if self.glyphs.get(font):
            self.glyphs[font].append(dataset)
        else:
            self.glyphs[font] = [dataset]
        layer_def["layout"].update({
            "text-font": [font],
            "text-size": TextPropertyExtractor.get_text_size(text_format, label_settings),
            "text-anchor": TextPropertyExtractor.get_text_anchor(label_settings),
            "text-justify": TextPropertyExtractor.get_text_justify(label_settings),
            "text-offset": TextPropertyExtractor.get_text_offset(label_settings),
            "text-allow-overlap": TextPropertyExtractor.get_text_allow_overlap(label_settings),
            "text-ignore-placement": TextPropertyExtractor.get_text_ignore_placement(),
            "text-optional": TextPropertyExtractor.get_text_optional(),
            "text-padding": TextPropertyExtractor.get_text_padding(),
            "text-line-height": TextPropertyExtractor.get_text_line_height(text_format),
            "text-letter-spacing": TextPropertyExtractor.get_text_letter_spacing(text_format),
            "text-transform": TextPropertyExtractor.get_text_transform(label_settings),
            "text-max-width": TextPropertyExtractor.get_text_max_width(label_settings),
            "text-max-angle": TextPropertyExtractor.get_text_max_angle(label_settings),
            "text-keep-upright": TextPropertyExtractor.get_text_keep_upright(),
            "text-rotate": TextPropertyExtractor.get_text_rotate(label_settings),
            "text-rotation-alignment": TextPropertyExtractor.get_text_rotation_alignment(),
            "text-pitch-alignment": TextPropertyExtractor.get_text_pitch_alignment(),
            "symbol-placement": IconPropertyExtractor.get_symbol_placement(label_settings),
            "symbol-spacing": IconPropertyExtractor.get_symbol_spacing(label_settings),
            "symbol-avoid-edges": IconPropertyExtractor.get_symbol_avoid_edges(),
            "symbol-sort-key": TextPropertyExtractor.get_text_sort_key(label_settings),
            "symbol-z-order": IconPropertyExtractor.get_symbol_z_order(),
            "visibility": "visible",
        })

        # Placements that let QGIS choose the side of the point. MapLibre
        # uses a radial offset instead of text-offset with them.
        variable_anchor = TextPropertyExtractor.get_text_variable_anchor(label_settings)
        if variable_anchor:
            radial = TextPropertyExtractor.get_text_radial_offset(label_settings)
            layer_def["layout"]["text-variable-anchor"] = variable_anchor
            layer_def["layout"]["text-radial-offset"] = radial
            # Renderers without variable anchors (OpenLayers' ol-mapbox-style,
            # QGIS's style converter) read text-anchor and text-offset
            # instead: the first position, at the same distance. MapLibre
            # ignores both when the variable anchors are set.
            layer_def["layout"]["text-anchor"] = variable_anchor[0]
            layer_def["layout"]["text-offset"] = TextPropertyExtractor.radial_to_offset(
                variable_anchor[0], radial
            )


        #         "text-halo-blur": TextPropertyExtractor.get_text_halo_blur(
        #     text_format, label_settings
        # ),
        layer_def["paint"].update({
            "text-color": TextPropertyExtractor.get_text_color(text_format, label_settings),
            "text-opacity": TextPropertyExtractor.get_text_opacity(text_format, label_settings),
            "text-halo-color": TextPropertyExtractor.get_text_halo_color(
                text_format, label_settings
            ),
            "text-halo-width": TextPropertyExtractor.get_text_halo_width(text_format, label_settings),
            "text-translate": TextPropertyExtractor.get_text_translate(),
            "text-translate-anchor": TextPropertyExtractor.get_text_translate_anchor(),
        })

        # Deep-copy label settings to avoid mutating the caller's object.
        label_settings = QgsPalLayerSettings(label_settings)
        label_format = QgsTextFormat(label_settings.format())
        background = QgsTextBackgroundSettings(label_format.background())
        if background.markerSymbol():
            background.setMarkerSymbol(background.markerSymbol().clone())
        label_format.setBackground(background)
        label_settings.setFormat(label_format)

        if background.enabled() and background.markerSymbol():
            self._apply_icon_from_background(layer_def, background, style_name)
        else:
            self._apply_default_icon_props(layer_def)

        self.style["layers"].append(layer_def)

    def _apply_icon_from_background(self, layer_def: dict, background, style_name: str):
        """Configure icon layout/paint from a label background marker symbol."""
        try:
            marker = background.markerSymbol() if hasattr(background, "markerSymbol") else None
            if marker and marker.type() == QgsSymbol.SymbolType.Marker:
                marker_name = f"marker_{self.marker_counter}"
                self.marker_counter += 1
                self.marker_symbols[marker_name] = marker.clone()
                layer_def["layout"]["icon-image"] = marker_name
            else:
                layer_def["layout"]["icon-image"] = style_name
        except (RuntimeError, AttributeError):
            layer_def["layout"]["icon-image"] = style_name

        text_fit = IconPropertyExtractor.get_icon_text_fit(background)
        if text_fit:
            layer_def["layout"]["icon-text-fit"] = text_fit

        text_fit_padding = IconPropertyExtractor.get_icon_text_fit_padding(background)
        if text_fit_padding:
            layer_def["layout"]["icon-text-fit-padding"] = text_fit_padding

        layer_def["layout"].update({
            "icon-anchor": IconPropertyExtractor.get_icon_anchor(),
            "icon-rotate": IconPropertyExtractor.get_icon_rotate(background=background),
            "icon-padding": IconPropertyExtractor.get_icon_padding(),
            "icon-rotation-alignment": IconPropertyExtractor.get_icon_rotation_alignment(),
            "icon-pitch-alignment": IconPropertyExtractor.get_icon_pitch_alignment(),
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(),
            "icon-ignore-placement": IconPropertyExtractor.get_icon_ignore_placement(),
            "icon-keep-upright": IconPropertyExtractor.get_icon_keep_upright(),
            "icon-offset": IconPropertyExtractor.get_icon_offset(background),
        })

        try:
            layer_def["paint"].update({
                "icon-opacity": IconPropertyExtractor.get_icon_opacity(background),
                "icon-color": IconPropertyExtractor.get_icon_color(background),
                "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(background),
                "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(background),
                "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
                "icon-translate": IconPropertyExtractor.get_icon_translate(),
                "icon-translate-anchor": IconPropertyExtractor.get_icon_translate_anchor(),
            })
        except (OSError, RuntimeError):
            pass

    def _apply_default_icon_props(self, layer_def: dict):
        """Apply default icon layout/paint when no background marker is present."""
        layer_def["layout"].update({
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(),
            "icon-ignore-placement": IconPropertyExtractor.get_icon_ignore_placement(),
            "icon-optional": IconPropertyExtractor.get_icon_optional(),
            "icon-padding": IconPropertyExtractor.get_icon_padding(),
        })
        layer_def["paint"].update({
            "icon-opacity": IconPropertyExtractor.get_icon_opacity(),
            "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(),
            "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(),
            "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
            "icon-translate": IconPropertyExtractor.get_icon_translate(),
            "icon-translate-anchor": IconPropertyExtractor.get_icon_translate_anchor(),
        })

    def to_json(self, indent: int = 2) -> str:
        """Serialise the in-memory style dict to a JSON string."""
        return json.dumps(self.style, indent=indent)

    def save_to_file(self, filename: str = "style.json", indent: int = 2) -> str:
        """Write the style JSON and the sprite sheet to the output directory.

        Args:
            filename: Filename for the JSON file (relative to ``style/``).
            indent:   JSON indentation level for human-readable output.

        Returns:
            The absolute path of the written ``style.json`` file.
        """
        style_dir = os.path.join(self.output_dir, "style")
        os.makedirs(style_dir, exist_ok=True)

        if self.marker_symbols:
            SpriteGenerator(
                self.marker_symbols, style_dir, _SPRITE_QUALITY, False
            ).generate()
        else:
            del self.style["sprite"]
        if self.glyphs:
            glyphs_dir = os.path.join(self.output_dir, "style", "glyphs")
            os.mkdir(glyphs_dir)
            fontstacks = GlyphGenerator(self.glyphs, 'q2vt_label', glyphs_dir).generate()
            self._use_fontstacks(fontstacks)
        else:
             del self.style["glyphs"]
        rounded_style = self.round_numeric_values(self.style)
        filepath = os.path.join(style_dir, filename)
        with open(filepath, "w", encoding="utf8") as f:
            json.dump(rounded_style, f, indent=indent)
        return filepath
    
    def _use_fontstacks(self, fontstacks: Dict[str, str]) -> None:
        """Point each label's text-font at the glyph folder written for it.

        A font without a style name (e.g. "MS Shell Dlg 2 ") would otherwise
        request glyphs from a folder that doesn't exist.
        """
        for layer_def in self.style["layers"]:
            layout = layer_def.get("layout", {})
            fonts = layout.get("text-font")
            if fonts:
                layout["text-font"] = [fontstacks.get(font, font.strip()) for font in fonts]

    def round_numeric_values(self, obj):
        """
        Recursively walk through dict/list values.
        Convert anything numeric-like to float and round to 2 decimals.
        """
        if isinstance(obj, dict):
            return {k: self.round_numeric_values(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.round_numeric_values(item) for item in obj]
        if isinstance(obj, bool):
            return obj
        try:
            # Try converting strings, ints, floats, etc.
            num = float(obj)

            # Keep integers as int if possible after rounding. Small values
            # (a map-unit width at zoom 0) keep 3 significant digits instead.
            rounded = round(num, 2) if abs(num) >= 0.1 or num == 0 else float(f"{num:.3g}")
            return int(rounded) if rounded.is_integer() else rounded

        except (ValueError, TypeError):
            # Not numeric → keep original
            return obj



if __name__ == "__console__":
    exporter = QgisMapLibreStyleExporter(output_dir=QgsProcessingUtils.tempFolder())
    output_file = exporter.export()