"""Convert QGIS Vector Tile Layer styles to MapLibre GL JSON style format."""

import json
import os
from typing import Any, Dict, List, Optional, Union

from qgis.PyQt.QtGui import QColor
from qgis.core import (
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
from .sprite_generator import SpriteGenerator


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
            if evaluation is not None and not qexpr.needsGeometry():
                return evaluation
            field_name = expression.replace('"', "")
            idx = field_name.find("q2vt")
            if field_name and idx != -1:
                return ["get", field_name[idx: idx + 13]]
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
        color_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.PropertyFillColor)
        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_line_width(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[float, List]:
        """Return ``line-width`` in pixels, resolving any data-defined override."""
        width_unit = PropertyExtractor.get_attribute(
            symbol_layer, "widthUnit", "widthUnits", "strokeWidthUnit", "strokeWidthUnits"
        )
        width_px = PropertyExtractor.convert_length_to_pixels(symbol_layer.width(), width_unit)
        width_prop = symbol_layer.dataDefinedProperties().property(
            QgsSymbolLayer.PropertyStrokeWidth
        )
        return PropertyExtractor.get_value_or_expression(width_px, width_prop)

    @staticmethod
    def get_line_opacity(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[float, List]:
        """Return ``line-opacity`` from the colour alpha channel.

        Honours a data-defined ``PropertyOpacity`` override when active, so
        per-feature opacity attributes are emitted as MapLibre expressions.
        """
        base_opacity = symbol_layer.color().alphaF()
        try:
            opacity_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.PropertyOpacity
            )
            return PropertyExtractor.get_value_or_expression(base_opacity, opacity_prop)
        except (AttributeError, RuntimeError):
            return base_opacity

    @staticmethod
    def get_line_cap(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Return ``line-cap`` mapped from the Qt pen-cap style."""
        cap_map = {0: "butt", 16: "square", 32: "round"}
        return cap_map.get(symbol_layer.penCapStyle(), "butt")

    @staticmethod
    def get_line_join(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Return ``line-join`` mapped from the Qt pen-join style."""
        join_map = {0: "miter", 64: "bevel", 128: "round"}
        return join_map.get(symbol_layer.penJoinStyle(), "miter")

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
        Lengths are converted to pixels using the dash unit if available,
        falling back to the line-width unit. When no dashing is configured,
        ``None`` is returned and ``line-dasharray`` should be omitted.
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
                symbol_layer, "dashUnit", "dashUnits", "customDashUnits"
            )
            width_unit = PropertyExtractor.get_attribute(
                symbol_layer, "widthUnit", "widthUnits", "strokeWidthUnit", "strokeWidthUnits"
            )
            return [
                PropertyExtractor.convert_length_to_pixels(d, dash_unit or width_unit)
                for d in dash_vector
            ]

        fixed_presets = {
            2: [4 * width_px, 2 * width_px],
            3: [1 * width_px, 1 * width_px],
            4: [4 * width_px, 2 * width_px, 1 * width_px, 2 * width_px],
            5: [4 * width_px, 2 * width_px, 1 * width_px, 2 * width_px, 1 * width_px, 2 * width_px],
        }
        if pen_style in fixed_presets:
            return fixed_presets[pen_style]

        return None

    @staticmethod
    def get_line_offset(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[float, List]:
        """Return ``line-offset`` in pixels (positive values offset to the right).

        Honours a data-defined ``PropertyOffset`` override when active. Note
        that data-defined offsets are emitted in QGIS units (pixels for the
        static path); for field-based overrides the user must supply pixel
        values to match the MapLibre rendering convention.
        """
        base_offset = 0.0
        try:
            offset = symbol_layer.offset()
            if offset != 0:
                offset_unit = PropertyExtractor.get_attribute(
                    symbol_layer, "offsetUnit", "offsetUnits"
                )
                base_offset = PropertyExtractor.convert_length_to_pixels(offset, offset_unit)
        except (RuntimeError, AttributeError):
            pass

        try:
            offset_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.PropertyOffset
            )
            return PropertyExtractor.get_value_or_expression(base_offset, offset_prop)
        except (AttributeError, RuntimeError):
            return base_offset

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
                QgsSymbolLayer.PropertyLayerEnabled
            )
        except (AttributeError, RuntimeError):
            pass
        return PropertyExtractor.get_value_or_expression(0, order_prop)

    @staticmethod
    def is_pattern_line(symbol_layer: QgsSymbolLayer) -> bool:
        """Return ``True`` if this line symbol layer represents a raster pattern.

        Pattern detection is performed by class-name inspection so QGIS
        versions lacking a particular subclass remain compatible.
        """
        pattern_class_names = {
            "QgsRasterLineSymbolLayer",
            "QgsLineburstSymbolLayer",
        }
        return type(symbol_layer).__name__ in pattern_class_names

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


class FillPropertyExtractor:
    """Extract fill paint and layout properties from QGIS fill symbol layers."""

    @staticmethod
    def get_fill_color(symbol_layer: QgsSimpleFillSymbolLayer) -> Union[str, List]:
        """Return ``fill-color`` resolving any data-defined override."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.color())
        color_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.PropertyFillColor)
        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_fill_opacity(symbol_layer: QgsSimpleFillSymbolLayer) -> Union[float, List]:
        """Return ``fill-opacity`` from the fill colour alpha channel.

        Honours a data-defined ``PropertyOpacity`` override when active, so
        per-feature opacity attributes are emitted as MapLibre expressions.
        """
        base_opacity = symbol_layer.color().alphaF()
        try:
            opacity_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.PropertyOpacity
            )
            return PropertyExtractor.get_value_or_expression(base_opacity, opacity_prop)
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
                QgsSymbolLayer.PropertyStrokeColor
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
                QgsSymbolLayer.PropertyLayerEnabled
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
        """Return ``icon-size`` honouring any data-defined override."""
        size_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.PropertySize)
        return PropertyExtractor.get_value_or_expression(default_size, size_prop)

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
                    QgsSymbolLayer.PropertyAngle
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
        return 2.0

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
    def get_icon_allow_overlap() -> bool:
        """Return ``icon-allow-overlap`` (MapLibre default: ``False``)."""
        return False

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
            placement = label_settings.placement
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
        return [f"{font.family()} {font.styleName()}"]

    @staticmethod
    def get_text_size(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[float, List]:
        """Return ``text-size`` in pixels, honouring data-defined overrides."""
        base_size = text_format.font().pointSizeF() * (96.0 / 72.0)
        size_prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Size)
        return PropertyExtractor.get_value_or_expression(base_size, size_prop)

    @staticmethod
    def get_text_color(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[str, List]:
        """Return ``text-color`` honouring any data-defined override."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(text_format.color())
        color_prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Color)
        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_text_opacity(
        text_format: QgsTextFormat,
        label_settings: QgsPalLayerSettings = None,
    ) -> Union[float, List]:
        """Return ``text-opacity`` from the text-colour alpha channel.

        Honours a data-defined ``FontOpacity`` override on ``label_settings``
        when active, so per-feature opacity attributes are emitted as
        MapLibre expressions. When the property does not exist on the
        running QGIS build, the static alpha value is returned.
        """
        base_opacity = text_format.color().alphaF()
        if label_settings is None:
            return base_opacity
        try:
            prop_key = getattr(QgsPalLayerSettings, "FontOpacity")
        except AttributeError:
            return base_opacity
        try:
            opacity_prop = label_settings.dataDefinedProperties().property(prop_key)
            return PropertyExtractor.get_value_or_expression(base_opacity, opacity_prop)
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
                QgsPalLayerSettings.BufferColor
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
        try:
            base_width = PropertyExtractor.convert_length_to_pixels(
                buffer.size(), buffer.sizeUnit()
            )
        except (AttributeError, RuntimeError):
            base_width = buffer.size()

        if label_settings is None:
            return base_width
        try:
            size_prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.BufferSize
            )
            return PropertyExtractor.get_value_or_expression(base_width, size_prop)
        except (AttributeError, RuntimeError):
            return base_width

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
        return anchor_map.get(label_settings.quadOffset, "center")

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
        base_justify = justify_map.get(justification, "left")
        try:
            justify_prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.MultiLineAlignment
            )
            return PropertyExtractor.get_value_or_expression(base_justify, justify_prop)
        except (AttributeError, RuntimeError):
            return base_justify

    @staticmethod
    def get_text_offset(label_settings: QgsPalLayerSettings) -> List[float]:
        """Return ``text-offset`` ``[x, y]`` in ems-equivalent pixel units."""
        x_offset = label_settings.xOffset
        y_offset = label_settings.yOffset
        if x_offset != 0 or y_offset != 0:
            offset_unit = PropertyExtractor.get_attribute(
                label_settings, "xOffsetUnit", "offsetUnit", "units"
            )
            return [
                PropertyExtractor.convert_length_to_pixels(x_offset, offset_unit),
                PropertyExtractor.convert_length_to_pixels(y_offset, offset_unit),
            ]
        return [0, 0]

    @staticmethod
    def get_text_radial_offset(label_settings: QgsPalLayerSettings) -> float:
        """Return ``text-radial-offset`` in pixels from the label distance setting.

        Used in combination with ``text-variable-anchor`` to position labels
        radially around an anchor point.
        """
        try:
            distance = label_settings.dist
        except AttributeError:
            return 0
        if not distance:
            return 0
        unit = PropertyExtractor.get_attribute(label_settings, "distUnits", "distUnit")
        return PropertyExtractor.convert_length_to_pixels(distance, unit)

    @staticmethod
    def get_text_variable_anchor(
        label_settings: QgsPalLayerSettings,
    ) -> Optional[List[str]]:
        """Return ``text-variable-anchor`` when QGIS uses a flexible placement.

        For ``AroundPoint`` (0) and ``OrderedPositionsAroundPoint`` (6), a
        list of candidate anchors is emitted so the renderer can pick the
        best fit. ``None`` is returned otherwise so MapLibre falls back to
        the static ``text-anchor`` value.
        """
        try:
            placement = label_settings.placement
        except AttributeError:
            return None
        if placement in (0, 6):
            return ["top", "bottom", "left", "right",
                    "top-left", "top-right", "bottom-left", "bottom-right"]
        return None

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
    def get_text_allow_overlap() -> bool:
        """Return ``text-allow-overlap`` (MapLibre default: ``False``)."""
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
        """Return ``text-padding`` in pixels (MapLibre default: 2)."""
        return 2

    @staticmethod
    def get_text_line_height() -> float:
        """Return ``text-line-height`` (MapLibre default: 1.2)."""
        return 1.2

    @staticmethod
    def get_text_letter_spacing() -> float:
        """Return ``text-letter-spacing`` in ems (default 0)."""
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

        Uses ``autoWrapLength`` when set; otherwise a very large value to
        effectively disable wrapping.
        """
        if label_settings.autoWrapLength > 0:
            return label_settings.autoWrapLength
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
            QgsPalLayerSettings.LabelRotation
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
        layer: Optional[QgsVectorTileLayer] = None,
        background_type: int = 0,
    ):
        """Initialise the exporter.

        Args:
            output_dir:      Directory where ``style.json`` and the sprite
                             folder will be written.
            layer:           The vector tile layer to export. If ``None``,
                             the active layer in the QGIS interface is used.
            background_type: Background tile preset:
                             ``0`` = OpenStreetMap raster,
                             ``1`` = NASA Blue Marble raster,
                             anything else = solid project background colour.
        """
        self.output_dir = output_dir
        self.marker_symbols: dict = {}
        self.marker_counter = 0

        self.layer = self._resolve_layer(layer)
        self.source_name = "q2vt_tiles"
        self.style = self._build_style_skeleton()
        self.style["layers"].append(self._build_background_layer(background_type))

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
                    "tiles": ["http://localhost:9000/tiles/tiles/{z}/{x}/{y}.pbf"],
                }
            },
            "glyphs": "local://{fontstack}/{range}.pbf",
            "sprite": "http://localhost:9000/style/sprite/sprite",
            "layers": [],
        }

    def _build_background_layer(self, background_type: int) -> dict:
        """Add a background source to the style and return its layer definition.

        Args:
            background_type: ``0`` for OSM, ``1`` for NASA Blue Marble; any
                             other value emits a solid-colour background
                             using the project's background colour.
        """
        if background_type == 0:
            self.style["sources"]["osm"] = {
                "type": "raster",
                "tiles": [
                    "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
                ],
                "tileSize": 256,
                "attribution": (
                    '&copy; <a href="https://www.openstreetmap.org/copyright">'
                    "OpenStreetMap</a> contributors"
                ),
            }
            return {"id": "osm-background", "type": "raster", "source": "osm",
                    "minzoom": 0, "maxzoom": 22}

        if background_type == 1:
            self.style["sources"]["bluemarbel"] = {
                "type": "raster",
                "tiles": [
                    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
                    "BlueMarble_ShadedRelief/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg"
                ],
                "tileSize": 256,
            }
            return {"id": "bluemarbel-background", "type": "raster", "source": "bluemarbel",
                    "minzoom": 0, "maxzoom": 22}

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
        self.layer.source()

        renderer = self.layer.renderer()
        if isinstance(renderer, QgsVectorTileBasicRenderer):
            for style in renderer.styles():
                self._convert_renderer_style(style)

        labeling = self.layer.labeling()
        if isinstance(labeling, QgsVectorTileBasicLabeling):
            for style in labeling.styles():
                self._convert_labeling_style(style)

        self.save_to_file()
        return self.style

    def _convert_renderer_style(self, style):
        """Convert a single ``QgsVectorTileBasicRendererStyle`` into MapLibre layer(s)."""
        if not style.isEnabled() or not style.symbol():
            return
        self._convert_symbol(
            style.symbol(), style.styleName(), style.layerName(),
            self.source_name, style.minZoomLevel(), style.maxZoomLevel() + 1,
        )

    def _convert_labeling_style(self, style):
        """Convert a single ``QgsVectorTileBasicLabelingStyle`` into a MapLibre symbol layer."""
        if not style.isEnabled() or not style.labelSettings():
            return
        min_zoom = style.minZoomLevel()
        max_zoom = style.maxZoomLevel()
        if min_zoom == max_zoom:
            max_zoom += 1
        self._convert_label(
            style.labelSettings(), style.styleName(), style.layerName(),
            self.source_name, min_zoom, max_zoom,
        )

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

        if symbol_type == QgsSymbol.Marker:
            self._convert_marker_symbol(
                symbol_layer, symbol, style_name, source_layer_name,
                source_name, min_zoom, max_zoom,
            )
        elif symbol_type == QgsSymbol.Line:
            self._convert_line_symbol(
                symbol_layer, style_name, source_layer_name, source_name, min_zoom, max_zoom
            )
        elif symbol_type == QgsSymbol.Fill:
            self._convert_fill_symbol(
                symbol_layer, style_name, source_layer_name, source_name, min_zoom, max_zoom
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
            self.marker_symbols[pattern_name] = symbol_or_layer.clone()
        except (AttributeError, RuntimeError):
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
        self.marker_symbols[marker_name] = symbol.clone()

        layer_def["layout"].update({
            "icon-image": IconPropertyExtractor.get_icon_image(marker_name),
            "icon-size": IconPropertyExtractor.get_icon_size(symbol_layer, 1.0),
            "icon-rotate": IconPropertyExtractor.get_icon_rotate(symbol_layer=symbol_layer),
            "icon-padding": IconPropertyExtractor.get_icon_padding(),
            "icon-rotation-alignment": IconPropertyExtractor.get_icon_rotation_alignment(),
            "icon-pitch-alignment": IconPropertyExtractor.get_icon_pitch_alignment(),
            "icon-anchor": IconPropertyExtractor.get_icon_anchor(),
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(),
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

    def _convert_line_symbol(
        self,
        symbol_layer: QgsSymbolLayer,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a QGIS line symbol into a MapLibre ``line`` layer."""
        layer_def = self._base_layer_def(
            "line", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        if isinstance(symbol_layer, QgsSimpleLineSymbolLayer):
            layer_def["paint"].update({
                "line-color": LinePropertyExtractor.get_line_color(symbol_layer),
                "line-width": LinePropertyExtractor.get_line_width(symbol_layer),
                "line-opacity": LinePropertyExtractor.get_line_opacity(symbol_layer),
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
        elif LinePropertyExtractor.is_pattern_line(symbol_layer):
            # Raster / pattern-based line: emit line-pattern instead of line-color.
            pattern_name = self._register_pattern(symbol_layer)
            layer_def["paint"]["line-pattern"] = pattern_name
            layer_def["paint"].update({
                "line-opacity": 1.0,
                "line-translate": LinePropertyExtractor.get_line_translate(),
                "line-translate-anchor": LinePropertyExtractor.get_line_translate_anchor(),
            })
            layer_def["layout"].update({
                "line-cap": "butt",
                "line-join": "miter",
                "line-miter-limit": LinePropertyExtractor.get_line_miter_limit(),
                "line-round-limit": LinePropertyExtractor.get_line_round_limit(),
                "visibility": "visible",
            })

        self.style["layers"].append(layer_def)

    def _convert_fill_symbol(
        self,
        symbol_layer: QgsSymbolLayer,
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
            layer_def["paint"].update({
                "fill-color": FillPropertyExtractor.get_fill_color(symbol_layer),
                "fill-opacity": FillPropertyExtractor.get_fill_opacity(symbol_layer),
                "fill-antialias": FillPropertyExtractor.get_fill_antialias(),
                "fill-translate": FillPropertyExtractor.get_fill_translate(),
                "fill-translate-anchor": FillPropertyExtractor.get_fill_translate_anchor(),
            })
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
                "fill-opacity": 1.0,
                "fill-antialias": FillPropertyExtractor.get_fill_antialias(),
                "fill-translate": FillPropertyExtractor.get_fill_translate(),
                "fill-translate-anchor": FillPropertyExtractor.get_fill_translate_anchor(),
            })
            layer_def["layout"].update({
                "fill-sort-key": FillPropertyExtractor.get_fill_sort_key(symbol_layer),
                "visibility": "visible",
            })

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
            "symbol", f"{style_name}_label", source_layer_name, source_name, min_zoom, max_zoom
        )

        text_format = label_settings.format()

        text_field = TextPropertyExtractor.get_text_field(label_settings)
        if text_field:
            layer_def["layout"]["text-field"] = text_field

        layer_def["layout"].update({
            "text-font": TextPropertyExtractor.get_text_font(text_format),
            "text-size": TextPropertyExtractor.get_text_size(text_format, label_settings),
            "text-anchor": TextPropertyExtractor.get_text_anchor(label_settings),
            "text-justify": TextPropertyExtractor.get_text_justify(label_settings),
            "text-offset": TextPropertyExtractor.get_text_offset(label_settings),
            "text-radial-offset": TextPropertyExtractor.get_text_radial_offset(label_settings),
            "text-allow-overlap": TextPropertyExtractor.get_text_allow_overlap(),
            "text-ignore-placement": TextPropertyExtractor.get_text_ignore_placement(),
            "text-optional": TextPropertyExtractor.get_text_optional(),
            "text-padding": TextPropertyExtractor.get_text_padding(),
            "text-line-height": TextPropertyExtractor.get_text_line_height(),
            "text-letter-spacing": TextPropertyExtractor.get_text_letter_spacing(),
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
            "symbol-sort-key": IconPropertyExtractor.get_symbol_sort_key(),
            "symbol-z-order": IconPropertyExtractor.get_symbol_z_order(),
            "visibility": "visible",
        })

        variable_anchor = TextPropertyExtractor.get_text_variable_anchor(label_settings)
        if variable_anchor:
            layer_def["layout"]["text-variable-anchor"] = variable_anchor

        layer_def["paint"].update({
            "text-color": TextPropertyExtractor.get_text_color(text_format, label_settings),
            "text-opacity": TextPropertyExtractor.get_text_opacity(text_format, label_settings),
            "text-halo-color": TextPropertyExtractor.get_text_halo_color(
                text_format, label_settings
            ),
            "text-halo-width": TextPropertyExtractor.get_text_halo_width(
                text_format, label_settings
            ),
            "text-halo-blur": TextPropertyExtractor.get_text_halo_blur(
                text_format, label_settings
            ),
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
            if marker and marker.type() == QgsSymbol.Marker:
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
                self.marker_symbols, style_dir, scale_factor=1, test_mode=False
            ).generate()
        else:
            del self.style["sprite"]

        filepath = os.path.join(style_dir, filename)
        with open(filepath, "w", encoding="utf8") as f:
            json.dump(self.style, f, indent=indent)

        return filepath


if __name__ == "__console__":
    exporter = QgisMapLibreStyleExporter(output_dir=QgsProcessingUtils.tempFolder())
    output_file = exporter.export()
