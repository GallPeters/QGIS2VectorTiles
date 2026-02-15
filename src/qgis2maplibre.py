"""Convert QGIS Vector Tile Layer styles to MapLibre GL JSON style format."""

import json
import os
from typing import Dict, Any, Optional, Union, List
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
    QgsTextBackgroundSettings,
)
from qgis.utils import iface
from .qgis2sprites import SpriteGenerator


class PropertyExtractor:
    """Utility class for extracting and converting PyQGIS properties to MapLibre format."""

    @staticmethod
    def get_value_or_expression(value: Any, prop: QgsProperty) -> Union[Any, List]:
        """
        Return either the static value or a MapLibre expression for data-defined properties.

        Args:
            value: The static value to return if not data-defined
            prop: The QgsProperty to check for data-defined behavior

        Returns:
            Either the static value or a MapLibre ["get", "field_name"] expression
        """
        if prop and prop.isActive():
            expression = prop.expressionString()
            evaluation = QgsExpression(expression).evaluate()
            if evaluation is not None:
                return evaluation
            field_name = expression.replace('"', "")
            if field_name:
                return ["get", field_name]
        return value

    @staticmethod
    def convert_qcolor_to_maplibre(color: QColor) -> str:
        """Convert QColor to MapLibre RGB format."""
        return f"rgba({color.red()}, {color.green()}, {color.blue()})"

    @staticmethod
    def convert_length_to_pixels(value: float, unit_obj=None) -> float:
        """
        Convert a length value from various QGIS units to pixels.

        Assumes 96 DPI. If unit_obj is an enum or string, attempts to infer
        unit from its string representation. Falls back to treating the value
        as millimeters (common default in QGIS symbol sizes).
        """
        if value is None:
            return value

        # If unit object provided, try to get a descriptive string
        unit_str = ""
        try:
            if unit_obj is not None:
                try:
                    unit_str = str(unit_obj).lower()
                except (OSError, RuntimeError):
                    unit_str = ""
        except (OSError, RuntimeError):
            unit_str = ""

        # Interpret unit strings heuristically
        if "millimeter" in unit_str or "mm" in unit_str:
            return value * 3.78
        if "inch" in unit_str or "in" in unit_str:
            return value * 96.0
        if "point" in unit_str or "pt" in unit_str:
            return value * (96.0 / 72.0)
        if "pixel" in unit_str or "px" in unit_str:
            return value

        # Try to handle known QgsUnitTypes enums by value
        try:
            if unit_obj == 0:
                return value * 3.78
            if unit_obj == 5:
                return value * 96.0
            if unit_obj == 4:
                return value * (96.0 / 72.0)
            if unit_obj == 2:
                return value
        except (RuntimeError, AttributeError):
            pass

        # Fallback: assume millimeters
        return value * 3.78


class LinePropertyExtractor:
    """Extract line properties from QGIS symbol layers."""

    @staticmethod
    def get_line_color(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[str, List]:
        """Get line-color property."""
        color = symbol_layer.color()
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(color)

        dd_props = symbol_layer.dataDefinedProperties()
        color_prop = dd_props.property(QgsSymbolLayer.PropertyFillColor)

        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_line_width(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[float, List]:
        """Get line-width property in pixels."""
        width = symbol_layer.width()

        # Detect unit for line width
        width_unit = None
        for attr in ("widthUnit", "widthUnits", "strokeWidthUnit", "strokeWidthUnits"):
            if hasattr(symbol_layer, attr):
                u = getattr(symbol_layer, attr)
                width_unit = u() if callable(u) else u
                break

        width_px = PropertyExtractor.convert_length_to_pixels(width, width_unit)

        dd_props = symbol_layer.dataDefinedProperties()
        width_prop = dd_props.property(QgsSymbolLayer.PropertyStrokeWidth)

        return PropertyExtractor.get_value_or_expression(width_px, width_prop)

    @staticmethod
    def get_line_opacity(symbol_layer: QgsSimpleLineSymbolLayer) -> float:
        """Get line-opacity property."""
        color = symbol_layer.color()
        return color.alphaF()

    @staticmethod
    def get_line_cap(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Get line-cap property."""
        pen_cap_style = symbol_layer.penCapStyle()
        cap_map = {
            0: "butt",  # Qt.FlatCap
            16: "square",  # Qt.SquareCap
            32: "round",  # Qt.RoundCap
        }
        return cap_map.get(pen_cap_style, "butt")

    @staticmethod
    def get_line_join(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Get line-join property."""
        pen_join_style = symbol_layer.penJoinStyle()
        join_map = {
            0: "miter",  # Qt.MiterJoin
            64: "bevel",  # Qt.BevelJoin
            128: "round",  # Qt.RoundJoin
        }
        return join_map.get(pen_join_style, "miter")

    @staticmethod
    def get_line_miter_limit() -> float:
        """Get line-miter-limit property (default MapLibre value)."""
        return 2.0

    @staticmethod
    def get_line_round_limit() -> float:
        """Get line-round-limit property (default MapLibre value)."""
        return 1.05

    @staticmethod
    def get_line_dasharray(
        symbol_layer: QgsSimpleLineSymbolLayer, width_px: float
    ) -> Optional[List[float]]:
        """Get line-dasharray property."""
        dash_vector = None
        try:
            dash_vector = symbol_layer.customDashVector()
        except (RuntimeError, AttributeError):
            dash_vector = None

        # Detect if custom dash pattern is enabled
        custom_dash_enabled = False
        try:
            if hasattr(symbol_layer, "useCustomDashPattern"):
                custom_dash_enabled = bool(symbol_layer.useCustomDashPattern())
            elif hasattr(symbol_layer, "customDashEnabled"):
                custom_dash_enabled = bool(symbol_layer.customDashEnabled())
            elif hasattr(symbol_layer, "isCustomDash"):
                custom_dash_enabled = bool(symbol_layer.isCustomDash())
        except (RuntimeError, AttributeError):
            custom_dash_enabled = False

        # Get pen style
        pen_style = None
        try:
            if hasattr(symbol_layer, "penStyle"):
                pen_style = symbol_layer.penStyle()
            elif hasattr(symbol_layer, "strokeStyle"):
                pen_style = symbol_layer.strokeStyle()
            elif hasattr(symbol_layer, "pen_style"):
                v = getattr(symbol_layer, "pen_style")
                pen_style = v() if callable(v) else v
        except (RuntimeError, AttributeError):
            pen_style = None

        fixed_dashed_types = {2, 3, 4, 5}

        # If custom dash is enabled, prefer the custom dash vector
        if custom_dash_enabled:
            if dash_vector and len(dash_vector) > 0:
                dash_unit = None
                for attr in ("dashUnit", "dashUnits", "customDashUnits"):
                    if hasattr(symbol_layer, attr):
                        u = getattr(symbol_layer, attr)
                        dash_unit = u() if callable(u) else u
                        break

                width_unit = None
                for attr in ("widthUnit", "widthUnits", "strokeWidthUnit", "strokeWidthUnits"):
                    if hasattr(symbol_layer, attr):
                        u = getattr(symbol_layer, attr)
                        width_unit = u() if callable(u) else u
                        break

                dasharray = [
                    PropertyExtractor.convert_length_to_pixels(d, dash_unit or width_unit)
                    for d in dash_vector
                ]
                return dasharray

        # If not custom but pen style is one of the fixed dashed presets
        elif pen_style in fixed_dashed_types:
            try:
                w = width_px
            except (RuntimeError, ValueError):
                w = 1.0
            preset = {
                2: [4 * w, 2 * w],
                3: [1 * w, 1 * w],
                4: [4 * w, 2 * w, 1 * w, 2 * w],
                5: [4 * w, 2 * w, 1 * w, 2 * w, 1 * w, 2 * w],
            }
            return preset.get(pen_style)

        return None

    @staticmethod
    def get_line_offset(symbol_layer: QgsSimpleLineSymbolLayer) -> float:
        """Get line-offset property in pixels."""
        try:
            offset = symbol_layer.offset()
            offset_unit = None
            for attr in ("offsetUnit", "offsetUnits"):
                if hasattr(symbol_layer, attr):
                    u = getattr(symbol_layer, attr)
                    offset_unit = u() if callable(u) else u
                    break
            if offset != 0:
                return PropertyExtractor.convert_length_to_pixels(offset, offset_unit)
        except (RuntimeError, AttributeError):
            pass
        return 0

    @staticmethod
    def get_line_blur() -> float:
        """Get line-blur property (default value)."""
        return 0

    @staticmethod
    def get_line_gap_width() -> float:
        """Get line-gap-width property (default value)."""
        return 0

    @staticmethod
    def get_line_translate() -> List[float]:
        """Get line-translate property (default value)."""
        return [0, 0]

    @staticmethod
    def get_line_translate_anchor() -> str:
        """Get line-translate-anchor property (default value)."""
        return "map"


class FillPropertyExtractor:
    """Extract fill properties from QGIS symbol layers."""

    @staticmethod
    def get_fill_color(symbol_layer: QgsSimpleFillSymbolLayer) -> Union[str, List]:
        """Get fill-color property."""
        fill_color = symbol_layer.color()
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(fill_color)

        dd_props = symbol_layer.dataDefinedProperties()
        color_prop = dd_props.property(QgsSymbolLayer.PropertyFillColor)

        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_fill_opacity(symbol_layer: QgsSimpleFillSymbolLayer) -> float:
        """Get fill-opacity property."""
        fill_color = symbol_layer.color()
        return fill_color.alphaF()

    @staticmethod
    def get_fill_outline_color(symbol_layer: QgsSimpleFillSymbolLayer) -> Union[str, List, None]:
        """Get fill-outline-color property."""
        stroke_width = symbol_layer.strokeWidth()
        stroke_style = symbol_layer.strokeStyle()

        # Only add outline if stroke width is greater than 0
        if stroke_width > 0 and stroke_style != 0:
            stroke_color = symbol_layer.strokeColor()
            base_color = PropertyExtractor.convert_qcolor_to_maplibre(stroke_color)

            dd_props = symbol_layer.dataDefinedProperties()
            color_prop = dd_props.property(QgsSymbolLayer.PropertyStrokeColor)

            return PropertyExtractor.get_value_or_expression(base_color, color_prop)
        return None

    @staticmethod
    def get_fill_antialias() -> bool:
        """Get fill-antialias property (default value)."""
        return True

    @staticmethod
    def get_fill_translate() -> List[float]:
        """Get fill-translate property (default value)."""
        return [0, 0]

    @staticmethod
    def get_fill_translate_anchor() -> str:
        """Get fill-translate-anchor property (default value)."""
        return "map"


class IconPropertyExtractor:
    """Extract icon properties for symbol layers."""

    @staticmethod
    def get_icon_image(marker_name: str) -> str:
        """Get icon-image property."""
        return marker_name

    @staticmethod
    def get_icon_size(
        symbol_layer: QgsSymbolLayer, default_size: float = 1.0
    ) -> Union[float, List]:
        """Get icon-size property."""
        dd_props = symbol_layer.dataDefinedProperties()
        size_prop = dd_props.property(QgsSymbolLayer.PropertySize)

        return PropertyExtractor.get_value_or_expression(default_size, size_prop)

    @staticmethod
    def get_icon_rotation_alignment() -> str:
        """Get icon-rotation-alignment property (default value)."""
        return "map"

    @staticmethod
    def get_icon_pitch_alignment() -> str:
        """Get icon-pitch-alignment property (default value)."""
        return "viewport"

    @staticmethod
    def get_icon_anchor() -> str:
        """Get icon-anchor property (default value)."""
        return "center"

    @staticmethod
    def get_icon_allow_overlap() -> bool:
        """Get icon-allow-overlap property (default value)."""
        return False

    @staticmethod
    def get_icon_ignore_placement() -> bool:
        """Get icon-ignore-placement property (default value)."""
        return False

    @staticmethod
    def get_icon_optional() -> bool:
        """Get icon-optional property (default value)."""
        return False

    @staticmethod
    def get_icon_keep_upright() -> bool:
        """Get icon-keep-upright property (default value)."""
        return True

    @staticmethod
    def get_icon_text_fit(background: QgsTextBackgroundSettings) -> Optional[str]:
        """Get icon-text-fit property."""
        if background.enabled() and background.sizeType() == 0:  # Buffer type
            return "both"
        return None

    @staticmethod
    def get_icon_text_fit_padding(background: QgsTextBackgroundSettings) -> Optional[List[float]]:
        """Get icon-text-fit-padding property."""
        if background.enabled() and background.sizeType() == 0:  # Buffer type
            buffer_size = background.size().width()
            buffer_unit = background.sizeUnit()
            buffer_px = PropertyExtractor.convert_length_to_pixels(buffer_size, buffer_unit)
            return [buffer_px * 2, buffer_px, buffer_px * 2, buffer_px]
        return None

    @staticmethod
    def get_icon_offset(background: QgsTextBackgroundSettings = None) -> List[float]:
        """Get icon-offset property."""
        if background and background.enabled():
            offset_unit = background.offsetUnit()
            offset = background.offset()
            offset_x = PropertyExtractor.convert_length_to_pixels(offset.x(), offset_unit)
            offset_y = PropertyExtractor.convert_length_to_pixels(offset.y(), offset_unit)
            return [offset_x, offset_y]
        return [0, 0]

    @staticmethod
    def get_icon_opacity(background: QgsTextBackgroundSettings = None) -> float:
        """Get icon-opacity property."""
        if background and background.enabled():
            try:
                return background.opacity()
            except (OSError, RuntimeError):
                pass
        return 1.0

    @staticmethod
    def get_icon_color(background: QgsTextBackgroundSettings = None) -> str:
        """Get icon-color property."""
        if background and background.enabled():
            try:
                bg_color = background.fillColor()
                return PropertyExtractor.convert_qcolor_to_maplibre(bg_color)
            except (OSError, RuntimeError):
                pass
        return "rgb(255, 255, 255)"

    @staticmethod
    def get_icon_halo_color() -> str:
        """Get icon-halo-color property (default value)."""
        return "rgb(0, 0, 0)"

    @staticmethod
    def get_icon_halo_width() -> float:
        """Get icon-halo-width property (default value)."""
        return 0

    @staticmethod
    def get_icon_halo_blur() -> float:
        """Get icon-halo-blur property (default value)."""
        return 0

    @staticmethod
    def get_icon_translate() -> List[float]:
        """Get icon-translate property (default value)."""
        return [0, 0]

    @staticmethod
    def get_icon_translate_anchor() -> str:
        """Get icon-translate-anchor property (default value)."""
        return "map"


class TextPropertyExtractor:
    """Extract text properties from QGIS label settings."""

    @staticmethod
    def get_text_field(label_settings: QgsPalLayerSettings) -> Optional[List]:
        """Get text-field property."""
        field_name = label_settings.fieldName
        if field_name:
            return ["get", field_name]
        return None

    @staticmethod
    def get_text_font(text_format: QgsTextFormat) -> List[str]:
        """Get text-font property."""
        font = text_format.font()
        return [f"{font.family()} {font.styleName()}"]

    @staticmethod
    def get_text_size(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[float, List]:
        """Get text-size property in pixels."""
        font = text_format.font()
        # Convert QGIS font point size to pixels (MapLibre uses pixels)
        # 1pt = 1/72in; at 96 DPI => px = pt * 96/72
        base_size = font.pointSizeF() * (96.0 / 72.0)

        dd_props = label_settings.dataDefinedProperties()
        size_prop = dd_props.property(QgsPalLayerSettings.Size)

        return PropertyExtractor.get_value_or_expression(base_size, size_prop)

    @staticmethod
    def get_text_color(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[str, List]:
        """Get text-color property."""
        color = text_format.color()
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(color)

        dd_props = label_settings.dataDefinedProperties()
        color_prop = dd_props.property(QgsPalLayerSettings.Color)

        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_text_opacity(text_format: QgsTextFormat) -> float:
        """Get text-opacity property."""
        color = text_format.color()
        return color.alphaF()

    @staticmethod
    def get_text_halo_color(text_format: QgsTextFormat) -> str:
        """Get text-halo-color property."""
        buffer = text_format.buffer()
        if buffer.enabled():
            return PropertyExtractor.convert_qcolor_to_maplibre(buffer.color())
        return "rgb(255, 255, 255)"

    @staticmethod
    def get_text_halo_width(text_format: QgsTextFormat) -> float:
        """Get text-halo-width property."""
        buffer = text_format.buffer()
        if buffer.enabled():
            return buffer.size()
        return 0

    @staticmethod
    def get_text_halo_blur(text_format: QgsTextFormat) -> float:
        """Get text-halo-blur property."""
        buffer = text_format.buffer()
        if buffer.enabled():
            return buffer.size() * 0.5
        return 0

    @staticmethod
    def get_text_anchor(label_settings: QgsPalLayerSettings) -> str:
        """Get text-anchor property."""
        quad = label_settings.quadOffset

        anchor_map = {
            0: "bottom-right",  # QuadrantAboveLeft
            1: "bottom",  # QuadrantAbove
            2: "bottom-left",  # QuadrantAboveRight
            3: "right",  # QuadrantLeft
            4: "center",  # QuadrantOver
            5: "left",  # QuadrantRight
            6: "top-right",  # QuadrantBelowLeft
            7: "top",  # QuadrantBelow
            8: "top-left",  # QuadrantBelowRight
        }

        return anchor_map.get(quad, "center")

    @staticmethod
    def get_text_justify(label_settings: QgsPalLayerSettings) -> str:
        """Get text-justify property."""
        try:
            justification = label_settings.multiLineAlignment
        except AttributeError:
            try:
                justification = label_settings.alignment
            except AttributeError:
                justification = 1

        justify_map = {
            0: "left",
            1: "center",
            2: "right",
            3: "center",  # Justify (MapLibre doesn't have justify, use center)
        }
        return justify_map.get(justification, "left")

    @staticmethod
    def get_text_offset(label_settings: QgsPalLayerSettings) -> List[float]:
        """Get text-offset property."""
        x_offset = label_settings.xOffset
        y_offset = label_settings.yOffset

        offset_unit = None
        for attr in ("xOffsetUnit", "offsetUnit", "units"):
            if hasattr(label_settings, attr):
                u = getattr(label_settings, attr)
                offset_unit = u() if callable(u) else u
                break

        if x_offset != 0 or y_offset != 0:
            return [
                PropertyExtractor.convert_length_to_pixels(x_offset, offset_unit),
                PropertyExtractor.convert_length_to_pixels(y_offset, offset_unit),
            ]
        return [0, 0]

    @staticmethod
    def get_text_allow_overlap() -> bool:
        """Get text-allow-overlap property (default value)."""
        return False

    @staticmethod
    def get_text_ignore_placement() -> bool:
        """Get text-ignore-placement property (default value)."""
        return False

    @staticmethod
    def get_text_optional() -> bool:
        """Get text-optional property (default value)."""
        return False

    @staticmethod
    def get_text_padding() -> float:
        """Get text-padding property (default value)."""
        return 0

    @staticmethod
    def get_text_line_height() -> float:
        """Get text-line-height property (default value)."""
        return 1.2

    @staticmethod
    def get_text_letter_spacing() -> float:
        """Get text-letter-spacing property (default value)."""
        return 0

    @staticmethod
    def get_text_transform() -> str:
        """Get text-transform property (default value)."""
        return "none"

    @staticmethod
    def get_text_max_width(label_settings: QgsPalLayerSettings) -> float:
        """Get text-max-width property."""
        if label_settings.autoWrapLength > 0:
            return label_settings.autoWrapLength
        return 999  # Effectively no max width

    @staticmethod
    def get_text_keep_upright() -> bool:
        """Get text-keep-upright property (default value)."""
        return True

    @staticmethod
    def get_text_rotate(label_settings: QgsPalLayerSettings) -> Union[float, List]:
        """Get text-rotate property."""
        base_rotation = label_settings.angleOffset if label_settings.angleOffset != 0 else 0

        dd_props = label_settings.dataDefinedProperties()
        rotation_prop = dd_props.property(QgsPalLayerSettings.LabelRotation)

        return PropertyExtractor.get_value_or_expression(base_rotation, rotation_prop)

    @staticmethod
    def get_text_rotation_alignment() -> str:
        """Get text-rotation-alignment property (default value)."""
        return "map"

    @staticmethod
    def get_text_pitch_alignment() -> str:
        """Get text-pitch-alignment property (default value)."""
        return "viewport"

    @staticmethod
    def get_text_translate() -> List[float]:
        """Get text-translate property (default value)."""
        return [0, 0]

    @staticmethod
    def get_text_translate_anchor() -> str:
        """Get text-translate-anchor property (default value)."""
        return "map"


class QgisMapLibreStyleExporter:
    """Export QGIS Vector Tile Layer styles to MapLibre GL style JSON."""

    def __init__(self, output_dir: str, layer: Optional[QgsVectorTileLayer] = None):
        """Initialize converter with output directory and optional QgsVectorTileLayer."""
        self.output_dir = output_dir
        self.marker_symbols = {}  # Dict to collect marker symbols for sprite generation
        self.marker_counter = 0  # Counter for unique marker names

        # Get layer (use active layer if not provided)
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

        self.layer = layer
        self.source_name = "q2vt_tiles"  # Fixed source name for simplicity
        # Initialize style structure
        self.style = {
            "version": 8,
            "name": f"{self.source_name}_style",
            "sources": {
                self.source_name: {
                    "type": "vector",
                    "tiles": ["http://localhost:9000/tiles/{z}/{x}/{y}.pbf"],
                }
            },
            "glyphs": "local://{fontstack}/{range}.pbf",
            "sprite": "http://localhost:9000/style/sprite/sprite",
            "layers": [],
        }

    def export(self) -> Dict[str, Any]:
        """Convert all styles from QgsVectorTileLayer to MapLibre GL style."""
        # Add source for the layer
        self.layer.source()

        # Extract renderer styles
        renderer = self.layer.renderer()
        if isinstance(renderer, QgsVectorTileBasicRenderer):
            for style in renderer.styles():
                self._convert_renderer_style(style)

        # Extract labeling styles
        labeling = self.layer.labeling()
        if isinstance(labeling, QgsVectorTileBasicLabeling):
            for style in labeling.styles():
                self._convert_labeling_style(style)

        self.save_to_file()
        return self.style

    def _get_symbol_type_code(self, symbol_type) -> str:
        """Convert QgsSymbol type to short code."""
        if symbol_type == QgsSymbol.Marker:
            return "marker"
        elif symbol_type == QgsSymbol.Line:
            return "line"
        elif symbol_type == QgsSymbol.Fill:
            return "fill"
        else:
            return "symbol"

    def _convert_renderer_style(self, style):
        """Convert a single renderer style from QgsVectorTileBasicRendererStyle."""
        layer_name = style.layerName()
        style_name = style.styleName()
        symbol = style.symbol()
        min_zoom = style.minZoomLevel()
        max_zoom = style.maxZoomLevel()
        if min_zoom == max_zoom:
            max_zoom += 1
        enabled = style.isEnabled()
        if not enabled or not symbol:
            return

        self._convert_symbol(symbol, style_name, layer_name, self.source_name, min_zoom, max_zoom)

    def _convert_labeling_style(self, style):
        """Convert a single labeling style from QgsVectorTileBasicLabelingStyle."""
        layer_name = style.layerName()
        style_name = style.styleName()
        label_settings = style.labelSettings()
        min_zoom = style.minZoomLevel()
        max_zoom = style.maxZoomLevel()
        if min_zoom == max_zoom:
            max_zoom += 1
        enabled = style.isEnabled()

        if not enabled or not label_settings:
            return

        self._convert_label(
            label_settings, style_name, layer_name, self.source_name, min_zoom, max_zoom
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
        """Convert QgsSymbol to MapLibre layer(s) with zoom levels."""
        symbol_type = symbol.type()

        if symbol.symbolLayerCount() > 0:
            symbol_layer = symbol.symbolLayer(0)

            if symbol_type == QgsSymbol.Marker:
                self._convert_marker_symbol(
                    symbol_layer,
                    symbol,
                    style_name,
                    source_layer_name,
                    source_name,
                    min_zoom,
                    max_zoom,
                )
            elif symbol_type == QgsSymbol.Line:
                self._convert_line_symbol(
                    symbol_layer, style_name, source_layer_name, source_name, min_zoom, max_zoom
                )
            elif symbol_type == QgsSymbol.Fill:
                self._convert_fill_symbol(
                    symbol_layer, style_name, source_layer_name, source_name, min_zoom, max_zoom
                )

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
        """Convert marker symbol to MapLibre symbol layer using property extractors."""
        layer_def = {
            "id": style_name,
            "type": "symbol",
            "source": source_name,
            "source-layer": source_layer_name,
            "paint": {},
            "layout": {},
        }

        # Add zoom levels if specified
        if min_zoom >= 0:
            layer_def["minzoom"] = min_zoom
        if max_zoom >= 0:
            layer_def["maxzoom"] = max_zoom

        # Collect marker for sprite generation
        marker_name = f"marker_{self.marker_counter}"
        self.marker_counter += 1
        self.marker_symbols[marker_name] = symbol.clone()

        # Icon properties using extractors
        layer_def["layout"]["icon-image"] = IconPropertyExtractor.get_icon_image(marker_name)
        layer_def["layout"]["icon-size"] = IconPropertyExtractor.get_icon_size(symbol_layer, 1.0)
        layer_def["layout"][
            "icon-rotation-alignment"
        ] = IconPropertyExtractor.get_icon_rotation_alignment()
        layer_def["layout"][
            "icon-pitch-alignment"
        ] = IconPropertyExtractor.get_icon_pitch_alignment()
        layer_def["layout"]["icon-anchor"] = IconPropertyExtractor.get_icon_anchor()
        layer_def["layout"]["icon-allow-overlap"] = IconPropertyExtractor.get_icon_allow_overlap()
        layer_def["layout"][
            "icon-ignore-placement"
        ] = IconPropertyExtractor.get_icon_ignore_placement()
        layer_def["layout"]["visibility"] = "visible"

        layer_def["paint"]["icon-opacity"] = IconPropertyExtractor.get_icon_opacity()
        layer_def["paint"]["icon-translate"] = IconPropertyExtractor.get_icon_translate()
        layer_def["paint"][
            "icon-translate-anchor"
        ] = IconPropertyExtractor.get_icon_translate_anchor()

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
        """Convert line symbol to MapLibre line layer using property extractors."""
        layer_def = {
            "id": style_name,
            "type": "line",
            "source": source_name,
            "source-layer": source_layer_name,
            "paint": {},
            "layout": {},
        }

        # Add zoom levels if specified
        if min_zoom >= 0:
            layer_def["minzoom"] = min_zoom
        if max_zoom >= 0:
            layer_def["maxzoom"] = max_zoom

        if isinstance(symbol_layer, QgsSimpleLineSymbolLayer):
            # Paint properties using extractors
            layer_def["paint"]["line-color"] = LinePropertyExtractor.get_line_color(symbol_layer)
            layer_def["paint"]["line-width"] = LinePropertyExtractor.get_line_width(symbol_layer)
            layer_def["paint"]["line-opacity"] = LinePropertyExtractor.get_line_opacity(
                symbol_layer
            )
            layer_def["paint"]["line-blur"] = LinePropertyExtractor.get_line_blur()
            layer_def["paint"]["line-gap-width"] = LinePropertyExtractor.get_line_gap_width()
            layer_def["paint"]["line-translate"] = LinePropertyExtractor.get_line_translate()
            layer_def["paint"][
                "line-translate-anchor"
            ] = LinePropertyExtractor.get_line_translate_anchor()

            # Line offset
            offset = LinePropertyExtractor.get_line_offset(symbol_layer)
            if offset != 0:
                layer_def["paint"]["line-offset"] = offset

            # Line dasharray
            width_value = layer_def["paint"]["line-width"]
            width_px = width_value if isinstance(width_value, (int, float)) else 1.0
            dasharray = LinePropertyExtractor.get_line_dasharray(symbol_layer, width_px)
            if dasharray:
                layer_def["paint"]["line-dasharray"] = dasharray

            # Layout properties using extractors
            layer_def["layout"]["line-cap"] = LinePropertyExtractor.get_line_cap(symbol_layer)
            layer_def["layout"]["line-join"] = LinePropertyExtractor.get_line_join(symbol_layer)
            layer_def["layout"]["line-miter-limit"] = LinePropertyExtractor.get_line_miter_limit()
            layer_def["layout"]["line-round-limit"] = LinePropertyExtractor.get_line_round_limit()
            layer_def["layout"]["visibility"] = "visible"

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
        """Convert fill symbol to MapLibre fill layer using property extractors."""
        layer_def = {
            "id": style_name,
            "type": "fill",
            "source": source_name,
            "source-layer": source_layer_name,
            "paint": {},
            "layout": {},
        }

        # Add zoom levels if specified
        if min_zoom >= 0:
            layer_def["minzoom"] = min_zoom
        if max_zoom >= 0:
            layer_def["maxzoom"] = max_zoom

        if isinstance(symbol_layer, QgsSimpleFillSymbolLayer):
            # Paint properties using extractors
            layer_def["paint"]["fill-color"] = FillPropertyExtractor.get_fill_color(symbol_layer)
            layer_def["paint"]["fill-opacity"] = FillPropertyExtractor.get_fill_opacity(
                symbol_layer
            )
            layer_def["paint"]["fill-antialias"] = FillPropertyExtractor.get_fill_antialias()
            layer_def["paint"]["fill-translate"] = FillPropertyExtractor.get_fill_translate()
            layer_def["paint"][
                "fill-translate-anchor"
            ] = FillPropertyExtractor.get_fill_translate_anchor()

            # Outline color (only if applicable)
            outline_color = FillPropertyExtractor.get_fill_outline_color(symbol_layer)
            if outline_color:
                layer_def["paint"]["fill-outline-color"] = outline_color

            # Layout properties
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
        """Convert QgsPalLayerSettings to MapLibre symbol layer using property extractors."""
        layer_def = {
            "id": f"{style_name}_label",
            "type": "symbol",
            "source": source_name,
            "source-layer": source_layer_name,
            "paint": {},
            "layout": {},
        }

        # Add zoom levels if specified
        if min_zoom >= 0:
            layer_def["minzoom"] = min_zoom
        if max_zoom >= 0:
            layer_def["maxzoom"] = max_zoom

        # Text format
        text_format = label_settings.format()

        # Layout properties using extractors
        text_field = TextPropertyExtractor.get_text_field(label_settings)
        if text_field:
            layer_def["layout"]["text-field"] = text_field

        layer_def["layout"]["text-font"] = TextPropertyExtractor.get_text_font(text_format)
        layer_def["layout"]["text-size"] = TextPropertyExtractor.get_text_size(
            text_format, label_settings
        )
        layer_def["layout"]["text-anchor"] = TextPropertyExtractor.get_text_anchor(label_settings)
        layer_def["layout"]["text-justify"] = TextPropertyExtractor.get_text_justify(
            label_settings
        )
        layer_def["layout"]["text-offset"] = TextPropertyExtractor.get_text_offset(label_settings)
        layer_def["layout"]["text-allow-overlap"] = TextPropertyExtractor.get_text_allow_overlap()
        layer_def["layout"][
            "text-ignore-placement"
        ] = TextPropertyExtractor.get_text_ignore_placement()
        layer_def["layout"]["text-optional"] = TextPropertyExtractor.get_text_optional()
        layer_def["layout"]["text-padding"] = TextPropertyExtractor.get_text_padding()
        layer_def["layout"]["text-line-height"] = TextPropertyExtractor.get_text_line_height()
        layer_def["layout"][
            "text-letter-spacing"
        ] = TextPropertyExtractor.get_text_letter_spacing()
        layer_def["layout"]["text-transform"] = TextPropertyExtractor.get_text_transform()
        layer_def["layout"]["text-max-width"] = TextPropertyExtractor.get_text_max_width(
            label_settings
        )
        layer_def["layout"]["text-keep-upright"] = TextPropertyExtractor.get_text_keep_upright()
        layer_def["layout"]["text-rotate"] = TextPropertyExtractor.get_text_rotate(label_settings)
        layer_def["layout"][
            "text-rotation-alignment"
        ] = TextPropertyExtractor.get_text_rotation_alignment()
        layer_def["layout"][
            "text-pitch-alignment"
        ] = TextPropertyExtractor.get_text_pitch_alignment()
        layer_def["layout"]["visibility"] = "visible"

        # Paint properties using extractors
        layer_def["paint"]["text-color"] = TextPropertyExtractor.get_text_color(
            text_format, label_settings
        )
        layer_def["paint"]["text-opacity"] = TextPropertyExtractor.get_text_opacity(text_format)
        layer_def["paint"]["text-halo-color"] = TextPropertyExtractor.get_text_halo_color(
            text_format
        )
        layer_def["paint"]["text-halo-width"] = TextPropertyExtractor.get_text_halo_width(
            text_format
        )
        layer_def["paint"]["text-halo-blur"] = TextPropertyExtractor.get_text_halo_blur(
            text_format
        )
        layer_def["paint"]["text-translate"] = TextPropertyExtractor.get_text_translate()
        layer_def["paint"][
            "text-translate-anchor"
        ] = TextPropertyExtractor.get_text_translate_anchor()

        # Background (icon) properties using extractors
        label_settings = QgsPalLayerSettings(label_settings)
        label_format = QgsTextFormat(label_settings.format())
        background = QgsTextBackgroundSettings(label_format.background())
        if background.markerSymbol():
            background.setMarkerSymbol(background.markerSymbol().clone())
        label_format.setBackground(background)
        label_settings.setFormat(label_format)

        if background.enabled() and background.markerSymbol():
            # Try to extract marker from background
            try:
                if hasattr(background, "markerSymbol"):
                    marker = background.markerSymbol()
                    if marker and marker.type() == QgsSymbol.Marker:
                        marker_name = f"marker_{self.marker_counter}"
                        self.marker_counter += 1
                        self.marker_symbols[marker_name] = marker.clone()
                        layer_def["layout"]["icon-image"] = marker_name
                    else:
                        layer_def["layout"]["icon-image"] = style_name
                else:
                    layer_def["layout"]["icon-image"] = style_name
            except (RuntimeError, AttributeError):
                layer_def["layout"]["icon-image"] = style_name

            # Icon layout properties
            icon_text_fit = IconPropertyExtractor.get_icon_text_fit(background)
            if icon_text_fit:
                layer_def["layout"]["icon-text-fit"] = icon_text_fit

            icon_text_fit_padding = IconPropertyExtractor.get_icon_text_fit_padding(background)
            if icon_text_fit_padding:
                layer_def["layout"]["icon-text-fit-padding"] = icon_text_fit_padding

            layer_def["layout"]["icon-anchor"] = IconPropertyExtractor.get_icon_anchor()
            layer_def["layout"][
                "icon-rotation-alignment"
            ] = IconPropertyExtractor.get_icon_rotation_alignment()
            layer_def["layout"][
                "icon-pitch-alignment"
            ] = IconPropertyExtractor.get_icon_pitch_alignment()
            layer_def["layout"][
                "icon-allow-overlap"
            ] = IconPropertyExtractor.get_icon_allow_overlap()
            layer_def["layout"][
                "icon-ignore-placement"
            ] = IconPropertyExtractor.get_icon_ignore_placement()
            layer_def["layout"][
                "icon-keep-upright"
            ] = IconPropertyExtractor.get_icon_keep_upright()
            layer_def["layout"]["icon-offset"] = IconPropertyExtractor.get_icon_offset(background)

            # Icon paint properties
            try:
                layer_def["paint"]["icon-opacity"] = IconPropertyExtractor.get_icon_opacity(
                    background
                )
                layer_def["paint"]["icon-color"] = IconPropertyExtractor.get_icon_color(background)
                layer_def["paint"]["icon-halo-color"] = IconPropertyExtractor.get_icon_halo_color()
                layer_def["paint"]["icon-halo-width"] = IconPropertyExtractor.get_icon_halo_width()
                layer_def["paint"]["icon-halo-blur"] = IconPropertyExtractor.get_icon_halo_blur()
                layer_def["paint"]["icon-translate"] = IconPropertyExtractor.get_icon_translate()
                layer_def["paint"][
                    "icon-translate-anchor"
                ] = IconPropertyExtractor.get_icon_translate_anchor()
            except (OSError, RuntimeError):
                pass
        else:
            # Default icon properties even without background
            layer_def["layout"][
                "icon-allow-overlap"
            ] = IconPropertyExtractor.get_icon_allow_overlap()
            layer_def["layout"][
                "icon-ignore-placement"
            ] = IconPropertyExtractor.get_icon_ignore_placement()
            layer_def["layout"]["icon-optional"] = IconPropertyExtractor.get_icon_optional()
            layer_def["paint"]["icon-opacity"] = IconPropertyExtractor.get_icon_opacity()
            layer_def["paint"]["icon-halo-color"] = IconPropertyExtractor.get_icon_halo_color()
            layer_def["paint"]["icon-halo-width"] = IconPropertyExtractor.get_icon_halo_width()
            layer_def["paint"]["icon-halo-blur"] = IconPropertyExtractor.get_icon_halo_blur()
            layer_def["paint"]["icon-translate"] = IconPropertyExtractor.get_icon_translate()
            layer_def["paint"][
                "icon-translate-anchor"
            ] = IconPropertyExtractor.get_icon_translate_anchor()

        self.style["layers"].append(layer_def)

    def to_json(self, indent: int = 2) -> str:
        """Convert the style to JSON string."""
        return json.dumps(self.style, indent=indent)

    def save_to_file(self, filename: str = "style.json", indent: int = 2):
        """Save style and sprites to same directory; add sprite source to style.json."""
        # Create style subdirectory
        style_subdir = "style"
        full_output_dir = os.path.join(self.output_dir, style_subdir)

        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)

        # Generate sprites if markers were collected
        if self.marker_symbols:
            sprite_gen = SpriteGenerator(
                self.marker_symbols, full_output_dir, scale_factor=1, test_mode=False
            )
            sprite_gen.generate()

        # Save style.json
        filepath = os.path.join(full_output_dir, filename)
        with open(filepath, "w", encoding="utf8") as f:
            json.dump(self.style, f, indent=indent)

        return filepath


# Example usage - run in QGIS Python console
if __name__ == "__console__":
    # Convert active layer to MapLibre style
    exporter = QgisMapLibreStyleExporter(output_dir=QgsProcessingUtils.tempFolder())
    output_file = exporter.export()
