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
    """Utility class for extracting and converting PyQGIS properties to MapLibre format."""

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
        """Return static value or a MapLibre ["get", field] expression for data-defined props."""
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
        """Convert QColor to a MapLibre rgba() string."""
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alphaF()})"

    @classmethod
    def convert_length_to_pixels(cls, value: float, unit_obj=None) -> float:
        """Convert a length value from QGIS units to pixels (assumes 96 DPI)."""
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

        return value * 3.78  # Default: treat as millimeters


class LinePropertyExtractor:
    """Extract line paint and layout properties from QGIS symbol layers."""

    @staticmethod
    def get_line_color(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[str, List]:
        """Get line-color."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.color())
        color_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.PropertyFillColor)
        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_line_width(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[float, List]:
        """Get line-width in pixels."""
        width_unit = None
        for attr in ("widthUnit", "widthUnits", "strokeWidthUnit", "strokeWidthUnits"):
            if hasattr(symbol_layer, attr):
                u = getattr(symbol_layer, attr)
                width_unit = u() if callable(u) else u
                break
        width_px = PropertyExtractor.convert_length_to_pixels(symbol_layer.width(), width_unit)
        width_prop = symbol_layer.dataDefinedProperties().property(
            QgsSymbolLayer.PropertyStrokeWidth
        )
        return PropertyExtractor.get_value_or_expression(width_px, width_prop)

    @staticmethod
    def get_line_opacity(symbol_layer: QgsSimpleLineSymbolLayer) -> float:
        """Get line-opacity from alpha channel."""
        return symbol_layer.color().alphaF()

    @staticmethod
    def get_line_cap(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Get line-cap from pen cap style."""
        cap_map = {0: "butt", 16: "square", 32: "round"}
        return cap_map.get(symbol_layer.penCapStyle(), "butt")

    @staticmethod
    def get_line_join(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Get line-join from pen join style."""
        join_map = {0: "miter", 64: "bevel", 128: "round"}
        return join_map.get(symbol_layer.penJoinStyle(), "miter")

    @staticmethod
    def get_line_miter_limit() -> float:
        return 2.0

    @staticmethod
    def get_line_round_limit() -> float:
        return 1.05

    @staticmethod
    def get_line_dasharray(
        symbol_layer: QgsSimpleLineSymbolLayer, width_px: float
    ) -> Optional[List[float]]:
        """Get line-dasharray from custom dash pattern or pen style preset."""
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

        pen_style = None
        try:
            for attr in ("penStyle", "strokeStyle", "pen_style"):
                if hasattr(symbol_layer, attr):
                    v = getattr(symbol_layer, attr)
                    pen_style = v() if callable(v) else v
                    break
        except (RuntimeError, AttributeError):
            pass

        if custom_dash_enabled and dash_vector:
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
    def get_line_offset(symbol_layer: QgsSimpleLineSymbolLayer) -> float:
        """Get line-offset in pixels."""
        try:
            offset = symbol_layer.offset()
            if offset != 0:
                offset_unit = None
                for attr in ("offsetUnit", "offsetUnits"):
                    if hasattr(symbol_layer, attr):
                        u = getattr(symbol_layer, attr)
                        offset_unit = u() if callable(u) else u
                        break
                return PropertyExtractor.convert_length_to_pixels(offset, offset_unit)
        except (RuntimeError, AttributeError):
            pass
        return 0

    @staticmethod
    def get_line_blur() -> float:
        return 0

    @staticmethod
    def get_line_gap_width() -> float:
        return 0

    @staticmethod
    def get_line_translate() -> List[float]:
        return [0, 0]

    @staticmethod
    def get_line_translate_anchor() -> str:
        return "map"


class FillPropertyExtractor:
    """Extract fill paint and layout properties from QGIS symbol layers."""

    @staticmethod
    def get_fill_color(symbol_layer: QgsSimpleFillSymbolLayer) -> Union[str, List]:
        """Get fill-color."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.color())
        color_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.PropertyFillColor)
        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_fill_opacity(symbol_layer: QgsSimpleFillSymbolLayer) -> float:
        """Get fill-opacity from alpha channel."""
        return symbol_layer.color().alphaF()

    @staticmethod
    def get_fill_outline_color(symbol_layer: QgsSimpleFillSymbolLayer) -> Union[str, List, None]:
        """Get fill-outline-color if stroke is visible."""
        if symbol_layer.strokeWidth() > 0 and symbol_layer.strokeStyle() != 0:
            base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.strokeColor())
            color_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.PropertyStrokeColor
            )
            return PropertyExtractor.get_value_or_expression(base_color, color_prop)
        return None

    @staticmethod
    def get_fill_antialias() -> bool:
        return True

    @staticmethod
    def get_fill_translate() -> List[float]:
        return [0, 0]

    @staticmethod
    def get_fill_translate_anchor() -> str:
        return "map"


class IconPropertyExtractor:
    """Extract icon paint and layout properties for symbol layers."""

    @staticmethod
    def get_icon_image(marker_name: str) -> str:
        return marker_name

    @staticmethod
    def get_icon_size(
        symbol_layer: QgsSymbolLayer, default_size: float = 1.0
    ) -> Union[float, List]:
        """Get icon-size, respecting data-defined overrides."""
        size_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.PropertySize)
        return PropertyExtractor.get_value_or_expression(default_size, size_prop)

    @staticmethod
    def get_icon_rotation_alignment() -> str:
        return "map"

    @staticmethod
    def get_icon_pitch_alignment() -> str:
        return "viewport"

    @staticmethod
    def get_icon_anchor() -> str:
        return "center"

    @staticmethod
    def get_icon_allow_overlap() -> bool:
        return False

    @staticmethod
    def get_icon_ignore_placement() -> bool:
        return False

    @staticmethod
    def get_icon_optional() -> bool:
        return False

    @staticmethod
    def get_icon_keep_upright() -> bool:
        return True

    @staticmethod
    def get_icon_text_fit(background: QgsTextBackgroundSettings) -> Optional[str]:
        """Get icon-text-fit if background uses buffer sizing."""
        if background.enabled() and background.sizeType() == 0:
            return "both"
        return None

    @staticmethod
    def get_icon_text_fit_padding(
        background: QgsTextBackgroundSettings,
    ) -> Optional[List[float]]:
        """Get icon-text-fit-padding from background buffer size."""
        if background.enabled() and background.sizeType() == 0:
            buf_px = PropertyExtractor.convert_length_to_pixels(
                background.size().width(), background.sizeUnit()
            )
            return [buf_px * 2, buf_px, buf_px * 2, buf_px]
        return None

    @staticmethod
    def get_icon_offset(background: QgsTextBackgroundSettings = None) -> List[float]:
        """Get icon-offset from background offset settings."""
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
        """Get icon-opacity from background settings."""
        if background and background.enabled():
            try:
                return background.opacity()
            except (OSError, RuntimeError):
                pass
        return 1.0

    @staticmethod
    def get_icon_color(background: QgsTextBackgroundSettings = None) -> str:
        """Get icon-color from background fill color."""
        if background and background.enabled():
            try:
                return PropertyExtractor.convert_qcolor_to_maplibre(background.fillColor())
            except (OSError, RuntimeError):
                pass
        return "rgb(255, 255, 255)"

    @staticmethod
    def get_icon_halo_color() -> str:
        return "rgb(0, 0, 0)"

    @staticmethod
    def get_icon_halo_width() -> float:
        return 0

    @staticmethod
    def get_icon_halo_blur() -> float:
        return 0

    @staticmethod
    def get_icon_translate() -> List[float]:
        return [0, 0]

    @staticmethod
    def get_icon_translate_anchor() -> str:
        return "map"


class TextPropertyExtractor:
    """Extract text paint and layout properties from QGIS label settings."""

    @staticmethod
    def get_text_field(label_settings: QgsPalLayerSettings) -> Optional[List]:
        """Get text-field expression."""
        if label_settings.fieldName:
            return ["get", label_settings.fieldName]
        return None

    @staticmethod
    def get_text_font(text_format: QgsTextFormat) -> List[str]:
        """Get text-font as [family + style] list."""
        font = text_format.font()
        return [f"{font.family()} {font.styleName()}"]

    @staticmethod
    def get_text_size(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[float, List]:
        """Get text-size in pixels, respecting data-defined overrides."""
        base_size = text_format.font().pointSizeF() * (96.0 / 72.0)
        size_prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Size)
        return PropertyExtractor.get_value_or_expression(base_size, size_prop)

    @staticmethod
    def get_text_color(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[str, List]:
        """Get text-color, respecting data-defined overrides."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(text_format.color())
        color_prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Color)
        return PropertyExtractor.get_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_text_opacity(text_format: QgsTextFormat) -> float:
        """Get text-opacity from alpha channel."""
        return text_format.color().alphaF()

    @staticmethod
    def get_text_halo_color(text_format: QgsTextFormat) -> str:
        """Get text-halo-color from buffer settings."""
        buffer = text_format.buffer()
        if buffer.enabled():
            return PropertyExtractor.convert_qcolor_to_maplibre(buffer.color())
        return "rgb(255, 255, 255)"

    @staticmethod
    def get_text_halo_width(text_format: QgsTextFormat) -> float:
        """Get text-halo-width from buffer size."""
        buffer = text_format.buffer()
        return buffer.size() if buffer.enabled() else 0

    @staticmethod
    def get_text_halo_blur(text_format: QgsTextFormat) -> float:
        """Get text-halo-blur as half the buffer size."""
        buffer = text_format.buffer()
        return buffer.size() * 0.5 if buffer.enabled() else 0

    @staticmethod
    def get_text_anchor(label_settings: QgsPalLayerSettings) -> str:
        """Get text-anchor from quadrant offset."""
        anchor_map = {
            0: "bottom-right", 1: "bottom",  2: "bottom-left",
            3: "right",        4: "center",  5: "left",
            6: "top-right",    7: "top",     8: "top-left",
        }
        return anchor_map.get(label_settings.quadOffset, "center")

    @staticmethod
    def get_text_justify(label_settings: QgsPalLayerSettings) -> str:
        """Get text-justify from multi-line alignment."""
        try:
            justification = label_settings.multiLineAlignment
        except AttributeError:
            try:
                justification = label_settings.alignment
            except AttributeError:
                justification = 1
        justify_map = {0: "left", 1: "center", 2: "right", 3: "center"}
        return justify_map.get(justification, "left")

    @staticmethod
    def get_text_offset(label_settings: QgsPalLayerSettings) -> List[float]:
        """Get text-offset in pixels."""
        x_offset = label_settings.xOffset
        y_offset = label_settings.yOffset
        if x_offset != 0 or y_offset != 0:
            offset_unit = None
            for attr in ("xOffsetUnit", "offsetUnit", "units"):
                if hasattr(label_settings, attr):
                    u = getattr(label_settings, attr)
                    offset_unit = u() if callable(u) else u
                    break
            return [
                PropertyExtractor.convert_length_to_pixels(x_offset, offset_unit),
                PropertyExtractor.convert_length_to_pixels(y_offset, offset_unit),
            ]
        return [0, 0]

    @staticmethod
    def get_text_allow_overlap() -> bool:
        return False

    @staticmethod
    def get_text_ignore_placement() -> bool:
        return False

    @staticmethod
    def get_text_optional() -> bool:
        return False

    @staticmethod
    def get_text_padding() -> float:
        return 0

    @staticmethod
    def get_text_line_height() -> float:
        return 1.2

    @staticmethod
    def get_text_letter_spacing() -> float:
        return 0

    @staticmethod
    def get_text_transform() -> str:
        return "none"

    @staticmethod
    def get_text_max_width(label_settings: QgsPalLayerSettings) -> float:
        """Get text-max-width; large value means no effective limit."""
        if label_settings.autoWrapLength > 0:
            return label_settings.autoWrapLength
        return 999

    @staticmethod
    def get_text_keep_upright() -> bool:
        return True

    @staticmethod
    def get_text_rotate(label_settings: QgsPalLayerSettings) -> Union[float, List]:
        """Get text-rotate, respecting data-defined overrides."""
        base_rotation = label_settings.angleOffset if label_settings.angleOffset != 0 else 0
        rotation_prop = label_settings.dataDefinedProperties().property(
            QgsPalLayerSettings.LabelRotation
        )
        return PropertyExtractor.get_value_or_expression(base_rotation, rotation_prop)

    @staticmethod
    def get_text_rotation_alignment() -> str:
        return "map"

    @staticmethod
    def get_text_pitch_alignment() -> str:
        return "viewport"

    @staticmethod
    def get_text_translate() -> List[float]:
        return [0, 0]

    @staticmethod
    def get_text_translate_anchor() -> str:
        return "map"


class QgisMapLibreStyleExporter:
    """Export QGIS Vector Tile Layer styles to a MapLibre GL style JSON."""

    def __init__(
        self,
        output_dir: str,
        layer: Optional[QgsVectorTileLayer] = None,
        background_type: int = 0,
    ):
        self.output_dir = output_dir
        self.marker_symbols: dict = {}
        self.marker_counter = 0

        self.layer = self._resolve_layer(layer)
        self.source_name = "q2vt_tiles"
        self.style = self._build_style_skeleton()
        self.style["layers"].append(self._build_background_layer(background_type))

    def _resolve_layer(self, layer: Optional[QgsVectorTileLayer]) -> QgsVectorTileLayer:
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
        """Add background tile source to style and return the corresponding layer definition."""
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
        """Convert all styles from the layer to MapLibre GL style and save to disk."""
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
        """Convert a single QgsVectorTileBasicRendererStyle to MapLibre layer(s)."""
        if not style.isEnabled() or not style.symbol():
            return
        self._convert_symbol(
            style.symbol(), style.styleName(), style.layerName(),
            self.source_name, style.minZoomLevel(), style.maxZoomLevel() + 1,
        )

    def _convert_labeling_style(self, style):
        """Convert a single QgsVectorTileBasicLabelingStyle to a MapLibre symbol layer."""
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
        """Dispatch symbol conversion by type."""
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
        """Build the common layer definition skeleton."""
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
        """Convert marker symbol to a MapLibre symbol layer."""
        layer_def = self._base_layer_def(
            "symbol", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        marker_name = f"marker_{self.marker_counter}"
        self.marker_counter += 1
        self.marker_symbols[marker_name] = symbol.clone()

        layer_def["layout"].update({
            "icon-image": IconPropertyExtractor.get_icon_image(marker_name),
            "icon-size": IconPropertyExtractor.get_icon_size(symbol_layer, 1.0),
            "icon-rotation-alignment": IconPropertyExtractor.get_icon_rotation_alignment(),
            "icon-pitch-alignment": IconPropertyExtractor.get_icon_pitch_alignment(),
            "icon-anchor": IconPropertyExtractor.get_icon_anchor(),
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(),
            "icon-ignore-placement": IconPropertyExtractor.get_icon_ignore_placement(),
            "visibility": "visible",
        })
        layer_def["paint"].update({
            "icon-opacity": IconPropertyExtractor.get_icon_opacity(),
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
        """Convert line symbol to a MapLibre line layer."""
        layer_def = self._base_layer_def(
            "line", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        if isinstance(symbol_layer, QgsSimpleLineSymbolLayer):
            layer_def["paint"].update({
                "line-color": LinePropertyExtractor.get_line_color(symbol_layer),
                "line-width": LinePropertyExtractor.get_line_width(symbol_layer),
                "line-opacity": LinePropertyExtractor.get_line_opacity(symbol_layer),
                # "line-blur": LinePropertyExtractor.get_line_blur(),
                "line-gap-width": LinePropertyExtractor.get_line_gap_width(),
                "line-translate": LinePropertyExtractor.get_line_translate(),
                "line-translate-anchor": LinePropertyExtractor.get_line_translate_anchor(),
            })

            offset = LinePropertyExtractor.get_line_offset(symbol_layer)
            if offset != 0:
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
        """Convert fill symbol to a MapLibre fill layer."""
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
            # outline_color = FillPropertyExtractor.get_fill_outline_color(symbol_layer)
            # if outline_color:
            #     layer_def["paint"]["fill-outline-color"] = outline_color

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
        """Convert QgsPalLayerSettings to a MapLibre symbol layer."""
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
            "text-allow-overlap": TextPropertyExtractor.get_text_allow_overlap(),
            "text-ignore-placement": TextPropertyExtractor.get_text_ignore_placement(),
            "text-optional": TextPropertyExtractor.get_text_optional(),
            "text-padding": TextPropertyExtractor.get_text_padding(),
            "text-line-height": TextPropertyExtractor.get_text_line_height(),
            "text-letter-spacing": TextPropertyExtractor.get_text_letter_spacing(),
            "text-transform": TextPropertyExtractor.get_text_transform(),
            "text-max-width": TextPropertyExtractor.get_text_max_width(label_settings),
            "text-keep-upright": TextPropertyExtractor.get_text_keep_upright(),
            "text-rotate": TextPropertyExtractor.get_text_rotate(label_settings),
            "text-rotation-alignment": TextPropertyExtractor.get_text_rotation_alignment(),
            "text-pitch-alignment": TextPropertyExtractor.get_text_pitch_alignment(),
            "visibility": "visible",
        })

        layer_def["paint"].update({
            "text-color": TextPropertyExtractor.get_text_color(text_format, label_settings),
            "text-opacity": TextPropertyExtractor.get_text_opacity(text_format),
            "text-halo-color": TextPropertyExtractor.get_text_halo_color(text_format),
            "text-halo-width": TextPropertyExtractor.get_text_halo_width(text_format),
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
                "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(),
                "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(),
                # "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
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
        })
        layer_def["paint"].update({
            "icon-opacity": IconPropertyExtractor.get_icon_opacity(),
            "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(),
            "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(),
            # "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
            "icon-translate": IconPropertyExtractor.get_icon_translate(),
            "icon-translate-anchor": IconPropertyExtractor.get_icon_translate_anchor(),
        })

    def to_json(self, indent: int = 2) -> str:
        """Serialize the style dict to a JSON string."""
        return json.dumps(self.style, indent=indent)

    def save_to_file(self, filename: str = "style.json", indent: int = 2) -> str:
        """Save style JSON and sprites to the output directory."""
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
