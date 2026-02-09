"""Convert QGIS Vector Tile Layer styles to MapLibre GL JSON style format."""

from qgis.core import (
    QgsProject,
    QgsVectorTileLayer,
    QgsVectorTileBasicRenderer,
    QgsVectorTileBasicLabeling,
    QgsSymbol,
    QgsPalLayerSettings,
    QgsSimpleMarkerSymbolLayer,
    QgsSimpleLineSymbolLayer,
    QgsSimpleFillSymbolLayer,
    QgsSvgMarkerSymbolLayer,
    QgsProperty,
    QgsWkbTypes,
    QgsSymbolLayer,
)
from qgis.PyQt.QtGui import QColor
import json
import os
from typing import Dict, Any, Union, Optional
from datetime import datetime


class QgisToMapLibreConverter:
    """Converts QGIS Vector Tile Layer styles to MapLibre GL style JSON."""

    def __init__(self, output_dir: str, layer: Optional[QgsVectorTileLayer] = None):
        """Initialize converter with output directory and optional QgsVectorTileLayer."""
        self.output_dir = output_dir

        # Get layer (use active layer if not provided)
        if layer is None:
            try:
                from qgis.utils import iface

                if iface and iface.activeLayer():
                    layer = iface.activeLayer()
                else:
                    raise ValueError("No active layer found and no layer provided")
            except ImportError:
                raise ValueError(
                    "Cannot access active layer (iface not available). Please provide a layer explicitly."
                )

        if not isinstance(layer, QgsVectorTileLayer):
            raise ValueError(f"Layer must be a QgsVectorTileLayer, got {type(layer).__name__}")

        self.layer = layer
        self.source_name = layer.name().replace(" ", "_").lower()

        # Initialize style structure
        self.style = {
            "version": 8,
            "name": f"{layer.name()} - Converted Style",
            "sources": {},
            "layers": [],
        }

    def convert(self) -> Dict[str, Any]:
        """Convert all styles from QgsVectorTileLayer to MapLibre GL style."""
        # Add source for the layer
        source_url = self.layer.source()
        self.style["sources"][self.source_name] = {
            "type": "vector",
            "tiles": (
                [source_url]
                if source_url
                else [f"{{source}}/{self.source_name}/{{z}}/{{x}}/{{y}}.pbf"]
            ),
        }

        # Extract renderer styles
        renderer = self.layer.renderer()
        if isinstance(renderer, QgsVectorTileBasicRenderer):
            for style_index, style in enumerate(renderer.styles()):
                self._convert_renderer_style(style, style_index)

        # Extract labeling styles
        labeling = self.layer.labeling()
        if isinstance(labeling, QgsVectorTileBasicLabeling):
            for style_index, style in enumerate(labeling.styles()):
                self._convert_labeling_style(style, style_index)

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

    def _convert_renderer_style(self, style, style_index: int):
        """Convert a single renderer style from QgsVectorTileBasicRendererStyle."""
        layer_name = style.layerName()
        symbol = style.symbol()
        min_zoom = style.minZoomLevel()
        max_zoom = style.maxZoomLevel()
        enabled = style.isEnabled()

        if not enabled or not symbol:
            return

        # Generate style name based on layer name and symbol type (no index - allows merging across zoom levels)
        symbol_type_code = self._get_symbol_type_code(symbol.type())
        style_name = f"{layer_name}_{symbol_type_code}"

        # Convert the symbol with zoom levels, using generated layer ID and layer_name as source-layer
        self._convert_symbol(symbol, style_name, layer_name, self.source_name, min_zoom, max_zoom)

    def _convert_labeling_style(self, style, style_index: int):
        """Convert a single labeling style from QgsVectorTileBasicLabelingStyle."""
        layer_name = style.layerName()
        label_settings = style.labelSettings()
        min_zoom = style.minZoomLevel()
        max_zoom = style.maxZoomLevel()
        enabled = style.isEnabled()

        if not enabled or not label_settings:
            return

        # Generate style name for labels (no index - allows merging across zoom levels)
        style_name = f"{layer_name}_label"

        # Convert the label settings with zoom levels, using generated layer ID and layer_name as source-layer
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

        # Get the first symbol layer (simplified approach)
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
                    symbol_layer,
                    symbol,
                    style_name,
                    source_layer_name,
                    source_name,
                    min_zoom,
                    max_zoom,
                )
            elif symbol_type == QgsSymbol.Fill:
                self._convert_fill_symbol(
                    symbol_layer,
                    symbol,
                    style_name,
                    source_layer_name,
                    source_name,
                    min_zoom,
                    max_zoom,
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
        """Convert marker symbol to MapLibre circle or symbol layer."""
        layer_def = {
            "id": style_name,
            "type": "circle",  # Default to circle
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

        if isinstance(symbol_layer, QgsSimpleMarkerSymbolLayer):
            # Circle/Symbol properties
            color = symbol_layer.color()
            stroke_color = symbol_layer.strokeColor()
            size = symbol_layer.size()

            layer_def["paint"]["circle-color"] = self._qcolor_to_maplibre(color)
            layer_def["paint"]["circle-radius"] = self._convert_size(size)
            layer_def["paint"]["circle-stroke-color"] = self._qcolor_to_maplibre(stroke_color)
            layer_def["paint"]["circle-stroke-width"] = symbol_layer.strokeWidth()
            layer_def["paint"]["circle-stroke-opacity"] = stroke_color.alphaF()
            layer_def["paint"]["circle-opacity"] = color.alphaF()
            
            # Circle additional properties
            layer_def["layout"]["visibility"] = "visible"
            layer_def["paint"]["circle-blur"] = 0  # Default blur
            layer_def["paint"]["circle-pitch-scale"] = "map"  # Default pitch scale
            layer_def["paint"]["circle-pitch-alignment"] = "viewport"  # Default pitch alignment
            layer_def["paint"]["circle-translate"] = [0, 0]  # Default no translation
            layer_def["paint"]["circle-translate-anchor"] = "map"  # Default anchor

            # Check for data-defined properties
            self._apply_data_defined_properties(
                symbol_layer,
                layer_def,
                {
                    "color": ("paint", "circle-color"),
                    "size": ("paint", "circle-radius"),
                    "stroke_color": ("paint", "circle-stroke-color"),
                    "stroke_width": ("paint", "circle-stroke-width"),
                },
            )

        elif isinstance(symbol_layer, QgsSvgMarkerSymbolLayer):
            # Use sprite-based icon
            layer_def["type"] = "symbol"
            layer_def["layout"]["icon-image"] = style_name  # Reference sprite
            layer_def["layout"]["visibility"] = "visible"

            size = symbol_layer.size()
            layer_def["layout"]["icon-size"] = (
                self._convert_size(size) / 24.0
            )  # Normalize to sprite size
            
            # Icon additional properties
            layer_def["layout"]["icon-rotation-alignment"] = "map"
            layer_def["layout"]["icon-pitch-alignment"] = "viewport"
            layer_def["layout"]["icon-anchor"] = "center"
            layer_def["layout"]["icon-allow-overlap"] = False
            layer_def["layout"]["icon-ignore-placement"] = False
            layer_def["paint"]["icon-opacity"] = 1.0

            # Check for data-defined size
            size_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.PropertySize)
            if size_prop and size_prop.isActive():
                field_name = self._get_field_from_property(size_prop)
                if field_name:
                    layer_def["layout"]["icon-size"] = ["get", field_name]

        self.style["layers"].append(layer_def)

    def _convert_line_symbol(
        self,
        symbol_layer: QgsSymbolLayer,
        symbol: QgsSymbol,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert line symbol to MapLibre line layer."""
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
            color = symbol_layer.color()
            width = symbol_layer.width()

            layer_def["paint"]["line-color"] = self._qcolor_to_maplibre(color)
            layer_def["paint"]["line-width"] = width
            layer_def["paint"]["line-opacity"] = color.alphaF()

            # Line cap and join
            pen_cap_style = symbol_layer.penCapStyle()
            pen_join_style = symbol_layer.penJoinStyle()

            layer_def["layout"]["line-cap"] = self._convert_line_cap(pen_cap_style)
            layer_def["layout"]["line-join"] = self._convert_line_join(pen_join_style)
            layer_def["layout"]["visibility"] = "visible"
            
            # Line miter and round limits
            layer_def["layout"]["line-miter-limit"] = 2.0  # Default MapLibre value
            layer_def["layout"]["line-round-limit"] = 1.05  # Default MapLibre value

            # Line dash pattern with custom dash vector
            dash_vector = symbol_layer.customDashVector()
            if dash_vector and len(dash_vector) > 0:
                # Convert QGIS dash vector to MapLibre line-dasharray
                # Scale the dash pattern by line width for proper visualization
                dasharray = [dash * width for dash in dash_vector]
                layer_def["paint"]["line-dasharray"] = dasharray
            
            # Line offset (QGIS offset property)
            try:
                offset = symbol_layer.offset()
                if offset != 0:
                    layer_def["paint"]["line-offset"] = offset
            except:
                pass
            
            # Line additional properties
            layer_def["paint"]["line-blur"] = 0  # Default blur
            layer_def["paint"]["line-gap-width"] = 0  # Default gap
            layer_def["paint"]["line-translate"] = [0, 0]  # Default no translation
            layer_def["paint"]["line-translate-anchor"] = "map"  # Default anchor

            # Check for data-defined properties
            self._apply_data_defined_properties(
                symbol_layer,
                layer_def,
                {"color": ("paint", "line-color"), "width": ("paint", "line-width")},
            )

        self.style["layers"].append(layer_def)

    def _convert_fill_symbol(
        self,
        symbol_layer: QgsSymbolLayer,
        symbol: QgsSymbol,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert fill symbol to MapLibre fill layer."""
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
            fill_color = symbol_layer.color()
            stroke_color = symbol_layer.strokeColor()
            stroke_width = symbol_layer.strokeWidth()
            stroke_style = symbol_layer.strokeStyle()

            layer_def["paint"]["fill-color"] = self._qcolor_to_maplibre(fill_color)
            layer_def["paint"]["fill-opacity"] = fill_color.alphaF()
            layer_def["layout"]["visibility"] = "visible"
            
            # Fill additional properties
            layer_def["paint"]["fill-antialias"] = True  # Default antialias enabled
            layer_def["paint"]["fill-translate"] = [0, 0]  # Default no translation
            layer_def["paint"]["fill-translate-anchor"] = "map"  # Default anchor

            # Only add outline if stroke width is greater than 0
            if stroke_width > 0 and stroke_style != 0:
                layer_def["paint"]["fill-outline-color"] = self._qcolor_to_maplibre(stroke_color)

            # Check for data-defined properties
            self._apply_data_defined_properties(
                symbol_layer,
                layer_def,
                {
                    "color": ("paint", "fill-color"),
                    "outline_color": ("paint", "fill-outline-color"),
                },
            )

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
        """Convert QgsPalLayerSettings to MapLibre symbol layer with text."""
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

        # Text field
        field_name = label_settings.fieldName
        if field_name:
            layer_def["layout"]["text-field"] = ["get", field_name]

        # Text format
        text_format = label_settings.format()
        font = text_format.font()

        layer_def["layout"]["text-font"] = [font.family()]
        layer_def["layout"]["text-size"] = font.pointSizeF()
        layer_def["layout"]["visibility"] = "visible"
        
        # Text style properties
        layer_def["layout"]["text-allow-overlap"] = not label_settings.isObstacleForPolygon()
        layer_def["layout"]["text-ignore-placement"] = False
        layer_def["layout"]["text-optional"] = False
        layer_def["layout"]["text-padding"] = 0
        
        # Text justification
        justification = label_settings.alignment
        layer_def["layout"]["text-justify"] = self._convert_text_justification(justification)
        
        # Text line height and letter spacing
        layer_def["layout"]["text-line-height"] = 1.2
        layer_def["layout"]["text-letter-spacing"] = 0  # Default no extra spacing
        
        # Text transform
        layer_def["layout"]["text-transform"] = "none"  # Default no transform
        
        # Text max width (wrap text)
        layer_def["layout"]["text-max-width"] = 10  # Default wrap at 10em

        # Text color
        color = text_format.color()
        layer_def["paint"]["text-color"] = self._qcolor_to_maplibre(color)
        layer_def["paint"]["text-opacity"] = color.alphaF()

        # Halo (buffer)
        buffer = text_format.buffer()
        if buffer.enabled():
            layer_def["paint"]["text-halo-width"] = buffer.size()
            layer_def["paint"]["text-halo-color"] = self._qcolor_to_maplibre(buffer.color())
            layer_def["paint"]["text-halo-blur"] = buffer.size() * 0.5
        else:
            layer_def["paint"]["text-halo-width"] = 0
            layer_def["paint"]["text-halo-color"] = "rgba(255, 255, 255, 1.00)"
            layer_def["paint"]["text-halo-blur"] = 0

        # Text offset
        x_offset = label_settings.xOffset
        y_offset = label_settings.yOffset
        if x_offset != 0 or y_offset != 0:
            layer_def["layout"]["text-offset"] = [x_offset, y_offset]
        else:
            layer_def["layout"]["text-offset"] = [0, 0]

        # Text anchor/alignment
        layer_def["layout"]["text-anchor"] = self._convert_text_anchor(label_settings)
        layer_def["layout"]["text-keep-upright"] = True

        # Text rotation
        if label_settings.angleOffset != 0:
            layer_def["layout"]["text-rotate"] = label_settings.angleOffset
        else:
            layer_def["layout"]["text-rotate"] = 0
        
        # Text rotation alignment
        layer_def["layout"]["text-rotation-alignment"] = "map"
        layer_def["layout"]["text-pitch-alignment"] = "viewport"

        # Text translate properties
        layer_def["paint"]["text-translate"] = [0, 0]  # Default no translation
        layer_def["paint"]["text-translate-anchor"] = "map"  # Default anchor

        # Check for data-defined properties
        dd_props = label_settings.dataDefinedProperties()

        # Size
        size_prop = dd_props.property(QgsPalLayerSettings.Size)
        if size_prop and size_prop.isActive():
            field = self._get_field_from_property(size_prop)
            if field:
                layer_def["layout"]["text-size"] = ["get", field]

        # Color
        color_prop = dd_props.property(QgsPalLayerSettings.Color)
        if color_prop and color_prop.isActive():
            field = self._get_field_from_property(color_prop)
            if field:
                layer_def["paint"]["text-color"] = ["get", field]

        # Rotation
        rotation_prop = dd_props.property(QgsPalLayerSettings.LabelRotation)
        if rotation_prop and rotation_prop.isActive():
            field = self._get_field_from_property(rotation_prop)
            if field:
                layer_def["layout"]["text-rotate"] = ["get", field]

        # Background (uses sprite if enabled)
        background = label_settings.format().background()
        if background.enabled():
            # Assume sprite contains background icon with style name
            layer_def["layout"]["icon-image"] = style_name
            layer_def["layout"]["icon-text-fit"] = "both"
            layer_def["layout"]["icon-text-fit-padding"] = [4, 2, 4, 2]
            layer_def["layout"]["icon-anchor"] = "center"
            layer_def["layout"]["icon-rotation-alignment"] = "map"
            layer_def["layout"]["icon-pitch-alignment"] = "viewport"
            layer_def["layout"]["icon-allow-overlap"] = False
            layer_def["layout"]["icon-ignore-placement"] = False
            layer_def["layout"]["icon-keep-upright"] = True
            try:
                layer_def["paint"]["icon-opacity"] = background.opacity()
                bg_color = background.fillColor()
                layer_def["paint"]["icon-color"] = self._qcolor_to_maplibre(bg_color)
                layer_def["paint"]["icon-halo-color"] = "rgba(0, 0, 0, 0.50)"
                layer_def["paint"]["icon-halo-width"] = 0
                layer_def["paint"]["icon-halo-blur"] = 0
                layer_def["paint"]["icon-translate"] = [0, 0]
                layer_def["paint"]["icon-translate-anchor"] = "map"
            except:
                pass
        else:
            # Default icon properties even without background
            layer_def["layout"]["icon-allow-overlap"] = False
            layer_def["layout"]["icon-ignore-placement"] = False
            layer_def["layout"]["icon-optional"] = True
            layer_def["paint"]["icon-opacity"] = 1.0
            layer_def["paint"]["icon-halo-color"] = "rgba(0, 0, 0, 0.00)"
            layer_def["paint"]["icon-halo-width"] = 0
            layer_def["paint"]["icon-halo-blur"] = 0
            layer_def["paint"]["icon-translate"] = [0, 0]
            layer_def["paint"]["icon-translate-anchor"] = "map"

        self.style["layers"].append(layer_def)

    def _apply_data_defined_properties(
        self,
        symbol_layer: QgsSymbolLayer,
        layer_def: Dict[str, Any],
        property_map: Dict[str, tuple],
    ):
        """Apply data-defined properties from QGIS to MapLibre layer."""
        dd_props = symbol_layer.dataDefinedProperties()

        # Map QGIS property types to their names
        qgis_property_map = {
            "size": QgsSymbolLayer.PropertySize,
            "color": QgsSymbolLayer.PropertyFillColor,
            "stroke_color": QgsSymbolLayer.PropertyStrokeColor,
            "width": QgsSymbolLayer.PropertyStrokeWidth,
            "outline_color": QgsSymbolLayer.PropertyStrokeColor,
        }

        for prop_name, (section, maplibre_prop) in property_map.items():
            if prop_name in qgis_property_map:
                qgis_prop = dd_props.property(qgis_property_map[prop_name])

                if qgis_prop and qgis_prop.isActive():
                    field_name = self._get_field_from_property(qgis_prop)

                    if field_name:
                        # Handle color properties specially
                        if "color" in prop_name.lower():
                            layer_def[section][maplibre_prop] = ["get", field_name]
                        else:
                            # Numeric properties
                            layer_def[section][maplibre_prop] = ["get", field_name]

    def _get_field_from_property(self, prop: QgsProperty) -> Optional[str]:
        """Extract field name from a QGIS property."""
        if prop.propertyType() == QgsProperty.FieldBasedProperty:
            return prop.field()
        elif prop.propertyType() == QgsProperty.ExpressionBasedProperty:
            # Try to extract field from simple expression
            expr = prop.expressionString()
            # Handle simple cases like "field_name" or '"field_name"'
            expr = expr.strip('"').strip("'")
            return expr if expr else None
        return None

    def _qcolor_to_maplibre(self, color: QColor) -> str:
        """Convert QColor to MapLibre rgba format."""
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alphaF():.2f})"

    def _convert_size(self, size: float) -> float:
        """Convert QGIS size (mm) to MapLibre size (pixels). 1mm ≈ 3.78 px at 96 DPI."""
        return size * 3.78

    def _convert_line_cap(self, cap_style) -> str:
        """Convert Qt pen cap style to MapLibre line-cap."""
        # Qt.FlatCap = 0, Qt.SquareCap = 16, Qt.RoundCap = 32
        cap_map = {
            0: "butt",  # Qt.FlatCap
            16: "square",  # Qt.SquareCap
            32: "round",  # Qt.RoundCap
        }
        return cap_map.get(cap_style, "butt")

    def _convert_line_join(self, join_style) -> str:
        """Convert Qt pen join style to MapLibre line-join."""
        # Qt.MiterJoin = 0, Qt.BevelJoin = 64, Qt.RoundJoin = 128
        join_map = {
            0: "miter",  # Qt.MiterJoin
            64: "bevel",  # Qt.BevelJoin
            128: "round",  # Qt.RoundJoin
        }
        return join_map.get(join_style, "miter")

    def _convert_text_anchor(self, label_settings: QgsPalLayerSettings) -> str:
        """Convert QGIS label placement quadrant to MapLibre text-anchor."""
        # This is a simplified conversion
        # QGIS has complex placement options, MapLibre has simpler anchors
        quad = label_settings.quadOffset

        # QgsPalLayerSettings.QuadrantPosition enum:
        # QuadrantAboveLeft = 0, QuadrantAbove = 1, QuadrantAboveRight = 2
        # QuadrantLeft = 3, QuadrantOver = 4, QuadrantRight = 5
        # QuadrantBelowLeft = 6, QuadrantBelow = 7, QuadrantBelowRight = 8

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

    def _convert_text_justification(self, alignment) -> str:
        """Convert QGIS text alignment to MapLibre text-justify."""
        # QgsPalLayerSettings alignment: 0=Left, 1=Center, 2=Right, 3=Justify
        justify_map = {
            0: "left",    # Left
            1: "center",  # Center
            2: "right",   # Right
            3: "center",  # Justify (MapLibre doesn't have justify, use center)
        }
        return justify_map.get(alignment, "left")

    def to_json(self, indent: int = 2) -> str:
        """Convert the style to JSON string."""
        return json.dumps(self.style, indent=indent)

    def save_to_file(self, filename: str = "style.json", indent: int = 2):
        """Save the style to a JSON file in a timestamped subdirectory."""
        # Create subdirectory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        style_subdir = f"style_{timestamp}"
        full_output_dir = os.path.join(self.output_dir, style_subdir)

        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)

        filepath = os.path.join(full_output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.style, f, indent=indent)

        return filepath

    def run(self, filename: str = "style.json", indent: int = 2) -> str:
        """Convert the layer styles and save to file."""
        self.convert()
        return self.save_to_file(filename, indent)


# Example usage - run in QGIS Python console
if __name__ == "__console__":
    # Convert active layer to MapLibre style
    from pathlib import Path

    converter = QgisToMapLibreConverter(
        output_dir=r"C:\Users\P0026701\OneDrive - Ness Israel\Desktop\09_02_2026_10_20_06_890302"
    )
    output_file = converter.run()
    print(f"Style saved successfully to: {output_file}")
