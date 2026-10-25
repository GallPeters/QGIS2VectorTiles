"""
tiles_styler.py

TilesStyler — takes the list of FlattenedRules produced by RulesFlattener /
RulesExporter and applies matching QgsVectorTileBasicRenderer /
QgsVectorTileBasicLabeling styles to a new QgsVectorTileLayer, which the
MapLibre style exporter then converts (the layer itself is never loaded).

Depends on: config, flattened_rule
"""

from os.path import join
from typing import Dict, List, Tuple

from qgis.core import (
    QgsVectorTileLayer,
    QgsVectorTileBasicRenderer,
    QgsVectorTileBasicRendererStyle,
    QgsVectorTileBasicLabeling,
    QgsVectorTileBasicLabelingStyle,
    QgsPalLayerSettings,
    QgsWkbTypes,
    QgsMarkerSymbol,
    QgsLineSymbol,
    QgsFillSymbol,
    QgsLabelThinningSettings,
    QgsProcessingUtils,
    QgsProperty,
    QgsSymbolLayer,
    Qgis,
)

from ..utils.config import _REMOVE_DUPLICATES_DISTANCE
from ..utils.zoom_levels import ZoomLevels
from ..utils.flattened_rule import (
    FlattenedRule, MARKER_ALL_VERTICES, MARKER_CENTRAL_POINT, MARKER_FIRST_VERTEX,
    MARKER_INNER_VERTICES, MARKER_LAST_VERTEX,
    marker_angle_field,
)
from .rules_exporter import RULE_FIELD


class TilesStyler:
    """Apply FlattenedRule styling to a new QgsVectorTileLayer and export the style."""

    _GEOM_TYPE_MAP = {
        0: QgsWkbTypes.PointGeometry,
        1: QgsWkbTypes.LineGeometry,
        2: QgsWkbTypes.PolygonGeometry,
    }
    _SYMBOL_CLASS_MAP = {
        0: QgsMarkerSymbol,
        1: QgsLineSymbol,
        2: QgsFillSymbol,
    }

    def __init__(self, flattened_rules: List[FlattenedRule], output_dir: str):
        self.flattened_rules = flattened_rules
        self.output_dir = output_dir or QgsProcessingUtils.tempFolder()
        self.tiles_layer = self._create_tiles_layer()
        self._draw_order: dict = {}
        self.renderer_styles: List[QgsVectorTileBasicRendererStyle] = []
        self.labeling_styles: List[QgsVectorTileBasicLabelingStyle] = []
        # Per style name (unique): the exact, fractional zooms its rule is
        # drawn from and until (exclusive), from the rule's scale range.
        # The styles' whole zoom levels are the tiles that hold the data.
        self.zoom_ranges: Dict[str, Tuple[float, float]] = {}

    def apply_styling(self) -> QgsVectorTileLayer:
        """Apply all rule styles to the tiles layer."""
        self._draw_order = self._get_draw_order()
        sorted_rules = reversed(sorted(self.flattened_rules, key=lambda x:self.get_label_priority(x)))
        for rule in sorted_rules:
            self._create_style_from_rule(rule)
        self._apply_styles_to_layer()
        return self.tiles_layer

    def _get_draw_order(self) -> dict:
        """Each rule's position in the top-to-bottom drawing order.

        The flattened rules are already top to bottom, but a layer drawn
        with symbol levels draws all its lower rendering passes first (all
        road casings below all road fills), so its renderer rules are
        reordered by rendering pass.
        """
        order = []
        done_layers = set()
        for flat_rule in self.flattened_rules:
            layer = flat_rule.layer
            if flat_rule.get_attr("t") != 0 or not self._uses_symbol_levels(layer):
                order.append(flat_rule)
                continue
            if layer.id() in done_layers:
                continue
            done_layers.add(layer.id())
            layer_rules = [
                r for r in self.flattened_rules
                if r.get_attr("t") == 0 and r.layer.id() == layer.id()
            ]
            order.extend(sorted(layer_rules, key=lambda r: -self._rendering_pass(r)))
        return {id(flat_rule): index for index, flat_rule in enumerate(order)}

    @staticmethod
    def _uses_symbol_levels(layer) -> bool:
        try:
            return bool(layer.renderer() and layer.renderer().usingSymbolLevels())
        except (AttributeError, RuntimeError):
            return False

    @staticmethod
    def _rendering_pass(flat_rule: FlattenedRule) -> int:
        try:
            return flat_rule.rule.symbol().symbolLayer(0).renderingPass()
        except (AttributeError, RuntimeError):
            return 0

    def get_label_priority(self, flat_rule: FlattenedRule) -> float:
        """A rule's sort key: renderer rules by drawing order, label rules by
        their (static) label priority, so higher-priority labels come later
        in the list and so above (and placed before) the others.

        Always a number: a priority read from a field or an expression is
        applied per feature by the style (symbol-sort-key), and the rule
        sorts by its static priority.
        """
        if flat_rule.get_attr("t") == 0:
            return self._draw_order.get(id(flat_rule), self.flattened_rules.index(flat_rule)) + 20
        label_settings = flat_rule.rule.settings()
        priority = label_settings.priority
        prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Property.Priority)
        if prop and prop.isActive():
            try:
                priority = round(float(prop.staticValue() if prop.propertyType() == 1
                                       else prop.expressionString().strip("'")), 2)
            except (ValueError, TypeError):
                pass
        return 11 - priority

    def _create_tiles_layer(self) -> QgsVectorTileLayer:
        """Create a vector tile layer and insert it at the top of the project legend."""
        output = join(self.output_dir, "tiles.mbtiles")
        tiles_path = f"type=mbtiles&url={output}"
        layer = QgsVectorTileLayer(tiles_path, "Vector Tiles")
        # QgsProject.instance().addMapLayer(layer, True)
        # QgsProject.instance().layerTreeRoot().insertLayer(0, layer)
        return layer

    def _create_style_from_rule(self, flat_rule: FlattenedRule):
        """Build and register a renderer or labeling style from the flat rule."""
        if flat_rule.get_attr("t") == 0:
            style = QgsVectorTileBasicRendererStyle()
            self._setup_renderer_style(style, flat_rule)
            self.renderer_styles.append(style)
        else:
            style = QgsVectorTileBasicLabelingStyle()
            self._setup_labeling_style(style, flat_rule)
            self.labeling_styles.append(style)

    def _setup_renderer_style(self, style: QgsVectorTileBasicRendererStyle, flat_rule: FlattenedRule):
        """Configure renderer style, resolving geometry type transformation if needed."""
        self._setup_base_style_properties(style, flat_rule)

        symbol = flat_rule.rule.symbol()
        new_symbol = symbol
        if not symbol:
            return

        symbol_layer = symbol.symbolLayers()[-1]
        sub_symbol = symbol_layer.subSymbol()
        sub_symbol_layer = sub_symbol.symbolLayers()[-1] if sub_symbol else None

        source_geom = int(flat_rule.get_attr("g"))
        target_geom = int(flat_rule.get_attr("c"))

        if source_geom != target_geom:
            if sub_symbol and symbol_layer.layerType() in (
                    "GeometryGenerator", "CentroidFill", "MarkerLine"):
                self._copy_data_driven_properties(symbol, sub_symbol)
                self._copy_data_driven_properties(symbol_layer, sub_symbol_layer)
                new_symbol = sub_symbol
                if symbol_layer.layerType() == "MarkerLine":
                    new_symbol = self._marker_line_points_symbol(flat_rule, symbol_layer, sub_symbol)
            else:
                new_symbol = self._create_transformed_symbol(target_geom, symbol_layer, symbol)
        elif target_geom == 2 and symbol_layer.type() == Qgis.SymbolType.Line:
            # A fill symbol's outline (line) layer, drawn along the polygons.
            new_symbol = self._create_transformed_symbol(1, symbol_layer, symbol)

        new_symbol = new_symbol.clone()
        # The source layer's opacity, which the tiles layer has no place for.
        try:
            new_symbol.setOpacity(new_symbol.opacity() * flat_rule.layer.opacity())
        except (AttributeError, RuntimeError):
            pass
        style.setSymbol(new_symbol)

    @staticmethod
    def _marker_line_points_symbol(flat_rule: FlattenedRule, symbol_layer, sub_symbol):
        """The marker of a marker line drawn at points, turned with the line
        by the exported rotation field when its markers follow the line."""
        marker = sub_symbol.clone()
        if (flat_rule.get_attr("p") in (MARKER_FIRST_VERTEX, MARKER_LAST_VERTEX, MARKER_CENTRAL_POINT,
                                        MARKER_ALL_VERTICES, MARKER_INNER_VERTICES)
                and symbol_layer.rotateSymbols()):
            angle = QgsProperty.fromExpression(f'"{marker_angle_field(flat_rule.get_attr("s") or 0)}"')
            for marker_layer in marker.symbolLayers():
                marker_layer.setDataDefinedProperty(QgsSymbolLayer.Property.PropertyAngle, angle)
        return marker

    def _create_transformed_symbol(self, target_geom: int, symbol_layer, orig_symbol):
        """Create a new symbol of the target geometry type from the given symbol layer."""
        symbol_class = self._SYMBOL_CLASS_MAP.get(target_geom, QgsMarkerSymbol)
        symbol = symbol_class()
        symbol.appendSymbolLayer(symbol_layer.clone())
        symbol.deleteSymbolLayer(0)
        self._copy_data_driven_properties(orig_symbol, symbol)
        if hasattr(symbol, "setOpacity") and hasattr(orig_symbol, "opacity"):
            symbol.setOpacity(orig_symbol.opacity())
        return symbol

    def _setup_labeling_style(
        self, style: QgsVectorTileBasicLabelingStyle, flat_rule: FlattenedRule
    ):
        """Configure labeling style, optionally de-duplicating labels."""
        self._setup_base_style_properties(style, flat_rule)
        settings = QgsPalLayerSettings(flat_rule.rule.settings())
        if Qgis.versionInt() >= 34400:
            self._apply_duplicate_removal(settings)
        style.setLabelSettings(settings)

    def _apply_duplicate_removal(self, settings: QgsPalLayerSettings):
        """Configure label thinning to suppress duplicate labels across tile boundaries."""
        thin = QgsLabelThinningSettings(settings.thinningSettings())
        thin.setAllowDuplicateRemoval(True)
        thin.setMinimumDistanceToDuplicate(_REMOVE_DUPLICATES_DISTANCE)
        thin.setMinimumDistanceToDuplicateUnit(Qgis.RenderUnit.Points)
        settings.setThinningSettings(thin)

    def _setup_base_style_properties(self, style, flat_rule: FlattenedRule):
        """Apply zoom levels, layer name, and geometry type to any style type."""
        style.setEnabled(True)
        style.setLayerName(flat_rule.output_dataset)
        name = flat_rule.rule.description()
        if name in self.zoom_ranges:  # Style names are the web layers' ids.
            name = next(f"{name}_{n}" for n in range(2, 10000) if f"{name}_{n}" not in self.zoom_ranges)
        style.setStyleName(name)
        low, high = flat_rule.get_attr("o"), flat_rule.get_attr("i")
        style.setMinZoomLevel(low)
        style.setMaxZoomLevel(high)
        self.zoom_ranges[name] = self._exact_zooms(flat_rule.rule, low, high)
        if flat_rule.member is not None:
            # A dataset shared by rules differing by filter: this rule's rows.
            style.setFilterExpression(f'"{RULE_FIELD}" = {flat_rule.member}')
        geom_code = flat_rule.get_attr("c")
        style.setGeometryType(
            self._GEOM_TYPE_MAP.get(geom_code, QgsWkbTypes.PointGeometry)
        )

    @staticmethod
    def _exact_zooms(rule, low: int, high: int) -> Tuple[float, float]:
        """The fractional zooms a rule is drawn from and until: its scale
        range as zooms, within its whole zoom levels low..high (high's
        tiles are drawn up to the next zoom)."""
        start, end = float(low), float(high + 1)
        if rule.minimumScale() > 0:
            start = max(start, ZoomLevels.fractional_zoom(rule.minimumScale()))
        if rule.maximumScale() > 0:
            end = min(end, ZoomLevels.fractional_zoom(rule.maximumScale()))
        return start, max(start, end)

    def _copy_data_driven_properties(self, source_obj, target_obj):
        """Copy all active data-driven properties from source to target object."""
        source_props = source_obj.dataDefinedProperties()
        target_props = target_obj.dataDefinedProperties()
        for prop_key in source_obj.propertyDefinitions():
            prop = source_props.property(prop_key)
            if prop.isActive():
                target_props.setProperty(prop_key, prop)
                target_props.property(prop_key).setActive(True)

    def _apply_styles_to_layer(self):
        """Assign collected renderer and labeling styles to the tiles layer."""
        renderer = QgsVectorTileBasicRenderer()
        renderer.setStyles(self.renderer_styles)

        labeling = QgsVectorTileBasicLabeling()
        labeling.setStyles(self.labeling_styles)

        self.tiles_layer.setRenderer(renderer)
        self.tiles_layer.setLabeling(labeling)
