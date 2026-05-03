"""
tiles_styler.py

TilesStyler — takes the list of FlattenedRules produced by RulesFlattener /
RulesExporter and applies matching QgsVectorTileBasicRenderer /
QgsVectorTileBasicLabeling styles to a new QgsVectorTileLayer, then saves
the layer style to a QLR file.

Depends on: config, flattened_rule
"""

from os import makedirs
from os.path import join
from typing import List, Optional

from qgis.core import (
    QgsProject,
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
    QgsLayerDefinition,
    QgsLabelThinningSettings,
    QgsProcessingUtils,
    Qgis,
)

from ..utils.config import _REMOVE_DUPLICATES_DISTANCE
from ..utils.flattened_rule import FlattenedRule


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

    def __init__(self, flattened_rules: List[FlattenedRule], output_dir: str, tiles_path: str):
        self.flattened_rules = flattened_rules
        self.output_dir = output_dir or QgsProcessingUtils.tempFolder()
        self.tiles_layer = self._create_tiles_layer(tiles_path)
        self.renderer_styles: List[QgsVectorTileBasicRendererStyle] = []
        self.labeling_styles: List[QgsVectorTileBasicLabelingStyle] = []

    def apply_styling(self) -> QgsVectorTileLayer:
        """Apply all rule styles to the tiles layer and save the QLR definition."""
        for rule in reversed(self.flattened_rules):
            self._create_style_from_rule(rule)

        self._apply_styles_to_layer()
        self._save_style()
        return self.tiles_layer

    def _create_tiles_layer(self, tiles_path: Optional[str]) -> QgsVectorTileLayer:
        """Create a vector tile layer and insert it at the top of the project legend."""
        layer = QgsVectorTileLayer(tiles_path, "Vector Tiles")
        layer = QgsProject.instance().addMapLayer(layer, False)
        QgsProject.instance().layerTreeRoot().insertLayer(0, layer)
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
        if not symbol:
            return

        symbol_layer = symbol.symbolLayers()[-1]
        sub_symbol = symbol_layer.subSymbol()
        sub_symbol_layer = sub_symbol.symbolLayers()[-1] if sub_symbol else None

        source_geom = int(flat_rule.get_attr("g"))
        target_geom = int(flat_rule.get_attr("c"))

        if source_geom != target_geom:
            if sub_symbol and symbol_layer.layerType() in ("GeometryGenerator", "CentroidFill"):
                self._copy_data_driven_properties(symbol, sub_symbol)
                self._copy_data_driven_properties(symbol_layer, sub_symbol_layer)
                symbol = sub_symbol
            else:
                symbol = self._create_transformed_symbol(target_geom, symbol_layer)

        style.setSymbol(symbol.clone())

    def _create_transformed_symbol(self, target_geom: int, symbol_layer):
        """Create a new symbol of the target geometry type from the given symbol layer."""
        symbol_class = self._SYMBOL_CLASS_MAP.get(target_geom, QgsMarkerSymbol)
        symbol = symbol_class()
        symbol.appendSymbolLayer(symbol_layer.clone())
        symbol.deleteSymbolLayer(0)
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
        style.setStyleName(flat_rule.rule.description())
        style.setMinZoomLevel(flat_rule.get_attr("o"))
        style.setMaxZoomLevel(flat_rule.get_attr("i"))
        geom_code = flat_rule.get_attr("c")
        style.setGeometryType(
            self._GEOM_TYPE_MAP.get(geom_code, QgsWkbTypes.PointGeometry)
        )

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

    def _save_style(self):
        """Export the tiles layer definition to a QLR file."""
        style_dir = join(self.output_dir, "style")
        makedirs(style_dir, exist_ok=True)
        qlr_path = join(style_dir, "tiles.qlr")
        node = QgsProject.instance().layerTreeRoot().findLayer(self.tiles_layer.id())
        QgsLayerDefinition().exportLayerDefinition(qlr_path, [node])
