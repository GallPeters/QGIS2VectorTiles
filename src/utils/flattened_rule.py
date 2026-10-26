"""
flattened_rule.py

FlattenedRule — shared data-transfer object (dataclass) that carries a
flattened QGIS rule together with its source layer and export dataset path.
Imported by RulesFlattener, RulesExporter, and TilesStyler.
"""

from dataclasses import dataclass
from typing import Optional, Union

from qgis.core import QgsRuleBasedRenderer, QgsRuleBasedLabeling, QgsSymbolLayer, QgsVectorLayer

from .config import _FIELD_PREFIX

# Marker line placements exported as points ("p" rule attribute): one rule
# per placement, its markers drawn at the points.
MARKER_FIRST_VERTEX = 1
MARKER_LAST_VERTEX = 2
MARKER_CENTRAL_POINT = 3
MARKER_ALL_VERTICES = 4
MARKER_INNER_VERTICES = 5
MARKER_SEGMENT_CENTERS = 6

# An outline line layer made from an offset fill (RulesFlattener) carries
# the fill's offset, as JSON, in an inactive data-defined property of this
# key (a fill colour, which a line never reads; inactive, so not exported
# as a field): the style converter draws the outline translated by it, as
# the fill. Its id would be simpler, but QGIS gives symbol layers new ids.
OFFSET_PROPERTY = QgsSymbolLayer.Property.PropertyFillColor

# A layer drawn as a heatmap becomes one rule whose (placeholder) marker
# carries the heatmap's settings, as JSON, in an inactive data-defined
# property of this key (its shape name, which the placeholder never reads);
# its size property, when active, is the weight. The style converter draws
# it as a MapLibre heatmap layer.
HEATMAP_PROPERTY = QgsSymbolLayer.Property.PropertyName

# A point cluster renderer becomes its own rules, for the points drawn
# alone, plus one drawing every point with the cluster symbol, whose symbol
# layers carry the clustering distance, as JSON, in an inactive
# data-defined property of this key (its "enabled", which a rule's symbol
# layers never read inactive). The tiles then hold, per zoom level, one point
# per cluster and the points left alone (GDALTilesGenerator).
CLUSTER_PROPERTY = QgsSymbolLayer.Property.PropertyLayerEnabled
# The number of points in a cluster, in the tiles.
CLUSTER_SIZE_FIELD = f"{_FIELD_PREFIX}_cluster_size"

# The style's metadata key holding, by dataset, the tile layer of another
# dataset it is written into (the style converter draws both as one layer).
TILE_LAYERS_KEY = "q2vt:tile-layers"


def cluster_settings(symbol) -> "dict | None":
    """The clustering settings a symbol carries (see CLUSTER_PROPERTY)."""
    import json  # pylint: disable=import-outside-toplevel
    try:
        prop = symbol.symbolLayer(0).dataDefinedProperties().property(CLUSTER_PROPERTY)
        if prop is None or prop.isActive() or not prop.staticValue():
            return None
        data = json.loads(prop.staticValue())
        return data if isinstance(data, dict) and "tolerance" in data else None
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return None


def pinned_field(label_field: str) -> str:
    """The dataset field flagging (1) the labels of a label text field
    placed at their data-defined position (X, Y)."""
    return label_field.replace(f"{_FIELD_PREFIX}_label", f"{_FIELD_PREFIX}_pinned", 1)


def marker_angle_field(symbol_layer_index: int) -> str:
    """The dataset field holding a marker line's per-point rotation."""
    return f"{_FIELD_PREFIX}_marker_angle_{int(symbol_layer_index):02d}"


@dataclass
class FlattenedRule:
    """A flattened rule with inherited properties from parent hierarchy."""

    rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule]
    layer: QgsVectorLayer
    output_dataset: Optional[str] = ""
    # In a dataset shared by rules differing by filter: this rule's number
    # (its rows' q2vt_rule value); None when the dataset is its own.
    member: Optional[int] = None
    # A label rule split by a renderer rule: that renderer rule, and the
    # label's own filter (without the renderer's). Such a label can be drawn
    # from the renderer rule's rows (see RulesExporter._attach_label).
    renderer: Optional["FlattenedRule"] = None
    label_filter: Optional[str] = None
    # A rule split into zoom bands by its @map_scale expressions: the number
    # its bands share (they may share a dataset, see RulesExporter).
    band_group: Optional[int] = None
    # A zoom band starting between two zoom levels (where its results start):
    # the zoom whose scale its @map_scale expressions stand for; None when
    # that is its first zoom level ("o").
    scale_zoom: Optional[int] = None
    # A label placed at its data-defined position: the X and Y expressions
    # (before the export rewrites the properties to read exported fields).
    label_position: Optional[tuple] = None

    def map_scale(self) -> str:
        """The scale @map_scale stands for in this rule's expressions."""
        from .zoom_levels import ZoomLevels  # pylint: disable=import-outside-toplevel
        zoom = self.scale_zoom if self.scale_zoom is not None else self.get_attr("o")
        return str(ZoomLevels.zoom_to_scale(zoom))

    def get_attr(self, char: str) -> Optional[int]:
        """Extract rule attribute from description by character prefix."""
        desc = self.rule.description()
        start = desc.find(char) + 1
        if start == 0:
            return None
        return int(desc[start : start + 2])

    def set_attr(self, char: str, value: int, geom_generator=None):
        """Set rule attribute in description."""
        value = int(value)
        new_attr = f"{char}{value:02d}"
        current = self.get_attr(char)

        desc = self.rule.description()
        if current is not None:
            old_attr = f"{char}{current:02d}"
            desc = desc.replace(old_attr, new_attr)
        else:
            desc = f"{desc}{new_attr}"

        self.rule.setDescription(desc)
        self.output_dataset = desc

        i = desc.find("s")
        if i >= 0 and geom_generator:
            self.output_dataset = self.output_dataset.replace(desc[i : i + 3], "s00")

    def get_description(self):
        """Construct rule description for labeling or renderer rule."""
        geom_desc = {0:'point', 1:'line', 2:'polygon', 3:'unknown'}
        lyr_name = f'layer: {self.layer.name() or self.layer.id()}'
        rule_type = f'type: {"symbology" if self.get_attr("t") == 0 else "labeling"}'
        rule_index = f'index: level {self.get_attr("d")} number {self.get_attr("r") + 1}'
        source_geom = geom_desc.get(self.get_attr("g"))
        target_geom = geom_desc.get(self.get_attr("c"))
        if source_geom == target_geom:
            geom_desc_str = 'conversion: none'
        else:
            geom_desc_str = f'conversion: {source_geom} to {target_geom}'
        return  "'{" + f'{lyr_name},{rule_type},{rule_index},{geom_desc_str}' + "}'"