"""
Extracts and flattens renderer and labeling rules from vector layers in the
current QGIS project with unified processing logic.
"""

from dataclasses import dataclass
from typing import List, Union, Optional, Callable
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsRuleBasedRenderer,
    QgsRuleBasedLabeling,
    QgsPalLayerSettings,
    QgsSymbol,
)


class ZoomLevels:
    """Manages predefined zoom levels and scale snapping."""

    LEVELS = [
        591657528,
        295828764,
        147914382,
        73957191,
        36978595,
        18489298,
        9244649,
        4622324,
        2311162,
        1155581,
        577791,
        288895,
        144448,
        72224,
        36112,
        18056,
        9028,
        4514,
        2257,
        1128,
        0.433333333333333,
        0.2375,
        0.139583333333333,
        0.090972222222222,
    ]

    @classmethod
    def snap_to_level(cls, scale, snap_up=True) -> float:
        """Snap scale to the nearest zoom level."""
        if scale <= 0:
            return cls.LEVELS[-1] if snap_up else cls.LEVELS[0]

        for i, level in enumerate(cls.LEVELS):
            if scale >= level:
                if i == 0 or not snap_up:
                    return level
                return level if snap_up else cls.LEVELS[i - 1]

        return cls.LEVELS[-1]


@dataclass
class FlatRule:
    """Represents a flattened rule with all inherited properties."""

    name: str
    flat_rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule]
    original_rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule]
    layer: QgsVectorLayer
    rule_type: str
    min_scale: float
    max_scale: float
    data: Union[QgsSymbol, QgsPalLayerSettings]


class RuleExtractor:
    """Extracts and flattens QGIS vector layer rules."""

    RULE_TYPES = ["renderer", "labeling"]

    def __init__(self):
        self.project = QgsProject.instance()
        self.layer_tree = self.project.layerTreeRoot()
        self.extracted_rules = []

    def extract_all_rules(self) -> List[FlatRule]:
        """Extract all rules from visible vector layers."""
        layers = self.project.mapLayers().values()

        for idx, layer in enumerate(layers):
            if not self._is_relevant_layer(layer):
                continue

            try:
                self._process_layer_rules(layer, idx)
            except Exception as e:
                raise (e)
                print(f"Error processing layer '{layer.name()}': {e}")

        return self.extracted_rules

    def _is_relevant_layer(self, layer) -> bool:
        """Check if layer is a visible vector layer."""
        is_vector = layer.type() == 0 and layer.geometryType() != 4
        is_visible = self.layer_tree.findLayer(layer.id()).isVisible()
        return is_vector and is_visible

    def _process_layer_rules(self, layer, layer_idx):
        """Process both renderer and labeling rules for a layer."""
        for rule_type in self.RULE_TYPES:
            rule_system = self._get_rule_system(layer, rule_type)
            if not rule_system:
                continue

            root_rule = self._get_root(rule_system, layer, layer_idx, rule_type)
            if root_rule:
                self._flatten_rules(root_rule, rule_type, layer)

    def _get_rule_system(self, layer, rule_type):
        """Get or convert layer system to rule-based."""
        system = layer.renderer() if rule_type == "renderer" else layer.labeling()

        if not system:
            return None

        # Return if already rule-based
        if isinstance(system, (QgsRuleBasedRenderer, QgsRuleBasedLabeling)):
            return system

        # Convert to rule-based
        if rule_type == "renderer":
            return QgsRuleBasedRenderer.convertFromRenderer(system)
        else:
            rule = QgsRuleBasedLabeling.Rule(system.settings())
            root = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
            root.appendChild(rule)
            return QgsRuleBasedLabeling(root)

    def _get_root(self, rule_system, layer, layer_idx, rule_type):
        """Prepare root rule with descriptive information."""
        root_rule = rule_system.rootRule()
        layer_name = layer.name() or "unnamed"
        description = f"layer: {layer_idx} ({layer_name}) > type: {rule_type}"
        root_rule.setDescription(description)
        return root_rule

    def _flatten_rules(self, rule, rule_type, layer, index=None) -> List:
        """Recursively flatten rules with inheritance."""
        flattened = []

        if rule.parent():
            clone = self._inherit_properties(rule, rule_type, index)

            # Split by symbol layers (renderer only) then by scales
            if rule_type == "renderer":
                splitted_rules = self._split_by_symbol_layers(clone)
            else:
                splitted_rules = [clone]

            for split_rule in splitted_rules:
                flattened.extend(self._split_by_scales(split_rule))

        # Generate flat rules
        for flat in flattened:
            self._create_flat_rule(flat, rule, rule_type, layer)

        # Process children recursively
        for index, child in enumerate(rule.children()):
            if child.active():
                self._flatten_rules(child, rule_type, layer, index)

        return flattened

    def _inherit_properties(self, rule, rule_type, rule_index):
        """Inherit all properties from parent rule."""
        clone = rule.clone()
        self._inherit_name(clone, rule, rule_type, rule_index)
        self._inherit_filter(clone, rule)
        self._inherit_scale(clone, rule, min)
        self._inherit_scale(clone, rule, max)
        if rule_type == "renderer":
            self._inherit_symbol(clone, rule)
        return clone

    def _inherit_name(self, clone, rule, rule_type, rule_index):
        """Inherit and combine rule names."""
        parent = rule.parent()
        is_root_child = not parent.parent()

        attr = "description" if is_root_child or rule_type == "renderer" else "label"

        rule_name = getattr(rule, attr)() or "unnamed"
        parent_name = getattr(parent, attr)() or ""

        prefix = f"{parent_name} > " if parent_name else ""
        clone.setDescription(f"{prefix}rule: {rule_index} ({rule_name})")

    def _inherit_filter(self, clone, rule):
        """Combine parent and rule filters with AND logic."""
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()

        if parent_filter and rule_filter:
            combined_filter = f"({parent_filter}) AND ({rule_filter})"
        else:
            combined_filter = parent_filter or rule_filter or ""

        clone.setFilterExpression(combined_filter)

    def _inherit_scale(self, clone, rule, comparator):
        """Inherit scale limits using min/max comparator."""
        scale_attr = f"{comparator.__name__}imumScale"
        rule_scale = getattr(rule, scale_attr)()
        parent_scale = getattr(rule.parent(), scale_attr)()

        if rule_scale > 0 and parent_scale > 0:
            inherited_scale = comparator(rule_scale, parent_scale)
        else:
            inherited_scale = rule_scale if rule_scale > 0 else parent_scale

        setter_name = f"set{comparator.__name__.capitalize()}imumScale"
        getattr(clone, setter_name)(inherited_scale)

    def _inherit_symbol(self, clone, rule):
        """Inherit symbol layers from parent."""
        clone_symbol = clone.symbol()
        parent_symbol = rule.parent().symbol()

        if parent_symbol and clone_symbol:
            for i in range(parent_symbol.symbolLayerCount()):
                symbol_layer = parent_symbol.symbolLayer(i).clone()
                clone_symbol.appendSymbolLayer(symbol_layer)

    def _split_by_symbol_layers(self, rule) -> List:
        """Split rule by individual symbol layers."""
        if not rule.symbol() or rule.symbol().symbolLayerCount() <= 1:
            return [rule]

        layer_count = rule.symbol().symbolLayerCount()
        split_rules = []

        for keep_idx in range(layer_count):
            clone = rule.clone()
            desc = f"{clone.description()} > symbol_layer: {keep_idx}"
            clone.setDescription(desc)

            # Remove all layers except the one to keep
            for remove_idx in reversed(range(layer_count)):
                if remove_idx != keep_idx:
                    clone.symbol().deleteSymbolLayer(remove_idx)

            split_rules.append(clone)

        return split_rules

    def _split_by_scales(self, rule) -> List:
        """Split rule by zoom levels if scale-dependent."""
        filter_expr = rule.filterExpression()
        min_scale = rule.minimumScale()
        max_scale = rule.maximumScale()

        # Handle scale-independent rules
        if "@map_scale" not in filter_expr:
            rule.setMinimumScale(ZoomLevels.snap_to_level(min_scale, True))
            rule.setMaximumScale(ZoomLevels.snap_to_level(max_scale, False))
            return [rule]

        # Split scale-dependent rules by zoom levels
        rule_lvls = [lvl for lvl in ZoomLevels.LEVELS if max_scale <= lvl <= min_scale]
        rules = []

        for i, level in enumerate(rule_lvls):
            clone = rule.clone()
            clone.setMinimumScale(level)

            next_max = rule_lvls[i + 1] if i + 1 < len(rule_lvls) else max_scale
            clone.setMaximumScale(next_max)

            # Replace scale variable in filter
            new_filter = filter_expr.replace("@map_scale", str(level))
            clone.setFilterExpression(new_filter)

            desc = f"{rule.description()} > subscale: {round(level, 2)}"
            clone.setDescription(desc)
            rules.append(clone)

        return rules

    def _create_flat_rule(self, flat_rule, original_rule, rule_type, layer):
        """Create and store a FlatRule instance."""
        data = flat_rule.symbol() if rule_type == "renderer" else flat_rule.settings()

        flat = FlatRule(
            name=flat_rule.description(),
            flat_rule=flat_rule,
            original_rule=original_rule,
            layer=layer,
            rule_type=rule_type,
            min_scale=flat_rule.minimumScale(),
            max_scale=flat_rule.maximumScale(),
            data=data,
        )
        self.extracted_rules.append(flat)


def main():
    """Console execution entry point."""
    extractor = RuleExtractor()
    print("Extracting rules from current QGIS project...")

    rules = extractor.extract_all_rules()
    print(f"Successfully extracted {len(rules)} rules")

    for rule in rules:
        print(f"  - {rule.name}")


if __name__ == "__console__":
    main()
