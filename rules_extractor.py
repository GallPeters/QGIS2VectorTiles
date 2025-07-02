from qgis.core import *
from dataclasses import *
from typing import *


class ZoomLevels:
    """Provides predefined zoom levels and scale snapping functionality."""

    levels = [
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
    def snap_scale_to_level(cls, scale, snap_up=True):
        """Snap scale to nearest zoom level."""
        if scale <= 0:
            return cls.levels[-1] if snap_up else cls.levels[0]

        # Find the closest level
        for i, level in enumerate(cls.levels):
            if scale >= level:
                if i == 0 or not snap_up:
                    return level
                # Return current level if snapping up, previous if snapping down
                return level if snap_up else cls.levels[i - 1]

        # Scale is smaller than all levels
        return cls.levels[-1]


@dataclass
class FlatRule:
    name: str
    flat: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule]
    orig: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule]
    layer: QgsVectorLayer
    container: str
    minscale: int
    maxscale: int
    data: Union[QgsSymbol, QgsPalLayerSettings]


class RuleExtractor:
    """
    Compact PyQGIS class that extracts and flattens all renderer and labeling rules
    from vector lyrs in the current QGIS project using unified processing logic.
    """

    def __init__(self):
        self.project = QgsProject.instance()
        self.root = QgsProject.instance().layerTreeRoot()
        self.rules_types = ["renderer", "labeling"]
        self.extracted_rules = []

    def extract_all_rules(self):
        """Extract all rules from all vector lyrs in the project."""
        for index, lyr in enumerate(self.project.mapLayers().values()):
            if self.lyr_is_relevant(lyr):
                try:
                    # Process both renderer and labeling using unified approach
                    for rule_type in self.rules_types:
                        rule_based = self._get_rule_based_system(lyr, rule_type)
                        root_rule = self._get_root(rule_based, rule_type, lyr, index)
                        if rule_based and root_rule:
                            self._flatten_rules(root_rule, rule_type, lyr)
                except Exception as e:
                    print(f"Error processing lyr {lyr.name()}: {str(e)}")
        return self.extracted_rules

    def lyr_is_relevant(self, lyr):
        is_vector = lyr.type() == 0 and lyr.geometryType() != 4
        is_visible = self.root.findLayer(lyr.id()).isVisible()
        return is_vector and is_visible

    def _get_root(self, rule_based, rule_type, lyr, index):
        """Get root rule."""
        lyr_name = lyr.name() if lyr.name() else "unnamed"
        root_rule = rule_based.rootRule()
        desc = f"lyr: {index} ({lyr_name}) > rule_type: {rule_type}"
        root_rule.setDescription(desc)
        return root_rule

    def _get_rule_based_system(self, lyr, rule_type):
        """Get or convert to rule-based system for both renderer and labeling."""
        system = lyr.renderer() if rule_type == "renderer" else lyr.labeling()
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

    def _flatten_rules(self, rule, rule_type, lyr):
        """Unified rule flattening for both renderer and labeling."""
        flattened = []
        if rule.parent():
            self._inherit_rule_name(rule, rule_type, len(flattened))
            self._inherit_rule_filter(rule)
            self._inherit_scale(rule, max)
            self._inherit_scale(rule, min)
            self._inherit_rule_data(rule, rule_type)

            # Split if needed and append flattened rule
            splitted = []
            if rule_type == "renderer":
                splitted.extend(self._split_rule_by_symbol_layers(rule))
            else:
                splitted.append(rule)
            for splitted_rule in splitted:
                flattened.extend(self._split_rule_by_scales(splitted_rule))
        self._generate_flat_rule(flattened, rule, rule_type, lyr)
        # Process children recursively or add leaf rule
        if rule.children():
            for child in rule.children():
                if child.active():
                    flattened.extend(self._flatten_rules(child, rule_type, lyr))
        return flattened

    def _generate_flat_rule(self, flattened, rule, rule_type, lyr):
        for flat in flattened:
            flat_rule = FlatRule(
                flat.description(),
                flat,
                rule,
                lyr,
                rule_type,
                flat.minimumScale(),
                flat.maximumScale(),
                flat.symbol() if rule_type == "renderer" else flat.settings(),
            )
            self.extracted_rules.append(flat_rule)

    def _inherit_rule_name(self, rule, rule_type, index):
        """Inherit rule-specific name (label or description)."""
        parent_is_root = rule.parent().parent()
        if not parent_is_root or rule_type == "labeling":
            attr = "description"
        else:
            attr = "label"
        rule_name = getattr(rule, attr)()
        if not rule_name:
            rule_name = "unnamed"
        parent_name = getattr(rule.parent(), attr)()
        prefix = f"{parent_name}" if parent_name else ""
        rule.setDescription(f"{prefix} > rule: {index} ({rule_name})")

    def _inherit_rule_filter(self, rule):
        """Combine filters using AND logic."""
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()

        if not parent_filter and not rule_filter:
            rule_filter = ""
        elif not parent_filter:
            rule_filter = rule_filter
        elif not rule_filter:
            rule_filter = parent_filter
        else:
            rule_filter = f"({parent_filter}) AND ({rule_filter})"
        rule.setFilterExpression(rule_filter)

    def _inherit_scale(self, rule, comparator):
        """Inherit scale using comparator function (min/max)."""
        rule_scale = getattr(rule, f"{comparator.__name__}imumScale")()
        parent_scale = getattr(rule.parent(), f"{comparator.__name__}imumScale")()
        if rule_scale > 0 and parent_scale > 0:
            rule_scale = comparator(rule_scale, parent_scale)
        rule_scale = rule_scale if rule_scale > 0 else parent_scale
        setattr(rule, f"set{comparator.__name__.capitalize()}imumScale", rule_scale)

    def _inherit_rule_data(self, rule, rule_type):
        """Inherit rule-specific data (symbol or settings)."""
        if rule_type == "renderer":
            rule_symbol = rule.symbol()
            parent_symbol = rule.parent().symbol()
            if parent_symbol and rule_symbol:
                for index in range(parent_symbol.symbolLayerCount()):
                    symbol_lyr = parent_symbol.symbolLayer(index).clone()
                    rule_symbol.appendSymbolLayer(symbol_lyr)

    def _split_rule_by_symbol_layers(self, rule):
        """Split a rule by symbol layers or return original if symbol contains a single layer."""
        if not rule.symbol():
            return [rule]

        layer_count = rule.symbol().symbolLayerCount()
        if layer_count <= 1:
            return [rule]

        splitted_rules = []
        for keep_idx in range(layer_count):
            clone = rule.clone()
            clone.setDescription(f"{clone.description()} > symlayer: {keep_idx}")

            # Remove all layers except the one we want to keep
            for remove_idx in reversed(range(layer_count)):
                if remove_idx != keep_idx:
                    clone.symbol().deleteSymbolLayer(remove_idx)

            splitted_rules.append(clone)

        return splitted_rules

    def _split_rule_by_scales(self, rule):
        """Split a rule by scale levels or return original if scale-independent."""
        filter_exp = rule.filterExpression()
        minscale = rule.minimumScale()
        maxscale = rule.maximumScale()

        # Scale-independent rules: just snap the scales
        if "@map_scale" not in filter_exp:
            rule.setMinimumScale(ZoomLevels.snap_scale_to_level(minscale, True))
            rule.setMaximumScale(ZoomLevels.snap_scale_to_level(maxscale, False))
            return [rule]

        # Scale-dependent rules: create rule for each applicable zoom level
        rules = []
        rule_lvls = [lvl for lvl in ZoomLevels.levels if maxscale <= lvl <= minscale]

        for i, lvl in enumerate(rule_lvls):
            clone = rule.clone()
            clone.setMinimumScale(lvl)
            new_maxscale = rule_lvls[i + 1] if i + 1 < len(rule_lvls) else maxscale
            clone.setMaximumScale(new_maxscale)
            clone.setFilterExpression(filter_exp.replace("@map_scale", str(lvl)))
            clone.setDescription(f"{rule.description()} > subscale: {round(lvl, 2)}")
            rules.append(clone)
        return rules


# Console execution
if __name__ == "__console__":
    extractor = RuleExtractor()
    print("Extracting rules from current QGIS project...")
    rules = extractor.extract_all_rules()
    print(f"Finished successfully")
    for flat_rule in extractor.extracted_rules:
        print(flat_rule.name)
