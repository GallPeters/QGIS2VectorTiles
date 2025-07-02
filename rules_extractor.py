from qgis.core import *


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


class RuleExtractor:
    """
    Compact PyQGIS class that extracts and flattens all renderer and labeling rules
    from vector lyrs in the current QGIS project using unified processing logic.
    """

    def __init__(self):
        self.project = QgsProject.instance()
        self.rules_types = ["renderer", "labeling"]
        self.extracted_rules = {}

    def extract_all_rules(self):
        """Extract all rules from all vector lyrs in the project."""

        for index, lyr in enumerate(self.project.mapLayers().values()):
            rules_dict = {}
            if lyr.type() == 0 and lyr.geometryType() != 4:
                try:
                    # Process both renderer and labeling using unified approach
                    for rule_type in self.rules_types:
                        rule_based = self._get_rule_based_system(lyr, rule_type)
                        lyr_name = lyr.name() if lyr.name() else "unnamed"
                        root_rule = rule_based.rootRule()
                        root_rule.setDescription(f"lyr: {index} ({lyr_name})")
                        if rule_based and root_rule:
                            rules = []
                            for rule in self._flatten_rules(root_rule, rule_type):
                                rules.extend(self._split_rule_by_scales(rule))
                            rules_dict[rule_type] = rules
                    self.extracted_rules[lyr] = rules_dict
                except Exception as e:
                    print(f"Error processing lyr {lyr.name()}: {str(e)}")
        return self.extracted_rules

    def print_rules(self):
        """Print extracted rules with elegant tree structure."""
        print("\n -------- Extracted rules --------\nProject: ")

        def get_tree_chars(is_last):
            return (" └─ ", "    ") if is_last else (" ├─ ", " │ ")

        for lyr_idx, (lyr, rules_dict) in enumerate(self.extracted_rules.items()):
            lyr_last = lyr_idx == len(self.extracted_rules) - 1
            lyr_conn, lyr_pre = get_tree_chars(lyr_last)
            print(f"{lyr_conn}layer {lyr_idx + 1}: {lyr.name()}")

            for type_idx, (rule_type, rules) in enumerate(rules_dict.items()):
                type_last = type_idx == len(rules_dict) - 1
                type_conn, type_pre = get_tree_chars(type_last)
                print(f"{lyr_pre}{type_conn}{rule_type}:")

                for rule_idx, rule in enumerate(rules):
                    rule_last = rule_idx == len(rules) - 1
                    rule_conn, rule_pre = get_tree_chars(rule_last)
                    rule_prefix = f"{lyr_pre}{type_pre}"
                    print(f"{rule_prefix}{rule_conn}rule {rule_idx + 1}:")

                    details = [
                        ("name", rule.description()),
                        ("minscale", rule.minimumScale()),
                        ("maxscale", rule.maximumScale()),
                        ("filter", rule.filterExpression()),
                    ]

                    for val_idx, (label, value) in enumerate(details):
                        val_conn = " └─ " if val_idx == 3 else " ├─ "
                        print(f"{rule_prefix}{rule_pre}{val_conn}{label}: {value}")

        print("\n -------- Extraction has been finished successfully --------")

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

    def _flatten_rules(self, rule, rule_type):
        """Unified rule flattening for both renderer and labeling."""
        flattened = []
        if rule.parent():
            clone = rule.clone()
            self._inherit_rule_name(rule, clone, rule_type, len(flattened))
            self._inherit_rule_filter(rule, clone)
            self._inherit_scale(rule, clone, max)
            self._inherit_scale(rule, clone, min)
            self._inherit_rule_data(rule, clone, rule_type)

            # Split if needed and append flattened rule
            if rule_type == "renderer":
                flattened.extend(self._split_rule_by_symbol_layers(clone))
            else:
                flattened.append(clone)
        else:
            clone = rule
        # Process children recursively or add leaf rule
        if clone.children():
            for child in clone.children():
                if child.active():
                    flattened.extend(self._flatten_rules(child, rule_type))
        return flattened

    def _inherit_rule_name(self, rule, clone, rule_type, index):
        """Inherit rule-specific name (label or description)."""
        name_attr = (
            "description"
            if not rule.parent().parent() or rule_type == "labeling"
            else "label"
        )
        rule_name = getattr(rule, name_attr)()
        if not rule_name:
            rule_name = "unnamed"
        parent_name = getattr(rule.parent(), name_attr)()
        prefix = f"{parent_name}" if parent_name else ""
        clone.setDescription(f"{prefix} > rule: {index} ({rule_name})")

    def _inherit_rule_filter(self, rule, clone):
        """Combine filters using AND logic."""
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()

        if not parent_filter and not rule_filter:
            clone_filter = ""
        elif not parent_filter:
            clone_filter = rule_filter
        elif not rule_filter:
            clone_filter = parent_filter
        else:
            clone_filter = f"({parent_filter}) AND ({rule_filter})"
        clone.setFilterExpression(clone_filter)

    def _inherit_scale(self, rule, clone, comparator):
        """Inherit scale using comparator function (min/max)."""
        rule_scale = getattr(rule, f"{comparator.__name__}imumScale")()
        parent_scale = getattr(rule.parent(), f"{comparator.__name__}imumScale")()
        if rule_scale > 0 and parent_scale > 0:
            clone_scale = comparator(rule_scale, parent_scale)
        clone_scale = rule_scale if rule_scale > 0 else parent_scale
        setattr(clone, f"set{comparator.__name__.capitalize()}imumScale", clone_scale)

    def _inherit_rule_data(self, rule, clone, rule_type):
        """Inherit rule-specific data (symbol or settings)."""
        if rule_type == "renderer":
            clone_symbol = clone.symbol()
            parent_symbol = rule.parent().symbol()
            if parent_symbol and clone_symbol:
                for index in range(parent_symbol.symbolLayerCount()):
                    symbol_lyr = parent_symbol.symbolLayer(index).clone()
                    clone_symbol.appendSymbolLayer(symbol_lyr)

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
            clone = rule.clone()
            clone.setMinimumScale(ZoomLevels.snap_scale_to_level(minscale, True))
            clone.setMaximumScale(ZoomLevels.snap_scale_to_level(maxscale, False))
            return [clone]

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
    print(f"Printing rules...")
    extractor.print_rules()
